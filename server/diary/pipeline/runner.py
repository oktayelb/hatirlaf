"""Runs :data:`diary.pipeline.stages.PIPELINE` for a session.

Threading model is unchanged from the lean-MVP brief: one daemon thread per
session, no Celery, no Redis. The ``Session`` row is the progress tracker.

The runner is deliberately ignorant of what the stages do. It owns
re-entrancy, thread lifetime, and failure bookkeeping; the stage list owns
the work and the feature gating.
"""

from __future__ import annotations

import logging
import threading
import traceback

from django.conf import settings

from ..models import Edge, EncounteredEntity, Session, SessionStatus
from ..services import session_pipeline as session_pipeline_mod
from . import flags
from .stages import PipelineAbort, StageContext, active_stages, stage

logger = logging.getLogger(__name__)

_active_lock = threading.Lock()
_active_processing: set[int] = set()
_active_eventification: set[int] = set()
_bulk_reextract_lock = threading.Lock()


# --- Re-entrancy -------------------------------------------------------------


def is_processing_active(session_id: int) -> bool:
    with _active_lock:
        return session_id in _active_processing


def is_eventification_active(session_id: int) -> bool:
    with _active_lock:
        return session_id in _active_eventification


def _try_mark_active(active_set: set[int], session_id: int) -> bool:
    with _active_lock:
        if session_id in active_set:
            return False
        active_set.add(session_id)
        return True


def _clear_active(active_set: set[int], session_id: int) -> None:
    with _active_lock:
        active_set.discard(session_id)


# --- Entry points ------------------------------------------------------------


def kickoff(session_id: int) -> None:
    """Process a freshly uploaded Session."""
    _spawn(run, session_id, name=f"hatirlaf-{session_id}")


def kickoff_transcript_reprocess(session_id: int) -> None:
    """Rebuild from the saved transcript without re-running speech-to-text."""
    _spawn(run_transcript_reprocess, session_id, name=f"hatirlaf-reparse-{session_id}")


def kickoff_eventification(session_id: int) -> None:
    """Run the deferred understanding stages on their own worker."""
    if not flags.nlp_enabled():
        return
    _spawn(
        run_eventification,
        session_id,
        name=f"hatirlaf-eventify-{session_id}",
        allow_sync=False,
    )


def _spawn(target, session_id: int, *, name: str, allow_sync: bool = True) -> None:
    if allow_sync and getattr(settings, "HATIRLAF_SYNC_PROCESSING", False):
        target(session_id)
        return
    threading.Thread(target=target, args=(session_id,), daemon=True, name=name).start()


# --- Runs --------------------------------------------------------------------


def run(session_id: int) -> None:
    _run_chain(session_id, reuse_transcript=False)


def run_transcript_reprocess(session_id: int) -> None:
    _run_chain(session_id, reuse_transcript=True)


def _run_chain(
    session_id: int,
    *,
    reuse_transcript: bool,
    queue_eventification: bool = True,
) -> None:
    if not _try_mark_active(_active_processing, session_id):
        logger.info("Session %s is already being processed; skipping duplicate.", session_id)
        return
    try:
        session = _load(session_id, "processing")
        if session is None:
            return
        _execute(
            StageContext(
                session=session,
                reuse_transcript=reuse_transcript,
                queue_eventification=queue_eventification,
            )
        )
    finally:
        _clear_active(_active_processing, session_id)


def _execute(ctx: StageContext) -> None:
    """Walk the synchronous stages, then hand off the deferred ones."""
    session_id = ctx.session.id
    try:
        for step in active_stages(deferred=False):
            logger.debug("Session %s → stage %s", session_id, step.key)
            step.run(ctx)
    except PipelineAbort as abort:
        _fail(ctx.session, str(abort))
        return
    except Exception:  # pragma: no cover - defensive
        detail = traceback.format_exc()
        logger.error("Session %s failed: %s", session_id, detail)
        session_pipeline_mod.mark_session_failed(session_id, detail)
        return

    if ctx.queue_eventification and active_stages(deferred=True):
        kickoff_eventification(session_id)


def run_eventification(session_id: int) -> None:
    """Execute the deferred stages (currently just the LLM eventifier)."""
    deferred = active_stages(deferred=True)
    if not deferred:
        return
    if not _try_mark_active(_active_eventification, session_id):
        logger.info("Session %s eventification already active; skipping duplicate.", session_id)
        return
    try:
        session = _load(session_id, "eventification")
        if session is None:
            return
        if not (session.transcript or "").strip():
            session_pipeline_mod.mark_eventification_empty(session_id)
            return

        session_pipeline_mod.mark_eventification_running(session_id)
        ctx = StageContext(session=session)
        try:
            for step in deferred:
                step.run(ctx)
        except Exception as exc:  # pragma: no cover - stage is isolated
            logger.exception("Eventification of session %s failed: %s", session_id, exc)
            session_pipeline_mod.mark_eventification_failed(session_id, str(exc))
    finally:
        _clear_active(_active_eventification, session_id)


# --- Maintenance -------------------------------------------------------------


def reextract_all_transcripts() -> dict:
    """Re-run the understanding stages over every saved transcript.

    A developer helper: it skips speech-to-text entirely and starts from the
    current ``Session.transcript`` values, rebuilding the encountered-entity
    registry so stale labels do not survive a parser change.
    """
    if not flags.nlp_enabled():
        return {
            "started": False,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "detail": "NLP hattı kapalı; yeniden çıkarım yapılmadı.",
        }

    if not _bulk_reextract_lock.acquire(blocking=False):
        return {
            "started": False,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "detail": "Toplu çıkarım zaten çalışıyor.",
        }

    processed = skipped = failed = 0
    try:
        EncounteredEntity.objects.all().delete()
        for session in Session.objects.order_by("id"):
            if not (session.transcript or "").strip():
                skipped += 1
                continue
            if is_processing_active(session.id) or is_eventification_active(session.id):
                skipped += 1
                continue

            session.status = SessionStatus.PARSING
            session.status_detail = "Debug: kişi/yer çıkarımı yeniden çalışıyor."
            session.structured_events = []
            session.save(
                update_fields=["status", "status_detail", "structured_events", "updated_at"]
            )
            try:
                Edge.objects.filter(session=session).delete()
                ctx = StageContext(
                    session=session,
                    reuse_transcript=True,
                    queue_eventification=False,
                )
                stage("understand").run(ctx)
                processed += 1
            except Exception:  # pragma: no cover - defensive
                failed += 1
                logger.error(
                    "Bulk re-extract for session %s failed: %s",
                    session.id,
                    traceback.format_exc(),
                )
                session_pipeline_mod.mark_session_failed(session.id, traceback.format_exc())

        return {
            "started": True,
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "detail": (
                f"{processed} kayıt yeniden çıkarıldı; "
                f"{skipped} kayıt atlandı; {failed} hata."
            ),
        }
    finally:
        _bulk_reextract_lock.release()


# --- Helpers -----------------------------------------------------------------


def _load(session_id: int, what: str) -> Session | None:
    try:
        return Session.objects.get(pk=session_id)
    except Session.DoesNotExist:
        logger.error("Session %s vanished before %s", session_id, what)
        return None


def _fail(session: Session, detail: str) -> None:
    session.status = SessionStatus.FAILED
    session.status_detail = detail
    session.save(update_fields=["status", "status_detail", "updated_at"])
