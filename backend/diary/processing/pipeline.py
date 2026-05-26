"""End-to-end orchestration: audio -> transcript -> mentions -> conflicts.

Runs in a daemon thread per the PDF's lean-MVP brief (no Celery, no Redis).
The session row itself is the progress tracker — writes to ``status`` and
``status_detail`` give the client a poll endpoint.
"""

from __future__ import annotations

import logging
import threading
import traceback

from django.conf import settings

from ..media import decrypted_file_path
from ..models import Edge, EncounteredEntity, Session, SessionStatus
from . import extractor as extractor_mod
from . import llm as llm_mod
from . import nlp as nlp_mod
from . import transcription as tx_mod
from .conflicts import detect_conflicts
from ..services import session_pipeline as session_pipeline_mod

logger = logging.getLogger(__name__)

_active_lock = threading.Lock()
_active_processing: set[int] = set()
_active_eventification: set[int] = set()
_bulk_reextract_lock = threading.Lock()


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


def kickoff(session_id: int) -> None:
    """Start processing a Session, either synchronously or in a thread."""
    if getattr(settings, "HATIRLAF_SYNC_PROCESSING", False):
        run(session_id)
        return
    t = threading.Thread(target=run, args=(session_id,), daemon=True, name=f"hatirlaf-{session_id}")
    t.start()


def kickoff_eventification(session_id: int) -> None:
    """Start the LLM/event-calendar stage independently of transcript parsing."""
    t = threading.Thread(
        target=run_eventification,
        args=(session_id,),
        daemon=True,
        name=f"hatirlaf-eventify-{session_id}",
    )
    t.start()


def kickoff_transcript_reprocess(session_id: int) -> None:
    """Rebuild NLP/event data from the saved transcript without re-running STT."""
    if getattr(settings, "HATIRLAF_SYNC_PROCESSING", False):
        run_transcript_reprocess(session_id)
        return
    t = threading.Thread(
        target=run_transcript_reprocess,
        args=(session_id,),
        daemon=True,
        name=f"hatirlaf-reparse-{session_id}",
    )
    t.start()


def run(session_id: int) -> None:
    if not _try_mark_active(_active_processing, session_id):
        logger.info("Session %s processing is already active; skipping duplicate.", session_id)
        return
    try:
        try:
            session = Session.objects.get(pk=session_id)
        except Session.DoesNotExist:
            logger.error("Session %s vanished before processing", session_id)
            return

        try:
            _process(session)
        except Exception:  # pragma: no cover
            logger.error("Processing session %s failed: %s", session_id, traceback.format_exc())
            session_pipeline_mod.mark_session_failed(session_id, traceback.format_exc())
    finally:
        _clear_active(_active_processing, session_id)


def run_transcript_reprocess(session_id: int) -> None:
    if not _try_mark_active(_active_processing, session_id):
        logger.info("Session %s transcript reprocess is already active; skipping duplicate.", session_id)
        return
    try:
        try:
            session = Session.objects.get(pk=session_id)
        except Session.DoesNotExist:
            logger.error("Session %s vanished before transcript reprocess", session_id)
            return

        try:
            _reprocess_saved_transcript(session)
        except Exception:  # pragma: no cover
            logger.error("Transcript reprocess for session %s failed: %s", session_id, traceback.format_exc())
            session_pipeline_mod.mark_session_failed(session_id, traceback.format_exc())
    finally:
        _clear_active(_active_processing, session_id)


def _process(session: Session) -> None:
    # Fast path: transcript was uploaded manually (e.g., tests, no-audio flow).
    if session.transcript and not session.audio_file:
        session.status = SessionStatus.PARSING
        session.status_detail = ""
        session.save(update_fields=["status", "status_detail", "updated_at"])
        _parse_and_store(session)
        return

    # 1. Transcribe.
    session.status = SessionStatus.TRANSCRIBING
    session.status_detail = ""
    session.save(update_fields=["status", "status_detail", "updated_at"])

    if not session.audio_file or not session.audio_file.path:
        session.status = SessionStatus.FAILED
        session.status_detail = "Ne ses ne de metin sağlandı."
        session.save(update_fields=["status", "status_detail", "updated_at"])
        return

    suffix = ""
    if session.audio_file.name:
        suffix = "." + session.audio_file.name.rsplit(".", 1)[-1]
    with decrypted_file_path(session.audio_file, suffix=suffix) as audio_path:
        result = tx_mod.transcribe(audio_path, language=session.language)

    session.transcript = result.text
    aligned = tx_mod.assign_word_timings(result.words, result.text)
    session.word_timings = aligned
    if result.duration and not session.duration_seconds:
        session.duration_seconds = result.duration
    session.save(
        update_fields=[
            "transcript",
            "word_timings",
            "duration_seconds",
            "updated_at",
        ]
    )

    # 2. Parse + store mentions. NLP is only a helper — it never blocks
    # pipeline completion. If the transcript is empty we still mark the
    # session as done so the calendar sees the day (with a placeholder event).
    session.status = SessionStatus.PARSING
    session.status_detail = f"Transkripsiyon backend: {result.backend}"
    session.save(update_fields=["status", "status_detail", "updated_at"])
    _parse_and_store(session)


def _reprocess_saved_transcript(session: Session) -> None:
    if not (session.transcript or "").strip():
        _process(session)
        return

    session.status = SessionStatus.PARSING
    session.status_detail = "Kaydedilmiş transkript yeniden işleniyor."
    session.structured_events = []
    session.save(
        update_fields=[
            "status",
            "status_detail",
            "structured_events",
            "updated_at",
        ]
    )
    _parse_and_store(session)


def _parse_and_store(session: Session, *, queue_eventification: bool = True) -> None:
    extraction = extractor_mod.extract(session.transcript, session.recorded_at)
    parsed = extraction.parse or nlp_mod.analyze(session.transcript)
    flagged = detect_conflicts(parsed.mentions, session.transcript, session.recorded_at)
    session_pipeline_mod.persist_parsing_result(
        session,
        extraction,
        parsed,
        flagged,
        queue_eventification=queue_eventification,
    )
    if queue_eventification:
        kickoff_eventification(session.id)


def reextract_all_transcripts() -> dict:
    """Re-run NLP/entity extraction for every saved transcript.

    This debug helper intentionally skips transcription and starts from the
    current ``Session.transcript`` values. It rebuilds the encountered-entity
    registry from the refreshed mentions so stale person/place labels do not
    survive parser changes.
    """
    if not _bulk_reextract_lock.acquire(blocking=False):
        return {
            "started": False,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "detail": "Toplu çıkarım zaten çalışıyor.",
        }

    processed = 0
    skipped = 0
    failed = 0
    try:
        EncounteredEntity.objects.all().delete()
        sessions = list(Session.objects.order_by("id"))
        for session in sessions:
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
                update_fields=[
                    "status",
                    "status_detail",
                    "structured_events",
                    "updated_at",
                ]
            )
            try:
                Edge.objects.filter(session=session).delete()
                _parse_and_store(session, queue_eventification=False)
                processed += 1
            except Exception:  # pragma: no cover
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


def run_eventification(session_id: int) -> None:
    if not _try_mark_active(_active_eventification, session_id):
        logger.info("Session %s eventification is already active; skipping duplicate.", session_id)
        return
    try:
        try:
            session = Session.objects.get(pk=session_id)
        except Session.DoesNotExist:
            logger.error("Session %s vanished before eventification", session_id)
            return

        if not session.transcript.strip():
            session_pipeline_mod.mark_eventification_empty(session_id)
            return

        session_pipeline_mod.mark_eventification_running(session_id)

        try:
            extraction = extractor_mod.extract(session.transcript, session.recorded_at)
            llm_result = llm_mod.run(extraction)
            session_pipeline_mod.persist_eventification_result(session, extraction, llm_result)
        except Exception as exc:  # pragma: no cover - defensive, stage is isolated
            logger.exception("Eventification session %s failed: %s", session_id, exc)
            session_pipeline_mod.mark_eventification_failed(session_id, str(exc))
    finally:
        _clear_active(_active_eventification, session_id)
