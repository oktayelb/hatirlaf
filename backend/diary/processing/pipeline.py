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
from django.db import transaction
from django.utils import timezone

from ..models import ConflictReason, Mention, MentionType, Node, NodeKind, Session, SessionStatus
from . import extractor as extractor_mod
from . import llm as llm_mod
from . import nlp as nlp_mod
from . import transcription as tx_mod
from .conflicts import detect_conflicts

logger = logging.getLogger(__name__)


def kickoff(session_id: int) -> None:
    """Start processing a Session, either synchronously or in a thread."""
    if getattr(settings, "HATIRLAF_SYNC_PROCESSING", False):
        run(session_id)
        return
    t = threading.Thread(target=run, args=(session_id,), daemon=True, name=f"hatirlaf-{session_id}")
    t.start()


def run(session_id: int) -> None:
    try:
        session = Session.objects.get(pk=session_id)
    except Session.DoesNotExist:
        logger.error("Session %s vanished before processing", session_id)
        return

    try:
        _process(session)
    except Exception:  # pragma: no cover
        logger.error("Processing session %s failed: %s", session_id, traceback.format_exc())
        Session.objects.filter(pk=session_id).update(
            status=SessionStatus.FAILED,
            status_detail=traceback.format_exc()[:2000],
            updated_at=timezone.now(),
        )


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

    result = tx_mod.transcribe(session.audio_file.path, language=session.language)

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


def _parse_and_store(session: Session) -> None:
    extraction = extractor_mod.extract(session.transcript, session.recorded_at)
    parsed = extraction.parse or nlp_mod.analyze(session.transcript)
    flagged = detect_conflicts(parsed.mentions, session.transcript, session.recorded_at)

    # LLM (or NLP-only fallback) produces the calendar-ready event log.
    try:
        llm_result = llm_mod.run(extraction)
    except Exception as exc:  # pragma: no cover — defensive, pipeline must survive
        logger.exception("LLM stage failed, using pure NLP fallback: %s", exc)
        llm_result = llm_mod._fallback_from_hints(extraction)
    session.structured_events = llm_result.get("olay_loglari", [])
    session.nlp_hints = extraction.to_json()

    with transaction.atomic():
        # Replace prior mentions on re-run.
        session.mentions.all().delete()

        for fm in flagged:
            m = fm.mention
            audio_start, audio_end = _span_to_audio(session.word_timings, m.char_start, m.char_end)
            mention = Mention.objects.create(
                session=session,
                surface=m.surface,
                lemma=m.lemma,
                char_start=m.char_start,
                char_end=m.char_end,
                mention_type=_safe_mention_type(m.mention_type),
                audio_start=audio_start,
                audio_end=audio_end,
                is_conflict=fm.is_conflict,
                conflict_reason=_safe_reason(fm.conflict_reason),
                conflict_hint=fm.conflict_hint,
            )
            # Auto-resolve when we have a confident grounding and no conflict.
            if not fm.is_conflict:
                _auto_resolve(mention, fm)

        session.processed_text = parsed.lemma_text
        # Always finish as COMPLETED — the NLP/LLM layer is a helper, not a
        # gate. Any remaining conflicts are surfaced by the review screen
        # but never hold the session in a pending state.
        session.status = SessionStatus.COMPLETED
        session.status_detail = (
            f"NLP: {parsed.nlp_backend}; NER: {parsed.ner_backend}; "
            f"LLM: {llm_result.get('backend', 'nlp-only')}; "
            f"{len(session.structured_events)} olay, "
            f"{sum(1 for fm in flagged if fm.is_conflict)} çatışma."
        )
        session.save(
            update_fields=[
                "processed_text",
                "status",
                "status_detail",
                "structured_events",
                "nlp_hints",
                "updated_at",
            ]
        )


def _auto_resolve(mention: Mention, fm) -> None:
    """Create/find a Node for a non-conflict mention and wire it up."""
    kind = _suggested_kind_to_node_kind(fm.suggested_kind or mention.mention_type)
    label = mention.surface.strip()
    if not label:
        return
    node, _ = Node.objects.get_or_create(kind=kind, label=label)
    if kind == NodeKind.TIME and fm.suggested_resolution.get("time_value"):
        node.time_value = fm.suggested_resolution["time_value"]
        node.save(update_fields=["time_value", "updated_at"])
    mention.node = node
    mention.resolved = True
    mention.resolution_action = "AUTO"
    mention.save(update_fields=["node", "resolved", "resolution_action"])


def _span_to_audio(word_timings: list[dict], char_start: int, char_end: int):
    if not word_timings:
        return None, None
    hits = [
        w for w in word_timings
        if w.get("char_start", -1) >= 0
        and w["char_start"] < char_end
        and w["char_end"] > char_start
    ]
    if not hits:
        return None, None
    return float(hits[0]["start"]), float(hits[-1]["end"])


def _safe_mention_type(value: str) -> str:
    valid = {c[0] for c in MentionType.choices}
    return value if value in valid else MentionType.PERSON


def _safe_reason(value: str) -> str:
    if not value:
        return ""
    valid = {c[0] for c in ConflictReason.choices}
    return value if value in valid else ""


def _suggested_kind_to_node_kind(kind: str) -> str:
    valid = {c[0] for c in NodeKind.choices}
    return kind if kind in valid else NodeKind.OTHER
