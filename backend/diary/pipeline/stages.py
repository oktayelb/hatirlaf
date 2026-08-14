"""The stages a diary entry passes through, declared in order.

Each stage is a small function plus metadata saying when it applies. The
runner walks :data:`PIPELINE`, skips the stages whose feature flag is off,
and executes the rest. Adding or removing a step means editing this list —
the runner itself has no knowledge of what any particular stage does.

Capture stages (``requires=NLP_ANY``) are the diary: audio in, text out.
Understanding stages (``requires=NLP_ON``) are the analysis layer, and are
inert while :func:`diary.pipeline.flags.nlp_enabled` returns False.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from ..media import decrypted_file_path
from ..models import EventificationStatus, Session, SessionStatus
from ..processing import extractor as extractor_mod
from ..processing import llm as llm_mod
from ..processing import nlp as nlp_mod
from ..processing import transcription as tx_mod
from ..processing.conflicts import detect_conflicts
from ..services import session_pipeline as session_pipeline_mod
from . import flags

logger = logging.getLogger(__name__)

# When a stage applies, relative to the NLP switch.
NLP_ANY = "any"  # always runs — this is the diary itself
NLP_ON = "on"  # only while the understanding pipeline is enabled
NLP_OFF = "off"  # only while it is disabled


class PipelineAbort(Exception):
    """Stop the run with a message meant for the user, not a traceback."""


@dataclass
class StageContext:
    """Mutable state threaded through one run of the pipeline."""

    session: Session
    #: False for bulk re-extraction, which drives eventification itself.
    queue_eventification: bool = True
    #: True when re-running from an already-saved transcript (no STT).
    reuse_transcript: bool = False
    notes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    run: Callable[[StageContext], None]
    requires: str = NLP_ANY
    #: Deferred stages are handed to their own worker after the synchronous
    #: chain returns, so a slow model never holds up the entry appearing.
    deferred: bool = False

    def applies(self) -> bool:
        if self.requires == NLP_ANY:
            return True
        if self.requires == NLP_ON:
            return flags.nlp_enabled()
        return not flags.nlp_enabled()


# --- Stage bodies ------------------------------------------------------------


def transcribe(ctx: StageContext) -> None:
    """Audio in, text out. The one stage the diary cannot do without."""
    session = ctx.session

    if ctx.reuse_transcript and (session.transcript or "").strip():
        ctx.notes["backend"] = "kaydedilmiş transkript"
        return

    # Typed entry: the text is already the transcript.
    if session.transcript and not session.audio_file:
        ctx.notes["backend"] = "yazılı giriş"
        return

    if not session.audio_file:
        raise PipelineAbort("Ne ses ne de metin sağlandı.")

    session.status = SessionStatus.TRANSCRIBING
    session.status_detail = "Ses yazıya çevriliyor."
    session.save(update_fields=["status", "status_detail", "updated_at"])

    suffix = ""
    if session.audio_file.name:
        suffix = "." + session.audio_file.name.rsplit(".", 1)[-1]
    with decrypted_file_path(session.audio_file, suffix=suffix) as audio_path:
        result = tx_mod.transcribe(audio_path, language=session.language)

    session.transcript = result.text
    session.word_timings = tx_mod.assign_word_timings(result.words, result.text)
    if result.duration and not session.duration_seconds:
        session.duration_seconds = result.duration
    session.save(
        update_fields=["transcript", "word_timings", "duration_seconds", "updated_at"]
    )
    ctx.notes["backend"] = result.backend


def archive(ctx: StageContext) -> None:
    """Close out a capture-only run: the entry is saved, and that is all."""
    session = ctx.session
    session.status = SessionStatus.COMPLETED
    session.status_detail = "Günlüğe kaydedildi."
    session.processed_text = ""
    session.structured_events = []
    session.nlp_hints = {}
    session.eventification_status = EventificationStatus.NOT_STARTED
    session.eventification_detail = ""
    session.save(
        update_fields=[
            "status",
            "status_detail",
            "processed_text",
            "structured_events",
            "nlp_hints",
            "eventification_status",
            "eventification_detail",
            "updated_at",
        ]
    )


def understand(ctx: StageContext) -> None:
    """Parse the transcript into mentions, resolve what it can, flag the rest."""
    session = ctx.session
    session.status = SessionStatus.PARSING
    backend = ctx.notes.get("backend", "")
    session.status_detail = f"Transkripsiyon backend: {backend}" if backend else ""
    session.save(update_fields=["status", "status_detail", "updated_at"])

    extraction = extractor_mod.extract(session.transcript, session.recorded_at)
    parsed = extraction.parse or nlp_mod.analyze(session.transcript)
    flagged = detect_conflicts(parsed.mentions, session.transcript, session.recorded_at)
    session_pipeline_mod.persist_parsing_result(
        session,
        extraction,
        parsed,
        flagged,
        queue_eventification=ctx.queue_eventification,
    )


def eventify(ctx: StageContext) -> None:
    """Ask the local LLM to turn the transcript into dated calendar events."""
    session = ctx.session
    extraction = extractor_mod.extract(session.transcript, session.recorded_at)
    llm_result = llm_mod.run(extraction)
    enrichment = llm_mod.run_mood_tags(extraction, llm_result.get("olay_loglari", []))
    llm_result["olay_loglari"] = llm_mod.merge_mood_tags(
        llm_result.get("olay_loglari", []),
        enrichment,
    )
    session_pipeline_mod.persist_eventification_result(
        session, extraction, llm_result, enrichment
    )


# --- The pipeline ------------------------------------------------------------

PIPELINE: tuple[Stage, ...] = (
    Stage(
        key="transcribe",
        label="Ses yazıya çevriliyor",
        run=transcribe,
        requires=NLP_ANY,
    ),
    Stage(
        key="archive",
        label="Günlüğe kaydediliyor",
        run=archive,
        requires=NLP_OFF,
    ),
    Stage(
        key="understand",
        label="Metin analiz ediliyor",
        run=understand,
        requires=NLP_ON,
    ),
    Stage(
        key="eventify",
        label="Olaylar çıkarılıyor",
        run=eventify,
        requires=NLP_ON,
        deferred=True,
    ),
)


def active_stages(*, deferred: bool) -> list[Stage]:
    """Return the stages that apply right now, in declared order."""
    return [s for s in PIPELINE if s.applies() and s.deferred is deferred]


def stage(key: str) -> Stage:
    for candidate in PIPELINE:
        if candidate.key == key:
            return candidate
    raise KeyError(key)
