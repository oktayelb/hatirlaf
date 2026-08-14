"""Diary processing pipeline.

Three modules, one job each:

* :mod:`.flags`   — the on/off switch for the NLP layer.
* :mod:`.stages`  — the ordered list of steps an entry passes through.
* :mod:`.runner`  — threads, re-entrancy, and failure bookkeeping.

Callers should import from this package rather than reaching into the
submodules, so the internals stay free to move.
"""

from .flags import is_enabled, nlp_enabled, snapshot as feature_snapshot
from .runner import (
    is_eventification_active,
    is_processing_active,
    kickoff,
    kickoff_eventification,
    kickoff_transcript_reprocess,
    reextract_all_transcripts,
    run,
    run_eventification,
    run_transcript_reprocess,
)
from .stages import PIPELINE, Stage, StageContext

__all__ = [
    "PIPELINE",
    "Stage",
    "StageContext",
    "feature_snapshot",
    "is_enabled",
    "is_eventification_active",
    "is_processing_active",
    "kickoff",
    "kickoff_eventification",
    "kickoff_transcript_reprocess",
    "nlp_enabled",
    "reextract_all_transcripts",
    "run",
    "run_eventification",
    "run_transcript_reprocess",
]
