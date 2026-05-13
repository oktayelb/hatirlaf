"""Turkish morphological analysis + Named Entity Recognition facade.

The implementation lives in smaller helper modules:
  * ``nlp_models`` — shared dataclasses and lexicons
  * ``nlp_morph``  — morphology, tokenization, entity label normalization
  * ``nlp_ner``    — named-entity extraction and mention deduplication

This module keeps the original public surface stable for existing callers.
"""

from __future__ import annotations

import logging
import threading

from django.conf import settings

from . import nlp_morph as morph_mod
from . import nlp_ner as ner_mod
from .nlp_models import (
    AMBIGUOUS_PRONOUNS,
    RELATIVE_LOCATION_WORDS,
    RELATIVE_TIME_WORDS,
    TR_CITIES,
    TR_COUNTRIES,
    TR_MONTHS,
    TR_WEEKDAYS,
    EntityMention,
    ParseResult,
    Token,
)

logger = logging.getLogger(__name__)

_analyzer_lock = threading.Lock()
_cached_analyzer = None
_cached_backend: str | None = None

_ner_lock = threading.Lock()
_cached_ner = None
_cached_ner_backend: str | None = None

# Public API re-exports.
_TOKEN_RE = morph_mod._TOKEN_RE
fallback_lemma = morph_mod.fallback_lemma
normalize_entity_lemma = morph_mod.normalize_entity_lemma
normalize_entity_label = morph_mod.normalize_entity_label
tokenize = morph_mod.tokenize
_dedupe_mentions = ner_mod._dedupe_mentions


def _load_analyzer():
    global _cached_analyzer, _cached_backend
    with _analyzer_lock:
        if _cached_analyzer is not None or _cached_backend == "rules":
            return _cached_analyzer, _cached_backend
        try:
            import zeyrek  # type: ignore

            _cached_analyzer = zeyrek.MorphAnalyzer()
            _cached_backend = "zeyrek"
            logger.info("Loaded Zeyrek morphological analyzer")
        except Exception as exc:
            logger.warning("Zeyrek unavailable (%s). Using suffix-stripper fallback.", exc)
            _cached_backend = "rules"
        return _cached_analyzer, _cached_backend


def _load_ner():
    global _cached_ner, _cached_ner_backend
    with _ner_lock:
        if _cached_ner is not None or _cached_ner_backend == "rules":
            return _cached_ner, _cached_ner_backend
        if not getattr(settings, "HATIRLAF_USE_TURKISH_NER", False):
            _cached_ner_backend = "rules"
            return None, "rules"
        try:
            from transformers import pipeline  # type: ignore

            model_name = getattr(
                settings,
                "HATIRLAF_TURKISH_NER_MODEL",
                "savasy/bert-base-turkish-ner-cased",
            )
            logger.info("Loading Turkish NER pipeline (%s)", model_name)
            _cached_ner = pipeline(
                "token-classification",
                model=model_name,
                aggregation_strategy="simple",
            )
            _cached_ner_backend = f"hf_ner:{model_name}"
        except Exception as exc:
            logger.warning("Turkish transformer NER unavailable (%s). Falling back to rules.", exc)
            _cached_ner_backend = "rules"
        return _cached_ner, _cached_ner_backend


def analyze(text: str) -> ParseResult:
    """Run the full local NLP pipeline on a Turkish transcript."""
    if not text.strip():
        return ParseResult()

    analyzer, nlp_backend = _load_analyzer()
    tokens = morph_mod.morph_analyze(text, analyzer, nlp_backend)
    mentions, ner_backend = ner_mod.extract_entities(text, tokens, _load_ner)
    mentions.extend(ner_mod._find_pronouns(text))
    mentions.extend(ner_mod._find_relative_times(text))
    mentions.extend(ner_mod._find_relative_locations(text))
    mentions.extend(ner_mod._find_absolute_times(text))
    mentions = ner_mod._dedupe_mentions(mentions)

    lemma_text = " ".join(t.lemma for t in tokens)
    return ParseResult(
        tokens=tokens,
        mentions=mentions,
        lemma_text=lemma_text,
        nlp_backend=nlp_backend or "rules",
        ner_backend=ner_backend,
    )
