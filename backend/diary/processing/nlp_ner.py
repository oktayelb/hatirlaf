"""Named entity extraction and label cleanup."""

from __future__ import annotations

import logging
import re
from typing import Callable

from . import savyar_adapter
from .nlp_models import (
    AMBIGUOUS_PRONOUNS,
    RELATIVE_LOCATION_WORDS,
    RELATIVE_TIME_WORDS,
    TR_CITIES,
    TR_COUNTRIES,
    TR_MONTHS,
    TR_WEEKDAYS,
    EntityMention,
    Token,
)
from .nlp_morph import _TOKEN_RE, normalize_entity_lemma

logger = logging.getLogger(__name__)


def extract_entities(
    text: str,
    tokens: list[Token],
    load_ner: Callable[[], tuple[object | None, str | None]],
) -> tuple[list[EntityMention], str]:
    ner, backend = load_ner()
    if backend and backend.startswith("hf_ner") and ner is not None:
        try:
            return _hf_ner_entities(text, ner), backend
        except Exception as exc:
            logger.warning("Turkish transformer NER inference failed (%s). Using rules.", exc)
    return _rule_entities(text, tokens), "rules"


def _hf_ner_entities(text: str, ner) -> list[EntityMention]:
    out: list[EntityMention] = []
    for ent in ner(text):
        label = (ent.get("entity_group") or ent.get("entity") or "").upper()
        if label.startswith("B-") or label.startswith("I-"):
            label = label[2:]
        mapping = {"PER": "PERSON", "LOC": "LOCATION", "ORG": "ORG", "MISC": "EVENT"}
        mtype = mapping.get(label, "")
        if not mtype:
            continue
        surface = ent.get("word", "").strip()
        start = int(ent.get("start", 0))
        end = int(ent.get("end", start + len(surface)))
        if not surface:
            continue
        out.append(
            EntityMention(
                surface=surface,
                lemma=normalize_entity_lemma(surface),
                char_start=start,
                char_end=end,
                mention_type=mtype,
                source="hf_ner",
                score=float(ent.get("score", 1.0)),
            )
        )
    return out


def _rule_entities(text: str, tokens: list[Token]) -> list[EntityMention]:
    out: list[EntityMention] = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        low = t.surface.lower()

        bare = _strip_apostrophe_suffix(low)
        if bare in AMBIGUOUS_PRONOUNS or bare in RELATIVE_TIME_WORDS or bare in RELATIVE_LOCATION_WORDS:
            i += 1
            continue

        if t.is_proper and len(t.surface) > 1:
            j = i
            while j + 1 < n and tokens[j + 1].is_proper:
                between = text[tokens[j].char_end : tokens[j + 1].char_start]
                if any(ch in ".!?\n" for ch in between):
                    break
                nxt_low = _strip_apostrophe_suffix(tokens[j + 1].surface.lower())
                if nxt_low in TR_WEEKDAYS or nxt_low in TR_MONTHS:
                    break
                if _has_bare_location_case(tokens[j + 1].surface):
                    break
                j += 1
            start = t.char_start
            end = tokens[j].char_end
            surface = text[start:end]

            bare_parts = [_strip_apostrophe_suffix(text[tk.char_start:tk.char_end].lower()) for tk in tokens[i : j + 1]]
            span_lower = " ".join(bare_parts)

            mtype = "PERSON"
            if span_lower in TR_CITIES or span_lower in TR_COUNTRIES:
                mtype = "LOCATION"
            elif any(p in TR_CITIES or p in TR_COUNTRIES for p in bare_parts):
                mtype = "LOCATION"
            elif len(bare_parts) == 1 and (
                _has_apostrophe_location_case(surface) or _has_bare_location_case(surface)
            ):
                mtype = "LOCATION"
                clean_surface = _strip_location_case(surface)
                if clean_surface:
                    surface = clean_surface
                    end = start + len(surface)

            out.append(
                EntityMention(
                    surface=surface,
                    lemma=normalize_entity_lemma(surface),
                    char_start=start,
                    char_end=end,
                    mention_type=mtype,
                    source="rules",
                )
            )
            i = j + 1
            continue

        if bare in TR_WEEKDAYS or bare in TR_MONTHS:
            out.append(
                EntityMention(
                    surface=t.surface,
                    lemma=normalize_entity_lemma(t.surface),
                    char_start=t.char_start,
                    char_end=t.char_end,
                    mention_type="TIME",
                    source="rules",
                )
            )
        elif bare in TR_CITIES or bare in TR_COUNTRIES:
            out.append(
                EntityMention(
                    surface=t.surface,
                    lemma=normalize_entity_lemma(t.surface),
                    char_start=t.char_start,
                    char_end=t.char_end,
                    mention_type="LOCATION",
                    source="rules",
                )
            )
        else:
            recovered = savyar_adapter.recover_known_proper_case(
                bare,
                set(TR_CITIES) | set(TR_COUNTRIES),
            )
            if recovered is not None:
                out.append(
                    EntityMention(
                        surface=recovered.label,
                        lemma=normalize_entity_lemma(recovered.label),
                        char_start=t.char_start,
                        char_end=t.char_start + len(recovered.label),
                        mention_type="LOCATION",
                        source="rules",
                        hint=f"STT biçiminden ayrıştırıldı: {recovered.suffix}",
                    )
                )
        i += 1
    return out


def _strip_apostrophe_suffix(text: str) -> str:
    """Turkish proper nouns take inflection after an apostrophe."""
    for marker in ("'", "’"):
        idx = text.find(marker)
        if idx > 0:
            return text[:idx]
    return text


def _has_apostrophe_location_case(text: str) -> bool:
    low = text.lower()
    for marker in ("'", "’"):
        idx = low.find(marker)
        if idx > 0:
            suffix = low[idx + 1 :]
            return suffix.startswith(("da", "de", "ta", "te", "dan", "den", "tan", "ten"))
    return False


def _has_bare_location_case(text: str) -> bool:
    if not text[:1].isupper():
        return False
    low = text.lower()
    suffix = _bare_location_suffix(low)
    return bool(suffix and len(text) > len(suffix) + 2)


def _strip_location_case(text: str) -> str:
    for marker in ("'", "’"):
        idx = text.find(marker)
        if idx > 0:
            return text[:idx]
    suffix = _bare_location_suffix(text.lower())
    if suffix and len(text) > len(suffix) + 2:
        return text[: -len(suffix)]
    return text


def _bare_location_suffix(low: str) -> str:
    for suffix in ("dan", "den", "tan", "ten", "da", "de", "ta", "te"):
        if low.endswith(suffix):
            return suffix
    return ""


def _find_pronouns(text: str) -> list[EntityMention]:
    out: list[EntityMention] = []
    tokens = list(_TOKEN_RE.finditer(text))
    for idx, m in enumerate(tokens):
        surface = m.group(0)
        low = surface.lower()
        if savyar_adapter.enabled():
            categories = savyar_adapter.closed_class_categories(low)
            if "pronoun" not in categories and low not in AMBIGUOUS_PRONOUNS:
                continue
            if _looks_like_savyar_determiner(tokens, idx, categories):
                continue
        if low in AMBIGUOUS_PRONOUNS or "pronoun" in savyar_adapter.closed_class_categories(low):
            out.append(
                EntityMention(
                    surface=surface,
                    lemma=normalize_entity_lemma(surface),
                    char_start=m.start(),
                    char_end=m.end(),
                    mention_type="PRONOUN",
                    source="pronoun",
                    hint="Belirsiz zamir — kime atıfta bulunuyor?",
                )
            )
    return out


def _looks_like_savyar_determiner(tokens, idx: int, categories: set[str]) -> bool:
    if "determiner" not in categories or idx + 1 >= len(tokens):
        return False
    low = tokens[idx].group(0).lower()
    if low not in {"o", "bu", "şu"}:
        return False
    next_surface = tokens[idx + 1].group(0)
    next_hint = savyar_adapter.infer_subject_tense(next_surface)
    if next_hint.person or next_hint.zaman_dilimi:
        return False
    return True


def _find_relative_times(text: str) -> list[EntityMention]:
    out: list[EntityMention] = []
    for m in _TOKEN_RE.finditer(text):
        low = m.group(0).lower()
        if low in RELATIVE_TIME_WORDS:
            out.append(
                EntityMention(
                    surface=m.group(0),
                    lemma=normalize_entity_lemma(m.group(0)),
                    char_start=m.start(),
                    char_end=m.end(),
                    mention_type="TIME",
                    source="time",
                    hint="Göreceli zaman — tam tarih gerekli mi?",
                )
            )
    return out


def _find_relative_locations(text: str) -> list[EntityMention]:
    out: list[EntityMention] = []
    for m in _TOKEN_RE.finditer(text):
        low = m.group(0).lower()
        if low in RELATIVE_LOCATION_WORDS:
            out.append(
                EntityMention(
                    surface=m.group(0),
                    lemma=normalize_entity_lemma(m.group(0)),
                    char_start=m.start(),
                    char_end=m.end(),
                    mention_type="LOCATION",
                    source="rules",
                    hint="Belirsiz yer — neresi kastediliyor?",
                )
            )
    return out


def _find_absolute_times(text: str) -> list[EntityMention]:
    out: list[EntityMention] = []
    month_alt = "|".join(TR_MONTHS)
    patterns = [
        re.compile(rf"\b(\d{{1,2}})\s+({month_alt})(?:\s+(\d{{4}}))?\b", re.IGNORECASE),
        re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{2,4})\b"),
        re.compile(r"\b(saat\s+)?\d{1,2}[:.]\d{2}\b", re.IGNORECASE),
    ]
    for pat in patterns:
        for m in pat.finditer(text):
            out.append(
                EntityMention(
                    surface=m.group(0),
                    lemma=normalize_entity_lemma(m.group(0)),
                    char_start=m.start(),
                    char_end=m.end(),
                    mention_type="TIME",
                    source="time",
                )
            )
    return out


def _dedupe_mentions(mentions: Iterable[EntityMention]) -> list[EntityMention]:
    """Keep longest/highest-scoring mention per (start,end) overlap window."""
    sorted_m = sorted(mentions, key=lambda m: (m.char_start, -(m.char_end - m.char_start)))
    out: list[EntityMention] = []
    for m in sorted_m:
        if any(
            prev.char_start <= m.char_start
            and m.char_end <= prev.char_end
            and (prev.char_end - prev.char_start) > (m.char_end - m.char_start)
            and _source_priority(prev.source) >= _source_priority(m.source)
            for prev in out
        ):
            continue
        out = [
            prev for prev in out
            if not (
                m.char_start <= prev.char_start
                and prev.char_end <= m.char_end
                and (m.char_end - m.char_start) > (prev.char_end - prev.char_start)
            )
        ]
        out = [
            prev for prev in out
            if not (
                prev.char_start <= m.char_start
                and m.char_end <= prev.char_end
                and (prev.char_end - prev.char_start) > (m.char_end - m.char_start)
                and _source_priority(m.source) > _source_priority(prev.source)
            )
        ]
        dup = next(
            (
                i for i, prev in enumerate(out)
                if prev.char_start == m.char_start and prev.char_end == m.char_end
            ),
            None,
        )
        if dup is not None:
            if _source_priority(m.source) > _source_priority(out[dup].source):
                out[dup] = m
            continue
        out.append(m)
    out.sort(key=lambda m: m.char_start)
    return out


def _source_priority(source: str) -> int:
    if source == "hf_ner":
        return 40
    if source == "gazetteer":
        return 30
    if source in {"time", "pronoun"}:
        return 25
    if source == "rules":
        return 20
    return 0
