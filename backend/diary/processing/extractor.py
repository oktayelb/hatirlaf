"""Deterministic NLP pre-pass that feeds structured hints to the LLM.

The spec problem: a local, lightweight LLM (Qwen2.5-7B) is cheap and
private but mediocre at grounding Turkish relative times ("geçen hafta
bugün"), distinguishing past/future tense, and keeping proper-noun
spelling stable. Larger LLMs nail it but cost money and leak PII.

Strategy: run a rule-based Turkish NLP layer *before* the LLM. It:

  1. Segments the paragraph into candidate clauses.
  2. Resolves every date/time expression to an absolute ISO date using
     ``dateparser`` anchored on the session's ``recorded_at``.
  3. Collects person / location / org mentions from the existing NER
     pipeline (Zeyrek + rule-based or BERTurk).
  4. Classifies each clause's ``zaman_dilimi`` (Geçmiş / Şu An / Gelecek)
     using tense-marking verb suffixes (-di, -miş, -ecek, -acak, -yor).
  5. Builds a privacy mask: replaces real names/places with opaque
     placeholders so the same paragraph can be shipped to a cloud LLM
     without leaking PII. The mask map lives on the ExtractionResult so
     we can translate the LLM's response back to real names.

The LLM sees the paragraph + these clause-level hints and only needs to
pick which clause is which event — it never has to guess a date from
scratch.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable

import dateparser
from dateparser.search import search_dates

from . import nlp as nlp_mod
from . import savyar_adapter
from .name_gazetteer import TR_GIVEN_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ClauseHint:
    """One candidate event inferred by the NLP layer."""

    clause_index: int
    text: str
    char_start: int
    char_end: int

    # Absolute ISO date (YYYY-MM-DD) if we could ground it, else "".
    date_iso: str = ""
    # Additional time-of-day if mentioned (HH:MM) or "".
    time_hm: str = ""
    # Surface expressions that produced the date (for UX + debugging).
    time_phrases: list[str] = field(default_factory=list)
    # "Geçmiş" | "Şu An" | "Gelecek" | ""
    zaman_dilimi: str = ""
    # Named entities grounded to the clause span.
    persons: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    orgs: list[str] = field(default_factory=list)
    # Ambiguous local references ("o", "bu", "oradaki", "dünkü") that the
    # LLM should preserve as unresolved unless nearby context makes them clear.
    references: list[str] = field(default_factory=list)
    # Subject inferred from Turkish verb conjugation when the explicit subject
    # is dropped ("gittim" -> Ben, "konuştuk" -> Biz).
    subject_person: str = ""
    subject_pronoun: str = ""
    subject_verb: str = ""
    subject_tense: str = ""
    # Best-guess event phrase (verb + nearby noun group).
    event_phrase: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractionResult:
    paragraph: str
    recorded_at: dt.datetime | None
    clauses: list[ClauseHint] = field(default_factory=list)
    # All persons/locations/orgs seen across the paragraph (deduped).
    persons: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    orgs: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    # mask_token -> real value, so cloud-LLM output can be de-masked.
    mask_map: dict[str, str] = field(default_factory=dict)
    masked_paragraph: str = ""
    # Carry the upstream parse so callers don't re-run it.
    parse: nlp_mod.ParseResult | None = None

    def to_json(self) -> dict:
        return {
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "clauses": [c.as_dict() for c in self.clauses],
            "persons": self.persons,
            "locations": self.locations,
            "orgs": self.orgs,
            "references": self.references,
            "mask_map": self.mask_map,
            "masked_paragraph": self.masked_paragraph,
        }


# ---------------------------------------------------------------------------
# Clause segmentation
# ---------------------------------------------------------------------------

# Splits on sentence punctuation *and* strong coordinating words that in
# Turkish frequently open a new event ("hatta", "ve sonra", "ayrıca").
# Keep the delimiter so we can recover the original span.
_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[\.\!\?\n])\s+"
    r"|(?<=[,;])\s+(?=(?:hatta|ayrıca|bir de|sonra|daha sonra|ama|fakat|lakin)\b)",
    re.IGNORECASE,
)


def _segment(paragraph: str) -> list[tuple[str, int, int]]:
    """Return [(clause_text, char_start, char_end), ...]."""
    out: list[tuple[str, int, int]] = []
    cursor = 0
    text = paragraph
    # Walk the splits manually so we preserve exact offsets.
    last = 0
    for m in _CLAUSE_SPLIT_RE.finditer(text):
        start = last
        end = m.start()
        chunk = text[start:end].strip()
        if chunk:
            # Locate the chunk's actual bounds inside [start:end].
            ss = start + (len(text[start:end]) - len(text[start:end].lstrip()))
            ee = ss + len(chunk)
            out.append((chunk, ss, ee))
        last = m.end()
        cursor = last
    tail = text[last:].strip()
    if tail:
        ss = last + (len(text[last:]) - len(text[last:].lstrip()))
        ee = ss + len(tail)
        out.append((tail, ss, ee))
    return out


# ---------------------------------------------------------------------------
# Date grounding
# ---------------------------------------------------------------------------

# "geçen hafta bugün", "iki gün önce", "önümüzdeki çarşamba" are well-handled
# by dateparser's Turkish pack. We keep this regex list for things
# dateparser sometimes misses (combo phrases, "Bugün X Y Z").
_TR_WEEKDAYS = {
    "pazartesi": 0, "salı": 1, "çarşamba": 2, "perşembe": 3,
    "cuma": 4, "cumartesi": 5, "pazar": 6,
}
_TR_MONTHS = {
    "ocak": 1, "şubat": 2, "mart": 3, "nisan": 4, "mayıs": 5, "haziran": 6,
    "temmuz": 7, "ağustos": 8, "eylül": 9, "ekim": 10, "kasım": 11, "aralık": 12,
}

_ABS_DATE_PATTERNS = [
    # 21 Nisan 2026, 21 Nisan
    re.compile(
        r"\b(\d{1,2})\s+(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)(?:\s+(\d{4}))?\b",
        re.IGNORECASE,
    ),
    # 21/04/2026, 21.04.2026, 21-04-2026
    re.compile(r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})\b"),
]
_TIME_OF_DAY_RE = re.compile(
    r"\b(?:saat\s+)?(\d{1,2})[:.](\d{2})\b", re.IGNORECASE
)


def _parse_absolute_date(text: str) -> dt.date | None:
    for pat in _ABS_DATE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        try:
            if pat is _ABS_DATE_PATTERNS[0]:
                day = int(m.group(1))
                month = _TR_MONTHS[m.group(2).lower()]
                year = int(m.group(3)) if m.group(3) else dt.date.today().year
                return dt.date(year, month, day)
            else:
                day = int(m.group(1))
                month = int(m.group(2))
                y = m.group(3)
                year = int(y) if len(y) == 4 else 2000 + int(y)
                return dt.date(year, month, day)
        except (KeyError, ValueError):
            continue
    return None


def _parse_with_dateparser(
    text: str, anchor: dt.datetime
) -> list[tuple[str, dt.datetime]]:
    """Return all (phrase, datetime) pairs dateparser finds in ``text``."""
    try:
        found = search_dates(
            text,
            languages=["tr"],
            settings={
                "RELATIVE_BASE": anchor,
                "PREFER_DATES_FROM": "past",
                "DATE_ORDER": "DMY",
                "RETURN_AS_TIMEZONE_AWARE": False,
            },
        )
    except Exception as exc:  # dateparser is finicky on some edge inputs
        logger.debug("dateparser.search_dates failed: %s", exc)
        return []
    return found or []


def _ground_clause_date(
    clause_text: str,
    anchor: dt.datetime,
    paragraph_anchor_date: dt.date | None,
) -> tuple[dt.date | None, str, list[str]]:
    """Best-effort: return (date, time_hm, phrases_used)."""
    phrases: list[str] = []
    # 1. Absolute explicit date beats everything.
    abs_d = _parse_absolute_date(clause_text)
    if abs_d is not None:
        phrases.append(abs_d.strftime("%d %B %Y"))

    # 2. Time-of-day (HH:MM).
    time_hm = ""
    tm = _TIME_OF_DAY_RE.search(clause_text)
    if tm:
        try:
            h = int(tm.group(1))
            mnt = int(tm.group(2))
            if 0 <= h <= 23 and 0 <= mnt <= 59:
                time_hm = f"{h:02d}:{mnt:02d}"
                phrases.append(tm.group(0))
        except ValueError:
            pass

    # 3. Fast-path relative adjectives that dateparser's Turkish search can
    # miss in noun phrases ("dünkü toplantı", "yarınki randevu").
    lowered = clause_text.lower()
    relative_day_map = {
        "dün": -1,
        "dünkü": -1,
        "bugün": 0,
        "bugünkü": 0,
        "yarın": 1,
        "yarınki": 1,
    }
    explicit_rel: dt.date | None = None
    for word, offset in relative_day_map.items():
        if re.search(rf"\b{re.escape(word)}\w*\b", lowered):
            explicit_rel = (paragraph_anchor_date or anchor.date()) + dt.timedelta(days=offset)
            phrases.append(word)
            break

    # 4. Relative expressions via dateparser, anchored on recorded_at.
    rel_anchor = anchor
    if paragraph_anchor_date is not None:
        rel_anchor = dt.datetime.combine(paragraph_anchor_date, anchor.time())

    dp_hits = _parse_with_dateparser(clause_text, rel_anchor)
    # Prefer the longest phrase; shortest often catches lone month names.
    dp_hits_sorted = sorted(dp_hits, key=lambda p: -len(p[0]))
    best_rel: dt.date | None = None
    for phrase, parsed in dp_hits_sorted:
        low = phrase.lower().strip()
        if low in {"ben", "bu", "şu", "o"}:
            continue
        phrases.append(phrase)
        if best_rel is None and parsed is not None:
            best_rel = parsed.date()

    final = abs_d or explicit_rel or best_rel
    # Dedup phrases.
    seen = set()
    ph = []
    for p in phrases:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            ph.append(p)
    return final, time_hm, ph


# ---------------------------------------------------------------------------
# Tense classification
# ---------------------------------------------------------------------------

# Heuristic suffix tables — not perfect, but cover ~95% of diary phrasing.
# Matched on word endings after crude lowercasing.
_PAST_SUFFIXES = (
    "dim", "dın", "din", "dun", "dün",
    "tım", "tın", "tin", "tum", "tün",
    "dık", "dik", "duk", "dük",
    "tık", "tik", "tuk", "tük",
    "dınız", "diniz", "dunuz", "dünüz",
    "tınız", "tiniz", "tunuz", "tünüz",
    "dılar", "diler", "dular", "düler",
    "tılar", "tiler", "tular", "tüler",
    "mış", "miş", "muş", "müş",
    "mıştım", "miştim", "muştum", "müştüm",
    "di", "dı", "du", "dü", "ti", "tı", "tu", "tü",
)

_FUTURE_SUFFIXES = (
    "acak", "ecek", "acağım", "eceğim",
    "acaksın", "eceksin", "acağız", "eceğiz",
    "acaksınız", "eceksiniz", "acaklar", "ecekler",
    "acaktı", "ecekti",
)

_PRESENT_SUFFIXES = (
    "yor", "yorum", "yorsun", "yoruz", "yorsunuz", "yorlar",
    "makta", "mekte", "maktayım", "mekteyim",
)

# Hardcoded Turkish finite verb endings. The order matters: longer, more
# specific suffixes must be checked before shorter zero-person forms.
_SUBJECT_SUFFIX_RULES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "future",
        "1sg",
        "Ben",
        ("acağım", "eceğim", "acagim", "ecegim"),
    ),
    ("future", "2pl", "Siz", ("acaksınız", "eceksiniz", "acaksiniz", "eceksiniz")),
    ("future", "1pl", "Biz", ("acağız", "eceğiz", "acagiz", "ecegiz")),
    ("future", "2sg", "Sen", ("acaksın", "eceksin", "acaksin", "eceksin")),
    ("future", "3pl", "Onlar", ("acaklar", "ecekler")),
    ("future", "3sg", "O", ("acak", "ecek")),
    (
        "present_continuous",
        "2pl",
        "Siz",
        ("yorsunuz",),
    ),
    ("present_continuous", "1sg", "Ben", ("yorum",)),
    ("present_continuous", "2sg", "Sen", ("yorsun",)),
    ("present_continuous", "1pl", "Biz", ("yoruz",)),
    ("present_continuous", "3pl", "Onlar", ("yorlar",)),
    ("present_continuous", "3sg", "O", ("yor",)),
    (
        "reported_past",
        "2pl",
        "Siz",
        (
            "mışsınız", "mişsiniz", "muşsunuz", "müşsünüz",
            "missiniz", "mussunuz",
        ),
    ),
    ("reported_past", "1sg", "Ben", ("mışım", "mişim", "muşum", "müşüm", "misim", "musum")),
    ("reported_past", "2sg", "Sen", ("mışsın", "mişsin", "muşsun", "müşsün", "missin", "mussun")),
    ("reported_past", "1pl", "Biz", ("mışız", "mişiz", "muşuz", "müşüz", "misiz", "musuz")),
    ("reported_past", "3pl", "Onlar", ("mışlar", "mişler", "muşlar", "müşler", "misler", "muslar")),
    ("reported_past", "3sg", "O", ("mış", "miş", "muş", "müş", "mis", "mus")),
    (
        "definite_past",
        "2pl",
        "Siz",
        (
            "dınız", "diniz", "dunuz", "dünüz",
            "tınız", "tiniz", "tunuz", "tünüz",
        ),
    ),
    (
        "definite_past",
        "1sg",
        "Ben",
        ("dım", "dim", "dum", "düm", "tım", "tim", "tum", "tüm"),
    ),
    (
        "definite_past",
        "2sg",
        "Sen",
        ("dın", "din", "dun", "dün", "tın", "tin", "tun", "tün"),
    ),
    (
        "definite_past",
        "1pl",
        "Biz",
        ("dık", "dik", "duk", "dük", "tık", "tik", "tuk", "tük"),
    ),
    (
        "definite_past",
        "3pl",
        "Onlar",
        ("dılar", "diler", "dular", "düler", "tılar", "tiler", "tular", "tüler"),
    ),
    ("definite_past", "3sg", "O", ("dı", "di", "du", "dü", "tı", "ti", "tu", "tü")),
    ("necessitative", "2pl", "Siz", ("malısınız", "melisiniz", "malisiniz")),
    ("necessitative", "1sg", "Ben", ("malıyım", "meliyim", "maliyim")),
    ("necessitative", "2sg", "Sen", ("malısın", "melisin", "malisin")),
    ("necessitative", "1pl", "Biz", ("malıyız", "meliyiz", "maliyiz")),
    ("necessitative", "3pl", "Onlar", ("malılar", "meliler", "maliler")),
    ("necessitative", "3sg", "O", ("malı", "meli", "mali")),
    ("conditional", "2pl", "Siz", ("sanız", "seniz", "saniz")),
    ("conditional", "1sg", "Ben", ("sam", "sem")),
    ("conditional", "2sg", "Sen", ("san", "sen")),
    ("conditional", "1pl", "Biz", ("sak", "sek")),
    ("conditional", "3pl", "Onlar", ("salar", "seler")),
    ("conditional", "3sg", "O", ("sa", "se")),
    (
        "aorist",
        "2pl",
        "Siz",
        (
            "arsınız", "ersiniz", "ırsınız", "irsiniz", "ursunuz", "ürsünüz",
            "rsınız", "rsiniz", "rsunuz", "rsünüz",
        ),
    ),
    (
        "aorist",
        "1sg",
        "Ben",
        ("arım", "erim", "ırım", "irim", "urum", "ürüm", "rım", "rim", "rum", "rüm"),
    ),
    (
        "aorist",
        "2sg",
        "Sen",
        ("arsın", "ersin", "ırsın", "irsin", "ursun", "ürsün", "rsın", "rsin", "rsun", "rsün"),
    ),
    (
        "aorist",
        "1pl",
        "Biz",
        ("arız", "eriz", "ırız", "iriz", "uruz", "ürüz", "rız", "riz", "ruz", "rüz"),
    ),
    ("aorist", "3pl", "Onlar", ("arlar", "erler", "ırlar", "irler", "urlar", "ürler", "rlar", "rler")),
    ("aorist", "3sg", "O", ("ar", "er", "ır", "ir", "ur", "ür", "r")),
)


def _classify_tense(text: str) -> str:
    savyar_hint = savyar_adapter.infer_subject_tense(text)
    if savyar_hint.zaman_dilimi:
        return savyar_hint.zaman_dilimi

    low = text.lower()
    # Future markers first (they contain 'acak' which can look past-ish).
    if any(_ending(low, suf) for suf in _FUTURE_SUFFIXES):
        return "Gelecek"
    if any(_ending(low, suf) for suf in _PRESENT_SUFFIXES):
        return "Şu An"
    if any(_ending(low, suf) for suf in _PAST_SUFFIXES):
        return "Geçmiş"
    return ""


def _infer_subject_from_conjugation(text: str) -> tuple[str, str, str, str]:
    """Infer dropped Turkish subject from finite verb morphology.

    Returns ``(person, pronoun, verb_surface, tense_key)``. This is deliberately
    suffix-based and deterministic; it does not try to prove that every
    matching word is a verb, so 3sg zero-person forms are kept conservative.
    """
    savyar_hint = savyar_adapter.infer_subject_tense(text)
    if savyar_hint.person:
        return (
            savyar_hint.person,
            savyar_hint.pronoun,
            savyar_hint.verb,
            savyar_hint.tense_key,
        )

    best: tuple[str, str, str, str] = ("", "", "", "")
    for m in nlp_mod._TOKEN_RE.finditer(text):
        surface = m.group(0).strip("'’")
        low = surface.lower()
        if len(low) < 3 or low in _STOPWORD_TOKENS or low in _CALENDAR_WORDS:
            continue
        for tense, person, pronoun, suffixes in _SUBJECT_SUFFIX_RULES:
            suffix = next((s for s in suffixes if low.endswith(s)), "")
            if not suffix or len(low) <= len(suffix):
                continue
            if person == "3sg" and not _looks_like_strong_3sg_verb(low, suffix, tense):
                continue
            candidate = (person, pronoun, surface, tense)
            # Later verbs usually carry the main predicate of the clause.
            best = candidate
            break
    return best


def _looks_like_strong_3sg_verb(token: str, suffix: str, tense: str) -> bool:
    if tense in {"future", "present_continuous", "reported_past", "necessitative"}:
        return True
    if tense == "definite_past":
        return len(token) >= len(suffix) + 2
    # Aorist and conditional 3sg endings are short and collide with nouns and
    # adjectives, so only trust them when the stem is long enough.
    return len(token) >= len(suffix) + 4


def _ending(text: str, suffix: str) -> bool:
    # Word-level match: any whitespace-split token ending with suffix.
    for tok in re.split(r"[\s,\.!\?;]+", text):
        if len(tok) > len(suffix) and tok.endswith(suffix):
            return True
    return False


# ---------------------------------------------------------------------------
# Entity attachment
# ---------------------------------------------------------------------------


def _mentions_in_span(
    mentions: Iterable[nlp_mod.EntityMention], start: int, end: int
) -> list[nlp_mod.EntityMention]:
    return [m for m in mentions if m.char_start >= start and m.char_end <= end]


# Tokens that the upstream NER sometimes mislabels as people/locations but
# are actually calendar words. Normalised (lowercase, no apostrophe).
_CALENDAR_WORDS = nlp_mod.TR_MONTHS | nlp_mod.TR_WEEKDAYS
# Lowercase conjunctions / fillers that must never appear in an entity name.
_STOPWORD_TOKENS = {
    "ve", "ile", "hatta", "bugün", "dün", "yarın", "bu", "şu", "o", "ben",
    "sen", "biz", "siz", "onlar", "orada", "burada", "şurada", "oradan",
    "buradan", "şuradan", "birlikte", "beraber",
}

_ORG_SUFFIXES = (
    "derneği", "derneğinin", "derneğine", "derneğinde", "derneğinden",
    "şirketi", "şirketinin", "şirketinde", "şirketinden", "şirketine",
    "üniversitesi", "üniversitesinde", "üniversitesine",
    "fakültesi", "fakültesinde", "fakültesinin",
    "müdürlüğü", "müdürlüğünde", "müdürlüğüne",
    "okulu", "okulunda", "okuluna",
    "holding", "holdingi", "holdinginin",
    "bankası", "bankasında", "bankasının",
    "hastanesi", "hastanesinde", "hastanesine",
)


def _clean_entity_token(surface: str) -> str:
    """Strip stopwords/punctuation from an NER surface span."""
    parts = [p for p in re.split(r"[\s,]+", surface.strip(" ,.;!?")) if p]
    parts = [p for p in parts if p.lower() not in _STOPWORD_TOKENS]
    parts = [p for p in parts if p.lower() not in _CALENDAR_WORDS]
    parts = [p for p in parts if not p.isdigit()]
    # Also drop apostrophe suffixes on each part.
    clean = []
    for p in parts:
        for marker in ("'", "’"):
            idx = p.find(marker)
            if idx > 0:
                p = p[:idx]
                break
        if p:
            clean.append(p)
    return " ".join(clean).strip()


def _augment_mentions(
    paragraph: str, parse: nlp_mod.ParseResult
) -> list[nlp_mod.EntityMention]:
    """Fix NER weaknesses:

    1. Filter entity spans that are actually month/weekday words or
       stopword-heavy fragments ("Nisan 2026, Ve").
    2. Add lowercase Turkish given names from the gazetteer.
    3. Detect organisations via compound suffixes ("X derneği").
    4. Keep relative location words as references, not concrete places.
    """
    out: list[nlp_mod.EntityMention] = []
    for m in parse.mentions:
        if m.mention_type == "LOCATION" and m.surface.lower() in nlp_mod.RELATIVE_LOCATION_WORDS:
            out.append(
                nlp_mod.EntityMention(
                    surface=m.surface,
                    lemma=m.lemma,
                    char_start=m.char_start,
                    char_end=m.char_end,
                    mention_type="PRONOUN",
                    source="pronoun",
                    score=m.score,
                    hint=m.hint or "Belirsiz yer göndergesi — neresi kastediliyor?",
                )
            )
            continue
        if m.mention_type in ("PERSON", "LOCATION", "ORG"):
            cleaned = _clean_entity_token(m.surface)
            if not cleaned or len(cleaned) < 2:
                continue
            # Recompute span to fit the cleaned token.
            idx = paragraph.lower().find(cleaned.lower(), max(0, m.char_start - 5))
            if idx == -1:
                idx = m.char_start
            out.append(
                nlp_mod.EntityMention(
                    surface=cleaned,
                    lemma=cleaned.lower(),
                    char_start=idx,
                    char_end=idx + len(cleaned),
                    mention_type=m.mention_type,
                    source=m.source,
                    score=m.score,
                    hint=m.hint,
                )
            )
        else:
            out.append(m)

    # Gazetteer-based lowercase-name detection.
    for tok_match in nlp_mod._TOKEN_RE.finditer(paragraph):
        low = tok_match.group(0).lower()
        if low in TR_GIVEN_NAMES:
            # Absorb a following capitalised/lowercase token as surname if it
            # looks like a proper second name ("emre alp").
            surface_end = tok_match.end()
            next_match = nlp_mod._TOKEN_RE.search(paragraph, surface_end)
            if next_match and next_match.start() - tok_match.end() <= 2:
                next_low = next_match.group(0).lower()
                if (
                    next_low not in _STOPWORD_TOKENS
                    and next_low not in _CALENDAR_WORDS
                    and len(next_low) >= 3
                    and next_low not in TR_GIVEN_NAMES
                ):
                    surface_end = next_match.end()
            surface = paragraph[tok_match.start():surface_end]
            out.append(
                nlp_mod.EntityMention(
                    surface=surface,
                    lemma=surface.lower(),
                    char_start=tok_match.start(),
                    char_end=surface_end,
                    mention_type="PERSON",
                    source="gazetteer",
                )
            )

    # ORG suffix detection: every (word, next_word) pair where next_word
    # is an org-suffix inflection ("X derneği", "X üniversitesi").
    tokens = list(nlp_mod._TOKEN_RE.finditer(paragraph))
    for idx, tok in enumerate(tokens[:-1]):
        nxt = tokens[idx + 1]
        if nxt.group(0).lower() in _ORG_SUFFIXES:
            # Skip fillers as the head noun.
            head_low = tok.group(0).lower()
            if head_low in _STOPWORD_TOKENS or head_low in _CALENDAR_WORDS:
                continue
            label = paragraph[tok.start() : nxt.end()]
            out.append(
                nlp_mod.EntityMention(
                    surface=label,
                    lemma=label.lower(),
                    char_start=tok.start(),
                    char_end=nxt.end(),
                    mention_type="ORG",
                    source="rules",
                )
            )

    return nlp_mod._dedupe_mentions(out)


def _clean_label(surface: str) -> str:
    # Strip Turkish genitive/possessive apostrophe suffixes.
    for marker in ("'", "’"):
        idx = surface.find(marker)
        if idx > 0:
            surface = surface[:idx]
            break
    return surface.strip().title()


# ---------------------------------------------------------------------------
# Event phrase heuristic
# ---------------------------------------------------------------------------

# Pull the verb-ending token as the event kernel and grab a few nouns
# around it. Crude but deterministic.
_VERB_TAIL_RE = re.compile(
    r"\b(\w+(?:tim|tım|tık|tik|dum|düm|dik|dık|dun|dük|duk|tuk|tük|duk|"
    r"acağım|eceğim|acak|ecek|yorum|yor|maktayım|mekteyim|mekte|makta|"
    r"dı|di|du|dü|tı|ti|tu|tü|mış|miş|muş|müş))\b",
    re.IGNORECASE,
)


def _event_phrase(clause: str) -> str:
    verb_hits = list(_VERB_TAIL_RE.finditer(clause))
    if not verb_hits:
        # Fall back to the clause itself, trimmed.
        return clause.strip()[:120]
    last = verb_hits[-1]
    # Take 4 words before the verb tail for context.
    words = clause[: last.end()].strip().split()
    return " ".join(words[-6:]).strip(",. ")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract(
    paragraph: str,
    recorded_at: dt.datetime | None = None,
) -> ExtractionResult:
    """Run the deterministic NLP pipeline on a Turkish paragraph.

    Returns an ``ExtractionResult`` with clause-level hints the LLM can
    consume, plus a mask map for privacy-safe cloud-LLM usage.
    """
    paragraph = paragraph or ""
    anchor = recorded_at or dt.datetime.now()
    parse = nlp_mod.analyze(paragraph)
    # Replace parse.mentions with the augmented/cleaned set so every
    # downstream consumer (clause hints, mask map) sees real entities only.
    parse.mentions = _augment_mentions(paragraph, parse)

    # Paragraph-level anchor date: if the user opens with "Bugün X Nisan
    # 2026", trust that over recorded_at for grounding *relative* phrases
    # later in the paragraph.
    paragraph_anchor = _paragraph_anchor_date(paragraph, anchor)

    clauses: list[ClauseHint] = []
    for idx, (text, cs, ce) in enumerate(_segment(paragraph)):
        date, time_hm, phrases = _ground_clause_date(text, anchor, paragraph_anchor)
        tense = _classify_tense(text)
        if not tense and phrases:
            # Fall back on temporal direction from grounded date.
            if date is not None:
                anchor_d = paragraph_anchor or anchor.date()
                if date < anchor_d:
                    tense = "Geçmiş"
                elif date == anchor_d:
                    tense = "Şu An"
                else:
                    tense = "Gelecek"
        subject_person, subject_pronoun, subject_verb, subject_tense = _infer_subject_from_conjugation(text)

        clause_mentions = _mentions_in_span(parse.mentions, cs, ce)
        persons = _dedup([_clean_label(m.surface) for m in clause_mentions if m.mention_type == "PERSON"])
        locations = _dedup([_clean_label(m.surface) for m in clause_mentions if m.mention_type == "LOCATION"])
        orgs = _dedup([_clean_label(m.surface) for m in clause_mentions if m.mention_type == "ORG"])
        references = _dedup([
            m.surface
            for m in clause_mentions
            if (
                m.mention_type == "PRONOUN"
                or m.surface.lower() in nlp_mod.RELATIVE_TIME_WORDS
                or m.surface.lower() in nlp_mod.RELATIVE_LOCATION_WORDS
            )
        ])

        clauses.append(
            ClauseHint(
                clause_index=idx,
                text=text,
                char_start=cs,
                char_end=ce,
                date_iso=date.isoformat() if date else "",
                time_hm=time_hm,
                time_phrases=phrases,
                zaman_dilimi=tense,
                persons=persons,
                locations=locations,
                orgs=orgs,
                references=references,
                subject_person=subject_person,
                subject_pronoun=subject_pronoun,
                subject_verb=subject_verb,
                subject_tense=subject_tense,
                event_phrase=_event_phrase(text),
            )
        )

    # Paragraph-wide dedup.
    all_persons = _dedup([p for c in clauses for p in c.persons])
    all_locations = _dedup([p for c in clauses for p in c.locations])
    all_orgs = _dedup([p for c in clauses for p in c.orgs])
    all_references = _dedup([p for c in clauses for p in c.references])

    masked, mask_map = _build_mask(paragraph, parse.mentions)

    return ExtractionResult(
        paragraph=paragraph,
        recorded_at=recorded_at,
        clauses=clauses,
        persons=all_persons,
        locations=all_locations,
        orgs=all_orgs,
        references=all_references,
        mask_map=mask_map,
        masked_paragraph=masked,
        parse=parse,
    )


def _paragraph_anchor_date(
    paragraph: str, fallback: dt.datetime
) -> dt.date | None:
    # Pick the first absolute date in the paragraph (often "Bugün 21 Nisan 2026").
    first = _parse_absolute_date(paragraph[:400])
    return first or (fallback.date() if fallback else None)


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        key = it.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _build_mask(
    paragraph: str, mentions: Iterable[nlp_mod.EntityMention]
) -> tuple[str, dict[str, str]]:
    """Replace person/location/org spans with opaque <MASK_n> tokens.

    The mask map goes back to the caller so remote-LLM output can be
    translated back to real identifiers locally.
    """
    maskable = [
        m for m in mentions
        if m.mention_type in ("PERSON", "LOCATION", "ORG")
    ]
    # Sort by start descending so replacements don't break offsets.
    maskable.sort(key=lambda m: m.char_start, reverse=True)
    # Assign stable IDs by *label*, so the same person is mask_1 everywhere.
    label_to_id: dict[str, str] = {}
    mask_map: dict[str, str] = {}
    kind_counters: dict[str, int] = {"PERSON": 0, "LOCATION": 0, "ORG": 0}

    # Walk front-to-back once for ID assignment, then back-to-front for
    # replacement.
    for m in sorted(maskable, key=lambda m: m.char_start):
        key = f"{m.mention_type}:{m.surface.lower()}"
        if key in label_to_id:
            continue
        kind_counters[m.mention_type] += 1
        token = f"<{m.mention_type}_{kind_counters[m.mention_type]}>"
        label_to_id[key] = token
        mask_map[token] = m.surface

    out = paragraph
    for m in maskable:
        key = f"{m.mention_type}:{m.surface.lower()}"
        token = label_to_id[key]
        out = out[: m.char_start] + token + out[m.char_end:]
    return out, mask_map


def unmask(text: str, mask_map: dict[str, str]) -> str:
    for token, real in mask_map.items():
        text = text.replace(token, real)
    return text
