"""Turkish morphological analysis + Named Entity Recognition.

Primary tools (per spec):
  * Zeyrek — pure-Python port of Zemberek for lemmatization / morphology.
  * BERTurk via Hugging Face transformers — token-level NER.

Graceful fallbacks:
  * If Zeyrek isn't installed, we use a lightweight suffix-stripper covering
    the most common Turkish cases. Good enough to power conflict detection.
  * If BERTurk isn't installed (or HATIRLAF_USE_BERTURK=0), we run a
    rule-based NER that relies on capitalization, a built-in Turkish
    location/person gazetteer, and Zeyrek-tagged proper-noun morphotags.

Both paths emit the same normalized output.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Iterable

from django.conf import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Token:
    surface: str
    lemma: str
    char_start: int
    char_end: int
    pos: str = ""
    is_proper: bool = False


@dataclass
class EntityMention:
    surface: str
    lemma: str
    char_start: int
    char_end: int
    mention_type: str  # PERSON | LOCATION | TIME | EVENT | ORG | PRONOUN
    source: str = ""  # "berturk" | "rules" | "pronoun" | "time"
    score: float = 1.0
    hint: str = ""


@dataclass
class ParseResult:
    tokens: list[Token] = field(default_factory=list)
    mentions: list[EntityMention] = field(default_factory=list)
    lemma_text: str = ""
    nlp_backend: str = "rules"
    ner_backend: str = "rules"


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# Turkish personal & demonstrative pronouns that frequently introduce
# ambiguous references in diary speech.
AMBIGUOUS_PRONOUNS = {
    # Third-person singular/plural personal & demonstratives
    "o", "onu", "ona", "onun", "onda", "ondan", "onunla", "onlar", "onları",
    "onlara", "onların", "onlarda", "onlardan",
    "bu", "bunu", "buna", "bunun", "bunda", "bundan", "bununla",
    "şu", "şunu", "şuna", "şunun", "şunda", "şundan", "şununla",
    "bunlar", "bunları", "bunlara", "bunların",
    "şunlar", "şunları", "şunlara", "şunların",
    "kendisi", "kendileri",
    # Demonstrative/locative referential adjectives. In speech these often
    # stand in for a person/place/event established in earlier context.
    "oradaki", "oradakini", "oradakine", "oradakinin", "oradakiler",
    "buradaki", "buradakini", "buradakine", "buradakinin", "buradakiler",
    "şuradaki", "şuradakini", "şuradakine", "şuradakinin", "şuradakiler",
}

# Surface forms that hint at a relative / vague time expression.
RELATIVE_TIME_WORDS = {
    "dün", "bugün", "yarın", "geçen", "önceki", "gelecek", "gelecekteki",
    "önce", "sonra", "biraz", "az", "şimdi", "az önce", "biraz önce",
    "demin", "az sonra", "öbür", "evvelki", "ertesi",
    "dünkü", "dünkünü", "dünküne", "dünkünün", "bugünkü", "bugünkünü",
    "bugünküne", "bugünkünün", "yarınki", "yarınkini", "yarınkine",
    "yarınkinin",
    "hafta", "ay", "yıl", "gün", "akşam", "sabah", "gece", "öğle", "öğleden",
}

# Relative location words — handled specially because "burası" is
# interpretable only with context.
RELATIVE_LOCATION_WORDS = {
    "burası", "orası", "şurası", "burada", "orada", "şurada",
    "buraya", "oraya", "şuraya", "buradan", "oradan", "şuradan",
    "buradaki", "oradaki", "şuradaki",
}

# Built-in gazetteer used by the rule-based NER fallback. Extend over time.
TR_CITIES = {
    "istanbul", "ankara", "izmir", "bursa", "antalya", "konya", "adana",
    "gaziantep", "şanlıurfa", "mersin", "diyarbakır", "kayseri", "eskişehir",
    "samsun", "denizli", "trabzon", "malatya", "erzurum", "kocaeli",
    "sakarya", "manisa", "hatay", "balıkesir", "tekirdağ", "van", "aydın",
    "muğla", "tokat", "elazığ", "sivas", "rize", "ordu", "giresun", "artvin",
}

TR_COUNTRIES = {
    "türkiye", "almanya", "fransa", "italya", "ispanya", "yunanistan",
    "bulgaristan", "rusya", "ukrayna", "amerika", "abd", "ingiltere",
    "japonya", "çin", "kore", "brezilya", "kanada", "hollanda", "belçika",
}

# Turkish months for absolute-time detection.
TR_MONTHS = {
    "ocak", "şubat", "mart", "nisan", "mayıs", "haziran",
    "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık",
}

TR_WEEKDAYS = {
    "pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi", "pazar",
}


# ---------------------------------------------------------------------------
# Zeyrek / fallback morphology
# ---------------------------------------------------------------------------

_analyzer_lock = threading.Lock()
_cached_analyzer = None
_cached_backend: str | None = None


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


# Ordered Turkish suffixes for the naive fallback stripper.
_FALLBACK_SUFFIXES = sorted(
    [
        "larımızdaki", "lerimizdeki", "larınızdaki", "lerinizdeki",
        "larımızdan", "lerimizden", "larınızdan", "lerinizden",
        "larımıza", "lerimize", "larınıza", "lerinize",
        "larımız", "lerimiz", "larınız", "leriniz",
        "larında", "lerinde", "larından", "lerinden",
        "larına", "lerine", "larını", "lerini",
        "mızın", "mizin", "nızın", "nizin",
        "larım", "lerim", "ların", "lerin",
        "ların", "lerin", "ları", "leri",
        "lar", "ler",
        "ımız", "imiz", "umuz", "ümüz",
        "ınız", "iniz", "unuz", "ünüz",
        "ında", "inde", "unda", "ünde",
        "ından", "inden", "undan", "ünden",
        "ına", "ine", "una", "üne",
        "ını", "ini", "unu", "ünü",
        "dan", "den", "tan", "ten",
        "yla", "yle",
        "nın", "nin", "nun", "nün",
        "ım", "im", "um", "üm",
        "ın", "in", "un", "ün",
        "da", "de", "ta", "te",
        "ya", "ye", "a", "e",
        "ı", "i", "u", "ü",
        "yi", "yı", "yu", "yü",
    ],
    key=len,
    reverse=True,
)


def fallback_lemma(word: str) -> str:
    low = word.lower()
    for suf in _FALLBACK_SUFFIXES:
        if low.endswith(suf) and len(low) > len(suf) + 1:
            return low[: -len(suf)]
    return low


# ---------------------------------------------------------------------------
# BERTurk / rule-based NER
# ---------------------------------------------------------------------------

_ner_lock = threading.Lock()
_cached_ner = None
_cached_ner_backend: str | None = None


def _load_ner():
    global _cached_ner, _cached_ner_backend
    with _ner_lock:
        if _cached_ner is not None or _cached_ner_backend == "rules":
            return _cached_ner, _cached_ner_backend
        if not getattr(settings, "HATIRLAF_USE_BERTURK", False):
            _cached_ner_backend = "rules"
            return None, "rules"
        try:
            from transformers import pipeline  # type: ignore

            logger.info("Loading BERTurk NER pipeline (savasy/bert-base-turkish-ner-cased)")
            _cached_ner = pipeline(
                "token-classification",
                model="savasy/bert-base-turkish-ner-cased",
                aggregation_strategy="simple",
            )
            _cached_ner_backend = "berturk"
        except Exception as exc:
            logger.warning("BERTurk NER unavailable (%s). Falling back to rules.", exc)
            _cached_ner_backend = "rules"
        return _cached_ner, _cached_ner_backend


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

# Matches Turkish word characters including dotted/undotted i variants.
_TOKEN_RE = re.compile(r"[A-Za-zÇĞİıÖŞÜçğıöşüÂâÎîÛû]+(?:['’][A-Za-zÇĞİıÖŞÜçğıöşüÂâÎîÛû]+)?")


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Analysis entry point
# ---------------------------------------------------------------------------


def analyze(text: str) -> ParseResult:
    """Run the full local NLP pipeline on a Turkish transcript."""
    if not text.strip():
        return ParseResult()

    analyzer, nlp_backend = _load_analyzer()
    tokens = _morph_analyze(text, analyzer, nlp_backend)

    mentions, ner_backend = _extract_entities(text, tokens)
    mentions.extend(_find_pronouns(text))
    mentions.extend(_find_relative_times(text))
    mentions.extend(_find_relative_locations(text))
    mentions.extend(_find_absolute_times(text))
    mentions = _dedupe_mentions(mentions)

    lemma_text = " ".join(t.lemma for t in tokens)
    return ParseResult(
        tokens=tokens,
        mentions=mentions,
        lemma_text=lemma_text,
        nlp_backend=nlp_backend or "rules",
        ner_backend=ner_backend,
    )


def _morph_analyze(text: str, analyzer, backend: str | None) -> list[Token]:
    tokens: list[Token] = []
    for surface, start, end in tokenize(text):
        lemma = surface.lower()
        pos = ""
        # Walk backwards past whitespace so "Foo. Bar" correctly flags Bar
        # as sentence-initial (not proper).
        prev = start - 1
        while prev >= 0 and text[prev] in " \t":
            prev -= 1
        at_sentence_start = prev < 0 or text[prev] in ".!?\n"
        is_proper = surface[:1].isupper() and not at_sentence_start
        if backend == "zeyrek" and analyzer is not None:
            try:
                parses = analyzer.analyze(surface)
                # zeyrek returns [[Parse, Parse, ...]] per sentence
                if parses and parses[0]:
                    best = parses[0][0]
                    lemma = (best.lemma or surface).lower()
                    pos = best.pos or ""
                    if "Prop" in (best.morphemes or []) or pos == "Noun" and surface[:1].isupper():
                        is_proper = True
            except Exception:
                lemma = fallback_lemma(surface)
        else:
            lemma = fallback_lemma(surface)
        tokens.append(
            Token(
                surface=surface,
                lemma=lemma,
                char_start=start,
                char_end=end,
                pos=pos,
                is_proper=is_proper,
            )
        )
    return tokens


def _extract_entities(text: str, tokens: list[Token]) -> tuple[list[EntityMention], str]:
    ner, backend = _load_ner()
    if backend == "berturk" and ner is not None:
        try:
            return _berturk_entities(text, ner), "berturk"
        except Exception as exc:
            logger.warning("BERTurk inference failed (%s). Using rules.", exc)
    return _rule_entities(text, tokens), "rules"


def _berturk_entities(text: str, ner) -> list[EntityMention]:
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
                lemma=surface.lower(),
                char_start=start,
                char_end=end,
                mention_type=mtype,
                source="berturk",
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

        # Skip capitalized pronouns/relative-time/location words — they're
        # the job of the dedicated pronoun/time detectors, not the
        # proper-noun-run classifier.
        bare = _strip_apostrophe_suffix(low)
        if bare in AMBIGUOUS_PRONOUNS or bare in RELATIVE_TIME_WORDS or bare in RELATIVE_LOCATION_WORDS:
            i += 1
            continue

        # Multi-word proper-noun run (consecutive capitalised tokens, each
        # *not* at sentence-start).
        if t.is_proper and len(t.surface) > 1:
            j = i
            while j + 1 < n and tokens[j + 1].is_proper:
                between = text[tokens[j].char_end : tokens[j + 1].char_start]
                if any(ch in ".!?\n" for ch in between):
                    break
                nxt_low = _strip_apostrophe_suffix(tokens[j + 1].surface.lower())
                if nxt_low in TR_WEEKDAYS or nxt_low in TR_MONTHS:
                    break
                j += 1
            start = t.char_start
            end = tokens[j].char_end
            surface = text[start:end]

            # Strip apostrophe suffixes from each sub-token when classifying.
            bare_parts = [_strip_apostrophe_suffix(text[tk.char_start:tk.char_end].lower()) for tk in tokens[i:j + 1]]
            span_lower = " ".join(bare_parts)

            mtype = "PERSON"
            if span_lower in TR_CITIES or span_lower in TR_COUNTRIES:
                mtype = "LOCATION"
            elif any(p in TR_CITIES or p in TR_COUNTRIES for p in bare_parts):
                mtype = "LOCATION"
            elif len(bare_parts) == 1 and _has_apostrophe_location_case(surface):
                mtype = "LOCATION"

            out.append(
                EntityMention(
                    surface=surface,
                    lemma=span_lower,
                    char_start=start,
                    char_end=end,
                    mention_type=mtype,
                    source="rules",
                )
            )
            i = j + 1
            continue

        # Weekday / month words (TIME) — case-insensitive.
        if bare in TR_WEEKDAYS or bare in TR_MONTHS:
            out.append(
                EntityMention(
                    surface=t.surface,
                    lemma=bare,
                    char_start=t.char_start,
                    char_end=t.char_end,
                    mention_type="TIME",
                    source="rules",
                )
            )
        # Known city / country (case-insensitive, apostrophe-stripped).
        elif bare in TR_CITIES or bare in TR_COUNTRIES:
            out.append(
                EntityMention(
                    surface=t.surface,
                    lemma=bare,
                    char_start=t.char_start,
                    char_end=t.char_end,
                    mention_type="LOCATION",
                    source="rules",
                )
            )
        i += 1
    return out


def _strip_apostrophe_suffix(text: str) -> str:
    """Turkish proper nouns take inflection after an apostrophe
    ("Ankara'ya"→"ankara"). Strip everything from the first apostrophe on."""
    for marker in ("'", "’"):
        idx = text.find(marker)
        if idx > 0:
            return text[:idx]
    return text


def _has_apostrophe_location_case(text: str) -> bool:
    """Return True for proper nouns carrying locative/ablative case.

    This catches STT/user text like "Kadıköy'de" or "Okuldan" when the
    gazetteer is incomplete, without treating dative person mentions
    ("Ayşe'ye") as places.
    """
    low = text.lower()
    for marker in ("'", "’"):
        idx = low.find(marker)
        if idx > 0:
            suffix = low[idx + 1 :]
            return suffix.startswith(("da", "de", "ta", "te", "dan", "den", "tan", "ten"))
    return False


def _find_pronouns(text: str) -> list[EntityMention]:
    out: list[EntityMention] = []
    for m in _TOKEN_RE.finditer(text):
        if m.group(0).lower() in AMBIGUOUS_PRONOUNS:
            out.append(
                EntityMention(
                    surface=m.group(0),
                    lemma=m.group(0).lower(),
                    char_start=m.start(),
                    char_end=m.end(),
                    mention_type="PRONOUN",
                    source="pronoun",
                    hint="Belirsiz zamir — kime atıfta bulunuyor?",
                )
            )
    return out


def _find_relative_times(text: str) -> list[EntityMention]:
    out: list[EntityMention] = []
    for m in _TOKEN_RE.finditer(text):
        low = m.group(0).lower()
        if low in RELATIVE_TIME_WORDS:
            out.append(
                EntityMention(
                    surface=m.group(0),
                    lemma=low,
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
                    lemma=low,
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
    # Patterns like "14 Mart", "14 Mart 2026", "14/03/2026", "14.03.2026"
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
                    lemma=m.group(0).lower(),
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
        # Skip if contained in a previously-kept mention.
        if any(
            prev.char_start <= m.char_start and m.char_end <= prev.char_end
            and (prev.char_end - prev.char_start) > (m.char_end - m.char_start)
            for prev in out
        ):
            continue
        # Remove previously-kept mentions strictly contained in this one.
        out = [
            prev for prev in out
            if not (
                m.char_start <= prev.char_start
                and prev.char_end <= m.char_end
                and (m.char_end - m.char_start) > (prev.char_end - prev.char_start)
            )
        ]
        # Replace exact duplicates, preferring BERTurk over rules.
        dup = next(
            (
                i for i, prev in enumerate(out)
                if prev.char_start == m.char_start and prev.char_end == m.char_end
            ),
            None,
        )
        if dup is not None:
            if out[dup].source == "rules" and m.source == "berturk":
                out[dup] = m
            continue
        out.append(m)
    out.sort(key=lambda m: m.char_start)
    return out
