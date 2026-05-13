"""Morphology, tokenization, and entity label normalization."""

from __future__ import annotations

import logging
import re

from . import savyar_adapter
from .name_gazetteer import TR_GIVEN_NAMES
from .nlp_models import TR_CITIES, TR_COUNTRIES, Token

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-zÇĞİıÖŞÜçğıöşüÂâÎîÛû]+(?:['’][A-Za-zÇĞİıÖŞÜçğıöşüÂâÎîÛû]+)?")

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

_ENTITY_SUFFIX_HINTS = (
    "da", "de", "ta", "te", "dan", "den", "tan", "ten",
    "la", "le", "yla", "yle", "ya", "ye",
    "a", "e", "ı", "i", "u", "ü", "yi", "yı", "yu", "yü",
    "ın", "in", "un", "ün", "nın", "nin", "nun", "nün",
)


def fallback_lemma(word: str) -> str:
    savyar_lemma = savyar_adapter.best_lemma(word)
    if savyar_lemma:
        return savyar_lemma

    low = word.lower()
    for suf in _FALLBACK_SUFFIXES:
        if low.endswith(suf) and len(low) > len(suf) + 1:
            return low[: -len(suf)]
    return low


def normalize_entity_lemma(text: str) -> str:
    """Return a neutral lowercase lemma for an entity phrase."""
    roots: list[str] = []
    for part in re.split(r"(\s+)", str(text or "").strip()):
        if not part:
            continue
        if part.isspace():
            roots.append(part)
            continue
        roots.append(_normalize_entity_token(part))
    return "".join(roots).strip()


def normalize_entity_label(text: str) -> str:
    """Return a user-facing neutral label for an entity phrase."""
    normalized = normalize_entity_lemma(text)
    if not normalized:
        return str(text or "").strip()
    return " ".join(_tr_title(token) for token in normalized.split())


def _strip_apostrophe_suffix(text: str) -> str:
    """Turkish proper nouns take inflection after an apostrophe."""
    for marker in ("'", "’"):
        idx = text.find(marker)
        if idx > 0:
            return text[:idx]
    return text


def _normalize_entity_token(token: str) -> str:
    raw = token.strip()
    bare = _strip_apostrophe_suffix(raw)
    if not bare:
        return ""
    low = bare.lower()
    root = savyar_adapter.best_lemma(bare)
    if not root:
        root = fallback_lemma(bare)
    if not root:
        root = low
    root = root.lower()
    if _should_keep_original_entity_form(raw, low, root):
        return low
    return root


def _should_keep_original_entity_form(raw: str, low: str, root: str) -> bool:
    if root == low:
        return True
    if "'" in raw or "’" in raw:
        return False
    if low in TR_GIVEN_NAMES or low in TR_CITIES or low in TR_COUNTRIES:
        return True
    if any(low.endswith(suffix) for suffix in _ENTITY_SUFFIX_HINTS):
        return False
    return True


def _tr_title(token: str) -> str:
    if not token:
        return ""
    low = token.lower()
    first = low[0]
    if first == "i":
        first = "İ"
    elif first == "ı":
        first = "I"
    elif first == "ş":
        first = "Ş"
    elif first == "ğ":
        first = "Ğ"
    elif first == "ü":
        first = "Ü"
    elif first == "ö":
        first = "Ö"
    elif first == "ç":
        first = "Ç"
    else:
        first = first.upper()
    return first + low[1:]


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def morph_analyze(text: str, analyzer, backend: str | None) -> list[Token]:
    tokens: list[Token] = []
    for surface, start, end in tokenize(text):
        lemma = surface.lower()
        pos = ""
        prev = start - 1
        while prev >= 0 and text[prev] in " \t":
            prev -= 1
        at_sentence_start = prev < 0 or text[prev] in ".!?\n"
        is_proper = surface[:1].isupper() and not at_sentence_start
        if backend == "zeyrek" and analyzer is not None:
            try:
                parses = analyzer.analyze(surface)
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
