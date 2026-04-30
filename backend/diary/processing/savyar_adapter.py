"""Optional sentence-level SAVYAR ML integration.

Hatirlaf does not call SAVYAR's low-level morphology internals here. It sends a
sentence to SAVYAR's own ML workflow through ``scripts/savyar_ml_bridge.py`` and
consumes the ranked candidates returned by that process.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SAVYAR_ROOT = _REPO_ROOT / "savyar"
_BRIDGE_PATH = _REPO_ROOT / "scripts" / "savyar_ml_bridge.py"
_DEFAULT_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class MorphCandidate:
    word: str
    root: str
    pos: str
    suffixes: tuple[str, ...]
    final_pos: str
    ml_score: float | None = None


@dataclass(frozen=True)
class SubjectTenseHint:
    person: str = ""
    pronoun: str = ""
    verb: str = ""
    tense_key: str = ""
    zaman_dilimi: str = ""
    root: str = ""
    suffixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProperCaseHint:
    label: str
    suffix: str


_PERSON_BY_SUFFIX = {
    "conjugation_1sg": ("1sg", "Ben"),
    "conjugation_2sg": ("2sg", "Sen"),
    "conjugation_3sg": ("3sg", "O"),
    "conjugation_1pl": ("1pl", "Biz"),
    "conjugation_2pl": ("2pl", "Siz"),
    "conjugation_3pl": ("3pl", "Onlar"),
}

_CASE_SUFFIXES = {
    "ablative_den": "ablative",
    "dative_e": "dative",
    "locative_de": "locative",
    "accusative_i": "accusative",
    "noun_compound": "genitive",
}


def enabled() -> bool:
    return bool(getattr(settings, "HATIRLAF_USE_SAVYAR", False))


def _python_executable() -> str:
    configured = getattr(settings, "HATIRLAF_SAVYAR_PYTHON", "")
    if configured:
        return configured
    bundled = _SAVYAR_ROOT / ".venv" / "bin" / "python"
    return str(bundled if bundled.exists() else sys.executable)


def _model_path() -> Path:
    return Path(
        getattr(
            settings,
            "HATIRLAF_SAVYAR_MODEL_PATH",
            _SAVYAR_ROOT / "ml" / "current_best.pt",
        )
    )


@lru_cache(maxsize=1024)
def _analyze_sentence_cached(text: str, timeout_seconds: float) -> tuple[tuple[MorphCandidate, ...], ...]:
    if not text.strip():
        return ()
    if not _BRIDGE_PATH.exists() or not _SAVYAR_ROOT.exists():
        logger.warning("SAVYAR bridge or root path is missing")
        return ()

    cmd = [
        _python_executable(),
        str(_BRIDGE_PATH),
        "--savyar-root",
        str(_SAVYAR_ROOT),
        "--model-path",
        str(_model_path()),
    ]
    try:
        completed = subprocess.run(
            cmd,
            input=json.dumps({"text": text}, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("SAVYAR ML sentence analysis timed out")
        return ()
    except OSError as exc:
        logger.warning("SAVYAR ML sentence analysis could not start: %s", exc)
        return ()

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        logger.warning("SAVYAR ML sentence analysis failed: %s", detail[:500])
        return ()

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        logger.warning("SAVYAR ML returned invalid JSON: %s", exc)
        return ()

    out: list[tuple[MorphCandidate, ...]] = []
    for word_info in payload.get("words", []):
        candidates = []
        for raw in word_info.get("candidates", []):
            candidates.append(
                MorphCandidate(
                    word=raw.get("word", word_info.get("word", "")),
                    root=raw.get("root", ""),
                    pos=raw.get("pos", ""),
                    suffixes=tuple(raw.get("suffixes", [])),
                    final_pos=raw.get("final_pos", ""),
                    ml_score=raw.get("ml_score"),
                )
            )
        out.append(tuple(candidates))
    return tuple(out)


def analyze_sentence(text: str, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS) -> tuple[tuple[MorphCandidate, ...], ...]:
    if not enabled():
        return ()
    return _analyze_sentence_cached(text, timeout_seconds)


def analyze_word(word: str, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS) -> tuple[MorphCandidate, ...]:
    sentence = analyze_sentence(word, timeout_seconds)
    return sentence[0] if sentence else ()


def best_lemma(word: str) -> str:
    candidates = [c for c in analyze_word(word) if not c.pos.startswith("cc_") and c.root]
    if not candidates:
        return ""
    return candidates[0].root


def closed_class_categories(word: str) -> set[str]:
    categories = set()
    for cand in analyze_word(word):
        if cand.pos.startswith("cc_"):
            categories.add(cand.pos[3:])
        for suffix in cand.suffixes:
            if suffix.startswith("cc_"):
                categories.add(suffix[3:])
    return categories


def infer_subject_tense(clause: str) -> SubjectTenseHint:
    words = list(_token_surfaces(clause))
    sentence = analyze_sentence(clause)
    best: SubjectTenseHint | None = None
    for idx, candidates in enumerate(sentence):
        surface = words[idx] if idx < len(words) else (candidates[0].word if candidates else "")
        if not candidates:
            continue
        hint = _hint_from_candidate(surface, candidates[0])
        if not hint.person and not hint.zaman_dilimi:
            continue
        best = hint
    return best or SubjectTenseHint()


def _token_surfaces(text: str):
    for match in re.finditer(r"[A-Za-zÇĞİıÖŞÜçğıöşüÂâÎîÛû]+(?:['’][A-Za-zÇĞİıÖŞÜçğıöşüÂâÎîÛû]+)?", text):
        yield match.group(0).strip("'’")


def _hint_from_candidate(surface: str, cand: MorphCandidate) -> SubjectTenseHint:
    suffixes = cand.suffixes
    person = pronoun = ""
    for suffix in reversed(suffixes):
        if suffix in _PERSON_BY_SUFFIX:
            person, pronoun = _PERSON_BY_SUFFIX[suffix]
            break

    tense_key = _tense_key(suffixes)
    zaman = _zaman_dilimi(suffixes, tense_key)
    return SubjectTenseHint(
        person=person,
        pronoun=pronoun,
        verb=surface if person or tense_key else "",
        tense_key=tense_key,
        zaman_dilimi=zaman,
        root=cand.root,
        suffixes=suffixes,
    )


def _tense_key(suffixes: tuple[str, ...]) -> str:
    suffix_set = set(suffixes)
    has_past = bool(suffix_set & {"pasttense_di", "pasttense_noundi"})
    if "nounifier_ecek" in suffix_set:
        return "future_past" if has_past else "future"
    if "continuous_iyor" in suffix_set:
        return "present_continuous_past" if has_past else "present_continuous"
    if "pastfactative_miş" in suffix_set or "copula_mis" in suffix_set:
        return "reported_past"
    if has_past:
        return "definite_past"
    if "infinitive_me" in suffix_set and "composessive_li" in suffix_set:
        return "necessitative"
    if "if_se" in suffix_set:
        return "conditional"
    if "factative_ir" in suffix_set:
        return "aorist"
    return ""


def _zaman_dilimi(suffixes: tuple[str, ...], tense_key: str) -> str:
    suffix_set = set(suffixes)
    if suffix_set & {"pasttense_di", "pasttense_noundi", "pastfactative_miş", "copula_mis"}:
        return "Geçmiş"
    if tense_key == "future":
        return "Gelecek"
    if tense_key.startswith("present_continuous"):
        return "Şu An"
    return ""


def recover_known_proper_case(surface: str, known_labels: set[str]) -> ProperCaseHint | None:
    lowered = surface.lower()
    for label in sorted(known_labels, key=len, reverse=True):
        low_label = label.lower()
        if not lowered.startswith(low_label) or lowered == low_label:
            continue
        tail = lowered[len(low_label):]
        tail_candidate = low_label + "'" + tail
        for cand in analyze_word(tail_candidate):
            case = next((_CASE_SUFFIXES[s] for s in cand.suffixes if s in _CASE_SUFFIXES), "")
            if case:
                return ProperCaseHint(label=label, suffix=case)
        if tail in {"dan", "den", "tan", "ten"}:
            return ProperCaseHint(label=label, suffix="ablative")
        if tail in {"da", "de", "ta", "te"}:
            return ProperCaseHint(label=label, suffix="locative")
        if tail in {"a", "e", "ya", "ye"}:
            return ProperCaseHint(label=label, suffix="dative")
    return None
