#!/usr/bin/env python3
"""Compare Hatirlaf subject/tense inference on BOUN treebank verbs.

This is a lightweight diagnostic, not a formal linguistic benchmark. It uses
BOUN UD features as labels, then runs the same Hatirlaf inference functions with
SAVYAR disabled and enabled so changes are measurable before wiring more of the
pipeline to the adapter.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
DEFAULT_CONLLU = REPO_ROOT / "savyar" / "data" / "boun_treebank" / "tr_boun-ud-dev.conllu"


@dataclass(frozen=True)
class VerbExample:
    sentence: str
    surface: str
    person: str
    zaman_dilimi: str


def _setup_django() -> None:
    sys.path.insert(0, str(BACKEND_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "diary_backend.settings")
    os.environ.setdefault("HATIRLAF_PRELOAD_MODELS", "0")
    import django

    django.setup()


def _parse_feats(raw: str) -> dict[str, str]:
    if not raw or raw == "_":
        return {}
    out = {}
    for part in raw.split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key] = value
    return out


def _expected_person(feats: dict[str, str]) -> str:
    person = feats.get("Person")
    number = feats.get("Number")
    if not person or not number:
        return ""
    suffix = "sg" if number == "Sing" else "pl" if number == "Plur" else ""
    return f"{person}{suffix}" if suffix else ""


def _expected_zaman(feats: dict[str, str]) -> str:
    tense = feats.get("Tense", "")
    aspect = feats.get("Aspect", "")
    evident = feats.get("Evident", "")
    if tense in {"Past", "Pqb", "Pqp"} or evident == "Nfh":
        return "Gecmis"
    if tense in {"Fut", "Future"}:
        return "Gelecek"
    if aspect == "Prog" or tense == "Pres":
        return "Su An"
    return ""


def _ascii_zaman(value: str) -> str:
    return value.replace("Geçmiş", "Gecmis").replace("Şu An", "Su An")


def load_examples(path: Path, limit_sentences: int) -> list[VerbExample]:
    examples: list[VerbExample] = []
    sentence_tokens: list[str] = []
    sentence_rows: list[tuple[str, str, dict[str, str]]] = []
    seen_sentences = 0

    def flush() -> None:
        nonlocal seen_sentences
        if not sentence_rows:
            return
        sentence = " ".join(sentence_tokens)
        for surface, upos, feats in sentence_rows:
            if upos not in {"VERB", "AUX"}:
                continue
            person = _expected_person(feats)
            zaman = _expected_zaman(feats)
            if person or zaman:
                examples.append(VerbExample(sentence, surface, person, zaman))
        seen_sentences += 1

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                flush()
                sentence_tokens = []
                sentence_rows = []
                if limit_sentences and seen_sentences >= limit_sentences:
                    break
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 6 or "-" in fields[0] or "." in fields[0]:
                continue
            surface = fields[1]
            upos = fields[3]
            feats = _parse_feats(fields[5])
            sentence_tokens.append(surface)
            sentence_rows.append((surface, upos, feats))

    if sentence_rows and (not limit_sentences or seen_sentences < limit_sentences):
        flush()
    return examples


def evaluate(examples: list[VerbExample], use_savyar: bool) -> dict[str, int]:
    from django.test import override_settings
    from diary.processing import extractor

    metrics = {
        "examples": 0,
        "subject_labeled": 0,
        "subject_correct": 0,
        "tense_labeled": 0,
        "tense_correct": 0,
    }
    with override_settings(HATIRLAF_USE_SAVYAR=use_savyar, HATIRLAF_USE_BERTURK=False):
        for example in examples:
            metrics["examples"] += 1
            person, _pronoun, _verb, _tense_key = extractor._infer_subject_from_conjugation(example.surface)
            zaman = _ascii_zaman(extractor._classify_tense(example.surface))

            if example.person:
                metrics["subject_labeled"] += 1
                if person == example.person:
                    metrics["subject_correct"] += 1
            if example.zaman_dilimi:
                metrics["tense_labeled"] += 1
                if zaman == example.zaman_dilimi:
                    metrics["tense_correct"] += 1
    return metrics


def _pct(num: int, den: int) -> str:
    return "n/a" if not den else f"{num / den * 100:.1f}%"


def print_metrics(label: str, metrics: dict[str, int]) -> None:
    print(f"\n{label}")
    print(f"examples: {metrics['examples']}")
    print(
        "subject: "
        f"{metrics['subject_correct']}/{metrics['subject_labeled']} "
        f"({_pct(metrics['subject_correct'], metrics['subject_labeled'])})"
    )
    print(
        "tense: "
        f"{metrics['tense_correct']}/{metrics['tense_labeled']} "
        f"({_pct(metrics['tense_correct'], metrics['tense_labeled'])})"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conllu", type=Path, default=DEFAULT_CONLLU)
    parser.add_argument("--limit-sentences", type=int, default=300)
    args = parser.parse_args()

    _setup_django()
    examples = load_examples(args.conllu, args.limit_sentences)
    if not examples:
        print(f"No verb examples found in {args.conllu}", file=sys.stderr)
        return 1

    print(f"Loaded {len(examples)} verb examples from {args.conllu}")
    baseline = evaluate(examples, use_savyar=False)
    savyar = evaluate(examples, use_savyar=True)
    print_metrics("Hatirlaf regex baseline", baseline)
    print_metrics("Hatirlaf + SAVYAR", savyar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

