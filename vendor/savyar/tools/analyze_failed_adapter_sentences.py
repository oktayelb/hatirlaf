import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_SPECS = [
    ("metu", "data/metu_treebank/treebank_adapted.jsonl", "data/metu_treebank/treebank_adapted_unmatched.jsonl"),
    ("google", "data/google_treebank/treebank_adapted.jsonl", "data/google_treebank/treebank_adapted_unmatched.jsonl"),
    ("boun", "data/boun_treebank/treebank_adapted.jsonl", "data/boun_treebank/treebank_adapted_unmatched.jsonl"),
]

MEASURE_TOKENS = {
    "gr", "kg", "g", "ml", "lt", "l", "cc", "dk", "adet", "kutu", "dal",
    "bardak", "kaşık", "tatlı", "çay", "yemek", "su", "paket",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain adapter failed_sentences by grouping them into fixable categories."
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Max example sentences per category.",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_failed_sentence(entry: Dict) -> bool:
    return not any(word.get("suffixes") for word in entry.get("words", []))


def normalize_token(token: str) -> str:
    return token.strip().lower()


def classify_failed_sentence(entry: Dict, unmappable_surfaces: set) -> Tuple[str, str]:
    sentence = entry.get("original_sentence", "")
    words = entry.get("words", [])
    tokens = [normalize_token(w.get("word", "")) for w in words if w.get("word", "")]

    if any(tok in unmappable_surfaces for tok in tokens):
        return (
            "contains_unmappable_token",
            "At least one token in the sentence has treebank features with no Savyar mapping yet.",
        )

    if any("http" in tok or ".org" in tok or ".com" in tok or "." in tok for tok in tokens):
        return (
            "url_or_symbolic_fragment",
            "Mostly URL / symbolic material; not a meaningful suffix-training sentence.",
        )

    if any(any(ch.isdigit() for ch in tok) for tok in tokens):
        if any(tok in MEASURE_TOKENS for tok in tokens):
            return (
                "numeric_measurement_fragment",
                "Recipe / dosage / tabular fragment with numbers and measure words; typically no trainable suffixes.",
            )
        return (
            "numeric_fragment",
            "Contains numerals or alphanumeric labels and no suffix-bearing token survived.",
        )

    if len(tokens) == 1:
        return (
            "single_token_fragment",
            "Single-token sentence fragment; often just a lemma/adverb/title with no suffix supervision value.",
        )

    if len(tokens) <= 4:
        return (
            "short_nominal_fragment",
            "Very short nominal/adjectival fragment with no suffix-bearing token.",
        )

    return (
        "long_suffixless_fragment",
        "Longer sentence fragment, but every token is bare or closed-class so there is no suffix token to train on.",
    )


def build_unmappable_surface_set(unmatched_entries: List[Dict]) -> set:
    surfaces = set()
    for entry in unmatched_entries:
        reason = str(entry.get("reason", ""))
        if "unmappable" in reason:
            surface = normalize_token(entry.get("surface", ""))
            if surface:
                surfaces.add(surface)
    return surfaces


def analyze_dataset(name: str, adapted_path: str, unmatched_path: str, limit: int) -> Dict:
    adapted_entries = load_jsonl(adapted_path)
    unmatched_entries = load_jsonl(unmatched_path)
    unmappable_surfaces = build_unmappable_surface_set(unmatched_entries)

    failed_entries = [entry for entry in adapted_entries if is_failed_sentence(entry)]
    categories = defaultdict(lambda: {"count": 0, "why": "", "examples": []})

    for entry in failed_entries:
        category, why = classify_failed_sentence(entry, unmappable_surfaces)
        bucket = categories[category]
        bucket["count"] += 1
        bucket["why"] = why
        if len(bucket["examples"]) < limit:
            bucket["examples"].append(
                {
                    "sentence": entry.get("original_sentence", ""),
                    "tokens": [w.get("word", "") for w in entry.get("words", [])],
                }
            )

    fix_advice = {
        "contains_unmappable_token": "Add the missing treebank→Savyar mapping in the adapter. Current known case is METU `Time`.",
        "url_or_symbolic_fragment": "Do not treat as adapter bug. Exclude from failed-sentence KPI or filter these fragments before adaptation.",
        "numeric_measurement_fragment": "Do not treat as adapter bug. Recipe/list fragments contain little or no suffix supervision; optionally filter them from evaluation stats.",
        "numeric_fragment": "Usually not worth fixing in morphology code. Consider excluding from failed-sentence KPI.",
        "single_token_fragment": "Usually not an adapter failure. If the token should be closed-class, add it to the closed-class lexicon; otherwise keep as suffixless fragment.",
        "short_nominal_fragment": "Usually just a suffixless phrase, not a mapping failure. Consider excluding such sentences from failed-sentence KPI.",
        "long_suffixless_fragment": "Inspect manually; if many tokens are actually closed-class, enrich the closed-class inventory. Otherwise treat as non-trainable input.",
    }

    ordered_categories = dict(
        sorted(categories.items(), key=lambda item: (-item[1]["count"], item[0]))
    )
    for category, info in ordered_categories.items():
        info["how_to_fix"] = fix_advice.get(category, "")

    return {
        "dataset": name,
        "adapted_path": adapted_path,
        "unmatched_path": unmatched_path,
        "failed_sentence_count": len(failed_entries),
        "unmappable_token_count": len(unmappable_surfaces),
        "categories": ordered_categories,
    }


def print_report(report: Dict) -> None:
    print(f"[{report['dataset']}] failed_sentences={report['failed_sentence_count']}")
    print(f"  adapted:   {report['adapted_path']}")
    print(f"  unmatched: {report['unmatched_path']}")
    print(f"  unmappable token types seen: {report['unmappable_token_count']}")
    print()

    for category, info in report["categories"].items():
        print(f"  {category}: {info['count']}")
        print(f"    why: {info['why']}")
        print(f"    fix: {info['how_to_fix']}")
        for ex in info["examples"]:
            print(f"    ex: {ex['sentence']}")
        print()


def main() -> None:
    args = parse_args()
    reports = [
        analyze_dataset(name, adapted, unmatched, args.limit)
        for name, adapted, unmatched in DEFAULT_SPECS
    ]
    for report in reports:
        print_report(report)

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(reports, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
