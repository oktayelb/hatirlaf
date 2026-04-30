import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Dict, Iterable, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_UNMATCHED = [
    "data/metu_treebank/treebank_adapted_unmatched.jsonl",
    "data/google_treebank/treebank_adapted_unmatched.jsonl",
    "data/boun_treebank/treebank_adapted_unmatched.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show unmappable treebank items in a readable grouped report."
    )
    parser.add_argument(
        "--unmatched",
        action="append",
        dest="paths",
        help="Path to a treebank_adapted_unmatched.jsonl file. May be passed multiple times.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max examples per unmapped feature.",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="Optional path to write the grouped report as JSON.",
    )
    return parser.parse_args()


def load_entries(paths: Iterable[str]) -> List[Dict]:
    entries: List[Dict] = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    item["_source_path"] = path_str
                    entries.append(item)
    return entries


def extract_unmapped_keys(reason: str) -> List[str]:
    if not reason:
        return []
    if reason.startswith("unmappable features: "):
        raw = reason.replace("unmappable features: ", "").strip()
        raw = raw.strip("[]")
        if not raw:
            return []
        return [piece.strip().strip("'") for piece in raw.split(",") if piece.strip()]
    if reason == "unmappable_features":
        return ["unmappable_features"]
    return [reason]


def build_report(entries: List[Dict], limit: int) -> Dict:
    grouped = defaultdict(lambda: {"count": 0, "examples": []})
    by_source = Counter()

    for entry in entries:
        reason = str(entry.get("reason", ""))
        if "unmappable" not in reason:
            continue
        by_source[entry["_source_path"]] += 1
        for key in extract_unmapped_keys(reason):
            bucket = grouped[key]
            bucket["count"] += 1
            if len(bucket["examples"]) < limit:
                bucket["examples"].append(
                    {
                        "source": entry["_source_path"],
                        "surface": entry.get("surface", ""),
                        "lemma": entry.get("lemma", ""),
                        "features": entry.get("features", []),
                        "reason": reason,
                    }
                )

    return {
        "total_unmappable_entries": sum(by_source.values()),
        "by_source": dict(by_source),
        "by_feature": dict(
            sorted(grouped.items(), key=lambda item: (-item[1]["count"], item[0]))
        ),
    }


def print_report(report: Dict) -> None:
    print(f"Total unmappable entries: {report['total_unmappable_entries']}")
    if report["by_source"]:
        print("By source:")
        for source, count in report["by_source"].items():
            print(f"  {source}: {count}")
    print()

    for feature, info in report["by_feature"].items():
        print(f"[{feature}] count={info['count']}")
        for ex in info["examples"]:
            print(
                f"  surface={ex['surface']:<20} lemma={ex['lemma']:<16} "
                f"features={ex['features']}"
            )
            print(f"    source: {ex['source']}")
        print()


def main() -> None:
    args = parse_args()
    paths = args.paths or list(DEFAULT_UNMATCHED)
    entries = load_entries(paths)
    report = build_report(entries, limit=args.limit)
    print_report(report)

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
