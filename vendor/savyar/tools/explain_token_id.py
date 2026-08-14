import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.ml_ranking_model import SPECIAL_BOS, SPECIAL_MASK, SPECIAL_PAD, SPECIAL_WORD_SEP, SUFFIX_OFFSET, _get_all_suffixes
from util.words.closed_class import CLOSED_CLASS_TOKEN_SPECS


def explain_token_id(token_id: int) -> str:
    special = {
        SPECIAL_PAD: "SPECIAL_PAD",
        SPECIAL_WORD_SEP: "SPECIAL_WORD_SEP",
        SPECIAL_BOS: "SPECIAL_BOS",
        SPECIAL_MASK: "SPECIAL_MASK",
    }
    if token_id in special:
        return special[token_id]

    suffixes = _get_all_suffixes()
    suffix_start = SUFFIX_OFFSET
    suffix_end = suffix_start + len(suffixes)
    if suffix_start <= token_id < suffix_end:
        suffix = suffixes[token_id - suffix_start]
        return f"SUFFIX:{suffix.name}"

    closed_class_start = suffix_end
    closed_class_end = closed_class_start + len(CLOSED_CLASS_TOKEN_SPECS)
    if closed_class_start <= token_id < closed_class_end:
        category, surface = CLOSED_CLASS_TOKEN_SPECS[token_id - closed_class_start]
        return f"CLOSED_CLASS:{category}:{surface}"

    return "OUT_OF_RANGE"


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain a model token ID.")
    parser.add_argument("token_ids", nargs="+", type=int)
    args = parser.parse_args()

    for token_id in args.token_ids:
        print(f"{token_id}: {explain_token_id(token_id)}")


if __name__ == "__main__":
    main()
