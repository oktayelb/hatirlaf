import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app.morphology_adapter as morph
from ml.ml_ranking_model import (
    CATEGORY_SPECIAL,
    GROUP_TO_ID,
    SPECIAL_BOS,
    SPECIAL_FEATURE_ID,
    SPECIAL_MASK,
    SPECIAL_PAD,
    SPECIAL_WORD_SEP,
    TYPE_TO_ID,
    WORD_FINAL_NO,
    SentenceDisambiguator,
    Trainer,
    _get_all_suffixes,
)
from util.words.closed_class import CLOSED_CLASS_TOKEN_SPECS


DEFAULT_DATASETS = [
    "data/metu_treebank/treebank_adapted.jsonl",
    "data/google_treebank/treebank_adapted.jsonl",
    "data/boun_treebank/treebank_adapted.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect concrete model failures for selected suffixes."
    )
    parser.add_argument(
        "--checkpoint",
        default="ml/model.pt",
        help="Model checkpoint to load.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="JSONL dataset to inspect. May be passed multiple times.",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        dest="suffixes",
        required=True,
        help="Suffix name to inspect. May be passed multiple times.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Max false-negative / false-positive examples per suffix.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=0,
        help="Optional cap on number of sentence entries to inspect.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of masked slots to score per forward pass.",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="Optional path to write the full report as JSON.",
    )
    return parser.parse_args()


def load_jsonl(paths: Iterable[str], max_sentences: int = 0) -> List[Dict]:
    entries: List[Dict] = []
    seen = 0
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                item["_source_path"] = path_str
                entries.append(item)
                seen += 1
                if max_sentences and seen >= max_sentences:
                    return entries
    return entries


def _empty_special_row() -> Tuple[int, int, int, int, int, int, int]:
    return (
        SPECIAL_WORD_SEP,
        CATEGORY_SPECIAL,
        SPECIAL_FEATURE_ID,
        SPECIAL_FEATURE_ID,
        SPECIAL_FEATURE_ID,
        SPECIAL_FEATURE_ID,
        WORD_FINAL_NO,
    )


def entry_to_sequence(entry: Dict, source_path: str):
    suffix_ids = [SPECIAL_BOS]
    category_ids = [CATEGORY_SPECIAL]
    group_ids = [SPECIAL_FEATURE_ID]
    comes_to_ids = [SPECIAL_FEATURE_ID]
    makes_ids = [SPECIAL_FEATURE_ID]
    word_pos_ids = [SPECIAL_FEATURE_ID]
    word_final = [WORD_FINAL_NO]
    position_meta: List[Optional[Dict]] = [None]

    words = entry.get("words", [])
    sentence_text = entry.get("original_sentence", "")

    for word_index, word_entry in enumerate(words):
        suffix_dicts = word_entry.get("suffixes", [])
        if not suffix_dicts:
            continue

        encoded = morph.encode_suffix_names(suffix_dicts)
        for suffix_index, token in enumerate(encoded):
            sid, cid, gid, ctid, mid, posid, is_final = token
            suffix_ids.append(sid)
            category_ids.append(cid)
            group_ids.append(gid)
            comes_to_ids.append(ctid)
            makes_ids.append(mid)
            word_pos_ids.append(posid)
            word_final.append(is_final)
            position_meta.append(
                {
                    "source": source_path,
                    "sentence": sentence_text,
                    "word": word_entry.get("word", ""),
                    "root": word_entry.get("root", ""),
                    "morphology_string": word_entry.get("morphology_string", ""),
                    "final_pos": word_entry.get("final_pos", ""),
                    "word_index": word_index,
                    "suffix_index": suffix_index,
                    "suffix_count": len(encoded),
                    "gold_suffix": suffix_dicts[suffix_index]["name"],
                    "gold_form": suffix_dicts[suffix_index].get("form", ""),
                }
            )

        sid, cid, gid, ctid, mid, posid, is_final = _empty_special_row()
        suffix_ids.append(sid)
        category_ids.append(cid)
        group_ids.append(gid)
        comes_to_ids.append(ctid)
        makes_ids.append(mid)
        word_pos_ids.append(posid)
        word_final.append(is_final)
        position_meta.append(None)

    return (
        (suffix_ids, category_ids, group_ids, comes_to_ids, makes_ids, word_pos_ids, word_final),
        position_meta,
    )


def predict_masked_slots(
    trainer: Trainer,
    sequence,
    batch_size: int,
) -> List[Tuple[int, int]]:
    suffix_ids, category_ids, group_ids, comes_to_ids, makes_ids, word_pos_ids, word_final = sequence
    eligible_positions = [
        idx for idx, sid in enumerate(suffix_ids)
        if sid not in (SPECIAL_PAD, SPECIAL_BOS, SPECIAL_WORD_SEP, SPECIAL_MASK)
    ]
    if not eligible_positions:
        return []

    device = trainer.device
    predictions: List[Tuple[int, int]] = []
    model = trainer.model
    model.eval()

    with torch.no_grad():
        for start in range(0, len(eligible_positions), batch_size):
            chunk = eligible_positions[start:start + batch_size]
            size = len(chunk)

            s = torch.tensor([suffix_ids] * size, dtype=torch.long, device=device)
            c = torch.tensor([category_ids] * size, dtype=torch.long, device=device)
            g = torch.tensor([group_ids] * size, dtype=torch.long, device=device)
            ct = torch.tensor([comes_to_ids] * size, dtype=torch.long, device=device)
            m = torch.tensor([makes_ids] * size, dtype=torch.long, device=device)
            wp = torch.tensor([word_pos_ids] * size, dtype=torch.long, device=device)
            wf = torch.tensor([word_final] * size, dtype=torch.long, device=device)
            pad_mask = torch.zeros((size, len(suffix_ids)), dtype=torch.bool, device=device)

            rows = torch.arange(size, device=device)
            cols = torch.tensor(chunk, dtype=torch.long, device=device)
            s[rows, cols] = SPECIAL_MASK

            logits = model(s, c, g, ct, m, wp, wf, pad_mask=pad_mask)
            pred_ids = logits[rows, cols].argmax(dim=-1).tolist()
            predictions.extend(zip(chunk, pred_ids))

    return predictions


def build_report(entries: List[Dict], target_suffixes: List[str], checkpoint: str, datasets: List[str], limit: int, batch_size: int) -> Dict:
    all_suffixes = _get_all_suffixes()
    id_to_name = {idx + 4: suffix.name for idx, suffix in enumerate(all_suffixes)}
    closed_class_offset = 4 + len(all_suffixes)
    for idx, (category, surface) in enumerate(CLOSED_CLASS_TOKEN_SPECS):
        id_to_name[closed_class_offset + idx] = f"CC:{category}:{surface}"
    name_set = {suffix.name for suffix in all_suffixes}
    unknown_suffixes = [name for name in target_suffixes if name not in name_set]
    if unknown_suffixes:
        raise ValueError(f"Unknown suffix names requested: {', '.join(unknown_suffixes)}")

    model = SentenceDisambiguator(
        suffix_vocab_size=len(all_suffixes),
        closed_class_vocab_size=len(CLOSED_CLASS_TOKEN_SPECS),
    )
    trainer = Trainer(model=model, path=checkpoint)

    report = {
        "checkpoint": checkpoint,
        "datasets": datasets,
        "target_suffixes": target_suffixes,
        "entries_scanned": len(entries),
        "suffixes": {},
    }

    suffix_stats: Dict[str, Dict] = {}
    for suffix_name in target_suffixes:
        suffix_stats[suffix_name] = {
            "gold_count": 0,
            "tp": 0,
            "fn": 0,
            "fp": 0,
            "fn_confusions": Counter(),
            "fp_sources": Counter(),
            "false_negatives": [],
            "false_positives": [],
        }

    for entry in entries:
        sequence, position_meta = entry_to_sequence(entry, entry.get("_source_path", ""))
        preds = predict_masked_slots(trainer, sequence, batch_size=batch_size)
        for pos, pred_id in preds:
            meta = position_meta[pos]
            if meta is None:
                continue

            gold_name = meta["gold_suffix"]
            pred_name = id_to_name.get(pred_id, f"UNK_{pred_id}")

            if gold_name in suffix_stats:
                stats = suffix_stats[gold_name]
                stats["gold_count"] += 1
                if pred_name == gold_name:
                    stats["tp"] += 1
                else:
                    stats["fn"] += 1
                    stats["fn_confusions"][pred_name] += 1
                    if len(stats["false_negatives"]) < limit:
                        stats["false_negatives"].append(
                            {
                                **meta,
                                "predicted_suffix": pred_name,
                                "predicted_id": pred_id,
                            }
                        )

            if pred_name in suffix_stats and pred_name != gold_name:
                stats = suffix_stats[pred_name]
                stats["fp"] += 1
                stats["fp_sources"][gold_name] += 1
                if len(stats["false_positives"]) < limit:
                    stats["false_positives"].append(
                        {
                            **meta,
                            "predicted_suffix": pred_name,
                            "predicted_id": pred_id,
                        }
                    )

    for suffix_name, stats in suffix_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        report["suffixes"][suffix_name] = {
            "gold_count": stats["gold_count"],
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "top_false_negative_confusions": stats["fn_confusions"].most_common(10),
            "top_false_positive_sources": stats["fp_sources"].most_common(10),
            "false_negative_examples": stats["false_negatives"],
            "false_positive_examples": stats["false_positives"],
        }

    return report


def print_report(report: Dict) -> None:
    print(f"Checkpoint: {report['checkpoint']}")
    print(f"Entries scanned: {report['entries_scanned']}")
    print(f"Datasets: {', '.join(report['datasets'])}")
    print()

    for suffix_name, stats in report["suffixes"].items():
        print(
            f"[{suffix_name}] gold={stats['gold_count']} "
            f"tp={stats['tp']} fn={stats['fn']} fp={stats['fp']} "
            f"P={stats['precision']:.4f} R={stats['recall']:.4f} F1={stats['f1']:.4f}"
        )
        if stats["top_false_negative_confusions"]:
            print("  Top FN confusions:")
            for confused, count in stats["top_false_negative_confusions"]:
                print(f"    {count:4d} -> {confused}")
        if stats["top_false_positive_sources"]:
            print("  Top FP sources:")
            for source, count in stats["top_false_positive_sources"]:
                print(f"    {count:4d} <- {source}")

        if stats["false_negative_examples"]:
            print("  False negatives:")
            for ex in stats["false_negative_examples"]:
                print(
                    f"    word={ex['word']:<20} gold={ex['gold_suffix']:<18} pred={ex['predicted_suffix']:<18} id={ex['predicted_id']:<4} "
                    f"morph='{ex['morphology_string']}'"
                )
                if ex["sentence"]:
                    print(f"      sentence: {ex['sentence']}")

        if stats["false_positive_examples"]:
            print("  False positives:")
            for ex in stats["false_positive_examples"]:
                print(
                    f"    word={ex['word']:<20} gold={ex['gold_suffix']:<18} pred={ex['predicted_suffix']:<18} id={ex['predicted_id']:<4} "
                    f"morph='{ex['morphology_string']}'"
                )
                if ex["sentence"]:
                    print(f"      sentence: {ex['sentence']}")
        print()


def main() -> None:
    args = parse_args()
    datasets = args.datasets or list(DEFAULT_DATASETS)
    entries = load_jsonl(datasets, max_sentences=args.max_sentences)

    report = build_report(
        entries=entries,
        target_suffixes=args.suffixes,
        checkpoint=args.checkpoint,
        datasets=datasets,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    print_report(report)

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
