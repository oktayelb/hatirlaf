#!/usr/bin/env python3
"""Run SAVYAR sentence-level ML analysis and return JSON.

This script is intentionally outside Django. Hatirlaf calls it as a small
process boundary so SAVYAR can run in its own Python environment with torch,
while Hatirlaf consumes only ranked morphology candidates.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path


def _candidate_to_json(word: str, decomp, ml_score: float | None) -> dict:
    root, pos, chain, final_pos = decomp
    suffixes = [getattr(s, "name", "") for s in chain if getattr(s, "name", "")]
    return {
        "word": word,
        "root": root,
        "pos": pos,
        "suffixes": suffixes,
        "final_pos": final_pos,
        "ml_score": ml_score,
    }


def analyze(text: str, savyar_root: Path, model_path: Path) -> dict:
    savyar_root = savyar_root.resolve()
    model_path = model_path.resolve()
    os.chdir(savyar_root)
    sys.path.insert(0, str(savyar_root))

    from app import analyzer
    from app import morphology_adapter as morph
    from app.input import sanitize_sentence
    from ml.config import config
    from ml.ml_ranking_model import SentenceDisambiguator, Trainer
    from util.words.closed_class import CLOSED_CLASS_TOKEN_SPECS

    config.model_path = model_path

    model = SentenceDisambiguator(
        suffix_vocab_size=len(morph._SUFFIX_TO_ID),
        closed_class_vocab_size=len(CLOSED_CLASS_TOKEN_SPECS),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        trainer = Trainer(model=model, path=str(model_path))
    trainer.model.eval()

    words = sanitize_sentence(text)
    if not words:
        return {"words": []}

    with contextlib.redirect_stdout(io.StringIO()):
        analyses = analyzer.analyze_words(words, include_closed_class=True)
    if any(not analysis["decomps"] for analysis in analyses):
        return {"words": [{"word": word, "candidates": []} for word in words]}

    encoded_by_word = [analysis["encoded_chains"] for analysis in analyses]
    with contextlib.redirect_stdout(io.StringIO()):
        predictions = trainer.sentence_predict(encoded_by_word)

    out_words = []
    for analysis, (_best_idx, scores) in zip(analyses, predictions):
        order = sorted(range(len(analysis["decomps"])), key=lambda idx: scores[idx], reverse=True)
        candidates = [
            _candidate_to_json(
                analysis["word"],
                analysis["decomps"][idx],
                float(scores[idx]) if idx < len(scores) else None,
            )
            for idx in order
        ]
        out_words.append({"word": analysis["word"], "candidates": candidates})
    return {"words": out_words}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--savyar-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()

    payload = json.load(sys.stdin)
    result = analyze(payload.get("text", ""), args.savyar_root, args.model_path)
    json.dump(result, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
