# Workflows: Orchestration Layer

## Overview

`WorkflowEngine` (`app/workflows.py`) is the central orchestrator that connects the rule-based decomposer, ML model, data persistence, and user interaction. It manages the full lifecycle of morphological analysis: decomposition generation, ML scoring, user correction, training, and checkpointing.

## Initialization

On startup, `WorkflowEngine.__init__()` sets up:

1. **DataManager** — handles file I/O for the dictionary, training logs, and sample files.
2. **SentenceDisambiguator** — the Transformer model, initialized with vocabulary sizes for both suffixes and closed-class words.
3. **Trainer** — wraps the model with optimizer, scheduler, and replay buffer. Attempts to load `ml/model.pt` if it exists.
4. **Replay buffer preloading** — if the loaded checkpoint has an empty replay buffer (fresh start or old checkpoint format), `_preload_replay_buffer()` reconstructs it from `sentence_valid_decompositions.jsonl`.

### Replay Buffer Preloading

`_preload_replay_buffer()` reads every entry from the JSONL training log and converts them back into token sequences:

- For **sentence entries**: iterates over each word, matches the logged decomposition against freshly-generated candidates via `morph.match_decompositions()`, and encodes the matching chain.
- For **word entries**: same matching logic but for a single word.
- **Fallback**: if decomposer output has changed and no match is found, it encodes directly from stored suffix names using `morph.encode_suffix_names()`.

This ensures the model always has past training data available for experience replay, even after code changes that alter decomposer output.

## Word-Level Pipeline

### `prepare_word_training(word)` — Generate and Rank Candidates

1. Calls `get_decompositions(word)` → runs `decompose_with_cc()` (cached in `self.decomp_cache`).
2. If only one decomposition exists → returns early (no ambiguity to resolve).
3. Encodes all suffix chains via `morph.encode_suffix_chain()`.
4. If the model has been trained (`training_count > 0`), calls `trainer.predict()` to score candidates.
5. Filters out zero-scored candidates (bare roots that produce trivial sequences).
6. Sorts by ML score and builds view models via `morph.reconstruct_morphology()` for display.

Returns a dict containing the sorted decompositions, their encoded chains, view models for display, and whether ML scores were available.

### `commit_word_training(word, correct_decomps, encoded_chains, original_indices)` — Learn from Correction

1. **Log decompositions**: For each confirmed-correct decomposition, extracts suffix names and surface forms, writes to JSONL via `DataManager`.
2. **Dictionary cleanup**: If the confirmed root differs from the surface word, deletes the surface form from `words.txt` (it was a derived word incorrectly listed as a standalone entry). Also checks infinitive forms.
3. **Train the model**: For each confirmed decomposition index, calls `trainer.train_sentence([confirmed_chain])` — the model does a replay-augmented gradient update.
4. **Checkpoint**: Every `checkpoint_frequency` examples, saves model state and training count.

Returns the final training loss and any deletion messages.

## Sentence-Level Pipeline

### `prepare_sentence_training(sentence)` — Decompose All Words

1. Splits the sentence into words.
2. For each word, generates all decompositions (including closed-class).
3. Encodes chains and builds display strings (`typing_strings`) for each candidate:
   - With suffixes: `"ev ler in den"` (root + space-separated suffix forms)
   - Bare root: `"ev"`

Returns a list of word data dicts, each containing decompositions, encoded chains, view models, and typing strings.

### `evaluate_sentence_target(word_data, target_str)` — Match User Input

Delegates to `find_matching_combinations()` in `sequence_matcher.py`. This performs a **DFS over the Cartesian product** of per-word decomposition candidates, pruning branches that don't match the user's typed target string token-by-token.

Matched combinations are then scored by the ML model (sum of log-probs over the full sentence token sequence) and returned sorted by score.

### `commit_sentence_training(sentence, words, word_data, correct_combo)` — Learn from Sentence

1. For each word, extracts the confirmed decomposition at the chosen index.
2. Logs all decompositions as a sentence entry via `DataManager.log_sentence_decompositions()`.
3. Encodes confirmed chains and calls `trainer.train_sentence(confirmed_chains)` — the model sees the full sentence as a single training sequence, learning cross-word suffix patterns.
4. Increments training count by the number of words and checkpoints if needed.

## Evaluation and Batch Processing

### `evaluate_word(word)` — Show ML Prediction

Generates decompositions, scores them, and returns the top-ranked candidate's view model. Used by the `eval` command.

### `sample_text(filename)` — Batch Word Processing

Processes a text file word by word:
1. Gets unique words from the file.
2. For unambiguous words (single decomposition), uses it directly.
3. For ambiguous words, scores candidates with `trainer.predict()` and picks the best.
4. Writes the decomposed output to file.

### `sample_sentences()` — Batch Sentence Processing

Processes a sentence file:
1. Splits text into sentences (by punctuation boundaries).
2. For each sentence, runs `prepare_sentence_training()` to get all candidates.
3. Uses `get_top_sentence_predictions()` (beam search, width 50, top 1) to find the best combination.
4. Formats and writes decomposed output.

`get_top_sentence_predictions()` uses **beam search**: at each word position, it expands the top `beam_width` partial combinations by appending each candidate, scores the extended sequences, and keeps the best `beam_width` results. This is more tractable than exhaustive DFS for long sentences.

### `relearn_all()` — Full Retraining

1. Reads all entries from the JSONL training log.
2. Encodes each entry directly from stored suffix names (bypasses the decomposer — faster and immune to decomposer changes).
3. Calls `trainer.train_bulk()` with all sequences in shuffled mini-batches for 70 epochs.
4. Saves checkpoint and updates training count.

This is used to retrain the model from scratch after code changes, new suffix definitions, or when the model has degraded.

## Data Flow Summary

```
User types word/sentence
         │
         ▼
  ┌──────────────┐
  │  Decomposer  │  Rule-based: generate ALL valid root+suffix interpretations
  └──────┬───────┘
         │ List[Tuple(root, pos, chain, final_pos)]
         ▼
  ┌──────────────┐
  │  Morphology  │  Encode suffix chains → (token_id, category_id) pairs
  │   Adapter    │  Reconstruct display strings for the CLI
  └──────┬───────┘
         │ List[(int, int)] per candidate
         ▼
  ┌──────────────┐
  │   ML Model   │  Score candidates using causal LM log-probabilities
  │  (Trainer)   │  Context-aware: uses left-context from committed words
  └──────┬───────┘
         │ Ranked candidates
         ▼
  ┌──────────────┐
  │     CLI      │  Present ranked options → User picks correct one
  └──────┬───────┘
         │ User's choice (index)
         ▼
  ┌──────────────┐
  │  Workflows   │  Log to JSONL, train model (replay-augmented), checkpoint
  └──────────────┘
```

## Caching

- **`decomp_cache`**: In-memory dict mapping `word → decompositions`. Avoids re-running the decomposer for repeated words within a session.
- **`@lru_cache` on `decompose()`**: Persists across calls in the same process. The workflow-level cache (`decomp_cache`) and the decomposer-level cache (`lru_cache`) are complementary — the former avoids the function call overhead, the latter avoids recomputation if the workflow cache is bypassed.
