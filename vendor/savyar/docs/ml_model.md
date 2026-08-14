# ML Model: Sentence-Level Morphological Disambiguation

## Overview

The ML layer (`ml/ml_ranking_model.py`) implements a **causal (decoder-only) Transformer** that learns to rank morphological decomposition candidates. It treats suffix sequences as a language modeling problem: given previously confirmed decompositions in a sentence, predict the most likely suffix sequence for the next word.

## Architecture — `SentenceDisambiguator`

### Model Class

A decoder-only Transformer with the following specifications (from `ml/config.py`):

| Hyperparameter     | Value | Rationale |
|--------------------|-------|-----------|
| Embedding dimension | 128   | Sized for a ~300-token vocabulary |
| Transformer layers  | 3     | Reduced from 4 to limit memorization on ~7k training sequences |
| Attention heads     | 8     | 16-dim per head at 128-dim model |
| FFN hidden dim      | 512   | Standard 4x expansion (128 * 4) |
| Dropout             | 0.2   | Increased for regularization on small dataset |
| Max sequence length  | 512   | Positional embedding limit |
| Activation          | GELU  | Smoother gradient flow than ReLU |
| Normalization       | Pre-LN | Layer norm before attention/FFN for training stability |

### Token Vocabulary

The vocabulary is a flat ID space encoding both suffixes and closed-class words:

```
[0]                             → PAD (padding token)
[1]                             → WORD_SEP (word boundary marker)
[2 .. suffix_vocab_size + 1]    → Suffix IDs (one per suffix type in ALL_SUFFIXES)
[suffix_vocab_size + 2 .. end]  → Closed-class word IDs (pronouns, conjunctions, etc.)
```

Each token carries two parallel IDs:
- **Token ID**: identifies the specific suffix or closed-class word
- **Category ID**: `0 = Noun`, `1 = Verb`, `2 = Special` (PAD/SEP), `3 = Closed-class`

### Input Representation

The model takes three embeddings at each position, concatenated and projected down:

```
input = Linear(embed_dim*3 → embed_dim)(
    concat(
        suffix_embed(token_id),      # what suffix/word this is
        category_embed(category_id), # noun/verb/special/closed-class
        pos_embed(position)          # absolute position in sequence
    )
)
```

This triple embedding allows the model to jointly learn:
- Suffix co-occurrence patterns (which suffixes follow which)
- POS transition preferences (noun→verb derivation patterns)
- Positional structure (suffixes earlier in a sentence vs. later)

### Causal Masking

An upper-triangular boolean mask ensures position `i` can only attend to positions `0..i`. This makes the model autoregressive: it predicts each token from its left context only, matching the left-to-right inference strategy.

### Weight Tying

The output projection (`lm_head`) shares weights with the token embedding (`suffix_embed`). This is a standard language model technique that:
- Halves the parameters in the embedding/output layers
- Creates a symmetric encoding: tokens that embed similarly are also predicted similarly

## Sequence Encoding

### Single Word

A suffix chain like `[plural, accusative]` is encoded as:

```
suffix_ids:   [plural_id, accusative_id, WORD_SEP]
category_ids: [0 (noun),   0 (noun),      2 (special)]
```

A bare root with no suffixes produces just `[WORD_SEP]`.

### Sentence

Multiple word chains are flattened with `WORD_SEP` delimiters:

```
Sentence: "evleri gördüm"
Token sequence: [plural_id, acc_id, WORD_SEP, past_id, 1sg_id, WORD_SEP]
```

The `build_sentence_sequence()` function handles this flattening.

## Training — `Trainer`

### Causal Language Modeling Objective

The model is trained with standard **next-token prediction** (cross-entropy loss):

```
Input:  token[0], token[1], ..., token[L-2]
Target: token[1], token[2], ..., token[L-1]
Loss:   CrossEntropy(logits, targets), ignoring PAD positions
```

This means the model learns `P(token_i | token_0, ..., token_{i-1})` — the probability of each suffix given all preceding suffixes in the sentence.

### Experience Replay

Rather than training repeatedly on a single new example (which causes catastrophic forgetting), the trainer uses an **experience replay buffer**:

1. **New example arrives** → added to the replay buffer (capacity: 7000 sequences).
2. **Sample `replay_k` (64) past examples** from the buffer.
3. **Form a mini-batch** of the new example + sampled past examples.
4. **Run `steps_per_update` (4) gradient steps** on this mixed batch.

When the buffer is full, eviction targets the first half (older entries), maintaining a mix of old and recent data.

This approach prevents the model from forgetting earlier decompositions while still learning from new corrections.

### Optimizer and Scheduling

- **AdamW** with `lr=3e-4`, `weight_decay=0.05`, betas `(0.9, 0.999)`
- **Cosine annealing with warm restarts**: `T_0=10`, `T_mult=2`, `eta_min=lr*0.01`
  - The learning rate follows a cosine curve that resets periodically with increasing period lengths (10, 20, 40, ... steps), preventing the model from getting stuck in local minima.
- **Gradient clipping** at `max_norm=1.0` to prevent exploding gradients.
- **Mixed-precision training** (`torch.amp`) when CUDA is available, using `GradScaler` for loss scaling.

### Bulk Training — `train_bulk()`

Used by the `relearn` command to retrain on all logged decompositions at once:

1. All sequences are loaded and added to the replay buffer.
2. Data is shuffled and processed in mini-batches of 128.
3. Runs for a configurable number of epochs (default 70 for relearn, 3 for standard bulk).
4. Reports average loss per epoch.

This is more efficient than per-sentence replay for large-scale retraining.

## Inference — Scoring and Prediction

### `score_candidates(context_chains, candidates, right_chains=None)`

Scores each candidate decomposition for a word given sentence context:

1. Build the **left context sequence** from already-committed word decompositions.
2. For each candidate, construct: `[left_context | candidate_tokens | WORD_SEP | right_context]`.
3. Run a forward pass and extract **log-probabilities** for the candidate's token positions only.
4. Sum these log-probs to get the candidate's score (higher = more likely).

The key insight: by conditioning on left context, the model performs **contextual disambiguation** — the same word gets different rankings depending on what precedes it in the sentence.

### `sentence_predict(all_candidates)`

Greedy left-to-right sentence-level disambiguation:

1. For word 1: score all candidates with no context → pick the best.
2. For word 2: score all candidates with word 1's chosen decomposition as context → pick the best.
3. Continue until all words are committed.

Each decision becomes part of the context for subsequent words, creating a chain of contextually-informed choices.

### `fast_batch_predict(all_candidates)`

GPU-optimized batch inference for processing large texts:

1. Flatten all (word, candidate) pairs into a single list.
2. Pad sequences to equal length.
3. Run a single batched forward pass (up to 512 candidates per batch).
4. Extract scores and pick the best candidate per word.

This is used when context is not needed (e.g., processing isolated words in sample files).

### Score Selection — `_get_best_index()`

When picking the best candidate, scores of exactly `0.0` are ignored. This prevents bare roots (which produce trivially short sequences) from winning over meaningful decompositions. The candidate with the highest non-zero score is selected.

## Checkpointing

The trainer saves and loads full state:

```python
{
    'model_state':     model weights,
    'optimizer_state':  AdamW state (momentum, variance),
    'scheduler_state':  cosine annealing position,
    'train_history':    list of training losses,
    'val_history':      list of validation losses,
    'best_val_loss':    best validation loss seen,
    'global_step':      total gradient steps taken,
    'replay_buffer':    all buffered (suffix_ids, category_ids) sequences,
}
```

Checkpoints are saved every `checkpoint_frequency` (1000) training examples and on `save`/`quit` commands. The replay buffer is persisted so it survives restarts.

## What the Model Actually Learns

The model learns a **statistical prior over Turkish suffix ordering and co-occurrence**, conditioned on sentence-level patterns. Concretely:

- **Suffix bigram/trigram preferences**: After a plural suffix, accusative is more likely than another plural.
- **POS transition flows**: Verb-to-noun derivations (participles) tend to be followed by noun inflection.
- **Sentence-level coherence**: In a sentence with past-tense verbs, the model will prefer past-tense decompositions for ambiguous words.
- **Closed-class context**: Seeing a pronoun or conjunction token influences what decompositions follow.

The model does **not** have access to word roots, surface forms, or semantic meaning — it operates entirely in the suffix/category token space. Disambiguation relies on morphological patterns rather than lexical semantics.
