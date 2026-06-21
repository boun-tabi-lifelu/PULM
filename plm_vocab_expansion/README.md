# Cost-Efficient Adaptation of ESM-2 to New Tokenizers

**Context:** We want to take pre-trained ESM-2 (35M parameters, amino acid tokenizer) and
adapt it to work with larger-vocabulary tokenizers — BPE or PUMA — without training from
scratch. The goal is to preserve the biochemical knowledge already encoded in ESM-2 while
enabling the model to operate on multi-residue token units.

---

## Why Not Train From Scratch?

Training ESM-2 35M on UniRef50 from scratch costs ~weeks of GPU compute. Our approach
reduces this to days by reusing the pre-trained backbone and only adapting the parts that
genuinely need to change when the tokenizer changes.

---

## The Core Problem

When we swap the tokenizer, three things break:

| What breaks | Why |
|---|---|
| **Embedding layer** | New vocabulary tokens have no pre-trained vectors |
| **Positional attention (RoPE)** | Token positions now span multiple residues; attention distance arithmetic is miscalibrated |
| **LM head** | The output projection was trained to predict over 33 AA tokens, not 800–51200 BPE/PUMA tokens |

Everything else — the transformer weights, FFN layers, attention value projections — is still
valid and should be preserved.

---

## Step 1 — Embedding Initialization

The most important question: *how do we initialize vectors for new multi-residue tokens?*

**Approach: mean pooling over constituent amino acids**

Every BPE/PUMA token is a sequence of amino acids (e.g., `"AGL"`, `"PRO"`). ESM-2 already
has high-quality embeddings for each individual amino acid. We initialize each new token
vector as the mean of its constituent AA embeddings.

```
e_new("AGL") = mean( e("A"), e("G"), e("L") )
```

**Norm correction on top of the mean, plus two special cases:**

1. **Per-token norm projection** — averaging shrinks the norm, but *not* by a fixed
   ~√k factor. The √k rule assumes the constituent AA embeddings are orthogonal and
   zero-mean; in ESM-2 they are strongly correlated by biochemistry (hydrophobicity,
   charge, aromaticity). A conserved hydrophobic motif (`"LIV"`, `"FAL"`) has nearly
   collinear constituents and barely shrinks — a rigid √k rescale would overshoot and
   blow the magnitude out of distribution — while a token mixing conflicting signals
   collapses *more* than √k. So instead of any analytical factor in k, we project each
   new embedding directly onto the norm shell of the original vocabulary:

   ```
   e_new = (ē / ‖ē‖₂) × mean_AA_norm
   ```

   where ē is the mean of the constituent AA embeddings and `mean_AA_norm` is the
   average L2 norm of the original 33 AA token embeddings. This lands every new token
   on the same hypersphere shell as the pre-trained vocabulary regardless of internal
   correlation, with no dependence on k.

**Special cases (not corrections to the mean — handled separately):**

- **Special token handling** — `<cls>`, `<pad>`, `<mask>`, etc. are copied directly
  from the old table. They are never mean-pooled or rescaled.

- **Fallback** — tokens with no AA decomposition (rare) get Gaussian initialization
  with matched mean and std.

**Why this works:** ESM-2's transformer layers have already learned to process vectors
in this embedding space. A well-initialized new token lands close to a semantically
reasonable region of that space, giving the model a head start.

---

### PUMA-Only Extension: Lineage-Aware Initialization

For the PUMA tokenizer, the AA-decomposition above is correct for two of the three unit
types in a PUMA genealogy — but not the third.

**Mutational parents and singletons** (units formed by a frequency merge, or units with
no surviving mutation candidates) are flat, BPE-equivalent strings of amino acids. They
use the mean-pool + norm-shell formula exactly as written above — no change.

**Mutational children** are different in kind: a child's identity is not an independent
composition of amino acids, it's a small perturbation of an already-formed parent. Since
a PUMA child can carry substitutions at more than one residue simultaneously, mean-
pooling the child's own mutated AA sequence and treating it as unrelated to the parent
throws away exactly the information the genealogy provides — and a per-residue additive
correction doesn't generalize cleanly once more than one residue can change at once.
Instead, blend the child's raw pooled vector with its parent's already-initialized
embedding, then project the *blend* — not either input alone — onto the shell:

```
e_pool(child) = mean( e(AA_1), e(AA_2), ..., e(AA_n) )     # raw, unprojected
e_child = project_to_shell( γ · e_pool(child) + (1 − γ) · e_parent )
```

`e_parent` is the already-projected parent embedding from above. Projecting *after* the
blend is required, not optional: `e_parent` already sits on the norm shell while
`e_pool(child)` does not, so blending them unprojected would land the child off the
shell and reintroduce the exact norm-inconsistency problem the projection step exists to
prevent.

`γ ∈ [0, 1]` starts as a flat hyperparameter (e.g. 0.6). A natural refinement is to tie
it to PUMA's own normalized alignment score `α` — the same score already gating mutation
acceptance at cutoff 0.7 — so that high-α (near-identical) children weight one term more
and low-α (more divergent but still-qualifying) children weight the other. The
*direction* of that relationship isn't obvious a priori — both "trust the pooled vector
more as divergence increases" and "trust it less" have a plausible rationale — so treat
`γ = f(α)` as a follow-up ablation against the flat-γ baseline, not a default.

> **Why this matters more for PUMA than for plain BPE:** without this, a child
> embedding's distance from its parent in the AA-pooled space is arbitrary — driven by
> which residues happened to change, not by how evolutionarily close PUMA judged the
> child to be. The blend makes the genealogy a literal geometric prior: siblings start
> training already clustered near their parent, mirroring the "center / centre" analogy
> by construction rather than by accident.

---

## Step 2 — LM Head Initialization

ESM-2's output head has this structure:

```
hidden state → Linear(d→d) → GELU → LayerNorm → Linear(d→V) + bias → logits
```

- The **inner layers** (Linear + LayerNorm) operate in the hidden space, which hasn't
  changed. We keep them exactly as-is from the pre-trained model.

- The **output projection** (Linear d→V) needs new rows for new tokens. We tie it to the
  embedding matrix (initialized, and kept, as its transpose) — the same weight-tying
  principle ESM-2 used during original training. Maintaining the tie throughout training,
  rather than only copying at init, also halves the new-vocabulary parameter cost at large
  V (see the budget note under Step 4).

- The **output bias** is initialized from log-frequencies of token counts in the training
  corpus. This gives the model a calibrated prior so it doesn't waste early steps
  learning that common tokens are common. Counts are Laplace-smoothed (add-ε) so that
  tokens absent from the corpus — likely at large V — do not produce `log(0) = −∞`.

```
bias[token_id] = log( (count(token) + ε) / (total_tokens + ε·V) )
```

---

### PUMA-Only Extension: Family-Smoothed Bias

Flat Laplace smoothing treats every rare token as equally uninformative. For PUMA
children, that's not true — a rare child still inherits the success of its whole family,
since families are defined as evolutionarily-supported variants of a shared parent.
Replace the raw count with a family-shrunk count before applying the existing smoothing:

```
c̃(token) = count(token) + δ · ( count(Family) / |Family| )

bias[token_id] = log( (c̃(token) + ε) / (total_tokens + ε·V) )
```

where `count(Family)` is the summed count of the parent plus all its children, and
`|Family|` is the family size. This applies to **children only** — parents are selected
by the merge step precisely because they're frequent, so they don't suffer the same
tail problem, and singletons have no family to borrow from and keep the unmodified
formula above.

Start with a flat `δ` (e.g. 0.1). As with `γ` in Step 1, tying `δ` to the alignment
score `α` is a reasonable follow-up — closer siblings borrow more family credit, more
divergent ones borrow less — but validate the flat version first so `γ`, `δ`, and the
regularizer weight below aren't three independently-tuned knobs before confirming even
one of them earns its keep.

---

## Step 3 — RoPE Positional Handling

ESM-2 uses Rotary Position Embeddings (RoPE). With single-AA tokens, position index 5
means "residue 5." With BPE/PUMA tokens, the *token* index no longer equals the
*residue* coordinate: a token at sequence position 5 may correspond to residue ~5 × k̄.
Every attention head's Q·K dot products were trained on true residue distances, so if we
feed sequential token indices the learned distance relationships are miscalibrated.

**Important: do not rescale the RoPE base frequency.** An earlier version of this
pipeline rescaled the base by k̄² (`new_base = 10000 × k̄²`). That is wrong in both
magnitude and direction:

- **Direction.** Tokenization makes sequences *shorter* in token units (a 300-residue
  protein → ~100 tokens at k̄=3). This is the opposite of context-length *extension*,
  where the base is raised to slow rotation. Here we are compressing, so raising the
  base slows rotation exactly when it needs to advance faster — the wrong way. (Even the
  NTK analogy, applied with effective factor s = 1/k̄ < 1, predicts base ÷ k̄, not × k̄².)
- **Variable token length.** BPE/PUMA tokens are *variable* length — a step over `"A"`
  (k=1) and a step over `"WLAMG"` (k=5) cover very different physical distances. No single
  global scalar (k̄, k̄², or otherwise) can encode a per-step distance that varies token
  by token.

**Fix: pass explicit residue-based position IDs.** Keep the base at 10000 and set each
token's position ID to the index of its first (or center) residue in the original
sequence. RoPE then sees true residue distances natively, variable token lengths are
handled exactly, and ESM-2's pre-trained attention calibration is preserved with no base
hacking.

| Tokenized sequence | `"M"` | `"AGL"` | `"V"` | `"KRE"` |
|---|---|---|---|---|
| Sequential token IDs (wrong) | 0 | 1 | 2 | 3 |
| Residue-based position IDs (use this) | 0 | 1 | 4 | 5 |

> **Note:** For the AA tokenizer (k̄ = 1.0, every token one residue), residue-based IDs
> are identical to sequential IDs, so this step is a no-op — preserving the
> collapse-to-standard-fine-tuning property.

Residue-based position IDs also keep max position well within ESM-2's trained range.
Any residual fine-grained recalibration is handled by LoRA on Q and K (next step); with
correct position IDs, a "no positional change at all" run is a meaningful baseline worth
ablating.

---

## Step 4 — LoRA Adapters

We freeze the entire transformer body and inject low-rank adapter matrices (LoRA) at
targeted weight matrices. The key insight is that *different weight matrices are affected
differently* by the tokenizer change, so we apply different ranks accordingly.

### Why these specific matrices?

| Matrix | Role | Affected by BPE/PUMA? |
|---|---|---|
| **W_q, W_k** | Compute attention scores via RoPE-encoded positions | ✅ Severely — positional mismatch |
| **W_v, W_o** | Aggregate and project token values | ✅ Moderately — value distribution shifted |
| **FFN W_up** | Pattern detectors; fire on token-level features | ✅ Moderately — single-AA patterns no longer apply |
| **FFN W_down** | Projects activated features to residual stream | ⚠️ Mildly — indirect exposure |

### Three presets

**Preset A — `uniform_r16`:** All six matrices, rank 16, all layers.
Best for initial exploration.

**Preset B — `layerwise` (recommended):** Tapering rank and coverage by layer depth.
Early layers see the raw token distribution and need the most capacity; late layers
are largely abstract and only need positional recalibration.

| Layer group | Matrices | Rank |
|---|---|---|
| Early (0 → L/3) | Q, K, V, W_o, FFN W_up | 16 |
| Middle (L/3 → 2L/3) | Q, K, V, W_o | 8 |
| Late (2L/3 → L) | Q, K only | 8 |

**Preset C — `attn_only_r16`:** Only Q and K, rank 16, all layers.
Targets only positional mismatch. Lowest parameter overhead.

### LoRA parameter budget (ESM-2 35M backbone)

| Preset | LoRA params | % of backbone |
|---|---|---|
| `uniform_r16` | ~2.2M | 6.3% |
| `layerwise` | ~1.1M | 3.1% |
| `attn_only_r16` | ~0.45M | 1.3% |

The backbone (and therefore LoRA overhead) is fixed at 35M regardless of vocabulary
size. Only the embedding matrix and LM head grow with larger vocabularies — and at
large V this is the *dominant* cost, not a footnote. At V = 51200 and d = 480, the
embedding table is ~24.6M params and an untied output projection another ~24.6M, ~49M
total — already larger than the 35M backbone. We therefore **tie the embedding and LM
head** (Step 2 already initializes the head as the transpose of the embedding), which
halves this to ~24.6M and keeps the two in sync. This also concentrates the
"cost-efficient" challenge where it actually lives: the new-vocabulary parameters, whose
Zipfian frequency tail means many rare tokens receive almost no gradient — which is
exactly why the analytical embedding init of Step 1 matters most for the rare-token tail.

> **Budget note:** the LoRA figures above should be verified against the implementation.
> For `uniform_r16` over six matrices (Q, K, V, W_o at d=480; FFN W_up, W_down at
> d↔1920), rank 16, 12 layers, a direct count gives ≈1.66M params (≈4.7% of backbone),
> not 2.2M/6.3%. Confirm whether the original figure double-counts adapters or assumes a
> different FFN width before quoting it.

---

## Step 5 — Staged Training

Rather than training all components simultaneously, we use three stages to avoid
instability and catastrophic forgetting.

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1 — Embedding Warm-up                (~5% of total steps)│
│  Trainable: word embeddings only                                │
│  LR: 5e-4 (moderate — embeddings are well-initialized, not random)│
│  Goal: settle new token vectors into ESM-2's representation space│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2 — Main Adaptation              (~75% of total steps)   │
│  Trainable: LoRA adapters + LM head + embeddings (low LR)       │
│  LR: 1e-4 (LoRA/head),  1e-5 (embeddings)                       │
│  Goal: recalibrate attention and FFN features to new tokens     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3 — Refinement                    (~20% of total steps)  │
│  Trainable: Stage 2 params + LayerNorms                         │
│  LR: 3e-5 (all), 1e-5 (LayerNorms)                              │
│  Goal: fine-tune normalization statistics to new token scale    │
└─────────────────────────────────────────────────────────────────┘
```

Unfreezing LayerNorms in Stage 3 is cheap (2 × d parameters per layer) but
consistently recovers meaningful perplexity.

---

### PUMA-Only Sub-component: Family Alignment Regularization

The embedding-init blend in Step 1 gives children a good *starting* position relative to
their parent, but says nothing about where they drift during training — and rare
children, by definition, get sparse MLM gradient and can drift unpredictably or barely
move at all. Add a small auxiliary loss during **Stage 1 only**:

```
L_PUMA = mean over all children ( 1 − cos(e_child, e_parent) )

L_total = L_MLM + λ · L_PUMA          (λ ≈ 0.01–0.05)
```

Cosine, not L2 — this constrains direction only, so it doesn't fight the norm-shell
projection already established at init.

**This must be computed over the full embedding table every step, not just the tokens
present in the current batch.** A batch-restricted version only ever touches tokens that
already receive an MLM gradient that step, in which case it adds almost nothing beyond
the init. Computed table-wide, it's the only mechanism in this pipeline that gives rare
children a gradient signal on steps where they don't appear in the corpus sample at all
— which is the actual rare-tail problem.

Restrict this to **Stage 1 by default.** By Stage 2 the embedding LR is already down to
1e-5 (see schedule above); at that rate the regularizer is competing with an
already-near-frozen parameter and is likely redundant. Extending it into Stage 2 is a
separate ablation, not a default.

`λ` should stay small enough that it acts as a gentle pull, not a constraint — the goal
is stabilizing the tail, not collapsing siblings into duplicate vectors.

---

## Step 6 — Masking Rate

It is tempting to think multi-residue tokens inflate the effective masking rate — that
masking 15% of tokens hides `15% × k̄` of residues. **This is wrong: the k̄ factors
cancel.** Tokens *tile* the sequence, so R residues produce T ≈ R/k̄ tokens. Masking a
fraction f of tokens hides `f · T · k̄ = f · R` residues, i.e. the residue-level masking
density is simply **f, independent of k̄** — because there are k̄× fewer tokens to begin
with.

Worked example: a 1200-residue protein at k̄ = 3 → 400 tokens. Masking 15% = 60 tokens =
180 residues = 15% of residues. Not 45%.

**Consequence:** the earlier `masking_rate = 0.15 / k̄` formula does *not* hold residue
density constant — it *drops* it to 0.15/k̄ (~5% of residues at k̄=3), under-masking and
weakening the training signal.

**Fix:** keep the token masking rate at the standard value so residue-level density
matches native ESM-2:

```
masking_rate = 0.15
```

This already gives 15% residue-level masking at any k̄. The genuine difference under
multi-residue tokens is *not* density but that each masked token is now a contiguous
span (a k̄-mer) and each prediction is higher-entropy over a larger vocabulary — a
per-prediction difficulty handled by training, not a rate to deflate. If span difficulty
proves to over- or under-stimulate the model in practice, tune the rate empirically
*around* 0.15 rather than dividing by k̄.

---

## What Stays Unchanged

To be explicit: the following components are frozen and never modified during adaptation.

- All transformer attention weight matrices W_q, W_k, W_v, W_o (except through LoRA deltas)
- All FFN weights (except through LoRA deltas)
- LM head dense layer and LayerNorm (until Stage 3)

---

## Summary

We adapt ESM-2 to a new tokenizer without retraining by addressing each failure mode
surgically:

| Problem | Solution |
|---|---|
| New tokens have no embeddings | Mean-pool constituent AAs + project onto AA norm shell |
| LM head vocabulary mismatch | Weight-tied init + smoothed log-frequency bias |
| RoPE positional miscalibration | Residue-based position IDs (no base change) + LoRA on W_q, W_k |
| Token distribution shift in attention | LoRA on W_v, W_o (moderate rank) |
| Token distribution shift in FFN | LoRA on W_up in early/middle layers |
| Training instability | Three-stage frozen → partial → full unfreezing |
| Multi-residue tokens | Keep token mask rate at 0.15 (residue density is k̄-invariant) |
| Rare PUMA children (Zipfian tail) | Lineage-aware embedding blend (Step 1) + family-smoothed bias (Step 2) + Stage-1 genealogy regularizer (Step 5) — PUMA only |

**The key principle:** every component that was not touched by the tokenizer change is
kept frozen. Every component that was touched is either re-initialized analytically
(embeddings, head) or given a small low-rank adapter to recalibrate (transformer body).
Nothing is trained from random initialization.

The approach scales to any vocabulary size from 800 to 51200 and is agnostic to the
specific tokenizer algorithm — BPE, PUMA, or AA are handled by the same pipeline with
the AA case collapsing to standard fine-tuning as a special case (k̄ = 1). The PUMA-only
extensions are additive on top of this, not a parallel pipeline: parent units and
singletons (no surviving mutation children) take the unmodified BPE-equivalent treatment
throughout, and the lineage-aware steps activate only for tokens where genealogy
actually exists.
