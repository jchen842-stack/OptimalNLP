# Research Diary — Summer D5.2: OptimalCE — Real Neuron Activations, and Why D5.1's Correction Was Half an Artifact

*Revises D5.1. D5.1 replaced synthetic masks with real SNLI token annotations but kept a
**proxy neuron** — an OR of three concepts plus noise — and concluded that D5's synthetic
sweep had overstated the frontier explosion. This entry replaces the proxy with real
per-token unit activations. The conclusion partly reverses: **the proxy was flattering the
result**, and at the very configuration D5.1 held up as evidence that D5 was too pessimistic,
real units time out.*

## 0. Headline

- **The proxy neuron was doing more work than the block structure was.** At `K=15, length=4,
  all categories` — D5.1's showcase "terminates comfortably in 6.5 s" case — real units give a
  **6.1× larger frontier, 22.5× the time, and 1 of 5 units does not terminate at all.**
- **Mechanism, visible in one column.** The proxy scores IoU 0.905 because it *is* an OR of
  three concepts; real units score 0.563–0.649. A strong best-so-far is what prunes this
  search, so an easily-explained target prunes hard and a realistic one does not.
- **D5.1 conflated two confounds** — block-structured overlap *and* an artificially easy
  target — and attributed the whole discrepancy to the first. D5's original synthetic
  prediction was closer to correct than D5.1 credited.
- **These units are untrained**, which is plausibly the *hardest* case, so this brackets the
  pessimistic end rather than settling the question. See §5.
- Still no corpus download and no training run: the random-weights arm is free.

## 1. Two code-reading errors in D5.1's plan (recorded, not quietly fixed)

D5.1 §9(A2) claimed the remaining work was to load a checkpoint and swap one array. Both
halves were wrong, and the correction is the reason this entry needed its own extractor.

**(a) `RANDOM_WEIGHTS` does not reach the NLI path.** `data/snli.py:load_for_analysis`
unconditionally `torch.load`s a checkpoint and takes **both** the weights and the vocabulary
(`stoi`/`itos`) from it. The flag lives in `loader.py`, the *machine-translation* loader —
and even there, `ltm()` calls `torch.load` before consulting the flag, because `model_opt`
and `vocab` come out of the checkpoint. There is no checkpoint-free route through their code.

**(b) Their NLI neurons are on the wrong axis for us.** `analyze.py:extract_features` calls
`model.get_final_reprs(s1, s1len, s2, s2len)`, which returns `mlp_input` — **one vector per
sentence *pair***. Upstream NLI explanations are therefore over *examples*, with token
features aggregated up (`get_max_ofis`). Our concept masks are over **tokens**. "A one-array
swap" was never available.

Both are recoverable because `models.TextEncoder.get_states()` already returns the
per-timestep LSTM outputs `(seq_len, batch, hidden_dim)` — exactly one activation vector per
token. So we build the encoder directly and derive the vocabulary from the corpus.

## 2. Method — `src/real_activations.py`

- **Vocabulary from the corpus**, not a checkpoint (427 types over the 200-sentence slice), so
  there are no OOV tokens. With random weights the embedding table is arbitrary anyway; what
  matters is that a token always maps to the *same* row, so activations carry real lexical
  structure rather than independent per-occurrence noise.
- `TextEncoder(vocab, 300, 512)` → `get_states()` → **(2 547 tokens, 512 units)**.
- **Binarization** follows upstream's default (`quantile_features` with `ALPHA=None`, i.e.
  threshold at 0); `--alpha` gives sparser masks if wanted.
- **Alignment is asserted, not assumed.** `real_token_masks.load_sentences` is now the single
  parser for both sides, and `verify_alignment` fails loudly if the activation row count and
  the mask token count disagree. Two independent parsers over the same file was the obvious
  way to silently mis-index every neuron.
- **Unit selection** is filtered to density ∈ [0.15, 0.85]; a unit that fires on ~everything or
  ~nothing has a degenerate optimum and would measure thresholding, not search behaviour.

**Density is matched, deliberately.** Real units average 0.504 density against the proxy's
0.43. Had the real masks been much sparser, the frontier difference could have been dismissed
as a density artifact rather than an explainability one.

## 3. Experiment — proxy vs real units (`results/real_units_K15.csv`)

`K=15, length=4, all categories`, 300 s budget, frontier cap 200 k. **Only the neuron changes.**

| neuron | density | peak frontier | visited | best IoU | time | outcome |
|---|---|---|---|---|---|---|
| proxy (D5.1) | 0.43 | 3 085 | 505 | **0.905** | 6.5 s | ✓ |
| unit 396 | 0.623 | 17 239 | 1 440 | 0.607 | 14.0 s | ✓ |
| unit 92 | 0.626 | 13 456 | 5 978 | 0.615 | 98.8 s | ✓ |
| unit 510 | 0.552 | 18 760 | 7 079 | 0.563 | 145.6 s | ✓ |
| unit 88 | 0.640 | 20 062 | 8 267 | 0.649 | 172.4 s | ✓ |
| unit 413 | 0.396 | 20 222 | — | — | 301 s | ⏱ **timeout** |

Median real unit: **6.1× the proxy's frontier, 22.5× its time.** Every real unit is worse than
the proxy on both, and the spread across units (14 s → non-termination) is itself larger than
the entire proxy-vs-disjoint gap D5.1 reported.

## 4. What this does to D5.1's claim

D5.1 §4 argued: *"the real K=15/length=4 case finishes in 6.5 s where D5's synthetic predicted
a timeout — uniform random overlap overstates the damage."* With a real neuron, that case
**does** time out. The sentence survives only for the proxy.

D5.1 varied the masks (synthetic → real) and held the target artificially easy, then
attributed the entire tractability gain to **block-structured overlap**. This entry shows the
target was carrying much of it. Both effects are real — block structure genuinely preserves
~80 % of disjoint pairs (D5.1 §5, unchanged) — but their relative sizes were misassigned. The
honest summary across three entries:

| entry | masks | neuron | K=15 / length 4 |
|---|---|---|---|
| D5 | synthetic | synthetic OR-target | timeout (131 s) |
| D5.1 | **real** | proxy OR-target | 6.5 s ✓ |
| D5.2 | **real** | **real unit** | 14 s – timeout (1/5) |

*Methodological note for the next one: D5.1 changed two things at once and read the result as
though it had changed one. The regime contrast was well controlled; the target was not
controlled at all.*

## 5. Scope — why this is the pessimistic bound, not the answer

The units are **untrained**. Random features have no reason to align with POS tags, deps, or
synsets, so they are close to the worst case for a *linguistic* concept vocabulary: no short
formula fits, the floor stays weak, pruning fails. A trained neuron may well be more
explainable, giving a stronger floor and a faster search.

So the truth is bracketed, not settled:

```
proxy neuron  (IoU 0.905, 6.5 s)  ..... trained units? ..... untrained units (IoU ~0.6, timeouts)
   optimistic bound                                              pessimistic bound
```

That is precisely why training the model is now worth the corpus download and the compute:
it is the only way to locate the real answer inside this interval, and the interval is wide
enough (6.5 s to non-termination) that the answer matters.

Other limits carried over from D5.1: 200 sentences / 2 547 tokens; single seed for mask
construction; 5 units sampled, which is enough to show the direction but not to characterise
the distribution over all 512.

## 6. Next steps

- **(A3) Trained units.** Fetch SNLI, train the Bowman classifier via `snli_train.py`, rerun
  §3 with trained units, and place the answer inside the bracket in §5. Same extractor —
  `real_activations.py --ckpt` already loads an `encoder.*`-prefixed state dict.
- **(A4) More units.** 5 of 512 shows direction only. Sweep ~50 and report the distribution of
  frontier/time, since the unit-to-unit spread is large.
- **(B) The beam fix under a realistic target.** D5.1's beam sweep used the proxy, so its clean
  IoU plateau at width 200 is now suspect for the same reason. Rerun on real units — this is
  the practically important one, since the beam width is what a user would actually set.

## 7. Artifacts

- Activations: `src/real_activations.py` → `results/real_activations.npz` (512 units × 2 547 tokens)
- Search with real units: `src/real_token_search.py --neuron real` → `results/real_units_K15.csv`
- Shared parser guaranteeing axis alignment: `real_token_masks.load_sentences`
- Unchanged from D5.1: masks, statistics, the beam patch
