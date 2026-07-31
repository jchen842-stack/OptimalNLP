# Research Diary — Summer D5.1: OptimalCE — Real SNLI Token Masks, the K-Axis Wall, and a Correction to D5

*Revises D5. D5 reproduced the predicted vision→NLP failure on **synthetic** masks, where
overlap was a knob we turned, and validated the §3.3 beam-fallback fix. This entry removes the
knob: every mask here is a real spaCy/benepar/WordNet annotation of SNLI dev tokens. It is
numbered D5.1 rather than D6 because the result is not a new direction but a **revision of D5's
central claim** — real overlap is block-structured, the synthetic generator overstated the
severity, and the wall lives on a different axis than D5 reported.*

> ### ⚠️ Corrections from later entries (added D5.5, in place — nothing deleted)
>
> - **Finding 4, "the wall arrives on the K axis": superseded by D5.5 §3.** The wall arrives on
>   the **length** axis. At M=24,199, exact search at length 3 terminates in 4.7–8.5 s even at
>   K=50 all-categories, while length 4 at K=15 takes 200–1,500 s with timeouts. K=15→50 at
>   length 3 costs ~4x; length 3→4 at K=15 costs 100x or more.
> - **Finding 3, "~half the concept pairs stay disjoint": needs qualifying (D5.5 §1).**
>   `disjoint_pairs` is a property of the *vocabulary*. The paper's §4.3 variable is the
>   fraction of *elements* carrying exactly one concept, which is **13.9%**, not ~51%, and it
>   falls as the corpus grows. The two statistics point opposite ways.
> - **The 2,547-token corpus is superseded** by 24,199 tokens (D5.5 §1).
> - Results here use the **proxy** neuron throughout.

## 0. Headline

- **No new dependencies.** The token annotations (`snli_1.0_dev.feats`) ship pre-computed in
  the Compositional Explanations of Neurons NLI codebase, so this needs no spaCy, no benepar,
  no GPU, and no model checkpoint. It ran on a laptop; **no Nautilus pod was used.**
- **The controlled contrast comes free with the data.** Single-valued annotation categories
  (`tag`, `dep`, `lemma` alone) are disjoint *by construction*; all categories at once overlap
  *by construction*. Only the admitted concept vocabulary changes between arms — the token set
  is identical. The regime shift is therefore a property of NLP annotation, not of our generator.
- **Validation:** the disjoint arms reproduce D5's synthetic control *exactly* (mean overlap
  1.00, common 0%, every pair disjoint, 2 nodes visited).
- **Correction to D5:** real overlap is more severe on paper (mean 4.20 vs synthetic 2.86,
  tokens in up to 7 concepts) yet *far more tractable*, because it is block-structured —
  concepts are mutually exclusive **within** a category, so most disjoint pairs survive and the
  fast-path is degraded rather than removed. **Uniform random overlap overstates the damage.**
- ⚠️ **Superseded — see the correction banner at the top of this entry.** The wall sits on
  the **length** axis. At M=24,199 exact search at length 3 terminates in 4.7–8.5 s even at
  K=50, while length 4 at K=15 takes 200–1,500 s with timeouts. The claim below is left as
  written.

  **The wall is real but sits on the K axis, not length alone.** K=50 / length 4 does not
  terminate; a K-matched disjoint control at the same size visits **2 nodes in 0.25 s**.
- **The fix holds on real data, and more cleanly than on synthetic** — IoU plateaus at beam 200
  (2.3 s) instead of wobbling with width.

## 1. Data: the annotations were already free

`nli/data/analysis/snli_1.0_dev.feats` (8 MB, 20 000 sentences) is checked into the NLI
codebase. Line format, one sentence per line, whitespace-separated tokens:

```
text|lemma|tag|dep|ent|synset|const
```

`const` is multi-valued (`;`-separated constituent labels); the rest are single-valued and may
be empty. A **concept** is a `(category, value)` pair, e.g. `("tag","NNS")`. Empty fields
contribute no concept — an absent entity tag is not the concept "no entity" — which keeps
`uncoverable` meaningful downstream.

This sidesteps the dependency wall that `annotate.py` would otherwise impose (spaCy 2.x API +
`benepar_en2` + WordNet). We consume the output, not the pipeline.

## 2. Method: change only the concept vocabulary

`src/real_token_masks.py` builds a `(K, M)` boolean concept matrix over 200 sentences
(2 547 tokens, chosen to sit near D5's synthetic `M=2048`). Concepts are the K most frequent
`(category, value)` pairs meeting `min_support=5`, which mirrors how the upstream analysis
drops the long tail of hapax lemmas and synsets.

`src/real_token_search.py` then feeds those masks to the *same* measurement core as D5
(`HeapProbe`, `StubConfig`, the quantity helpers), so peak-frontier / visited / time / IoU are
directly comparable to `results/overlap_sweep.csv` and `results/beam_sweep.csv`.

**The neuron is a proxy** — an OR of three real concepts plus 5 % label noise, mirroring D5's
`make_neuron`. The target concepts are drawn from the middle of the frequency order so the
target is neither near-empty nor near-universal. This bounds the claim: we are measuring how
the **search** behaves under real token concept structure, which is where the combinatorics
live, *not* what real trained neurons encode.

## 3. Experiment 1 — where does real SNLI actually sit? (`results/real_token_stats.csv`)

`K=15`, 2 547 tokens; only the admitted categories change.

| arm | mean overlap | common % | disjoint pairs | max overlap | coverage |
|---|---|---|---|---|---|
| *synthetic disjoint (D5, p_add=0)* | 1.00 | 0 % | **210** | 1 | — |
| `tag` only | **1.00** | **0 %** | **210** | 1 | 0.951 |
| `dep` only | **1.00** | **0 %** | **210** | 1 | 0.929 |
| `lemma`+`synset` | 1.59 | 59 % | 200 | 2 | 0.400 |
| `tag`+`dep` | 1.80 | 80 % | 174 | 2 | 0.858 |
| all categories | **3.18** | 81 % | **110** | **7** | 0.973 |
| *synthetic token (D5, p_add=1)* | 2.86 | 100 % | **0** | — | — |

Two things stand out. The disjoint arms land on D5's synthetic control **exactly** — same mean
overlap, same common fraction, same 210 disjoint pairs — which is a real validation of the
generator. But the all-categories arm **exceeds** the synthetic worst case on mean overlap
(3.18 > 2.86) while retaining **110 disjoint pairs where synthetic had 0**. That surviving
structure is the story of this entry.

## 4. Experiment 2 — the search on real masks (`results/real_token_search.csv`, `real_K{30,50}.csv`)

Frontier cap 200 k. Disjoint control vs token regime at each size.

| K | length | regime | mean overlap | peak frontier | visited | time | outcome |
|---|---|---|---|---|---|---|---|
| 15 | 3 | disjoint (`tag`) | 1.00 | 34 | 2 | 0.01 s | ✓ |
| 15 | 3 | token (all) | 3.18 | 336 | 12 | 0.07 s | ✓ |
| 15 | **4** | disjoint (`tag`) | 1.00 | 58 | 2 | 0.02 s | ✓ |
| 15 | **4** | token (all) | 3.18 | 3 085 | 505 | 6.5 s | ✓ |
| 30 | **4** | token (all) | 3.84 | 15 027 | 590 | 115.7 s | ✓ (barely) |
| 50 | **4** | disjoint (`lemma`) | 1.00 | 257 | **2** | **0.25 s** | ✓ |
| 50 | **4** | token (all) | 4.20 | **48 014** | — | 220 s | ⏱ **timeout** |

**The disjoint control still never breaks** — 2 nodes visited at every size, exactly as in D5.
That part of the thesis is untouched by real data.

**But the length-4 result contradicts D5.** D5 predicted length ≥ 4 × overlap as the wall, and
its synthetic K=15/length=4 token case *timed out at 131 s*. The real K=15/length=4 case
**finishes in 6.5 s**. The wall does not arrive until K grows: 6.5 s → 115.7 s → non-termination
across K = 15 → 30 → 50. Length is still an accelerant, but on real masks **K is the axis that
decides termination.**

At the wall, against a K-matched disjoint control: 48 014 frontier vs 257, i.e. **187×** — and
the qualitative gap matters more than the ratio. At *identical* K and length, the disjoint
search visits two nodes and returns in a quarter second; the overlapping one never returns.

## 5. Why: overlap is block-structured, not uniform

D5's generator added memberships to *random* concepts, which drove `disjoint_pairs` 210 → 0 the
instant overlap appeared — the fast-path was **removed**. Real annotation cannot do that. A
token has exactly one POS tag, so `tag=NN` and `tag=DT` are mutually exclusive *by definition*;
the same holds within `dep`, within `lemma`, within `synset`. Overlap exists only **across**
categories. The concept graph is therefore a union of cliques of mutual exclusion, and
`disjoint_pairs` stays high: 110/210 at K=15 (52 %), 1 954/2 450 at K=50 — **80 % of pairs at
K=50 still
disjoint at the wall.**

So the fast-path is **degraded, not deleted**, and the search survives much further than the
synthetic sweep implied. This is a correction to D5's §4, and it cuts against our own earlier
result: *the synthetic model was pessimistic in a way that flattered the finding.* The wall is
real, but D5 located it too early and attributed it to the wrong variable.

## 6. Experiment 3 — the beam fallback on real data (`results/real_beam_K50.csv`)

The real wall case: `K=50, length=4, all categories`, 300 s budget.

| beam width | frontier | terminates? | visited | best IoU | time |
|---|---|---|---|---|---|
| none (exact) | 48 014 | ✗ | — | — | 220 s timeout |
| **100** | 232 | ✓ | 55 | 0.688 | **0.54 s** |
| **200** | 334 | ✓ | 22 | **0.738** | **2.3 s** |
| 500 | 639 | ✓ | 46 | 0.738 | 9.0 s |
| 1000 | 1 139 | ✓ | 371 | 0.739 | 18.6 s |

The §3.3 escape hatch transfers to real data intact, and behaves **better** here than on
synthetic masks. D5 found IoU wobbling with width (0.377 → 0.368 → 0.424 → 0.420) and beam 2000
still timing out; on real masks IoU rises once and then **plateaus flat at 0.738 from beam 200
onward**, while time grows roughly linearly. That gives a clean, defensible operating point —
beam 200 buys the full attainable explanation quality in 2.3 s against a search that otherwise
never returns — instead of D5's "tune it and hope." Plausibly the same block structure is
responsible: with 80 % of pairs still disjoint, the top of the frontier is better ordered, so a
narrow beam is less likely to discard the eventual winner.

## 7. Methodological notes (two things to not repeat)

**A control that silently stopped being a control.** The first K-sweep used `tag` as the
disjoint control at every K. But `tag` is closed-class: only ~23 POS tags meet `min_support`, so
at K=30 and K=50 the "control" was still K=23 while the token arm grew. That inflated the
reported ratio to **649×**. Re-running against `lemma` — also single-valued and therefore
disjoint, but open-class and so able to scale — gives a properly K-matched **187×**. The
superseded figure is recorded here deliberately. *Closed-class categories cannot be K-matched
controls; check that a control still varies with the variable you are sweeping.*

**IoU is not comparable across arms.** The proxy neuron is built from each arm's *own*
vocabulary, so its target differs per arm. Frontier size, visited count, and wall time are the
cross-arm comparable metrics; IoU is only comparable *within* an arm (e.g. down a beam sweep).

## 8. Scope

- Concept masks are real; **the neuron is a proxy**. These are not explanations of real trained
  neurons. Real activations need a trained SNLI model — see §9(A2); **no external download is
  involved**, contrary to an earlier draft of this entry.
- 200 sentences / 2 547 tokens, to hold M near D5's synthetic 2 048. Scaling M is untested.
- Single seed. The regime gaps are orders of magnitude, so they are unlikely to be noise, but
  the beam-width IoU plateau in §6 deserves repetition across seeds before it is leaned on.

## 9. Next steps

- **(A2) Real activations.** Replace the proxy neuron with real unit activations over the same
  token axis. This upgrades the claim from "real token concept *structure* breaks the search"
  to "real neurons in a trained model do." **Cheaper than first assessed.** An earlier draft
  said this required fetching the OpenNMT en-de BiLSTM from `nli/models/README.md`; that
  checkpoint belongs to the **MT** half of that codebase and the NLI path never touches it.
  `settings.py` points at `models/bowman_snli/6.pth` with `MODEL_TYPE = "bowman"` — a Bowman
  SNLI entailment classifier (`BowmanEntailmentClassifier`, `models.py`) produced locally by
  `snli_train.py`, which writes `{epoch}.pth`; the checked-in `preds/6_snli_1.0_dev.csv` is
  that epoch-6 model's output. So the step is **train a small LSTM, not download anything**,
  and it is CPU-feasible. The plumbing already lines up: `analyze.py:extract_features()`
  collects per-token hidden states `(n_tokens, n_units)` and `quantile_features()` thresholds
  each unit into a boolean mask over tokens — precisely the `bitmaps` array
  `real_token_search.py` currently fills with the proxy, so the swap is one array plus a
  training run. Our own task7 Bowman LSTM (0.775 dev) is the same architecture family and is
  an alternative source of units.
- **(B) Sharpen the fix.** Auto-select beam width from a time budget, or add a total-expansion
  budget guaranteeing termination regardless of width. The real-data plateau at 200 suggests
  width selection may be easier on real inputs than the synthetic sweep implied.
- **(C) Re-examine D5's scale curve** under block-structured overlap: rebuild the synthetic
  generator to respect within-category mutual exclusion and check whether it then reproduces the
  real K-axis wall. If it does, we have a generator worth trusting for sizes real data cannot reach.

## 10. Artifacts

- Masks + statistics: `src/real_token_masks.py` → `results/real_token_stats.csv`
- Search on real masks: `src/real_token_search.py` → `results/real_token_search.csv`,
  `results/real_K30.csv`, `results/real_K50.csv`, `results/real_lemma_control_K50.csv`
- Beam validation: `results/real_beam_K50.csv`
- Fix (unchanged from D5): `patches/0001-frontier-beam-fallback.patch`
- Source annotations: `snli_1.0_dev.feats` from the NLI codebase (not vendored here)
- Upstream pin: `70805299` (`UPSTREAM`)
