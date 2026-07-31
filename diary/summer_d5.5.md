# Research Diary — Summer D5.5: The Activation Range Was the Bug; Corrected Protocol, and an Audit

*Revises D5.4. D5.4 established that our IoUs were tracking density and that the search was
returning high-coverage blankets. This entry finds why — we deviated from the paper on the
one parameter that determines whether an IoU means anything — corrects it, re-runs the
protocol at a 10x corpus, scores three pre-registered predictions (all three fail), and puts
the pipeline through an adversarial correctness audit that turned up one documentation error
and one property of the upstream method.*

---

## 0. Headline

The paper (arXiv 2511.20934 §B) defines the neuron activation range as the **top 0.005
quantile**. Our pipeline used `ALPHA=None` — threshold at 0 — which for a tanh-bounded LSTM
state splits ~50/50. A 100x deviation on the one parameter that sets the chance baseline.

Correcting it: **lift rises monotonically as the range tightens**, 1.05 → 1.99 → 2.42 → 3.41
in the trained arm. The near-chance result was an artifact of the range, not a property of the
neurons. But normalised fit peaks at ~20% and falls again, so these units are not solved —
just no longer trivial.

Second headline, from matching the paper's own max formula length: the NLP optimality gap is
**+5.05%**, against their vision band of +5.1–6.5%. A clean cross-domain replication, and it
only appears once length is matched.

---

## 1. Corpus scaling — forced, not optional

At M = 2,547, alpha = 0.005 yields ~13 firing tokens. Not computable. Density and corpus size
are coupled, so the corpus had to grow. 200 → 2,000 sentences, **24,199 tokens**.

| | 200 sents | 2,000 sents |
|---|---|---|
| tokens | 2,547 | 24,199 |
| (cat,val) pairs clearing `min_support=5` | 264 | **1,168** |
| K=15 vocabulary | — | 14 of 15 shared; `tag=JJ` out, `lemma=.` in |

Trained-checkpoint OOV against the probe corpus: **8.8%** (2,137/24,199), with `stoi` taken
from the checkpoint.

### Unique elements — the statistic we had been reporting was the wrong one

The paper's §4.3 variable is the fraction of **elements** carrying exactly one admitted
concept. We had been quoting `disjoint_pairs`, which counts **concept pairs** that never
co-occur — a property of the vocabulary, not the elements.

| | 2,547 tokens | 24,199 tokens |
|---|---|---|
| **unique-element fraction** | 18.1% | **13.9%** |
| disjoint_pairs | 52.4% | 51.4% |

They point opposite ways. ~51% of pairs are disjoint while only ~14% of tokens are unique, and
scaling the corpus makes the element-level picture **worse** while the vocabulary-level number
barely moves. D5.1 finding 3 ("~half the concept pairs stay disjoint, so the fast-path
partially survives") is the vocabulary-level reading and needs this qualification.

`results/unique_elements.csv`.

---

## 2. The alpha sweep (`results/alpha_sweep_K15.csv`)

Pre-registered: alpha ∈ {0.5, 0.2, 0.1, 0.05}, plus 0.005 attempted. Beam 200, 5 units per
arm, K=15, length 4, M=24,199. 40 runs, **no timeouts**.

| arm | alpha | density | mean cov | **mean lift** | mean nfit% |
|---|---|---|---|---|---|
| trained | 0.5 | 0.499 | 0.852 | **1.051** | 5.1 |
| trained | 0.2 | 0.186 | 0.505 | **1.985** | 20.4 |
| trained | 0.1 | 0.083 | 0.217 | **2.423** | 9.9 |
| trained | 0.05 | 0.047 | 0.114 | **3.405** | 10.6 |
| untrained | 0.5 | 0.500 | 0.847 | **1.033** | 3.3 |
| untrained | 0.2 | 0.200 | 0.420 | **1.723** | 18.1 |
| untrained | 0.1 | 0.100 | 0.194 | **2.102** | 12.2 |
| untrained | 0.05 | 0.049 | 0.205 | **2.074** | 5.6 |

**alpha = 0.5 at M = 24,199 reproduces the old alpha = 0 result at M = 2,547 exactly** — lift
1.03–1.05, coverage ~0.85, single-leaf blanket formulas like `(NOT tag=IN)` at coverage 0.87.
The degeneracy was the activation range, not the corpus size, and it survives a 9.5x corpus
change.

Two things that do **not** improve with alpha:

- **Normalised fit is non-monotonic**, peaking at alpha=0.2 then falling. Lift and fit
  disagree because the headroom term 1/d − 1 grows faster than lift. Formulas become far more
  selective than chance while capturing a *smaller* share of available headroom. At best
  ~20% of the way from chance to "the formula is the neuron".
- **Anchoring on `const=NP`/`const=VP` gets worse**, 1/5 → 5/5 trained. But the character
  changes completely: at alpha=0.5 the winners are one-leaf negations covering ~85% of tokens;
  at alpha=0.05 they are genuine compositions where `const=NP` appears mostly *inside a NOT*,
  as a restrictor rather than a blanket.

### alpha = 0.005 is not reachable at this corpus

Pre-registered statistical floor: exclude any (unit, alpha) with < 200 firing tokens.
At alpha = 0.005, the **maximum** firing count over all 512 units is 121. **512/512 excluded
in both arms.** M ≥ 40,000 tokens (~3,300 sentences) would be required, and that only puts the
median unit at the floor. Exclusions elsewhere: 0/512 at alpha 0.5/0.2/0.1; 7/512 trained at
alpha 0.05.

Also note **realised density drifts below nominal alpha** — tanh saturation puts ties at the
quantile threshold. Unit 413 realises 0.026 at a nominal 0.1; the worst case is an 18x
shortfall at alpha=0.05. Density is therefore taken **per run**, never per arm; an arm mean
produces lifts that cannot be reconciled against their own IoU.

---

## 3. Beam vs exact (`results/beam_vs_exact_K15.csv`, `_L3_K15.csv`)

Following the paper's own §4.3 advice — beam first, refine the interpretable units — Phase A
was beam over the full grid, Phase B exact on the 27 (unit, alpha) pairs with lift > 1.2.

**Length 4: 4 of 27 timed out** at the 1,500s cap and are excluded, giving n=23. They are
reported, not dropped. The cap is also *soft* — checked once per 256 heap pushes, and one run
overshot to 2,085s.

### Three pre-registered predictions, all NOT SUPPORTED

| prediction | stratified rho | permutation p | verdict |
|---|---|---|---|
| #1 primary — IoU gap vs lift | +0.011 | 0.5230 | NOT SUPPORTED |
| #1 secondary — lift gap vs lift | −0.180 | 0.2442 | NOT SUPPORTED |
| #2 — exact time vs lift | −0.259 | 0.1507 | NOT SUPPORTED |

The registered mechanism was that a strong incumbent found early prunes harder. It is not
supported. An **inverted** reading was also registered in advance, before the remaining
batches, because batch 1 pointed that way at n=3 — it is likewise not supported (rho +0.011,
indistinguishable from zero).

Scoring is stratified within (arm, alpha) because alpha drives lift and density together
(rho(density, lift) = −1.00 trained), so a pooled correlation is rho(y, density) sign-flipped.
p comes from 10,000 within-stratum permutations. The full reasoning, the amendments made
before the data existed, and the errors caught along the way are in
`results/METHOD_NOTES.md`.

### Q3 falsified: search cost rises as alpha falls

Predicted down, on the theory that a better-scoring formula prunes harder. Observed **up**, on
the within-unit design that removes the unit confound: 7 of 8 series show `visited` and
`time_s` rising as alpha falls, and 3 of 4 timeouts are at the lowest alpha.

### The band comparison, and why length matters

Table 4 reports a **ratio of averages** restricted to units where beam and exact find
different solutions. Our first comparison used an average of per-unit ratios and all pairs —
two errors that both inflate.

| | ratio-of-averages |
|---|---|
| length 4, differing pairs (n=22) | +8.77% |
| **length 3, differing pairs (n=7)** | **+5.05%** |
| length 3, all 27 pairs | +0.96% |
| paper Table 4 (vision) | +5.1% – 6.5% |

Table 4 is at max length 3; we were at 4. **At matched length the NLP result lands on the
vision band.** Most of the apparent NLP-over-vision excess was length, not domain.

The larger matched-length finding: **20 of 27 pairs agree exactly at length 3**, against 1 of
23 at length 4. Beam-200 finds the optimum for three quarters of units at the paper's length —
and we used a 40x wider beam than they did (200 vs 5, their Appendix B), which should make any
residual gap *smaller*, not larger.

### Length dominates cost, not K

`results/L3` grid, M=24,199, exact, 5 units:

| arm | peak frontier | visited | time |
|---|---|---|---|
| K=15 all cats | 412–1,728 | 340–866 | 0.4–1.9 s |
| K=30 all cats | 1,439–4,402 | 801–1,279 | 2.0–4.5 s |
| K=50 all cats | 3,249–7,537 | 956–1,578 | 4.7–8.5 s |
| K=50 lemma (disjoint control) | 41–474 | 1–2 | 0.0–0.3 s |

No timeouts anywhere, including K=50 all-categories. Against length 4 at K=15 taking
200–1,500 s with 4 timeouts: **K=15→50 at length 3 costs ~4x; length 3→4 at K=15 costs 100x or
more.** D5.1 finding 4 ("the wall arrives on the K axis") is superseded — the wall arrives on
the *length* axis, and K is the milder variable.

---

## 4. Correctness audit (`VERIFICATION.md`, `verify/run_all.sh`)

Ten checks, each recomputing a quantity a second independent way rather than re-reading the
code that produced it. `./verify/run_all.sh` runs them all and prints a PASS / FAIL / CANNOT
VERIFY table. **All pass.**

Highlights:

- **The patch is a genuine no-op.** Against a clean clone of upstream `70805299`, identical
  formula, `best_iou` hex `0x1.f51db8af0e455p-2`, visited 2,063, expanded 4,982, estimated
  48,937, peak 8,239. Bit-identical, not close.
- **Alignment.** `verify_alignment` compared row *counts*, which cannot detect a reordering.
  It now samples 50 rows and compares surface tokens; a deliberate two-row swap raises.
- **Padding.** Batched output matches unpadded single-sentence runs to 6e-8 on a 17.7x
  length-ratio batch, with a negative control 7 orders of magnitude larger.
- **One documentation error found.** `models/README.md` claimed the checkpoint came from an
  all-defaults training run. It did not: `--max_data` defaults to 100,000 → 16,669-type
  vocabulary, and the checkpoint stores 33,671, which only `--max_data 0` produces. Dev
  accuracy reproduced *exactly* either way (0.7934362934362934) — the metric passed and said
  nothing about provenance; a fingerprint caught it.
- **The OOV inversion.** Correct checkpoint vocabulary → 8.8% OOV. Rebuilding `stoi` from the
  probe corpus → **0.0%**, and that is the broken configuration: a vocabulary built from the
  probe corpus covers it perfectly, so every token maps to some row and every row is wrong,
  silently. **A suspiciously clean OOV rate is the alarm, not the all-clear.**

### The oracle, and a property of the method

`tests/test_bruteforce_oracle.py` enumerates the **entire** formula space and scores every
formula with plain numpy, then asserts the max equals what the search returns. This validates
the concept masks, the quantity helpers, the heuristic and the frontier end to end without a
second implementation.

At length 3 it matches to full precision on all three cases. **At length 4 it flagged a
mismatch** on untrained unit92 — 0.19523729099814222 unrestricted vs 0.19492814877430262 from
the search. Diagnosed rather than adjusted:

`expand_node` (`optimal.py:554-582`) grows a formula by exactly three moves, with candidate
labels being plain leaves — `Or(label, leaf)`, `And(label, leaf)`, `And(label, Not(leaf))`. So
formulas are **left-deep**, negation appears **only as AND-NOT**, there is **no `OR NOT`**, and
the leftmost term is never negated. Enumerating exactly that space reproduces the search's
value to full precision *and recovers the identical formula string*.

**The search is optimal in the space it can construct.** The unreachable formula is
`(tag=NN AND (dep=ROOT OR NOT const=VP)) OR dep=punct`, and `OR-NOT` is the distinguishing
move — isolated by running a left-deep enumeration that permits it (reaches the unrestricted
max) against one that forbids it (lands exactly on the search's value). Tree shape is not
involved.

Expressiveness gap: **+0.0000%** at length 3 on all three cases, **+0.1586%** at length 4 on
one unit. No reported number changes. The wording does: "exact search" means *optimal within
the method's formula grammar*.

---

## 5. What this means for the project's claim

- The tractability story (D5–D5.1) stands, with its axis corrected: **length**, not K.
- The explanation-quality story is rebuilt. Real units at a correct activation range reach
  lift 3.4x, and the trained arm separates from untrained (3.41 vs 2.07 at alpha=0.05) where at
  alpha=0.5 the two are indistinguishable. **That trained/untrained separation is a real
  finding the old setting could not have detected** — and it is the D5.3 question, reopened
  with an instrument that works.
- The optimality gap replicates the paper's vision result in NLP at matched length. This is
  the first NLP measurement of a quantity they report only for vision.
- Nothing here required modifying the heuristic. That remains the open engineering question.

---

## 6. Scope and caveats

- **n = 23 at length 4 is machine-dependent.** Four runs hit a wall-clock budget on this
  hardware; faster hardware gives a different n and different statistics. The length-3 grid has
  no timeouts and is the hardware-independent one.
- 5 units per arm per alpha. Small.
- alpha = 0.005, the paper's actual setting, remains **unmeasured** — it needs M ≥ 40,000.
- Untrained-arm weights depend on the torch RNG and therefore the torch version. Now recorded
  (`results/ENVIRONMENT.md`); it was not recorded for the original runs.
- The training seed is unrecoverable from surviving artifacts.

---

## 7. Next steps

1. **Lengths 5–6.** Length is the cost axis and the expressiveness gap is length-dependent;
   both arguments point here.
2. **alpha = 0.005 at M ≥ 80,000** (~6,600 sentences), beam-only. The paper's own setting,
   finally measurable.
3. **The `OR-NOT` gap.** Cheap to test whether adding `Or(label, Not(leaf))` to `expand_node`
   closes it, and whether it costs tractability.
4. **The #2 arm asymmetry** — per-stratum rho splits cleanly by arm (trained −1.000/−0.800/
   −0.400 vs untrained −0.400/+0.200/+0.500). Not evidence at n=3–4, but the only structured
   signal in three tests. Needs more units per arm.
5. Fixed-work comparison at equal expansions, as a cross-check on the Q3 falsification.

---

## 8. Artifacts

- Corpus + unique elements: `src/unique_elements.py` → `results/unique_elements.csv`
- Activations: `src/real_activations.py --alphas` → `results/acts2k_{arm}_a{alpha}.npz` (gitignored)
- Phase A: `results/alpha_sweep_K15.csv` via `src/alpha_sweep_report.py`
- Phase B: `results/beam_vs_exact_K15.csv`, `results/beam_vs_exact_L3_K15.csv`,
  `results/phaseB_report.txt`, `results/phaseB_report_L3.txt` via `src/phaseB_report.py`
- Corrected metrics: `src/corrected_metrics.py`
- Audit: `VERIFICATION.md`, `verify/run_all.sh`, `verify/*.py`
- Oracle: `tests/test_bruteforce_oracle.py` → `results/oracle_L3.txt`, `results/oracle_L4.txt`
- Method record, pre-registrations and error post-mortems: `results/METHOD_NOTES.md`
- Environment: `results/ENVIRONMENT.md`
- Reproduction: `REPRODUCE.md`
