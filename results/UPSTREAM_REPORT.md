# Report to upstream — `optimal-compositional-explanations` @ `70805299`

**Status: three verified claims. Reported as an unsound bound, not a caller error.**

Measured on SNLI token-level concepts: K=15 concepts, M=24,199 tokens (2,000 sentences),
`min_support=5`, formula length 3, 27 (arm, alpha, unit) pairs. Repo, code and raw output:
`results/METHOD_NOTES.md`, `tests/test_bruteforce_oracle_all27.py`, `src/exp_partition.py`.

---

## Claim 1 — `optimal.py` returns non-optimal answers at the paper's own maximum length

Enumerating **all 30,375 in-grammar length-3 formulas** in integer popcount arithmetic — the
exact space `expand_node` constructs, i.e. left-deep with `Or(f,leaf)`, `And(f,leaf)`,
`And(f,Not(leaf))` — across all 27 pairs, the search misses its own in-grammar optimum on 2:

```
trained a=0.2  unit88   in-grammar 0.25454105110196174   returned 0.2522022213711222   +0.9274%
trained a=0.05 unit86   in-grammar 0.21660649819494585   returned 0.20679723502304148  +4.7434%
```

The other 25 tie to float64 equality. Both missed optima have the form
`((dep=ROOT OR dep=nsubj) AND C)` — constructible by the three moves — and **upstream's own
`compositional/beam_optimal.py` finds both**, returning a higher IoU than the exact search.
At width 200 `beam_optimal` is optimal on **27/27**; the exact search is optimal on 25/27.

Reproduce: `python tests/test_bruteforce_oracle_all27.py`.

## Claim 2 — the cause is an inadmissible aggregated bound, pruned before refinement

Tracing the prefix `P = (dep=ROOT OR dep=nsubj)` that carries both optima:

```
unit88   assigned ceiling 0.232677 < true_max(P) 0.254541   removed by reduce_frontier at threshold 0.232934
unit86   assigned ceiling 0.203398 < true_max(P) 0.216606   removed by reduce_frontier at threshold 0.203735
```

At the moment of removal the node still carried the **aggregated** (`"sum"`) estimate — it had
not been popped and refined. `reduce_frontier` acts on the unrefined aggregated bound
(Algorithm 1 lines 11, 52), so **whether the optimum survives depends on pop ordering**: on
whether the node is expanded before a stale copy is pruned.

Admissibility of the aggregated form requires `Bott_1(E^C)_x = 0`. Section E.2.2 states the
violating case is *"a rare degenerate case (not observed in any of the datasets tested in this
paper)"*. **Measured on this corpus: `Bott_1(E^C)_x = 859`, on 100% of samples.** The common
region is 83.8% of tokens (20,280 / 24,199) and every one of the 15 concepts has thousands of
common locations where the neuron is silent. `min_support` + top-K selection on a token-level
corpus makes the violation universal rather than rare.

## Claim 3 — correct partitioning does not repair it

The aggregated estimate is **partition-invariant**: `SUM_x |I^C_max(L)_x|` counts over all
`(x,j)` pairs, and `Top^A_t` / `Bott^A_1` are concept-wise over dataset-wide totals. None
depends on how elements are grouped into samples.

**Confirmed empirically**: repartitioning from one sample to one-sample-per-sentence leaves
both ceilings **byte-identical** (`0.232677`, `0.203398`), and both prefixes are still dropped.

Under the corrected partition, `Bott_1(E^C)_x = 0` compliance rises from 0% to a median of
69.7% of sentences — **and 13 of 27 optimum-carrying prefixes still have an inadmissible
ceiling, with one pair worse than before.** Observed losses go 2 → 0, but not because the
bound improved: that is a pop-ordering effect, and 0 of 9 vulnerable pairs is within chance of
a 2/9 base rate (p ≈ 0.10).

**Our own harness reached the degenerate case by construction** (`bitmaps = reshape(1, M)`,
one sample holding the whole corpus). We report that as ours. But it **amplified exposure; it
did not cause it** — the bound is inadmissible whenever the condition fails, the condition is
universal, and a compliant partition does not restore soundness.

---

## Remedies

**(a) Do not prune on an unrefined aggregated estimate.** Refine before `reduce_frontier`
acts, or restrict pruning to refined nodes. This removes the order-dependence and addresses
the defect rather than its detection.

**Its cost, stated with it:** per Section C, sample computation is on the order of `|D|x` more
arithmetic per estimate than the aggregated form, and this pays it on **every node**, not only
popped ones — precisely the expense the aggregated path exists to avoid. **The remedy converts
a soundness defect into a runtime cost.** That is a real trade, and we present it as one.

**(b) Assert or warn** at `get_optimal_heuristic_info` when `Bott_1(E^C)_x != 0` on any sample,
naming E.2.2. Cheap — already a by-product of the quantity helpers. But it only *reports* the
violation, and per Claim 3 a compliant partition is not sufficient either.

## A diagnostic users can run with no code change

`n_prefixes` — the number of distinct construction paths to a formula — is a function of the
**operator signature alone**, readable off a published formula **by inspection**:

> **If the returned formula's two operators differ, there is exactly one construction path to
> it and no redundancy against an unsound prune.**

Identical adjacent operators commute and give alternatives: an OR-run of length `r` gives
`r+1` paths, an AND-NOT run gives `r` (the leftmost term is never negated, so the positive base
is pinned).

**Support:** 2 of 9 single-path units lost their optimum (22%); **0 of 18** units with a
redundant path did. The vulnerable share of signature space is **2/3 at every length**, so this
does not become less relevant for longer formulas.

---

## Scope and limits

- Verified at **length 3 only**. Length-4 optimality is **unverified**: our one-sided check
  (a length-4 result below the length-3 optimum is a definite miss) found 0 of 23 terminating
  pairs, but its median detection margin is 2.5%, so it is near-blind to losses of the smaller
  observed magnitude (0.93%) and excludes 4 timeouts.
- The 2/27 rate is **not** a failure rate for the method. Among the 9 pairs whose optimum has a
  single construction path it is **2/9**; among the other 18 it is **0/18**.
- One corpus, one model, one concept vocabulary. We have not tested the vision datasets.

---

## Observation, explicitly NOT a claim — the |D|x cost model did not hold here

Section C's cost model implies sample computation is on the order of `|D|x` more arithmetic per
estimate than the aggregated form. We measured the opposite:

```
matched median wall clock, length 4 :  flat (|D| = 1)  639.4s  ->  per-sentence (|D| = 2000)  247.2s
                                                                    0.39x -- 2.6x FASTER
matched median nodes expanded       :  8,340 -> 8,553  (1.03x, unchanged)
```

**One hypothesis, untested:** the `|D|x` factor may fall on cheap per-sample **scalars**, while
the expensive **mask** operations span the same 24,199 bits under either partition — so
partitioning multiplies the cheap term and leaves the dominant term untouched. If that is
right, Section C's model assumes many samples with small per-sample masks, which is a
**vision-shaped assumption** sitting alongside the `Bott_1(E^C)_x = 0` condition in E.2.2.

We have not tested this and do not claim it. We report the measurement because it is the
opposite of what the model predicts, and because if the explanation is right it would mean two
of the method's assumptions are shaped by the vision setting rather than stated as
preconditions.

Related, and better established: peak frontier rose 1.19-1.39x under the corrected partition
while **distinct nodes expanded stayed flat** (1.03x). The cause is refinement churn —
Algorithm 1 line-18 re-insertions rose **8.6x** (median 72 -> 622 per unit). The search is not
larger; it re-queues the same nodes more often.
