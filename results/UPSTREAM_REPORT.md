# Report to upstream — `optimal-compositional-explanations` @ `70805299`

Measured on SNLI token-level concepts, M = 24,199 tokens (2,000 sentences), `min_support = 5`,
formula length 3, 27 (arm, alpha, unit) pairs, one sample per sentence.
Code and raw output: `tests/test_bruteforce_oracle_all27.py`, `src/exp_k50_oracle.py`,
`src/exp_event_ordering.py`, `src/exp_refined_estimate.py`, `results/METHOD_NOTES.md`.

**Two observations. We are not claiming a cause for either, and we have retracted the
mechanism we first proposed.**

---

## 1. A complete formula, with admissible bounds, discarded unevaluated

On `trained a=0.2 unit88` at K = 15, the formula `((dep=ROOT OR dep=nsubj) AND const=NP)`:

```
flagged FINAL (next_op == "INDIVIDUAL")     yes
its exact IoU                               0.25454105110196174
incumbent threshold at that moment          0.25056904400606983
it exceeds the incumbent by                 0.003972007095891905
```

It was popped, carried through an estimation path, and **never evaluated**. Its exact IoU was
one mask operation away, and had it been computed it would have become the new incumbent.

**This is not a bound problem.** We logged all four path estimates for its parent, pre-filter,
and every one bounds the true reachable value from above:

```
parent P = (dep=ROOT OR dep=nsubj)      true reachable from P = 0.25454105110196174
   INDIVIDUAL 0.23267674991206472  (= IoU(P), the correct stop-here bound)
   OR         0.5360534646500176
   AND        0.4175506268081003
   NOT        0.40520673813169983
```

**It reproduces at K = 50**, where our other observation does not: **445,644 of 525,933
refinement calls (84.7%) carried `next_op == "INDIVIDUAL"`** — complete formulas being
estimated rather than evaluated. We have not established that this causes anything. We report
it because a formula flagged final, whose exact IoU is computable in one operation and exceeds
the incumbent, going unevaluated seems worth your attention regardless of consequence.

**The event rate does not predict harm, on any normalisation we tried.** Raw counts are not
comparable across configurations (K=50 explores 37x the space and pops 5.7x as many nodes), so
normalised:

```
config       events      pops     per-pop      per-formula    MISSES
flat K=15        45    37,096   1.213e-03       5.487e-05        2
part K=15        76    49,599   1.532e-03       9.267e-05        0
part K=50       191   209,964   9.097e-04       6.288e-06        0
```

**The one configuration with harm sits in the middle on both rates** — neither highest nor
lowest. Moving the rate in either direction lands on a configuration with zero harm. **We
therefore do not suggest "discard fewer formulas" as a remedy**: nothing here indicates harm is
a function of the discard rate.

## 2. Non-optimal returns at K = 15, which do NOT persist at K = 50

Enumerating all 30,375 in-grammar length-3 formulas in integer arithmetic across 27 pairs at
K = 15, the search misses its own in-grammar optimum on 2:

```
trained a=0.2  unit88   in-grammar 0.25454105110196174   returned 0.2522022213711222   +0.9274%
trained a=0.05 unit86   in-grammar 0.21660649819494585   returned 0.20679723502304148  +4.7434%
```

The other 25 tie to float64 equality. Your `beam_optimal.py` finds both.

**At K = 50 the same enumeration (1,125,000 formulas per pair, float64) gives 0 misses out of
27, and neither of these two persists.** So this is specific to our K = 15 configuration.

**Confound we cannot resolve:** K = 50 changes two things at once — it is a 37x larger space,
**and** the `Bott_1(E^C)_x = 0` precondition of E.2.2 holds there while it does not at K = 15.
We have not run the matched-K control that would separate them.

### What we retracted

We first reported that the aggregated bound was inadmissible and caused these misses.
**Both halves are withdrawn.** The inadmissibility "demonstration" compared the ceiling of the
**INDIVIDUAL (stop-here)** frontier entry against the best value reachable by **extending** the
node — different quantities. `estimate_paths_iou` emits four entries per node, and only the
stop-here entry was dropped, correctly, at its own `IoU(P)`. **We have not shown a single
inadmissible estimate.**

Five candidate explanations have been tested and all five failed: the disjoint fork in
`estimate_label_quantities`, aggregated-bound inadmissibility, the `reduce_frontier` threshold
prune, the `:699-709` refinement discard, and all three chain estimators (`:368`, `:509`,
`:581`). **The cause is unknown.**

---

## Scope and limits

- **Length 3 only.** Length-4 optimality is unverified; our one-sided check is near-blind to
  losses below ~2.5%.
- **`2/27` is not a failure rate**, and it is K = 15 specific. At K = 50 it is 0/27.
- Among the 9 K = 15 pairs whose optimum has a single construction path the rate is 2/9, and
  0/18 among the rest — **a correlation on n = 2 with no established mechanism.**
- **One corpus, one model, one vocabulary. We have not tested the vision datasets and make no
  claim about them.**
- The E.2.2 precondition holds on **every sentence at K >= 50** on our corpus, and at the full
  `min_support >= 5` vocabulary (K = 1168) `Bott^A_1(E^C) = 0`. **The violation we hit is
  produced by our own configuration, not by text.**
