# Report to upstream — `optimal-compositional-explanations` @ `70805299`

Measured on SNLI token-level concepts: K = 15 concepts, M = 24,199 tokens (2,000 sentences),
`min_support = 5`, formula length 3, 27 (arm, alpha, unit) pairs. Code and raw output:
`tests/test_bruteforce_oracle_all27.py`, `src/exp_event_ordering.py`,
`src/exp_refined_estimate.py`, `results/METHOD_NOTES.md`.

**Two claims are measured. A third — that the first causes the second — is NOT established,
and is marked as such throughout.**

---

## Lead observation — mechanism-independent

On `trained a=0.2 unit88`, the formula `((dep=ROOT OR dep=nsubj) AND const=NP)`:

```
flagged FINAL (next_op == "INDIVIDUAL")   yes
its exact IoU                             0.25454105110196174
incumbent threshold at that moment        0.25056904400606983
it exceeds the incumbent by               0.003972007095891905
```

**A complete formula, already flagged final, whose exact IoU exceeds the incumbent — was
carried through an estimation path and never evaluated.** Its exact IoU was one mask operation
away. Had it been evaluated it would have become the new incumbent.

This observation depends on **no** hypothesis about which prune fired. It is a direct
instrumentation record of a complete formula being estimated rather than computed.

---

## Claim (a) — the aggregated bound is inadmissible when `Bott_1(E^C)_x != 0`. MEASURED.

Admissibility of the aggregated form requires `Bott_1(E^C)_x = 0`. Section E.2.2 describes the
violating case as *"a rare degenerate case (not observed in any of the datasets tested in this
paper)"*. In **our configuration** we measure `Bott_1(E^C)_x = 859` on 100% of samples.

**We must be clear that this is our configuration and not a property of text.** Counting
`Bott_1` per sentence across vocabulary sizes on the same corpus:

```
K = 15   (ours)   83.5% of sentences compliant, max Bott_1 = 1
K = 50            100% compliant
K = 100           100% compliant
K = 1168 (full min_support >= 5 vocabulary)   100% compliant, Bott^A_1 = 0
```

**At the paper's own vocabulary scale the precondition holds on every sentence**, because most
concepts are absent from most sentences and supply the zero — 24 of 1,168 concepts have a
dataset-wide total of zero. Top-K selection on high-support features removes exactly those.
The violation we hit is produced by our `K = 15` and, dominantly, by our single-sample
partition.

A worked instance, on the prefix `P = (dep=ROOT OR dep=nsubj)`:

```
ceiling assigned to P     0.232677
best IoU reachable from P 0.254541     <- the bound is BELOW what its own subtree reaches
```

The bound is not an upper bound **given a violating input**. That the estimator is
inadmissible under a violated precondition is measured and stands. **We do not claim the
precondition is violated at realistic vocabulary scale — we measure that it is not.**

One limit worth stating: `Bott_1` is measurable at K = 1168 because it is counting, but the
consequence is not testable there — an exhaustive in-grammar oracle at K = 1168 is ~1.4e10
formulas per unit. **The precondition can be checked at paper scale; whether the search is
optimal there cannot**, and we do not extrapolate our 2/27 to it.

**It is also partition-invariant.** `SUM_x |I^C_max(L)_x|` counts over all `(x, j)`; `Top^A_t`
and `Bott^A_1` are concept-wise over dataset-wide totals. Repartitioning our data from one
sample to one-sample-per-sentence leaves both ceilings **byte-identical**, and 13 of 27
optimum-carrying prefixes remain inadmissible with one pair worse. **Correcting the partition
does not restore admissibility.**

## Claim (b) — the search returns non-optimal answers at length 3. MEASURED.

Enumerating all 30,375 in-grammar length-3 formulas in integer popcount arithmetic across all
27 pairs, the search misses its own in-grammar optimum on 2:

```
trained a=0.2  unit88   in-grammar 0.25454105110196174   returned 0.2522022213711222   +0.9274%
trained a=0.05 unit86   in-grammar 0.21660649819494585   returned 0.20679723502304148  +4.7434%
```

The other 25 tie to float64 equality. Upstream's own `beam_optimal.py` finds both, and at
width 200 is optimal on 27/27 where the exact search is optimal on 25/27.

## Claim (c) — that (a) causes (b). **NOT ESTABLISHED.**

We drafted this report once with (a) presented as the cause of (b). **Instrumentation does not
support that, and the causal wording has been removed.**

Per-node event logging on the losing pairs shows:

```
prefix P   CREATED 4x, DROPPED once, EXPANDED 3x   -- the inadmissible ceiling did not stop it
child      CREATED, POPPED, then no SCORED and no DROPPED event at all
```

**The child was produced regardless of the prefix-level prune**, and it died with no DROPPED
event and with an **admissible** refined estimate. Whatever removed it, it was not the
inadmissible aggregated ceiling acting on the prefix.

We are reporting (a) and (b) as two measured facts about the same code, without asserting the
link between them.

---

## Exclusions — three paths ruled out by instrumentation

For the child formula on both losing pairs:

```
reduce_frontier threshold prune   no DROPPED event under hooks covering all six
                                  reduce_frontier call sites (:408 :485 :743 :794 :828 :866)
the :697 incumbent skip           EXCLUDED   ceiling 0.4175506268081003 > threshold 0.25056904400606983
the :699-709 refinement discard   EXCLUDED   refined 0.4175506268081003 is above the threshold
                                             AND above the exact IoU it bounds -- admissible
```

**On the first line: absence of an event is evidence only under complete hook coverage, and our
coverage is not complete.** Two removal classes are unhooked — a silent non-append at
`estimate_iou_frontier:389` when a path estimate is `<= 0`, and six `continue` statements in
`perform_search` (`:679 :709 :747 :753 :804 :871`) after which a popped node is gone with no
event. The `:697` and `:699-709` exclusions are unaffected: they rest on positive
measurements, not on absence.

**Two paths remain un-excluded**, stated as remaining and not as suspects:

```
recent_nodes dedup      :749-757
the distributive path   :731-767
```

We have not instrumented these, and we make no claim about them.

---

## Remedies, offered against claim (a) only

**(a) Do not prune on an unrefined aggregated estimate.** Refine before `reduce_frontier`
acts, or restrict pruning to refined nodes.

**Its cost, stated with it:** per Section C, sample computation is on the order of `|D|x` more
arithmetic per estimate, and this pays it on every node rather than only popped ones —
precisely the expense the aggregated path exists to avoid. **The remedy converts a soundness
property into a runtime cost.** That is a real trade and we present it as one.

**(b) Assert or warn** at `get_optimal_heuristic_info` when `Bott_1(E^C)_x != 0` on any sample,
naming E.2.2. Cheap — already a by-product of the quantity helpers — but it only reports, and
per the partition-invariance result a compliant partition is not sufficient.

Neither remedy is claimed to fix the misses in (b), because the link is not established.

## A diagnostic users can run with no code change

`n_prefixes` — the number of distinct construction paths to a formula — is a function of the
operator signature alone, readable off a published formula by inspection:

> **If the returned formula's two operators differ, there is exactly one construction path to
> it and no redundancy against a lost node.**

Identical adjacent operators commute: an OR-run of length `r` gives `r+1` paths, an AND-NOT run
gives `r` (the leftmost term is never negated, so the positive base is pinned).

**Support:** 2 of 9 single-path units lost their optimum; **0 of 18** with a redundant path
did. The vulnerable share of signature space is **2/3 at every length**.

---

## Observation, explicitly not a claim — the `|D|x` cost model did not hold

```
matched median wall clock, length 4 :  |D| = 1  639.4s  ->  |D| = 2000  247.2s   (2.6x FASTER)
matched median nodes expanded       :  8,340 -> 8,553  (1.03x, unchanged)
```

One hypothesis, untested: the `|D|x` factor may fall on cheap per-sample scalars while the
expensive mask operations span the same 24,199 bits under either partition. If so, Section C's
model assumes many samples with small per-sample masks — a vision-shaped assumption alongside
the E.2.2 condition. We have not tested this.

Better established: peak frontier rose 1.19-1.39x under the corrected partition while distinct
nodes expanded stayed flat, because Algorithm 1 line-18 re-insertions rose **8.6x** (median 72
-> 622 per unit). The search is not larger; it re-queues the same nodes more often.

## Scope and limits

- Verified at **length 3 only**. Length-4 optimality is unverified: our one-sided check found 0
  definite misses among 23 (flat) and 26 (per-sentence) terminating pairs, but its median
  detection margin is ~2.5%, so it is near-blind to losses of the smaller observed magnitude
  (0.93%), and it excludes timeouts.
- **2/27 is not a failure rate.** Among the 9 pairs whose optimum has a single construction
  path it is 2/9; among the other 18 it is 0/18.
- One corpus, one model, one concept vocabulary. **We have not tested the vision datasets and
  make no claim about them.**
