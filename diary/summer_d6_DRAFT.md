# D6 — DRAFT. Numbers and structure; the entry is yours to write.

Two sections marked **[PENDING]** are gaps awaiting runs, named so they are not forgotten.

## Opening — the suite's correctness coverage was one check, and it was defective

`verify/run_all.sh` runs 10 checks. **Nine examine inputs or plumbing.** Exactly one examines
a **result**: the brute-force oracle. That one sampled 3 cases out of 27 and had a **14.5%**
chance of firing (`1 - C(25,2)/C(27,2)`).

**"11/11, no CANNOT VERIFY" was true, and near-uninformative about correctness.** It said the
right data went in. It did not say the right answer came out, and could not have.

## 1. Soundness — `optimal.py` returns non-optimal answers at the published length

Exhaustive enumeration of all 30,375 in-grammar length-3 formulas (K=15, M=24,199,
min_support=5), integer popcount arithmetic, all 27 (arm, alpha, unit) pairs:

```
trained a=0.2  unit88   in-grammar 0.25454105110196174   search 0.2522022213711222   +0.9274%
trained a=0.05 unit86   in-grammar 0.21660649819494585   search 0.20679723502304148  +4.7434%
```

The other 25 tie to float64 equality. Both missed optima are `((dep=ROOT OR dep=nsubj) AND C)`
— squarely in grammar — and upstream's own `beam_optimal.py` finds them while "exact" does not.

**Cause.** The aggregated ceiling is inadmissible. Traced on both pairs:

```
unit88  ceiling 0.232677 < true_max(prefix) 0.254541   dropped at threshold 0.232934
unit86  ceiling 0.203398 < true_max(prefix) 0.216606   dropped at threshold 0.203735
```

Removed by `reduce_frontier` both times, with the node still carrying the **aggregated**
(`"sum"`) estimate — pruned **before** refinement (Alg 1 lines 11, 52). Admissibility requires
`Bott_1(E^C)_x = 0`; measured here it is **859, on 100% of samples**, against E.2.2's claim
that the violating case is rare and unobserved.

## 2. Partition-invariance — repartitioning cannot repair it

The aggregated estimate is invariant under repartition: `SUM_x |I^C_max(L)_x|` counts over all
`(x,j)`; `Top^A_t` and `Bott^A_1` are concept-wise over dataset-wide totals. **Confirmed
empirically — both ceilings are byte-identical across two sample axes.**

Our harness used `bitmaps = reshape(1, M)`, one sample holding all 24,199 tokens, which is
E.2.2's degenerate case **by construction**. Correcting it to one sample per sentence:

```
Bott_1(E^C)_x == 0 : 0% of samples -> median 69.7% of sentences
inadmissible ceilings (set-based): 14/27 -> 13/27
losses: 2/27 -> 0/27
```

But **the misses did not recover because the bound improved.** Both ceilings are unchanged and
both prefixes are still dropped. Recovery is a **pop-ordering effect** — the prefix is expanded
before a stale copy is pruned. Two independent reasons not to read it as a fix: the ceilings
are identical, and 0-of-9-vulnerable against a 2/9 base rate has **p ~ 0.10**.

**`reshape(1, M)` amplified exposure; it did not cause it.**

## 3. The measurement-defect class, and the checklist

Eleven instances (`summer_d6.md`). **Headline: four of them produced results *favourable* to
the hypothesis under test, and every one was caught by an unrelated later measurement
disagreeing — never by a check designed to catch it.**

> **The pipeline had no mechanism for detecting a defect that agreed with us.**

Deliverable: six comparison preconditions — **Quantities, Membership, Reference,
Discrimination, Power, Sentinels** — carried by every verdict-producing comparison, with
Discrimination extended to decisions (a claim that an effect is large or small enough requires
a computed magnitude).

Retro-validation **7/7**, recorded as weak evidence: the checklist was derived from those seven
instances, so the pass is near-tautological. **Instances 8-11 are prospective** — found by
applying the fields to registrations that had already passed review, including P8's sentinel
gap, P3's missing power check, and P9's circular subject.

## 4. The `n_prefixes` mechanism

`n_prefixes` = the number of leaves that could have been added **last**, which requires
identical adjacent operators. AND-NOT gives run-length rather than run-length+1 because the
grammar never negates the leftmost term.

```
losses at n_prefixes = 1 : 2 of 2      vulnerable population : 9 of 27
rate among vulnerable    : 2/9 = 22%   (not 2/27 = 7%)
vulnerable share of signature space : 6/9 = 18/27 = 2/3, CONSTANT with length
```

All four pairs with every optimal prefix pruned had it pruned; two survived. **`n_prefixes` is
not protection from pruning — it is the number of independent chances to be expanded before a
prune fires. Redundancy and ordering are the same mechanism counted twice.**

One design choice, `And(label, Not(leaf))`, produces both the expressiveness gap (+0.1586% at
length 4) and the reduced redundancy.

## 5. Upstream: diagnostic and two remedies

**Diagnostic, inspection-only, no code change:** `n_prefixes` is a function of the operator
signature alone. **If the returned formula's two operators differ, there is one construction
path and no redundancy.** Support: 2 of 9 single-prefix units lost their optimum; **0 of 18**
with redundancy did.

**Remedy (a): do not prune on an unrefined aggregated estimate.** Removes the order-dependence.
**Cost, stated with it:** sample computation is ~|D|x more arithmetic per estimate (Section C),
paid on every node — exactly what the aggregated path exists to avoid. **It converts a
soundness defect into a runtime cost.**

**Remedy (b): assert when `Bott_1(E^C)_x != 0`.** Cheap, already a by-product of the quantity
helpers — but it only reports, and per section 2 a compliant partition is not sufficient.

## 6. Item 4 — the harm measurement **[PENDING]**

Three exposure measures exist (14/27 set-based, 4/27 all-prefixes-pruned, 31 leaf-ancestor
prunes across 24/27 pairs). **All count opportunities; all are nearly uncorrelated with
outcome** — 2 actual losses flat, 0 per-sentence. The event-ordering metric (was the prefix
expanded before any copy was dropped) is the section's **missing dependent variable**.

## 7. P9 — length-scaling of vulnerability **[PENDING]**

Returned-formula form, baseline **57.7%** (15 of 26 returned L3 formulas are mixed). Rising =
the bug concentrates with length; falling = longer formulas buy redundancy. The **true-optima**
form requires the length-4 oracle and is **D7**.

## Deferred to D7

Length-4 oracle; fork-only rerun at `optimal_utils.py:271`; queue items D/E/F; queue items 3
(50 units/arm), 4 (fixed work), 5 (alpha=0.005, beam-only). **None changes what D6 concludes.**
