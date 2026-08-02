# Message — rewritten from the report, NOT patched. NOT SENT.

Subject: A final-flagged formula discarded unevaluated in `optimal.py`

---

Hello — we've been applying `optimal-compositional-explanations` (@ `70805299`) to token-level
concepts on SNLI and instrumenting the search. One observation seems worth your time on its
own. On one unit, the formula `((dep=ROOT OR dep=nsubj) AND const=NP)` is flagged final
(`next_op == "INDIVIDUAL"`), has an exact IoU of `0.25454105110196174`, and exceeds the
incumbent threshold of `0.25056904400606983` by `0.003972007095891905` — and it is popped,
carried through an estimation path, and never evaluated. Its exact IoU was one mask operation
away, and computing it would have made it the new incumbent. This is not a bound problem: we
logged all four path estimates for its parent pre-filter and every one bounds the true
reachable value from above. It also reproduces at K = 50, where 445,644 of 525,933 refinement
calls carried `next_op == "INDIVIDUAL"`.

Separately, at K = 15 we find the search misses its own in-grammar optimum on 2 of 27
(unit, alpha) pairs, by +0.93% and +4.74% IoU, verified by enumerating all 30,375 in-grammar
length-3 formulas — your `beam_optimal.py` finds both. **This does not persist at K = 50**: the
same exhaustive enumeration over 1,125,000 formulas per pair gives 0 misses out of 27. So it is
specific to our small-vocabulary configuration, and we cannot separate the two things K = 50
changes at once (a 37x larger space, and the E.2.2 precondition holding there but not at
K = 15). We should say plainly that we first wrote to you attributing these misses to an
inadmissible aggregated bound, and we have **retracted that**: the comparison behind it put a
stop-here bound against an extend-from-here maximum, and we have since tested five candidate
explanations — the disjoint fork, bound inadmissibility, `reduce_frontier`, the refinement
discard, and all three chain estimators — and all five failed. We do not know the cause.

We are not asking you to act on the second item; it may well be a property of our
configuration and it disappears at a realistic vocabulary size. The first one we do think
stands on its own, and we would value your read on whether a final-flagged formula reaching
the estimation path rather than the evaluation path is intended. We have not tested the vision
datasets and make no claim about them. Happy to share code, event logs, or run anything useful.
