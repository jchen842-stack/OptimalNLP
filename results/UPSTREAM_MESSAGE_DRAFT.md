# Three-paragraph message — to send by hand. NOT SENT.

Subject: A complete formula discarded unevaluated in `optimal.py`, plus an inadmissible aggregated bound

---

Hello — we've been applying `optimal-compositional-explanations` (@ `70805299`) to token-level
concepts on SNLI, and instrumenting the search turned up something we think is worth your
time, independent of any theory about why it happens. On `trained a=0.2 unit88`, the formula
`((dep=ROOT OR dep=nsubj) AND const=NP)` is flagged final (`next_op == "INDIVIDUAL"`), has an
exact IoU of `0.25454105110196174`, and exceeds the incumbent threshold of
`0.25056904400606983` by `0.003972007095891905` — and it is carried through an estimation path
and never evaluated. Its exact IoU was one mask operation away, and had it been computed the
formula would have become the new incumbent. We have the per-node event log for this.

Separately, we have two measured facts that we are deliberately **not** connecting. First, the
aggregated bound is inadmissible on our data: admissibility needs `Bott_1(E^C)_x = 0`, which
Section E.2.2 describes as rare and unobserved, whereas on a token-level corpus with
`min_support` + top-K selection we measure `Bott_1(E^C)_x = 859` on 100% of samples, and we
have a prefix assigned a ceiling of `0.232677` whose own subtree reaches `0.254541`. Second,
enumerating all 30,375 in-grammar length-3 formulas exhaustively across 27 (unit, alpha) pairs,
the search misses its own in-grammar optimum on 2 of them, by +0.93% and +4.74% IoU — your own
`beam_optimal.py` finds both, which is how we noticed. **We drafted this message once with the
first presented as the cause of the second, and our instrumentation does not support that**, so
we have removed the claim: the prefix in question was created four times and expanded three
times despite its bad ceiling, the optimal child was produced anyway, and that child then died
with no drop event and with a refined estimate that is admissible. We have ruled out
`reduce_frontier`, the incumbent skip at `:697`, and the refinement discard at `:699-709`;
`recent_nodes` dedup at `:749-757` and the distributive path remain unexamined. We are not
guessing further.

One thing worth flagging about the inadmissibility on its own: it is partition-invariant. We
reached the most extreme form of the E.2.2 violation ourselves, by putting the whole corpus in
a single sample, and that is our error — but repartitioning to one sample per sentence leaves
the ceilings byte-identical and still leaves 13 of 27 prefixes inadmissible, so correcting the
partition does not restore admissibility. We'd value your read on whether refusing to prune on
an unrefined aggregated estimate is worth its cost (it pays the ~|D|× sample-computation on
every node, which is what the aggregated path exists to avoid), and on the unevaluated-final-
formula observation in the first paragraph, which we think stands on its own. We have not
tested the vision datasets and make no claim about them. Happy to share code, the event logs,
or run anything useful.
