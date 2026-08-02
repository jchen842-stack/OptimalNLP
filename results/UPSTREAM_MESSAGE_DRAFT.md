# Three-paragraph message — to send by hand

Subject: Non-optimal returns from `optimal.py` at length 3 — inadmissible aggregated bound

---

Hello — we've been applying `optimal-compositional-explanations` (@ `70805299`) to
token-level concepts on SNLI, and we've found that the exact search returns non-optimal
formulas at the paper's own maximum length. Enumerating all 30,375 in-grammar length-3
formulas exhaustively across 27 (unit, alpha) pairs, `optimal.py` misses its own in-grammar
optimum on 2 of them, by +0.93% and +4.74% IoU. Your own `beam_optimal.py` finds both, which
is how we noticed. Details and a reproduction are in the attached report.

The cause appears to be that `reduce_frontier` prunes on the **unrefined aggregated** estimate
(Alg 1 lines 11, 52), and that estimate is inadmissible on our data. Admissibility needs
`Bott_1(E^C)_x = 0`; Section E.2.2 describes the violating case as rare and not observed in
your datasets, but on a token-level corpus with `min_support` + top-K concept selection it
holds on **100%** of samples — we measure `Bott_1(E^C)_x = 859`, with 83.8% of tokens covered
by more than one concept. We reached the most extreme form of this ourselves by putting the
whole corpus in a single sample, which is our error and we say so in the report. But the
estimate is **partition-invariant** — repartitioning to one sample per sentence leaves both
ceilings byte-identical and 13 of 27 prefixes still inadmissible — so correcting the partition
does not restore soundness.

We're reporting this as a bound-soundness issue rather than a bug, and we'd value your read on
two possible remedies: refusing to prune on an unrefined aggregated estimate (sound, but pays
the ~|D|× sample-computation cost on every node, which is what the aggregated path exists to
avoid), or asserting when the E.2.2 condition fails (cheap, but only detects). We have not
tested the vision datasets and make no claim about them. Happy to share code, raw output, or
run anything that would be useful.
