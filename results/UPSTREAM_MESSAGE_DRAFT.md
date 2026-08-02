# Message — final rewrite from the report. NOT SENT.

Subject: `optimal.py:747` discards a final node on an estimate below its own IoU

---

Hello — we've been applying `optimal-compositional-explanations` (@ `70805299`) to token-level
concepts on SNLI, and instrumenting the search turned up something we think is worth your time.
At `optimal.py:712-747`, `apply_distributive_property` rewrites a node's label, the rewritten
form is re-estimated, and if that estimate falls below the incumbent threshold the node is
neither re-pushed nor scored — it is dropped at `:747`. The estimate is computed on the
transformed label; the node discarded is the original. We traced every qualifying discard in a
run and all 45 exit at that one site.

The part that concerns us is that these are FINAL nodes (`next_op == "INDIVIDUAL"`). A final
node has nothing left to extend, so its reachable maximum is just its own exact IoU — and on
41 of the 45 the estimate is below that value, several of them exactly `0.0` for formulas whose
actual IoU is 0.24–0.25. In one case the discarded formula had IoU `0.25454105110196174`
against an incumbent of `0.25056904400606983`, so scoring it would have made it the new best.
This holds whether or not the distributive transform preserves equivalence: if it does, the
estimate under-bounds its own formula; if it does not, an estimate for one formula was used to
discard another.

Usually this costs nothing — something better is found by another route, and 14 of the 16
affected pairs still return the true in-grammar optimum. On 2 of 27 pairs at our K = 15 setting
nothing better was found and the discarded formula beat the returned answer, by +0.93% and
+4.74% IoU, confirmed by exhaustive enumeration of all 30,375 in-grammar length-3 formulas. We
are **not** claiming the discard caused those two; it is the only surviving candidate after we
tested and eliminated five others, and we should say plainly that we twice proposed a cause and
were wrong both times, most recently by comparing a stop-here bound against an
extend-from-here maximum. The discard event itself occurs at every configuration we measured,
including K = 50 where no miss occurs at all, so the event and the harm are not the same thing.
Happy to share code, the event logs, or run anything useful — and we have not tested the vision
datasets, so we make no claim about them.
