# Report to upstream — `optimal-compositional-explanations` @ `70805299`

SNLI token-level concepts, M = 24,199 tokens (2,000 sentences), `min_support = 5`, length 3,
27 (arm, alpha, unit) pairs. Code and raw output: `src/exp_exit_paths.py`,
`src/exp_final_discard.py`, `tests/test_bruteforce_oracle_all27.py`, `results/METHOD_NOTES.md`.

**One finding with two consequences.**

---

## The finding — `optimal.py:747` discards a final node without evaluating it

At `:712-747`, `apply_distributive_property` transforms a node's label, the transformed form is
re-estimated (`:718-719`, `node_after_distr`), and if `new_max < -e_node` and
`new_max < minimum_threshold` the node is **neither re-pushed nor scored** — it is dropped at
`:747`. **The estimate is computed on the transformed label; the node discarded is the
original.**

We traced every qualifying discard in a run — a formula flagged FINAL
(`next_op == "INDIVIDUAL"`), popped, whose evaluated mask is never scored anywhere in the run,
and whose exact IoU exceeds the incumbent at that moment. **All 45 exit at `:747`.** The other
four `continue` sites and the `recent_nodes` memory account for zero.

**These are FINAL nodes** (`next_op == "INDIVIDUAL"`) — nothing left to extend, so the node's
reachable maximum is just its own exact IoU, which is computable in one mask operation and is
never computed.

**We cannot tell you what the estimate was.** `new_max` is `0.0` on 41 of the 45, and `0.0` is
a sentinel, not a value: `path_heuristic.py:48-50` computes `max_iou` and then returns `0.0,
0.0` if it is below `minimum_threshold`, and `:172-174` returns `0.0, 0.0` when quantities were
never computed. On the 4 events where `new_max` carries a real number, **it is above the
discarded formula's exact IoU — i.e. sound.**

So the observation is narrower than "the bound is wrong": **the discard decision at `:747` is
taken on a zeroed flag, and the underlying value is not recoverable at that site.** Whether it
was a genuine below-threshold estimate or a never-computed default we do not know, and we did
not add a hook inside `path_heuristic` to find out.

## Consequence 1 — usually harmless

Something better is generally found by another route. 14 of the 16 affected pairs at K = 15
flat still return the true in-grammar optimum, as do all 17 affected pairs at K = 15
per-sentence and all 12 at K = 50.

## Consequence 2 — on 2 of 27 pairs, nothing better was found

```
trained a=0.2  unit88   discarded 0.25454105110196174   returned 0.2522022213711222   +0.9274%
trained a=0.05 unit86   discarded 0.21660649819494585   returned 0.20679723502304148  +4.7434%
```

Verified independently by enumerating all 30,375 in-grammar length-3 formulas in integer
arithmetic. The exhaustive oracle and the per-node event log — different hooks, different code
paths — agree on exactly these two pairs and these magnitudes.

**We are not claiming the discard caused the misses.** It is the only surviving candidate after
five failed attributions, the exit path is 45/45, and the estimate is measurably unsound on a
final node. The causal step is supported by the data and not compelled by it.

---

## Scope

- **The discard event occurs at every configuration measured**, including K = 50, where the
  E.2.2 `Bott_1(E^C)_x = 0` precondition holds on every sentence and no miss occurs
  (191 events, 12 of 27 pairs).
- **The harm is K = 15 flat specific.** At K = 50 the exhaustive oracle finds 0 misses in 27.
- **The event rate does not order configurations by harm** on any normalisation — raw count,
  per node popped, or per enumerated formula. The single harmful configuration is middling on
  all three, so we do not suggest "discard less" as a remedy.
- Counts are **lower bounds**: the metric counts popped nodes, and nodes can also vanish
  pre-pop at `estimate_iou_frontier:389`.
- **Length 3 only. One corpus, one model, one vocabulary. We have not tested the vision
  datasets and make no claim about them.**

## What we retracted

We twice proposed a cause and both were wrong. First, the disjoint fork in
`estimate_label_quantities` — falsified by forcing `are_disjoint` to `False`, which recovered
neither miss. Second, inadmissibility of the aggregated bound — retracted as a category error:
we had compared the **stop-here** (`INDIVIDUAL`) bound of one frontier entry against the best
value reachable by **extending** the node. `estimate_paths_iou` emits four entries per node and
only the stop-here entry was dropped, correctly, at its own `IoU(P)`.

Five candidate mechanisms were tested and failed before this one: the disjoint fork, bound
inadmissibility, the `reduce_frontier` threshold prune, the `:699-709` refinement discard, and
all three chain estimators (`:368`, `:509`, `:581`) — every extension path bounds the reachable
value from above.

## Method note

The trace used a logging-only build of `optimal.py` generated at run time and never written
into our tree, with `git diff --numstat` showing `5 0` (five `_EXIT_LOG` insertions, zero
deletions), and a determinism gate requiring the patched and unpatched runs to match on
returned IoU, pop counts **and** expanded counts across all 27 pairs.
