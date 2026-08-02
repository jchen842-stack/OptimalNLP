# Report to upstream — `optimal-compositional-explanations` @ `70805299`

SNLI token-level concepts, M = 24,199 tokens (2,000 sentences), `min_support = 5`, length 3,
27 (arm, alpha, unit) pairs. Code and raw output: `src/exp_exit_paths.py`,
`src/exp_final_discard.py`, `tests/test_bruteforce_oracle_all27.py`, `results/METHOD_NOTES.md`.

**One finding with two consequences.**

---

## The finding — `optimal.py:747` discards a final node on an under-estimate of its own value

At `:712-747`, `apply_distributive_property` transforms a node's label, the transformed form is
re-estimated (`:718-719`, `node_after_distr`), and if `new_max < -e_node` and
`new_max < minimum_threshold` the node is **neither re-pushed nor scored** — it is dropped at
`:747`. **The estimate is computed on the transformed label; the node discarded is the
original.**

We traced every qualifying discard in a run — a formula flagged FINAL
(`next_op == "INDIVIDUAL"`), popped, whose evaluated mask is never scored anywhere in the run,
and whose exact IoU exceeds the incumbent at that moment. **All 45 exit at `:747`.** The other
four `continue` sites and the `recent_nodes` memory account for zero.

**A FINAL node has nothing left to extend, so its reachable maximum is its exact IoU. On 41 of
45 events the estimate is below that value:**

```
      new_max    minimum_threshold    exact IoU of the discarded formula
          0.0   0.23293365307753797         0.240215366
          0.0   0.25056904400606983         0.254541051
          0.0   0.12466358057917967         0.125174300
   0.16014669...  0.1453263274336283         0.129822531
   ...
new_max <  exact IoU :  41 of 45
new_max >= exact IoU :   4 of 45   (sound discards)
```

**Several estimates are exactly `0.0` for formulas whose actual IoU is 0.24-0.25.**

This holds **whether or not the distributive transform preserves equivalence.** If it does,
both forms share that IoU and the estimate under-bounds its own formula. If it does not, an
estimate for one formula was used to discard another.

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
