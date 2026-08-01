# D6 — the measurement-defect class

**Record, not a conclusion.** The six instances and the shape they share are set out below;
the entry itself is yours to write.

Every result cited here is in `results/METHOD_NOTES.md` with its raw output committed.

## The shape

**In each case the reference stopped matching the measured quantity, and nothing in the
pipeline flagged it.**

Not one of the six is a coding error in the thing being measured. In every one, the *thing
compared against* silently drifted out of correspondence with the thing measured — different
units, different population, different depth, different sample axis, or a sentinel that is not
a number at all — and the comparison still returned a value, so the pipeline reported a
verdict rather than an error.

## The six instances

| # | instance | the reference | the measured quantity | how it surfaced |
|---|---|---|---|---|
| 1 | **D5.4 density artifact** | trained arm sampled the full [0.15, 0.85] density band | untrained arm did not | the trained/untrained IoU gap was read as a training effect |
| 2 | **Oracle at 2 units** | 3 cases (2 real units + proxy), all among the 25 that tie | 27 pairs, 2 of which miss | check 10 passed honestly on a sample that could not see the failure |
| 3 | **Expanded-count vs formula-space** | `K*(3K)^(L-1)` formula-space ratios (24.0x, 12.4x) | measured **wall-clock** ratios (83.2x, 36.6x) | the P1 "cancellation" finding, withdrawn |
| 4 | **`nan` sentinel** | `true - x > tol` | `x` was `nan` / `None` / no-solution | 12 no-label runs scored as successes; 10 comparison sites still unguarded |
| 5 | **Treatment-dependent median membership** | all-27 median | timed-out peaks are truncated, and membership moves with the treatment | corrected to a matched set before the L4 run (A1) |
| 6 | **Depth-3 reference under a depth-4 search** | `in_grammar_max` enumerated at hardcoded depth 3 | a `PART_LENGTH=4` search | `part=0.146338 > true=0.145326` in the run log |

### Detection power for instance 2, since it is the one with a number

The oracle drew 2 real units. Two of the 27 pairs miss. Probability of drawing at least one
bad pair:

```
1 - C(25,2)/C(27,2) = 1 - 300/351 = 0.1453
```

**14.5%.** The check was not weak by accident of which units were drawn; it had a one-in-seven
chance of ever firing.

## What separates these from ordinary bugs

Each one **produced a number**. None raised, none returned an obviously wrong magnitude, and
four of the six produced a number that was *favourable* to the hypothesis under test —
instances 3, 4, 5 and 6 would all have been reported as supporting results.

Three were caught only because a later, unrelated measurement disagreed with them:
2 by running upstream's own `beam_optimal`, 3 by learning what `expanded` actually counts,
6 by reading a run log while waiting.

## Not on this list

The 2/27 exact-search misses are **not** an instance. They are a real property of an
inadmissible estimator, measured correctly, and they survive every correction above. Their
attributed cause changed twice; their existence did not.
