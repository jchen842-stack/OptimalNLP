"""Adjudicate experiment C against a decision rule fixed BEFORE the numbers were seen.

Reads `results/noprune_L3.csv` only. No searching, no re-running.

## The rule, pre-committed

Applied in this order; the first that fires decides.

  1. `noprune_expanded > 31,065` on ANY pair
        -> CONTAMINATED. The no-prune build altered the search rather than only relaxing it.
           C is VOID **for correctness as well as timing** -- a build that generates nodes
           the grammar cannot contain is not measuring the grammar's optimum.
  2. `noprune_expanded < published_expanded` on ANY pair
        -> BROKEN. Removing prunes cannot cut work. The build dropped something the
           published run kept. C VOID.
  3. `noprune_expanded == published_expanded` on EITHER known-miss pair
        -> the build did not take effect on the pairs it exists to test. C VOID FOR THOSE
           PAIRS, regardless of the IoU returned. A matching IoU there would be luck, not
           evidence.
  4. otherwise
        -> C is VALID FOR CORRECTNESS ONLY.

## The ceiling test is ONE-SIDED, and this is recorded rather than glossed

31,065 = 15 + 15*45 + 15*45^2 is the total node count of the in-grammar length-3 space at
K=15: every length-1, length-2 and length-3 formula the three moves of `expand_node` can
construct.

Exceeding it proves contamination. **Staying under it proves nothing.** Two mechanisms
legitimately reduce `expanded` and are deliberately left enabled in the no-prune build so
that a failure localises:

  * the `recent_nodes` dedup in `perform_search` -- skips re-expanding a node already
    expanded at the same estimate;
  * `apply_distributive_property` -- rewrites a node and re-inserts it rather than expanding.

So a count comfortably below 31,065 is equally consistent with "pruning is off and dedup is
working" and with "a prune is still live". **"Below ceiling" does not certify that pruning is
off.** Only conditions 2 and 3 -- strict containment of the published run, and a strict
change on the pairs under test -- provide that, and they are what the verdict rests on.

## `estimated` is NOT captured, on purpose

Upstream returns `(best_label, best_iou, visited, expanded, estimated)`. The C harness
discards `estimated`. It counts heuristic estimate calls, which informs **per-node cost** --
and C's runtime is explicitly not being reported as a measurement of anything, because the
no-prune build pays full enumeration cost by construction. Capturing it would only refine a
number that is not being quoted. Marked as not captured rather than silently omitted.

Usage::

    python src/exp_noprune_adjudicate.py
"""

import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results", "noprune_L3.csv")

K, LENGTH = 15, 3
CEILING = K + K * (3 * K) + K * (3 * K) ** 2          # 15 + 675 + 30375 = 31,065

KNOWN_MISSES = {("trained", "0.2", "unit88"), ("trained", "0.05", "unit86")}


def main():
    if not os.path.exists(CSV):
        print(f"{CSV} not found -- experiment C has not finished writing yet.")
        return 2
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        for k in ("published_expanded", "noprune_expanded", "published_visited",
                  "noprune_visited", "published_peak", "noprune_peak"):
            r[k] = int(float(r[k]))
        for k in ("published_time_s", "noprune_time_s"):
            r[k] = float(r[k])
        for k in ("in_grammar_max", "published_iou", "noprune_iou"):
            r[k] = float(r[k])
        r["key"] = (r["arm"], r["alpha"], r["unit"])

    print("=" * 104)
    print(f"EXPERIMENT C — ADJUDICATION (K={K}, length={LENGTH}, M=24,199, min_support=5, "
          f"{len(rows)} pairs)")
    print(f"in-grammar node ceiling = {K} + {K}*{3*K} + {K}*{3*K}^2 = {CEILING:,}")
    print("=" * 104)

    print(f"\n{'pair':>30} {'pub_exp':>9} {'npr_exp':>9} {'pub_vis':>9} {'npr_vis':>9} "
          f"{'pub_peak':>9} {'npr_peak':>9} {'pub_t':>8} {'npr_t':>9}")
    for r in sorted(rows, key=lambda r: (r["arm"], -float(r["alpha"]), r["unit"])):
        mark = "  <-- known miss" if r["key"] in KNOWN_MISSES else ""
        over = "  !! OVER CEILING" if r["noprune_expanded"] > CEILING else ""
        print(f"{r['arm'] + ' a=' + r['alpha'] + ' ' + r['unit']:>30} "
              f"{r['published_expanded']:>9,} {r['noprune_expanded']:>9,} "
              f"{r['published_visited']:>9,} {r['noprune_visited']:>9,} "
              f"{r['published_peak']:>9,} {r['noprune_peak']:>9,} "
              f"{r['published_time_s']:>8.2f} {r['noprune_time_s']:>9.2f}{mark}{over}")

    print("\n  estimated: NOT CAPTURED (informs per-node cost only; C's runtime is not "
          "reported as a measurement)")

    # ---- the pre-committed rule, in order --------------------------------------------
    print("\n--- DECISION RULE (pre-committed, applied in order) ---")
    over = [r for r in rows if r["noprune_expanded"] > CEILING]
    cut = [r for r in rows if r["noprune_expanded"] < r["published_expanded"]]
    inert = [r for r in rows
             if r["key"] in KNOWN_MISSES
             and r["noprune_expanded"] == r["published_expanded"]]

    def line(n, ok, txt):
        print(f"  {n}. {'TRIGGERED' if not ok else 'not triggered'}: {txt}")

    line(1, not over,
         f"noprune_expanded > {CEILING:,} on {len(over)} pairs"
         + ("" if not over else
            " -> " + ", ".join(f"{r['unit']}({r['noprune_expanded']:,})" for r in over[:5])))
    line(2, not cut,
         f"noprune_expanded < published_expanded on {len(cut)} pairs"
         + ("" if not cut else
            " -> " + ", ".join(f"{r['unit']}({r['noprune_expanded']:,}<"
                               f"{r['published_expanded']:,})" for r in cut[:5])))
    line(3, not inert,
         f"noprune_expanded == published_expanded on {len(inert)} of the 2 known-miss pairs"
         + ("" if not inert else
            " -> " + ", ".join(f"{r['arm']} a={r['alpha']} {r['unit']}" for r in inert)))

    print("\n--- VERDICT ---")
    if over:
        print("  CONTAMINATED — C is VOID for correctness AND timing.")
        print("  The build generated nodes the in-grammar space cannot contain, so it was")
        print("  not searching the space the oracle enumerated. Stop and report.")
        verdict = "CONTAMINATED"
    elif cut:
        print("  BROKEN — C is VOID. Removing prunes cannot reduce work; the build dropped")
        print("  something the published run kept.")
        verdict = "BROKEN"
    elif inert:
        print("  VOID ON THE PAIRS UNDER TEST — the build did not change the search on "
              f"{len(inert)} of the 2 known-miss pairs.")
        print("  Their returned IoU carries no evidence either way. Remaining pairs stand.")
        verdict = "VOID_ON_MISS_PAIRS"
    else:
        print("  VALID FOR CORRECTNESS ONLY.")
        verdict = "VALID_CORRECTNESS_ONLY"

    if not over and not cut:
        still = [r for r in rows if r["in_grammar_max"] - r["noprune_iou"] > 1e-12]
        usable = [r for r in rows if r["key"] not in {x["key"] for x in inert}]
        still_usable = [r for r in still if r["key"] in {x["key"] for x in usable}]
        print(f"\n  correctness, on the {len(usable)} pairs C is valid for: "
              f"{len(usable) - len(still_usable)}/{len(usable)} match the brute-force oracle")
        if still_usable:
            print("  STILL MISSING with pruning off:")
            for r in still_usable:
                print(f"    {r['arm']} a={r['alpha']} {r['unit']}: "
                      f"true={r['in_grammar_max']!r} noprune={r['noprune_iou']!r} "
                      f"(+{100 * (r['in_grammar_max'] / r['noprune_iou'] - 1):.4f}%)")
            print("\n  *** The fault is NOT the ceiling estimate. With pruning off the")
            print("      optimum is still unreachable, so it is in expansion, dedup, or the")
            print("      grammar walk. LARGER FINDING — report before anything else.")
        else:
            print("  All valid pairs reach the optimum: consistent with the fault being")
            print("  entirely in the ceiling estimate. Runtime still not a measurement.")

    print("\n--- ONE-SIDEDNESS, recorded ---")
    print(f"  Exceeding {CEILING:,} proves contamination. Staying under it proves NOTHING:")
    print("  recent_nodes dedup and apply_distributive_property legitimately reduce")
    print("  expanded and are deliberately left enabled. 'Below ceiling' does not certify")
    print("  that pruning is off — conditions 2 and 3 do, and the verdict rests on them.")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
