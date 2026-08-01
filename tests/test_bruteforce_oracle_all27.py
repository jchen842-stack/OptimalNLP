"""Wide brute-force oracle: is the search optimal in its own grammar, on ALL 27 pairs?

`test_bruteforce_oracle.py` asserts `search == in-grammar max` on **three** cases — trained
unit88 a=0.1, untrained unit92 a=0.1, and the proxy neuron. All three pass. This file runs
the same assertion over all **27** (arm, alpha, unit) pairs of the Phase B length-3 grid,
and finds that two of them do NOT pass.

Found 2026-08-01 while running upstream's own `beam_optimal.py` (experiment 2b): the beam
returned a HIGHER IoU than "exact" on two pairs, which is impossible if exact is optimal.

**The search is not optimal within its own grammar.** The cause is upstream and is an
inadmissible bound, not a bug in this project's pipeline:

`optimal_sample_heuristic.can_improve_or_iou_disjoint_case` (lines 62-75) reasons that for
DISJOINT A and B, "any formula obtainable by (A OR B) is guaranteed to be <= the one
obtainable by only A or only B". That holds for the OR node itself. It is FALSE once the
node is extended by AND, because the AND removes different false positives from each branch:

    unit86  IoU(nsubj AND NN) = 0.203735   IoU(ROOT AND NN) = 0.056926
            IoU((nsubj OR ROOT) AND NN) = 0.216606          -- beats both
    unit88  IoU(nsubj AND NP) = 0.219736   IoU(ROOT AND NP) = 0.056530
            IoU((nsubj OR ROOT) AND NP) = 0.254541          -- beats both

`dep` is single-valued, so `dep=nsubj` and `dep=ROOT` are exactly disjoint (0 shared tokens)
and take that code path. The OR node is therefore given a ceiling BELOW what its own subtree
can reach, `reduce_frontier` drops it as soon as the incumbent passes that ceiling, and the
branch holding the optimum is destroyed:

    unit86  ceiling 0.203398 dropped at threshold 0.203735, subtree could reach 0.216606
    unit88  ceiling 0.232677 dropped at threshold 0.232934, subtree could reach 0.254541

This file is a REGRESSION TEST against that recorded behaviour, in the same spirit as the
change in commit ef2138c which made the oracle assert against the method's actual formula
grammar and report the gap rather than assert an idealisation. It passes when the set of
missed pairs is exactly the recorded set. A new miss, or a miss that disappears (e.g. after
an upstream fix), fails and must be looked at.

Usage::

    python tests/test_bruteforce_oracle_all27.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("OPTIMALCE_UPSTREAM",
                                  os.path.expanduser("~/projects/optimalce")))
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, _SRC)
_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

import real_token_masks as rtm            # noqa: E402
import real_token_search as rts           # noqa: E402
from exp_beam_width import PAIRS, K, LENGTH, MAX_SENTS, MIN_SUPPORT  # noqa: E402

FEATS = os.environ.get("FEATS", os.path.expanduser(
    "~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats"))

# The recorded set of pairs where the search misses its own in-grammar optimum, with the
# in-grammar max and the value the search returns. Pasted from the first run of this file.
RECORDED_MISSES = {
    ("trained", "0.2", 88): (0.25454105110196174, 0.2522022213711222),
    ("trained", "0.05", 86): (0.21660649819494585, 0.20679723502304148),
}


def grammar_max_all(dense, neurons):
    """Exhaustive max over the in-grammar length-3 space, for every neuron at once.

    Exactly `expand_node`'s three moves: Or(label, leaf), And(label, leaf),
    And(label, Not(leaf)). Enumerates masks, so commutativity collapses for free. Integer
    popcounts throughout -- a float32 dot product here is accurate to ~1e-3 of a count and
    that is enough to flip a tie between two formulas.
    """
    Kc = dense.shape[0]
    N = np.array(neurons)
    nsz = N.sum(1).astype(np.int64)
    best = np.zeros(len(neurons), dtype=np.float64)
    leaves = [dense[i] for i in range(Kc)]

    def moves(m):
        for j in range(Kc):
            yield m | leaves[j]
            yield m & leaves[j]
            yield m & ~leaves[j]

    n_forms = 0
    for i in range(Kc):
        for m2 in moves(leaves[i]):
            for m3 in moves(m2):
                n_forms += 1
                inter = (N & m3).sum(1).astype(np.int64)
                size = np.int64(m3.sum())
                best = np.maximum(best, inter / np.maximum(size + nsz - inter, 1))
    return best, n_forms


def main():
    toks = rtm.load_tokens(FEATS, MAX_SENTS)
    concepts = rtm.select_concepts(toks, rtm.CATEGORIES, K, MIN_SUPPORT)
    dense = rtm.build_dense(toks, concepts)
    print(f"K={len(concepts)} M={dense.shape[1]} length={LENGTH}  "
          f"({sum(len(u) for _, _, u in PAIRS)} pairs)")

    keys, neurons = [], []
    for arm, alpha, units in PAIRS:
        path = os.path.join(_REPO, "results", f"acts2k_{arm}_a{alpha}.npz")
        if not os.path.exists(path):
            print(f"CANNOT VERIFY: {path} missing (gitignored; see REPRODUCE.md step 3)")
            return 0
        z = np.load(path)
        for u in units:
            keys.append((arm, alpha, u))
            neurons.append(z["acts"][u].astype(bool))

    print("enumerating the in-grammar space ...", flush=True)
    gram, n_forms = grammar_max_all(dense, neurons)
    print(f"  {n_forms:,} in-grammar formulas x {len(keys)} neurons")

    print("running the search on each pair ...", flush=True)
    misses = {}
    for (arm, alpha, u), neuron, g in zip(keys, neurons, gram):
        res = rts.run_one(dense, neuron, LENGTH, cap=200000, time_budget=1500,
                          beam_cap=None, concepts=concepts)
        # run_one rounds to 4dp for the CSV; recover full precision from the kept counts.
        exact = res["n_inter"] / (res["n_fires"] + int(neuron.sum()) - res["n_inter"])
        if g - exact > 1e-12:
            misses[(arm, alpha, u)] = (g, exact)
            print(f"  MISS  {arm} a={alpha} unit{u}: in-grammar {g!r} > search {exact!r} "
                  f"({100 * (g / exact - 1):+.4f}%)  search returned {res['formula']}")

    print(f"\n  pairs where the search missed its own in-grammar optimum: "
          f"{len(misses)}/{len(keys)}")
    ok = set(misses) == set(RECORDED_MISSES)
    if ok:
        for k, (g, e) in sorted(misses.items(), key=str):
            rg, re_ = RECORDED_MISSES[k]
            if abs(g - rg) > 1e-12 or abs(e - re_) > 1e-12:
                ok = False
                print(f"  VALUE DRIFT at {k}: recorded {(rg, re_)}, got {(g, e)}")
    if not ok:
        print(f"  recorded: {sorted(RECORDED_MISSES, key=str)}")
        print(f"  observed: {sorted(misses, key=str)}")
        print("  The set of misses changed. This is upstream behaviour, not a pipeline bug "
              "-- see this file's docstring. Investigate before adjusting anything.")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
