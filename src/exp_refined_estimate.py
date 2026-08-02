"""Closes item 4 — what killed the winning child, measured rather than inferred.

The event log established: the optimal formula on trained a=0.2 unit88 was POPPED carrying an
aggregated ceiling of 0.4175506268081003 and then discarded without being SCORED. Not by
`reduce_frontier`, and not by the `:697` threshold skip. The only remaining path is
`optimal.py:699-709`, where the node is re-estimated with the SAMPLE heuristic and, if
`new_max < -e_node` and `new_max < minimum_threshold`, is neither re-pushed nor scored.

The ordering at that moment, verified:

    ceiling    0.4175506268081003
    IoU        0.25454105110196174     <- its OWN exact IoU
    threshold  0.25056904400606983
    ceiling > IoU > threshold

The IoU exceeds the incumbent threshold by 0.00397201. **Had the formula been scored it would
have become the new incumbent.** It was discarded while strictly better than the incumbent it
was being compared against.

PRE-REGISTERED, before `path_heuristic.update_paths_iou` was hooked:

  W1  refined estimate for the winning child < 0.25056904400606983, while its exact IoU is
      0.25454105110196174
      -> THE REFINED SAMPLE ESTIMATE IS INADMISSIBLE ON A COMPLETE FORMULA. An upper bound
         below the value it bounds.

  W2  refined estimate >= 0.25056904400606983
      -> it died by another path. Mechanism UNIDENTIFIED, nothing is written to
         UPSTREAM_REPORT.md, and the mechanism goes to D7 unidentified.

CAPTURED PER EVENT: refined estimate, minimum_threshold, exact IoU, and whether the node was
flagged FINAL (`next_op == "INDIVIDUAL"`) at the time of refinement.

THE FINAL FLAG MATTERS ON ITS OWN, and gets a different sentence either way:
  * flagged final and still estimated -> a COMPLETE formula was bounded instead of evaluated.
    That is a CONTROL-FLOW defect: its exact IoU was computable and was not computed.
  * not flagged final -> the node was still a partial path, and an inadmissible bound on it is
    a BOUND defect.

Usage:  python src/exp_refined_estimate.py
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.environ.get(
    "OPTIMALCE_UPSTREAM", os.path.expanduser("~/projects/optimalce")))
import real_token_masks as rtm, real_token_search as rts
from compositional import optimal
from exp_beam_width import K, LENGTH, MAX_SENTS, MIN_SUPPORT, CAP

FEATS = os.path.expanduser("~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats")
CASES = [("trained", "0.2", 88, 0.25454105110196174, 0.25056904400606983),
         ("trained", "0.05", 86, 0.21660649819494585, 0.2037351443123939)]


def main():
    tok = rtm.load_tokens(FEATS, MAX_SENTS)
    _, cats = rts.ARMS["all"]
    con = rtm.select_concepts(tok, cats, K, MIN_SUPPORT)
    dense = rtm.build_dense(tok, con)
    print(__doc__[__doc__.index("PRE-REGISTERED"):__doc__.index("Usage:")])

    for arm, alpha, unit, true_iou, known_thr in CASES:
        nb = np.load(os.path.join(REPO, "results", f"acts2k_{arm}_a{alpha}.npz"))["acts"][unit].astype(bool)
        target = None
        Kc = dense.shape[0]; lm = [dense[i] for i in range(Kc)]
        nn = int(nb.sum())

        def iou(m): return int((m & nb).sum()) / max(int((m | nb).sum()), 1)

        def mv(m):
            for j in range(Kc):
                yield m | lm[j]
                yield m & lm[j]
                yield m & ~lm[j]
        best = 0.0
        for i in range(Kc):
            best = max(best, iou(lm[i]))
            for m2 in mv(lm[i]):
                best = max(best, iou(m2))
                for m3 in mv(m2): best = max(best, iou(m3))
        tset = set()
        for i in range(Kc):
            for m2 in mv(lm[i]):
                for m3 in mv(m2):
                    if abs(iou(m3) - best) < 1e-12: tset.add(m3.tobytes())

        hits, cache = [], {}
        PH = optimal.path_heuristic
        ORIG = PH.update_paths_iou

        def hook(*a, **kw):
            out = ORIG(*a, **kw)
            node = kw.get("node") or (a[1] if len(a) > 1 else None)
            if node is not None:
                f = node[2]
                kk = rts.render(f, con)
                if kk not in cache:
                    cache[kk] = (rts.eval_formula(f, dense).tobytes() in tset) if len(f) <= 3 else False
                if cache[kk]:
                    mx = out[0]
                    mx = max(x[0] for x in mx) if isinstance(mx, (list, tuple)) and mx and isinstance(mx[0], (list, tuple)) else mx
                    hits.append(dict(label=kk, next_op=node[1], refined=mx,
                                     thr=kw.get("minimum_threshold"),
                                     heur=kw.get("heuristic_name")))
            return out

        PH.update_paths_iou = hook
        try:
            r = rts.run_one(dense, nb, LENGTH, CAP, 1500.0, None, concepts=con)
        finally:
            PH.update_paths_iou = ORIG

        print(f"\n=== {arm} a={alpha} unit{unit} ===")
        print(f"  exact IoU of the optimum : {true_iou!r}")
        print(f"  search returned          : {r['formula']}  IoU={r['best_iou']}")
        print(f"  refinement events on the optimal formula: {len(hits)}")
        for h in hits:
            fin = (h["next_op"] == "INDIVIDUAL")
            print(f"    refined={h['refined']!r}  threshold={h['thr']!r}  "
                  f"next_op={h['next_op']}  FINAL={fin}  heuristic={h['heur']}")
            if isinstance(h["refined"], float) and h["thr"] is not None:
                below = h["refined"] < h["thr"]
                print(f"      refined < threshold ? {below}   "
                      f"refined < exact IoU ? {h['refined'] < true_iou}")
                print(f"      -> {'W1' if below else 'W2'}")
            if fin:
                print("      NOTE: node flagged FINAL and still ESTIMATED -- a complete "
                      "formula bounded instead of evaluated (CONTROL-FLOW defect).")
        if not hits:
            print("    NO refinement event recorded on the optimal formula.")
            print("    -> W2 by default: it did not die at optimal.py:699-709.")
            print("       Mechanism UNIDENTIFIED. Nothing written to UPSTREAM_REPORT.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
