"""R1/R2 — the MEASURING version of the unevaluated-final-formula event, at K=15 and K=50.

WHAT WAS WRONG. The event is: a formula flagged FINAL, POPPED, never SCORED, whose exact IoU
exceeded the incumbent at that moment. The test previously used was "refinement calls carrying
next_op == INDIVIDUAL" (84.7% at K=50). That is normal behaviour -- every node is estimated on
all four paths and scored on pop -- and it contains neither "never scored" nor "exceeded the
incumbent". It measured the wrong quantity, and it had already been elevated to "the only
finding that survives at paper-adjacent settings" before being checked.

MISSING CONDITION, added 2026-08-02. The first version keyed "never scored" by RENDERED LABEL.
optimal.py:762 `if label_node not in visited:` skips a node whose label equals an
already-visited one (`visited` at :652, appended :782/:816, compared by commutativity-aware
formula __eq__). That skip is CORRECT behaviour. "This copy never scored" and "this formula's
value was never computed" are different quantities; the metric measured the first, the claim
needs the second. Verified against source before acting on it.

The filter is at MASK level, which is broader than the code's own label-equality dedup: IoU
depends only on the mask, so if any formula with the same mask was scored anywhere in the run,
that value WAS computed.

PRE-REGISTERED:
  S1  filtered count >> 2  -> section 1 is an independent observation
  S2  filtered count == the 2 miss pairs only -> section 1 is the 2/27 seen from inside the
      search, not a second finding. The report has ONE finding.
  S3  filtered count == 0  -> the metric was measuring duplicates throughout

  R1/R2 (configuration-freeness) are judged on the FILTERED count at both K, never on a
  filtered-vs-unfiltered comparison -- that would be the Reference field failing.

MEASURED PER FORMULA: flagged FINAL (next_op == "INDIVIDUAL") AND popped AND never scored AND
exact IoU > incumbent threshold at the moment of the pop.

UNCOVERED-REMOVAL GATE. A zero is uninterpretable if formulas can vanish before ever being
popped. The silent non-append at estimate_iou_frontier:389 is hooked here by diffing the input
frontier against the returned estimates. The six `continue` sites are NOT individually hooked;
they act AFTER a pop, so they cannot hide a node from the pop counter -- they are why a popped
node ends up unscored, which is the thing being counted, not a way to miss one.

Usage:  python src/exp_final_discard.py 15 | 50
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.environ.get(
    "OPTIMALCE_UPSTREAM", os.path.expanduser("~/projects/optimalce")))
import real_token_masks as rtm, real_token_search as rts, synthetic_overlap_sweep as sos
from compositional import optimal
from utils import mask_utils as MU
import exp_partition as EP
from exp_beam_width import PAIRS, MAX_SENTS, MIN_SUPPORT, CAP

FEATS = os.path.expanduser("~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats")


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    print(__doc__)
    print(f"### K = {K} ###\n")
    tok = rtm.load_tokens(FEATS, MAX_SENTS)
    _, cats = rts.ARMS["all"]
    con = rtm.select_concepts(tok, cats, K, MIN_SUPPORT)
    dense = rtm.build_dense(tok, con)
    sid, pos, N_, L_, _ = EP.build_partition(rtm, tok)
    dense2 = np.stack([EP.to_2d(dense[c], sid, pos, N_, L_) for c in range(dense.shape[0])])
    keys, neur = [], []
    for arm, alpha, units in PAIRS:
        z = np.load(os.path.join(REPO, "results", f"acts2k_{arm}_a{alpha}.npz"))
        for u in units: keys.append((arm, alpha, f"unit{u}")); neur.append(z["acts"][u].astype(bool))

    OPOP, ORED, OGFM = sos.HeapProbe.heappop, optimal.reduce_frontier, MU.get_formula_mask_and_tree
    OEST = optimal.estimate_iou_frontier
    total_hits, per_pair, nonappend = [], [], [0]

    for k, nb in zip(keys, neur):
        nn = int(nb.sum()); thr = [0.0]; popped_final = {}; scored = set(); cache = {}
        scored_masks = set()
        def key(f):
            kk = rts.render(f, con)
            if kk not in cache: cache[kk] = kk
            return kk
        def exact(f):
            m = rts.eval_formula(f, dense)
            return int((m & nb).sum()) / max(int((m | nb).sum()), 1)
        def red(fr, t):
            thr[0] = max(thr[0], t); return ORED(fr, t)
        def pop(self, h):
            it = OPOP(self, h)
            if it[1] == "INDIVIDUAL":
                kk = key(it[2])
                if kk not in popped_final:
                    m = rts.eval_formula(it[2], dense)
                    popped_final[kk] = (int((m & nb).sum()) / max(int((m | nb).sum()), 1),
                                        thr[0], m.tobytes())
            return it
        def gfm(f, masks, path_masks=None, device=None):
            out = OGFM(f, masks, path_masks, device) if device is not None else OGFM(f, masks, path_masks)
            for lab in out:
                scored.add(key(lab))
                try: scored_masks.add(rts.eval_formula(lab, dense).tobytes())
                except Exception: pass
            return out
        def est(**kw):
            out = OEST(**kw)
            nonappend[0] += max(0, len(kw.get("frontier", [])) * 4 - len(out[0]))
            return out
        sos.HeapProbe.heappop = pop; optimal.reduce_frontier = red
        MU.get_formula_mask_and_tree = gfm; optimal.estimate_iou_frontier = est
        try:
            EP.NEURON_FLAT[0] = nb
            r = EP.run_partitioned(dense2, EP.to_2d(nb, sid, pos, N_, L_), con, dense)
        finally:
            sos.HeapProbe.heappop = OPOP; optimal.reduce_frontier = ORED
            MU.get_formula_mask_and_tree = OGFM; optimal.estimate_iou_frontier = OEST
        unfil = [(lab, v[0], v[1]) for lab, v in popped_final.items()
                 if lab not in scored and v[0] > v[1] + 1e-12]
        hits = [(lab, v[0], v[1]) for lab, v in popped_final.items()
                if lab not in scored and v[0] > v[1] + 1e-12 and v[2] not in scored_masks]
        per_pair.append((k, len(popped_final), len(hits), hits, len(unfil)))
        total_hits += [(k,) + h for h in hits]
        if hits:
            print(f"  {k[0]} a={k[1]} {k[2]}: {len(hits)} of {len(popped_final)} popped-FINAL "
                  f"never scored with IoU > incumbent")
            for lab, iou, t in hits[:3]:
                print(f"      IoU={iou!r} > incumbent={t!r}   {lab}")
    n = len(total_hits); nu = sum(p[4] for p in per_pair)
    print(f"\n  UNFILTERED (upper bound, known overcount): {nu} events, "
          f"{sum(1 for p in per_pair if p[4])} of {len(keys)} pairs")
    print(f"  FILTERED (mask never scored anywhere): {n} events, "
          f"{sum(1 for p in per_pair if p[2])} of {len(keys)} pairs")
    if n:
        print(f"  pairs with a filtered event: {sorted({h[0] for h in total_hits})}")
    print(f"  silent non-appends observed at estimate_iou_frontier: {nonappend[0]} "
          f"(gate: nodes CAN vanish pre-pop, so a zero would need this to be 0)")
    KNOWN = {("trained","0.2","unit88"),("trained","0.05","unit86")}
    aff = {h[0] for h in total_hits}
    print(f"  -> {'S3' if n == 0 else ('S2' if aff and aff <= KNOWN else 'S1')}"
          f"   (R1/R2 on the filtered count: {'R1' if n > 0 else 'R2'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
