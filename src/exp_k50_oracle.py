"""K=50 exhaustive oracle, per-sentence partition, length 3, all 27 pairs.

WHY THIS RUN DECIDES THE FINDING. B1 showed the E.2.2 precondition holds on every sentence at
K >= 50, so K = 50 is a regime the paper's assumptions cover. If the 2 misses persist there,
they are not caused by the Bott_1 violation.

PRE-REGISTERED, before running:

  K1  the 2 misses PERSIST at K = 50 (per-sentence)
      -> not caused by the Bott_1 violation. The finding is alive in a regime the paper's
         assumptions cover, and the mechanism hunt (M1/M2/M3) matters.

  K2  the misses VANISH at K = 50
      -> a K = 15 artifact. The report narrows to "the admissibility precondition is easy to
         violate by vocabulary configuration and nothing checks it", the mechanism hunt is a
         K = 15 artifact, and D6 closes without it.

  SAME RUN, second question: does the unevaluated-FINAL-formula observation reproduce at
  K = 50? It was observed at K = 15 and has NOT been shown to be configuration-free. Counted
  as refinement calls carrying next_op == "INDIVIDUAL".

  CHECKLIST. Quantities: per-pair boolean miss, both sides; IoU in float64. Membership: all 27
  pairs, per-sentence partition, stated. Reference: the K=50 exhaustive optimum, computed here,
  not reused from K=15. Discrimination: K1 and K2 are exhaustive and mutually exclusive.
  Power: n=27, 2 prior events; descriptive. Sentinels: timeouts and no-label returns bucketed
  separately, never folded into "no miss".

Enumeration: bottom-up, sharing length-2 sub-masks. 50 leaves -> 7,500 length-2 -> 1,125,000
length-3 per pair. Intersections via float32 GEMM against all 27 neurons at once.
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.environ.get(
    "OPTIMALCE_UPSTREAM", os.path.expanduser("~/projects/optimalce")))
import real_token_masks as rtm, real_token_search as rts
import exp_partition as EP
from compositional import optimal
from exp_beam_width import PAIRS, MAX_SENTS, MIN_SUPPORT, CAP

FEATS = os.path.expanduser("~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats")
K = 50
KNOWN = {("trained", "0.2", "unit88"), ("trained", "0.05", "unit86")}


def main():
    print(__doc__)
    tok = rtm.load_tokens(FEATS, MAX_SENTS)
    _, cats = rts.ARMS["all"]
    con = rtm.select_concepts(tok, cats, K, MIN_SUPPORT)
    dense = rtm.build_dense(tok, con)
    Kc, M = dense.shape
    sid, pos, N_, L_, _ = EP.build_partition(rtm, tok)
    dense2 = np.stack([EP.to_2d(dense[c], sid, pos, N_, L_) for c in range(Kc)])

    keys, neur = [], []
    for arm, alpha, units in PAIRS:
        z = np.load(os.path.join(REPO, "results", f"acts2k_{arm}_a{alpha}.npz"))
        for u in units: keys.append((arm, alpha, f"unit{u}")); neur.append(z["acts"][u].astype(bool))
    # float64, NOT float32. Counts are <= 24,199 and exactly representable in float64; in
    # float32 the GEMM result lands a few ULPs above the true integer count, which made all
    # 27 pairs look like misses with a +0.0000% gap. Precision artifact, not a finding.
    Nf = np.array(neur, dtype=np.float64)                 # (27, M)
    nsz = Nf.sum(1)
    print(f"[k50] {M} tokens, K={Kc}, {len(keys)} pairs, per-sentence grid {N_}x{L_}\n", flush=True)

    # ---- exhaustive in-grammar length-3 optimum, bottom-up ----
    t0 = time.time()
    lm = [dense[i] for i in range(Kc)]
    best = np.zeros(len(neur), dtype=np.float64)

    def upd(chunk):                                       # chunk: (B, M) bool
        cf = chunk.astype(np.float64)
        inter = Nf @ cf.T                                 # (27, B)
        msz = cf.sum(1)                                   # (B,)
        iou = inter / np.maximum(msz[None, :] + nsz[:, None] - inter, 1.0)
        np.round(iou, 12, out=iou)
        np.maximum(best, iou.max(1), out=best)

    lvl2 = []
    for i in range(Kc):
        upd(lm[i][None, :])
        block = np.empty((3 * Kc, M), dtype=bool)
        for j in range(Kc):
            block[3 * j] = lm[i] | lm[j]
            block[3 * j + 1] = lm[i] & lm[j]
            block[3 * j + 2] = lm[i] & ~lm[j]
        upd(block); lvl2.append(block)
    lvl2 = np.concatenate(lvl2, 0)                        # (Kc*3Kc, M)
    print(f"  level-2 masks: {lvl2.shape[0]:,}  ({time.time()-t0:.0f}s)", flush=True)
    for s in range(0, lvl2.shape[0], 25):
        sub = lvl2[s:s + 25]
        block = np.empty((sub.shape[0] * 3 * Kc, M), dtype=bool)
        r = 0
        for m2 in sub:
            for j in range(Kc):
                block[r] = m2 | lm[j]; block[r + 1] = m2 & lm[j]; block[r + 2] = m2 & ~lm[j]
                r += 3
        upd(block)
        if s % 1500 == 0: print(f"    level-3 {s}/{lvl2.shape[0]}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  exhaustive optimum done ({time.time()-t0:.0f}s)\n", flush=True)

    # ---- search, per-sentence, K=50 ----
    PH = optimal.path_heuristic; ORIG = PH.update_paths_iou
    final_est = [0]; total_est = [0]

    def hook(*a, **kw):
        node = kw.get("node") or (a[1] if len(a) > 1 else None)
        total_est[0] += 1
        if node is not None and node[1] == "INDIVIDUAL": final_est[0] += 1
        return ORIG(*a, **kw)

    miss, sent = [], []
    print(f"  {'pair':>27} {'returned':>12} {'K50 optimum':>12} {'gap%':>8}  status")
    for k, nb, b in zip(keys, neur, best):
        PH.update_paths_iou = hook
        try:
            EP.NEURON_FLAT[0] = nb
            r = EP.run_partitioned(dense2, EP.to_2d(nb, sid, pos, N_, L_), con, dense)
        finally:
            PH.update_paths_iou = ORIG
        if r["halted"] != "no" or r["iou"] != r["iou"]:
            sent.append(k); print(f"  {k[0]+' a='+k[1]+' '+k[2]:>27} {'SENTINEL':>12} {b:>12.6f}        {r['halted']}")
            continue
        m = b - r["iou"] > 1e-9        # float64 GEMM on exact integer counts
        if m: miss.append((k, r["iou"], b))
        print(f"  {k[0]+' a='+k[1]+' '+k[2]:>27} {r['iou']:>12.6f} {b:>12.6f} {100*(b/r['iou']-1):>7.4f}%"
              f"  {'MISS' if m else 'ok'}{'  <- known at K=15' if k in KNOWN else ''}")

    print(f"\n  misses at K=50 per-sentence: {len(miss)}/27   sentinels: {len(sent)}")
    for k, got, b in miss: print(f"    {k}: returned {got!r} vs optimum {b!r}  (+{100*(b/got-1):.4f}%)")
    kn = {k for k, _, _ in miss} & KNOWN
    print(f"\n  the 2 K=15 known misses at K=50: {sorted(kn) if kn else 'NEITHER misses'}")
    print(f"  -> {'K1' if kn else 'K2'}")
    print(f"\n  unevaluated-FINAL observation: {final_est[0]} of {total_est[0]} refinement calls "
          f"carried next_op == INDIVIDUAL  -> {'REPRODUCES at K=50' if final_est[0] else 'does NOT reproduce'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
