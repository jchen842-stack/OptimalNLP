"""Item 4 — the EVENT-ORDERING HARM MEASUREMENT. Length 3, both partitions.

WHAT IT REPLACES. Every exposure figure in this project counts OPPORTUNITIES:
14/27 inadmissible ceilings, 4/27 all-prefixes-pruned bound, 31 leaf-ancestor prunes across
24/27 pairs -- against 2 actual losses flat and 0 per-sentence. The tightest bound overstates
by 2x and by 4->infinity. None of them measures harm. This does.

METRIC. Per node, over a run: CREATED (pushed), EXPANDED (popped and children generated),
DROPPED (removed, with the removing call site). Node identity is (evaluated mask, label
string) -- NOT object identity, because F.Or(a,b) == F.Or(b,a) already produced a false
"never entered frontier" on all 27 pairs once.

  per optimal FORMULA : was it ever SCORED (reached the INDIVIDUAL branch at optimal.py:765)?
  LOSS (pair-level)   : no optimal formula was ever scored

  SUBJECT CORRECTED, 2026-08-02. The metric was first built on optimal PREFIXES, as specified.
  It failed its own validation gate: on trained a=0.2 unit88 the prefix (dep=ROOT OR dep=nsubj)
  was CREATED 4x, DROPPED once, and EXPANDED THREE TIMES -- and the search still missed the
  optimum. The harm is one level DOWN: the expansions created the optimal length-3 child, and
  the child was dropped before it was ever scored. Prefix-level events cannot see that.
  Prefix outcomes are retained as a secondary column.

CHECKLIST FIELDS, carried with the result:
  Quantities    per-pair boolean LOSS on both sides. Event counts are INPUTS, never compared
                against node counts.
  Membership    all 27 pairs, both partitions, fixed. No pair is excluded from the table.
  Reference     optimal prefix sets from ONE enumeration, shared by both runs by object
                identity (asserted at runtime), not recomputed per partition.
  Discrimination the validation gate below. A metric that cannot separate the 2 known losses
                from the 25 non-losses measures nothing, and nothing else is reported.
  Power         27 pairs, 2 events. DESCRIPTIVE ONLY. No rate is computed, no between-
                partition significance is claimed.
  Sentinels     three buckets, counted separately, folded into neither losses nor non-losses:
                (a) prefix never CREATED, (b) pair timed out, (c) prefix created but the run
                ended before either expansion or drop.

Usage:  python src/exp_event_ordering.py
"""
import os, sys, heapq as H
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.environ.get(
    "OPTIMALCE_UPSTREAM", os.path.expanduser("~/projects/optimalce")))
import real_token_masks as rtm, real_token_search as rts, synthetic_overlap_sweep as sos
from compositional import optimal
import exp_partition as EP
from exp_beam_width import PAIRS, K, LENGTH, MAX_SENTS, MIN_SUPPORT, CAP

FEATS = os.path.expanduser("~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats")
KNOWN_LOSSES = {("trained", "0.2", "unit88"), ("trained", "0.05", "unit86")}


def optimal_formula_sets(dense, neurons):
    """Per-pair set of optimal FORMULA masks (bytes) -- the thing whose loss is the harm."""
    Kc = dense.shape[0]; lm = [dense[i] for i in range(Kc)]
    N = np.array(neurons); nsz = N.sum(1).astype(np.int64)

    def mv(m):
        for j in range(Kc):
            yield m | lm[j]
            yield m & lm[j]
            yield m & ~lm[j]

    def iouv(m):
        it = (N & m).sum(1).astype(np.int64); s = np.int64(m.sum())
        return it / np.maximum(s + nsz - it, 1)

    best = np.zeros(len(neurons))
    for i in range(Kc):
        best = np.maximum(best, iouv(lm[i]))
        for m2 in mv(lm[i]):
            best = np.maximum(best, iouv(m2))
            for m3 in mv(m2): best = np.maximum(best, iouv(m3))
    out = [set() for _ in neurons]
    for i in range(Kc):
        for k in np.where(np.abs(iouv(lm[i]) - best) < 1e-12)[0]: out[k].add(lm[i].tobytes())
        for m2 in mv(lm[i]):
            for k in np.where(np.abs(iouv(m2) - best) < 1e-12)[0]: out[k].add(m2.tobytes())
            for m3 in mv(m2):
                for k in np.where(np.abs(iouv(m3) - best) < 1e-12)[0]: out[k].add(m3.tobytes())
    return out


def optimal_prefix_sets(dense, neurons):
    """ONE enumeration. Returns per-pair set of optimal PREFIX masks (bytes). Shared by both
    partitions -- optimal masks are partition-invariant."""
    Kc = dense.shape[0]; lm = [dense[i] for i in range(Kc)]
    N = np.array(neurons); nsz = N.sum(1).astype(np.int64)

    def mv(m):
        for j in range(Kc):
            yield m | lm[j]
            yield m & lm[j]
            yield m & ~lm[j]

    def iouv(m):
        it = (N & m).sum(1).astype(np.int64); s = np.int64(m.sum())
        return it / np.maximum(s + nsz - it, 1)

    best = np.zeros(len(neurons))
    for i in range(Kc):
        best = np.maximum(best, iouv(lm[i]))
        for m2 in mv(lm[i]):
            best = np.maximum(best, iouv(m2))
            for m3 in mv(m2): best = np.maximum(best, iouv(m3))
    prefs = [set() for _ in neurons]
    for i in range(Kc):
        v1 = iouv(lm[i])
        for k in np.where(np.abs(v1 - best) < 1e-12)[0]: prefs[k].add(lm[i].tobytes())
        for m2 in mv(lm[i]):
            v2 = iouv(m2)
            for k in np.where(np.abs(v2 - best) < 1e-12)[0]: prefs[k].add(lm[i].tobytes())
            for m3 in mv(m2):
                v3 = iouv(m3)
                for k in np.where(np.abs(v3 - best) < 1e-12)[0]: prefs[k].add(m2.tobytes())
    return prefs


def run_with_events(mode, dense, dense2, sid, pos, N, L, nb, concepts, target_masks,
                    formula_masks):
    """One run, returning {mask_bytes: [(pop_clock, event, callsite, label_str)]}.

    SCORED is hooked at `mask_utils.get_formula_mask_and_tree`, whose only call site in the
    search is `optimal.py:765`, inside the `next_op_node == "INDIVIDUAL"` branch. Its returned
    dict also carries ancestors, which the propagation block scores, so every key is recorded.
    """
    ev = {}; clock = [0]; cache = {}; thr = [0.0]

    watch = set(target_masks) | set(formula_masks)

    def key(f):
        # Cache by VALUE, never by id(). CPython reuses ids after GC, and these formula
        # objects are short-lived, so an id-keyed cache returns stale answers -- the same
        # identity defect as F.Or(a,b) vs F.Or(b,a), one layer down.
        if len(f) > 3: return None
        kk = rts.render(f, concepts)
        if kk not in cache:
            b = rts.eval_formula(f, dense).tobytes()
            cache[kk] = b if b in watch else None
        return cache[kk]

    def rec(f, kind, site, ceil=None):
        b = key(f)
        if b is not None:
            ev.setdefault(b, []).append((clock[0], kind, site, rts.render(f, concepts),
                                         ceil, thr[0]))

    def rec_many(nodes, kind, site):
        for n in nodes: rec(n[2], kind, site, -n[0] if n[0] is not None else None)

    from utils import mask_utils as MU
    OP, ORIG_RED, ORIG_EXP = sos.HeapProbe.heappush, optimal.reduce_frontier, optimal.expand_node
    ORIG_GFM = MU.get_formula_mask_and_tree

    def gfm(f, masks, path_masks=None, device=None):
        out = ORIG_GFM(f, masks, path_masks, device) if device is not None else ORIG_GFM(f, masks, path_masks)
        for lab in out:
            rec(lab, "SCORED", "get_formula_mask_and_tree")
        return out

    ORIG_HEAPIFY = sos.HeapProbe.heapify

    def push(self, heap, item):
        rec(item[2], "CREATED", "heappush", -item[0] if item[0] is not None else None)
        return OP(self, heap, item)

    def heapify(self, x):
        # update_frontier admits new nodes via heapify, not heappush (optimal.py:514).
        # Missing this made every length-3 node invisible to the CREATED event.
        rec_many(x, "CREATED", "heapify")
        return ORIG_HEAPIFY(self, x)

    def red(fr, t):
        thr[0] = max(thr[0], t)
        for n in fr:
            if -n[0] < t: rec(n[2], "DROPPED", "reduce_frontier", -n[0])
        return ORIG_RED(fr, t)

    def exp(node, *, candidate_labels, max_length):
        rec(node[2], "EXPANDED", "expand_node", -node[0] if node[0] is not None else None)
        return ORIG_EXP(node, candidate_labels=candidate_labels, max_length=max_length)

    class Clock:
        def __getattr__(s, n): return getattr(H, n)
        def heappush(s, h, i): return H.heappush(h, i)
        def heappop(s, h):
            clock[0] += 1; return H.heappop(h)
        def heapify(s, x): return H.heapify(x)

    sos.HeapProbe.heappush = push; optimal.reduce_frontier = red; optimal.expand_node = exp
    MU.get_formula_mask_and_tree = gfm; sos.HeapProbe.heapify = heapify
    base_pop = sos.HeapProbe.heappop

    def popped(self, h):
        # optimal.py:697 pops a node and SKIPS it when -e_node < minimum_threshold. That is a
        # removal path reduce_frontier never sees, and it is how the two known losses die.
        clock[0] += 1
        it = base_pop(self, h)
        rec(it[2], "POPPED", "heappop", -it[0] if it[0] is not None else None)
        return it

    sos.HeapProbe.heappop = popped
    try:
        if mode == "flat":
            r = rts.run_one(dense, nb, LENGTH, CAP, 1500.0, None, concepts=concepts)
            halted = r["halted"]
        else:
            EP.NEURON_FLAT[0] = nb
            r = EP.run_partitioned(dense2, EP.to_2d(nb, sid, pos, N, L), concepts, dense)
            halted = r["halted"]
    finally:
        sos.HeapProbe.heappush = OP; optimal.reduce_frontier = ORIG_RED
        optimal.expand_node = ORIG_EXP; sos.HeapProbe.heappop = base_pop
        MU.get_formula_mask_and_tree = ORIG_GFM; sos.HeapProbe.heapify = ORIG_HEAPIFY
    return ev, halted


def classify_formulas(events, fmasks, halted):
    """LOSS = no optimal formula was ever SCORED."""
    scored, dropped_unscored, never, created_only = [], [], [], []
    for b in fmasks:
        e = events.get(b, [])
        if not e: never.append(b); continue
        if any(x[1] == "SCORED" for x in e): scored.append(b)
        elif any(x[1] == "DROPPED" for x in e): dropped_unscored.append(b)
        else: created_only.append(b)
    lost = bool(fmasks) and not scored
    drop_info = []
    for b in fmasks:
        e_ = events.get(b, [])
        if any(x[1] == "SCORED" for x in e_): continue
        for e in e_:
            if e[1] in ("DROPPED", "POPPED") and e[4] is not None and e[4] < e[5]:
                drop_info.append((e[1], e[4], e[5])); break
    headroom = []
    for b in scored:
        for e in events.get(b, []):
            if e[1] == "POPPED" and e[4] is not None:
                headroom.append(e[4] - e[5]); break
    return lost, dict(scored=len(scored), dropped_unscored=len(dropped_unscored),
                      never_created=len(never), created_only=len(created_only),
                      timed_out=int(halted != "no"), drop_info=drop_info,
                      headroom=headroom)


def classify(events, prefs, halted):
    """Per-prefix outcome + pair-level LOSS + sentinel buckets + ordering margins."""
    out = {}; margins = []
    for b in prefs:
        e = events.get(b, [])
        if not e: out[b] = ("NEVER_CREATED", None); continue
        exps = [x[0] for x in e if x[1] == "EXPANDED"]
        drops = [x[0] for x in e if x[1] == "DROPPED"]
        if exps:
            m = (min(drops[-1:] or [None]) - exps[0]) if drops else None
            out[b] = ("EXPANDED_THEN_DROPPED" if drops else "EXPANDED_NEVER_DROPPED", m)
            if m is not None: margins.append(m)
        elif drops:
            out[b] = ("DROPPED_BEFORE_EXPANSION", None)
        else:
            out[b] = ("CREATED_ONLY", None)
    saved = [b for b, (s, _) in out.items() if s.startswith("EXPANDED")]
    created_only = [b for b, (s, _) in out.items() if s == "CREATED_ONLY"]
    never = [b for b, (s, _) in out.items() if s == "NEVER_CREATED"]
    lost = bool(prefs) and not saved and not created_only and not never
    return out, lost, margins, dict(never_created=len(never), created_only=len(created_only),
                                    timed_out=int(halted != "no"))


def main():
    tok = rtm.load_tokens(FEATS, MAX_SENTS)
    _, cats = rts.ARMS["all"]
    con = rtm.select_concepts(tok, cats, K, MIN_SUPPORT)
    dense = rtm.build_dense(tok, con)
    sid, pos, N, L, _ = EP.build_partition(rtm, tok)
    dense2 = np.stack([EP.to_2d(dense[c], sid, pos, N, L) for c in range(dense.shape[0])])

    keys, neur = [], []
    for arm, alpha, units in PAIRS:
        z = np.load(os.path.join(REPO, "results", f"acts2k_{arm}_a{alpha}.npz"))
        for u in units: keys.append((arm, alpha, f"unit{u}")); neur.append(z["acts"][u].astype(bool))

    prefs = optimal_prefix_sets(dense, neur)          # ONE enumeration
    fmasks = optimal_formula_sets(dense, neur)        # ONE enumeration
    res = {}
    for mode in ("flat", "part"):
        rows = []
        for k, nb, pf in zip(keys, neur, prefs):
            assert pf is prefs[keys.index(k)], "Reference: prefix set object must be shared"
            fm = fmasks[keys.index(k)]
            ev, halted = run_with_events(mode, dense, dense2, sid, pos, N, L, nb, con, pf, fm)
            out, _pl, margins, _ps = classify(ev, pf, halted)
            lost, sent = classify_formulas(ev, fm, halted)
            rows.append((k, len(pf), out, lost, margins, sent))
        res[mode] = rows

    # ---------- VALIDATION GATE ----------
    print("=" * 90); print("VALIDATION GATE — the metric must reproduce the known answer"); print("=" * 90)
    ok = True
    for mode, expect in (("flat", KNOWN_LOSSES), ("part", set())):
        got = {k for k, n, o, lost, m, s in res[mode] if lost}
        good = got == expect
        ok &= good
        print(f"  {mode:>5}: losses {sorted(got)}   expected {sorted(expect)}   {'OK' if good else 'MISMATCH'}")
    if not ok:
        print("\n  *** METRIC IS WRONG, not the search. Nothing else is reported.")
        return 1
    print("  GATE PASSED — the metric separates the 2 known losses from the 25 non-losses.\n")

    # ---------- THE TABLE ----------
    print("=" * 90)
    print("OPPORTUNITY vs HARM — length 3, K=15, M=24,199, all 27 pairs, both partitions")
    print("=" * 90)
    print(f"  {'measure':<46} {'flat':>12} {'per-sentence':>14}")
    print(f"  {'-'*46} {'-'*12} {'-'*14}")
    print(f"  {'OPPORTUNITY: inadmissible ceilings (set-based)':<46} {'14/27':>12} {'13/27':>14}")
    print(f"  {'OPPORTUNITY: all optimal prefixes pruned (bound)':<46} {'4/27':>12} {'4/27':>14}")
    print(f"  {'OPPORTUNITY: optimal leaf ancestors pruned':<46} {'31 / 24 pairs':>12} {'-':>14}")
    for mode, lab in (("flat", "flat"), ("part", "per-sentence")):
        pass
    lf = sum(1 for k, n, o, lost, m, s in res["flat"] if lost)
    lp = sum(1 for k, n, o, lost, m, s in res["part"] if lost)
    print(f"  {'HARM: pairs that actually lost the optimum':<46} {lf:>12} {lp:>14}")
    import statistics as st
    for mode, lab in (("flat", "flat"), ("part", "per-sentence")):
        mg = [x for k, n, o, lost, m, s in res[mode] for x in m]
        col = f"{min(mg)} / {st.median(mg):.0f} / {max(mg)}" if mg else "-"
        if mode == "flat": mf = col
        else: mp = col
    print(f"  {'ORDERING MARGIN, pops (min/median/max)':<46} {mf:>12} {mp:>14}")
    for b, lab in (("never_created", "SENTINEL: prefix never CREATED"),
                   ("created_only", "SENTINEL: created, run ended before expand/drop"),
                   ("timed_out", "SENTINEL: pair timed out")):
        f_ = sum(s[b] for k, n, o, lost, m, s in res["flat"])
        p_ = sum(s[b] for k, n, o, lost, m, s in res["part"])
        print(f"  {lab:<46} {f_:>12} {p_:>14}")

    print("\n  POWER: 27 pairs, 2 events. DESCRIPTIVE ONLY — no rate is computed and no")
    print("  between-partition significance is claimed. MEMBERSHIP: all 27 pairs, both")
    print("  partitions, none excluded. REFERENCE: one enumeration, shared by object identity.")

    print("\n--- PER-PAIR (flat) ---")
    for k, n, o, lost, m, s in res["flat"]:
        st_ = ",".join(sorted({v[0] for v in o.values()}))
        print(f"  {k[0]+' a='+k[1]+' '+k[2]:>27} n_pref={n} {'LOSS' if lost else '   .'}  "
              f"margins={sorted(m)}  {st_}")
    print("\n--- PER-PAIR (per-sentence) ---")
    for k, n, o, lost, m, s in res["part"]:
        st_ = ",".join(sorted({v[0] for v in o.values()}))
        print(f"  {k[0]+' a='+k[1]+' '+k[2]:>27} n_pref={n} {'LOSS' if lost else '   .'}  "
              f"margins={sorted(m)}  {st_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
