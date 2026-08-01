"""Experiment — warm-start the exact search from the beam_optimal solution, all 27 pairs.

The paper suggests initialising the optimal search from beam output. The targeted trace
showed `reduce_frontier` drops a node when its ceiling falls below the incumbent threshold,
and that the ceiling is inadmissible on some nodes. A HIGHER starting threshold therefore
strictly enlarges the set of wrongly-dropped nodes.

PRE-REGISTERED, before running:

  W1  MISS COUNT > 2.
      Warm-starting the incumbent at beam_optimal's IoU makes the search miss its own
      in-grammar optimum on MORE than the 2 pairs it misses cold.
      Rationale, recorded in advance: `reduce_frontier` drops on `-iou < threshold`, so
      raising the initial threshold can only grow the dropped set, never shrink it. The two
      observed cold drops sat only 89.3% and 25.7% of the way into windows of width 0.021864
      and 0.013209; four further pairs have an inadmissible ceiling and escaped only because
      an equal-valued optimum was reachable by another prefix. Those four are the obvious
      candidates to flip.
      SUPPORTED if miss count > 2.

  W2  NO PAIR IMPROVES.
      No pair that missed cold now finds the optimum. Warm-starting cannot repair an
      inadmissible ceiling.
      SUPPORTED if the cold miss set is a subset of the warm miss set.

If W1 holds it contradicts the paper's own suggested mitigation: seeding from beam output
makes in-grammar optimality worse, not better.
"""
import os, re, sys, types, csv
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(HERE); sys.path.insert(0,HERE)
from exp_beam_width import PAIRS, K, LENGTH, MAX_SENTS, MIN_SUPPORT, CAP
OUT=os.path.join(REPO,"results","warmstart_L3.csv")

SUBS=[(r"    best_results = \(-1\.0, None\)  # \(IoU, label\)",
       "    best_results = (WARM_START_IOU, None)  # WARM START", 1),
      (r"    minimum_threshold = 0\.0\n",
       "    minimum_threshold = WARM_START_IOU  # WARM START\n", 1)]

def build():
    src=open(os.path.join(REPO,".upstream-clean","compositional","optimal.py")).read()
    for p,r,w in SUBS:
        src,n=re.subn(p,r,src); assert n==w,(p,n)
    m=types.ModuleType("optimal_warm"); m.__dict__["__name__"]="optimal_warm"
    m.__dict__["WARM_START_IOU"]=0.0
    exec(compile(src,"optimal_warm","exec"),m.__dict__); return m

def main():
    print(__doc__,flush=True)
    sys.path.insert(0,os.environ.get("OPTIMALCE_UPSTREAM",os.path.expanduser("~/projects/optimalce")))
    import real_token_masks as rtm, real_token_search as rts
    import exp_noprune as C
    warm=build()
    tok=rtm.load_tokens(os.path.expanduser("~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats"),MAX_SENTS)
    _,cats=rts.ARMS["all"]; con=rtm.select_concepts(tok,cats,K,MIN_SUPPORT); dense=rtm.build_dense(tok,con)
    Kc=dense.shape[0]; lm=[dense[i] for i in range(Kc)]
    keys,neur=[],[]
    for arm,alpha,units in PAIRS:
        z=np.load(os.path.join(REPO,"results",f"acts2k_{arm}_a{alpha}.npz"))
        for u in units: keys.append((arm,alpha,f"unit{u}")); neur.append(z["acts"][u].astype(bool))
    N=np.array(neur); nsz=N.sum(1).astype(np.int64)
    def iouv(m):
        it=(N&m).sum(1).astype(np.int64); s=np.int64(m.sum()); return it/np.maximum(s+nsz-it,1)
    best=np.zeros(len(neur))
    def mv(m):
        for j in range(Kc):
            yield m|lm[j]
            yield m&lm[j]
            yield m&~lm[j]
    for i in range(Kc):
        for m2 in mv(lm[i]):
            for m3 in mv(m2): best=np.maximum(best,iouv(m3))
    bo={}
    for r in csv.DictReader(open(os.path.join(REPO,"results","beam_optimal_L3.csv"))):
        if int(r["beam"])==200: bo[(r["arm"],r["alpha"],r["unit"])]=float(r["beam_IoU"])
    rows=[]
    for (k,nb,g) in zip(keys,neur,best):
        seed=bo[k]; warm.WARM_START_IOU=seed
        w=C.run_with(warm,dense,nb,con)
        rows.append(dict(arm=k[0],alpha=k[1],unit=k[2],in_grammar_max=repr(g),
            seed_iou=repr(seed),warm_iou=repr(w["iou"]),
            warm_missed=int(g-w["iou"]>1e-12),warm_formula=w["formula"],
            warm_time_s=w["time_s"],warm_halted=w["halted"]))
        print(f"  {k[0]} a={k[1]} {k[2]}: seed={seed} warm={w['iou']:.6f} true={g:.6f} "
              f"{'MISS' if g-w['iou']>1e-12 else 'ok'}",flush=True)
    with open(OUT,"w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    COLD={("trained","0.2","unit88"),("trained","0.05","unit86")}
    warmset={(r["arm"],r["alpha"],r["unit"]) for r in rows if r["warm_missed"]}
    print(f"\nwrote {OUT}")
    print(f"\n  cold miss set ({len(COLD)}): {sorted(COLD)}")
    print(f"  warm miss set ({len(warmset)}): {sorted(warmset)}")
    print(f"\n  W1: {'SUPPORTED' if len(warmset)>2 else 'NOT SUPPORTED'}  "
          f"warm miss count = {len(warmset)} vs cold 2")
    print(f"  W2: {'SUPPORTED' if COLD<=warmset else 'NOT SUPPORTED'}  "
          f"cold misses still missing: {sorted(COLD&warmset)}; repaired: {sorted(COLD-warmset)}")
    return 0
if __name__=="__main__": raise SystemExit(main())
