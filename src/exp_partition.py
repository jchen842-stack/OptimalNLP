"""Per-SENTENCE sample partition: does the one-sample configuration explain the behaviour?

DEVIATION (registered 2026-08-01, third protocol item, never previously registered):

The paper's D is the set of dataset inputs -- images in vision, sentences for SNLI.
`real_token_search.run_one` builds `bitmaps = neuron_bits.reshape(1, M)`: ONE sample holding
all 24,199 tokens. Section E.2.2's degenerate case is "a single sample contains all the
concepts in the dataset" -- which this configuration satisfies BY CONSTRUCTION, not because
NLP is different from vision.

Distinct from the two earlier protocol items: alpha was a miss, K was a scope choice. This
one was never registered at all.

This script rebuilds the sample axis as one sample per sentence (~2,000), leaving concepts,
masks, alpha and K untouched, and re-runs length 3 on all 27 pairs.

PRE-REGISTERED:

  V0  CONTROL, checked FIRST. IoU is partition-invariant (Lemma 3.6), so every one of the 27
      pairs must return the same in-grammar optimum IoU as the brute-force oracle, and the
      oracle value itself is unchanged. If IoU moves, the repartition is wrong and the
      finding is not implicated. Nothing else is read until this passes.

  P1  Bott_1(E^C)_x = 0 on a substantial fraction of sentences, against 0 of 1 now.
      "Substantial" fixed in advance as >= 50% of sentences.

  P2  The 6/27 dropped-prefix exposure FALLS (fewer than 6 pairs have an inadmissible
      ceiling on the optimum-carrying prefix).

  P3  Both known misses recover: trained a=0.2 unit88 and trained a=0.05 unit86 reach the
      true in-grammar optimum.

  P4  Expanded-node counts fall at length 3 -- tighter bounds mean more pruning. Scored on
      the median over the 27 pairs.
"""
import os, sys, time, csv
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(HERE); sys.path.insert(0,HERE)
from exp_beam_width import PAIRS, K, MAX_SENTS, MIN_SUPPORT, CAP
LENGTH=int(os.environ.get("PART_LENGTH","3"))
TIME_BUDGET=1500.0
OUT=os.path.join(REPO,"results",f"partition_L{LENGTH}.csv")
FEATS=os.path.expanduser("~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats")

def build_partition(rtm, tokens):
    """(sent_id per token, N, L) with sentences padded to the longest."""
    sents=rtm.load_sentences(FEATS, MAX_SENTS)
    lens=[len(s) for s in sents]
    assert sum(lens)==len(tokens), f"{sum(lens)} != {len(tokens)}"
    N,L=len(lens),max(lens)
    sid=np.concatenate([np.full(n,i) for i,n in enumerate(lens)])
    pos=np.concatenate([np.arange(n) for n in lens])
    return sid,pos,N,L,lens

def to_2d(vec, sid, pos, N, L):
    out=np.zeros((N,L),dtype=bool); out[sid,pos]=vec; return out

def run_partitioned(dense2, neuron2, concepts, dense_flat):
    import scipy.sparse as sparse, torch, heapq
    from synthetic_overlap_sweep import HeapProbe, StubConfig, _Halt
    from compositional import optimal
    import real_token_search as rts
    Kc,N,L=dense2.shape
    masks=[sparse.csr_matrix(dense2[c]) for c in range(Kc)]
    se=dense2.astype(np.int32).sum(0)
    common=torch.from_numpy(se>1); unique=torch.from_numpy(se==1); unc=torch.from_numpy(se==0)
    dj=np.ones((Kc,Kc),dtype=bool)
    for a in range(Kc):
        for b in range(Kc):
            dj[a,b]=False if a==b else not bool((dense2[a]&dense2[b]).any())
    bitmaps=torch.from_numpy(neuron2)
    probe=HeapProbe(CAP,time_budget=TIME_BUDGET); optimal.heapq=probe
    optimal.MAX_FRONTIER_SIZE=None
    class Cfg:
        def get_length(self): return LENGTH
        def get_mask_shape(self): return (N,L)
        def get_device(self): return "cpu"
    t0=time.time(); halt=""; lab=None
    dn=open(os.devnull,"w"); so=sys.stdout
    try:
        sys.stdout=dn
        lab,iou,vis,exp,est=optimal.compute_optimal_explanations(
            bitmaps=bitmaps,masks=masks,masks_info=(common,unique,unc),
            disjoint_info=torch.from_numpy(dj),config=Cfg())
    except _Halt as h:
        halt=h.reason; iou,vis,exp=float("nan"),-1,-1
    finally:
        sys.stdout=so; dn.close(); optimal.heapq=heapq; dt=time.time()-t0
    f=None; exact=float("nan")
    if lab is not None:
        m=rts.eval_formula(lab,dense_flat)
        exact=int((m&NEURON_FLAT[0]).sum())/int((m|NEURON_FLAT[0]).sum())
        f=rts.render(lab,concepts)
    return dict(iou=exact,formula=f,visited=vis,expanded=exp,time_s=round(dt,2),
                peak=probe.peak,halted=halt or "no")

NEURON_FLAT=[None]

def main():
    print(__doc__,flush=True)
    sys.path.insert(0,os.environ.get("OPTIMALCE_UPSTREAM",os.path.expanduser("~/projects/optimalce")))
    import real_token_masks as rtm, real_token_search as rts
    tokens=rtm.load_tokens(FEATS,MAX_SENTS)
    _,cats=rts.ARMS["all"]; con=rtm.select_concepts(tokens,cats,K,MIN_SUPPORT)
    dense=rtm.build_dense(tokens,con)
    sid,pos,N,L,lens=build_partition(rtm,tokens)
    print(f"[partition] {len(tokens)} tokens -> {N} sentences, max len {L}, "
          f"padded grid {N}x{L}={N*L} ({N*L-len(tokens)} pad cells)\n",flush=True)
    dense2=np.stack([to_2d(dense[c],sid,pos,N,L) for c in range(dense.shape[0])])
    keys,neur=[],[]
    for arm,alpha,units in PAIRS:
        z=np.load(os.path.join(REPO,"results",f"acts2k_{arm}_a{alpha}.npz"))
        for u in units: keys.append((arm,alpha,f"unit{u}")); neur.append(z["acts"][u].astype(bool))
    # brute-force in-grammar optimum (partition-invariant; computed on the flat axis)
    Nn=np.array(neur); nsz=Nn.sum(1).astype(np.int64); lm=[dense[i] for i in range(K)]
    def iouv(m):
        it=(Nn&m).sum(1).astype(np.int64); s=np.int64(m.sum()); return it/np.maximum(s+nsz-it,1)
    best=np.zeros(len(neur))
    def mv(m):
        for j in range(K):
            yield m|lm[j]
            yield m&lm[j]
            yield m&~lm[j]
    for i in range(K):
        for m2 in mv(lm[i]):
            for m3 in mv(m2): best=np.maximum(best,iouv(m3))
    # P1: Bott_1(E^C)_x per sentence
    seall=dense2.astype(np.int32).sum(0); C2=seall>1
    rows=[]
    for i,(k,nb) in enumerate(zip(keys,neur)):
        nb2=to_2d(nb,sid,pos,N,L); NEURON_FLAT[0]=nb
        EC=np.stack([(C2&dense2[j]&~nb2).sum(1) for j in range(K)])   # (K,N)
        bott=EC.min(0)
        r=run_partitioned(dense2,nb2,con,dense)
        rows.append(dict(arm=k[0],alpha=k[1],unit=k[2],in_grammar_max=repr(best[i]),
            part_iou=repr(r["iou"]),missed=int(best[i]-r["iou"]>1e-12) if r["iou"]==r["iou"] else -1,
            bott_zero_frac=round(float((bott==0).mean()),4),bott_min=int(bott.min()),
            visited=r["visited"],expanded=r["expanded"],peak=r["peak"],
            time_s=r["time_s"],halted=r["halted"],formula=r["formula"]))
        print(f"  {k[0]} a={k[1]} {k[2]}: part={r['iou']:.6f} true={best[i]:.6f} "
              f"{'MISS' if best[i]-r['iou']>1e-12 else 'ok'}  bott0={rows[-1]['bott_zero_frac']:.3f} "
              f"exp={r['expanded']} t={r['time_s']}s {r['halted']}",flush=True)
    with open(OUT,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}")
    import statistics as st
    bad=[r for r in rows if r["missed"]!=0]
    print(f"\n  V0 CONTROL: {len(rows)-len(bad)}/{len(rows)} pairs reach the brute-force optimum")
    print(f"      {'PASS' if not bad else 'FAIL -- repartition is wrong, finding NOT implicated'}")
    fr=[r["bott_zero_frac"] for r in rows]
    print(f"  P1: median fraction of sentences with Bott_1(E^C)_x == 0 = {st.median(fr):.4f}"
          f"  -> {'SUPPORTED' if st.median(fr)>=0.5 else 'NOT SUPPORTED'} (>=0.50); was 0/1 flat")
    COLD={("trained","0.2","unit88"),("trained","0.05","unit86")}
    stillmiss={(r['arm'],r['alpha'],r['unit']) for r in rows if r['missed']==1}
    print(f"  P3: cold misses recovered: {sorted(COLD-stillmiss)}; still missing: {sorted(COLD&stillmiss)}"
          f"  -> {'SUPPORTED' if not (COLD&stillmiss) else 'NOT SUPPORTED'}")
    ex=[int(r['expanded']) for r in rows if int(r['expanded'])>0]
    print(f"  P4: median expanded (partitioned) = {st.median(ex):.0f}   [flat baseline printed separately]")
    return 0
if __name__=="__main__": raise SystemExit(main())
