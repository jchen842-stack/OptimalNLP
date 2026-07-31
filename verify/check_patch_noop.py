"""Check 3: is the MAX_FRONTIER_SIZE patch a no-op when None?

Runs one identical configuration against whichever `compositional` package OPTIMALCE_UPSTREAM
points at, printing every search output at FULL precision so patched and clean can be
compared exactly.
"""
import os, sys, time, heapq
sys.path.insert(0, os.environ['OPTIMALCE_UPSTREAM'])
sys.path.insert(0, 'src')
import numpy as np
from synthetic_overlap_sweep import HeapProbe, StubConfig, compute_disjoint_info, compute_quantities
import real_token_masks as rtm
from compositional import optimal
import scipy.sparse as sparse, torch

FEATS = os.path.expanduser('~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats')
toks = rtm.load_tokens(FEATS, 2000)
cons = rtm.select_concepts(toks, rtm.CATEGORIES, 15, 5)
dense = rtm.build_dense(toks, cons)
z = np.load('results/acts2k_trained_a0.2.npz')
neuron = z['acts'][413].astype(bool)

K, M = dense.shape
masks=[sparse.csr_matrix(dense[c].reshape(1,M)) for c in range(K)]
common,unique,uncoverable,_ = compute_quantities(dense, M)
dj = compute_disjoint_info(dense, K)
probe = HeapProbe(200000, time_budget=1500.0)
optimal.heapq = probe
patched = hasattr(optimal, 'MAX_FRONTIER_SIZE')
if patched:
    optimal.MAX_FRONTIER_SIZE = None
devnull=open(os.devnull,'w'); saved=sys.stdout
t0=time.time()
try:
    sys.stdout=devnull
    label, iou, vis, exp, est = optimal.compute_optimal_explanations(
        bitmaps=torch.from_numpy(neuron.reshape(1,M)), masks=masks,
        masks_info=(common,unique,uncoverable), disjoint_info=dj,
        config=StubConfig(4, M))
finally:
    sys.stdout=saved; optimal.heapq=heapq
dt=time.time()-t0

def render(f, c):
    from compositional import formula as F
    if isinstance(f, F.Leaf):  return f"{c[f.val][0]}={c[f.val][1]}"
    if isinstance(f, F.Not):   return f"(NOT {render(f.val,c)})"
    return f"({render(f.left,c)} {f.op} {render(f.right,c)})"

print(f"build            : {'PATCHED (MAX_FRONTIER_SIZE=None)' if patched else 'CLEAN UPSTREAM 7080529'}")
print(f"formula          : {render(label, cons)}")
print(f"best_iou  (repr) : {iou!r}")
print(f"best_iou  (hex)  : {float(iou).hex()}")
print(f"visited          : {vis}")
print(f"expanded         : {exp}")
print(f"estimated        : {est}")
print(f"peak_frontier    : {probe.peak}")
print(f"wall_s (informational, not compared): {dt:.1f}")
