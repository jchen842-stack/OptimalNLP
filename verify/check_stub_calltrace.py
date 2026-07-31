"""Check 6 (stronger): which FILES execute during a search, and do any of them import
the stubbed vision deps?

sys.settrace records every function call during the search. Cross-referenced against the
set of upstream files that mention detectron2/cityscapesscripts.
"""
import os, sys, subprocess
sys.path.insert(0, os.path.expanduser('~/projects/optimalce'))
sys.path.insert(0, 'src')
import synthetic_overlap_sweep as sos
import numpy as np, torch, scipy.sparse as sparse, heapq
import real_token_masks as rtm
from compositional import optimal

UP = os.path.expanduser('~/projects/optimalce')
tainted = subprocess.run(
    ['grep','-rl','-e','detectron2','-e','cityscapesscripts',UP,'--include=*.py'],
    capture_output=True, text=True).stdout.split()
print(f"upstream files importing/mentioning vision deps ({len(tainted)}):")
for t in tainted: print("   ", os.path.relpath(t, UP))

FEATS = os.path.expanduser('~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats')
toks = rtm.load_tokens(FEATS, 2000)
cons = rtm.select_concepts(toks, rtm.CATEGORIES, 15, 5)
dense = rtm.build_dense(toks, cons)
z = np.load('results/acts2k_trained_a0.1.npz'); neuron = z['acts'][88].astype(bool)
K, M = dense.shape
masks=[sparse.csr_matrix(dense[c].reshape(1,M)) for c in range(K)]
common,unique,unc,_ = sos.compute_quantities(dense, M)
dj = sos.compute_disjoint_info(dense, K)

FILES=set()
def tracer(frame, event, arg):
    if event=='call':
        FILES.add(frame.f_code.co_filename)
    return None

probe = sos.HeapProbe(200000, time_budget=600.0)
optimal.heapq = probe
devnull=open(os.devnull,'w'); saved=sys.stdout
try:
    sys.stdout=devnull
    sys.settrace(tracer)
    out = optimal.compute_optimal_explanations(
        bitmaps=torch.from_numpy(neuron.reshape(1,M)), masks=masks,
        masks_info=(common,unique,unc), disjoint_info=dj, config=sos.StubConfig(3, M))
finally:
    sys.settrace(None); sys.stdout=saved; optimal.heapq=heapq

print(f"\nsearch completed best_iou={out[1]:.6f} visited={out[2]}")
proj = sorted(f for f in FILES if f.startswith(UP))
print(f"\nupstream files EXECUTED during the search ({len(proj)}):")
for f in proj: print("   ", os.path.relpath(f, UP))
overlap = sorted(set(proj) & set(os.path.realpath(t) for t in tainted) |
                 (set(proj) & set(tainted)))
print(f"\nintersection (executed AND vision-tainted): {[os.path.relpath(f,UP) for f in overlap]}")
print("RESULT:", "PASS (no tainted file executes)" if not overlap else "FAIL")
