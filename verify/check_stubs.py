"""Check 6: is any stubbed vision module ever ATTRIBUTE-ACCESSED on the NLP path?

Method: after the stub finder installs, wrap _AnyModule.__getattr__ to log every attribute
access with a stack frame. Then run a real search end to end.
"""
import os, sys
sys.path.insert(0, os.path.expanduser('~/projects/optimalce'))
sys.path.insert(0, 'src')
import traceback
import synthetic_overlap_sweep as sos

ACCESSES = []
_AnyModule = type(sys.modules.get('detectron2')) if 'detectron2' in sys.modules else None
print("stub modules present after import:",
      sorted(m for m in sys.modules if m.split('.')[0] in ('detectron2','cityscapesscripts')))

target = sys.modules.get('detectron2')
if target is not None:
    cls = type(target)
    orig = cls.__getattr__
    def spy(self, name):
        if not name.startswith('__'):
            ACCESSES.append((getattr(self, '_name', '?'), name,
                             ''.join(traceback.format_stack(limit=4)[:-1])))
        return orig(self, name)
    cls.__getattr__ = spy
    print(f"instrumented {cls.__name__}.__getattr__")
else:
    print("no stub module was imported at all")

# Run a real end-to-end search.
import numpy as np, torch, scipy.sparse as sparse, heapq
import real_token_masks as rtm
from compositional import optimal
FEATS = os.path.expanduser('~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats')
toks = rtm.load_tokens(FEATS, 2000)
cons = rtm.select_concepts(toks, rtm.CATEGORIES, 15, 5)
dense = rtm.build_dense(toks, cons)
z = np.load('results/acts2k_trained_a0.1.npz'); neuron = z['acts'][88].astype(bool)
K, M = dense.shape
masks=[sparse.csr_matrix(dense[c].reshape(1,M)) for c in range(K)]
common,unique,unc,_ = sos.compute_quantities(dense, M)
dj = sos.compute_disjoint_info(dense, K)
probe = sos.HeapProbe(200000, time_budget=600.0)
optimal.heapq = probe
devnull=open(os.devnull,'w'); saved=sys.stdout
try:
    sys.stdout=devnull
    out = optimal.compute_optimal_explanations(
        bitmaps=torch.from_numpy(neuron.reshape(1,M)), masks=masks,
        masks_info=(common,unique,unc), disjoint_info=dj, config=sos.StubConfig(3, M))
finally:
    sys.stdout=saved; optimal.heapq=heapq
print(f"search completed, best_iou={out[1]:.6f}, visited={out[2]}")
print(f"\nstub attribute accesses during the NLP path: {len(ACCESSES)}")
for mod, name, st in ACCESSES[:10]:
    print(f"  {mod}.{name}\n{st}")
print("RESULT:", "PASS (no stub touched)" if not ACCESSES else "FAIL (stubs touched)")
