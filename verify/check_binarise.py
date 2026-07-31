"""Check 8: is the quantile PER UNIT or global? And does realised density match alpha?

Independent method: recompute the threshold per unit with np.percentile (a different call
than binarize's np.quantile) and compare masks; then construct a synthetic case where a
per-unit and a global threshold MUST differ, to prove the test is sensitive.
"""
import os, sys
sys.path.insert(0,'src')
import numpy as np, real_activations as ra

# --- sensitivity first: units with deliberately different scales -------------------
rng = np.random.default_rng(0)
states = np.concatenate([rng.normal(0,1,(1000,1)), rng.normal(50,1,(1000,1))], axis=1)
per_unit = ra.binarize(states, 0.1)
glob_thr = np.quantile(states, 0.9)
global_mask = (states > glob_thr).T
print("SENSITIVITY (unit0 ~N(0,1), unit1 ~N(50,1), alpha=0.1):")
print(f"  per-unit densities : {per_unit.mean(axis=1)}   <- both ~0.10 if PER UNIT")
print(f"  global-threshold   : {global_mask.mean(axis=1)}   <- 0.0 / 0.2 if GLOBAL")
is_per_unit = np.allclose(per_unit.mean(axis=1), 0.1, atol=0.02)
print(f"  verdict: {'PER UNIT' if is_per_unit else 'GLOBAL'}")

# --- real states, independent recomputation ----------------------------------------
import torch, real_token_masks as rtm
FEATS = os.path.expanduser('~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats')
sents = rtm.load_sentences(FEATS, 2000)
ck = torch.load('models/bowman_snli_best.pth', map_location='cpu')
real = ra.extract_states(sents, ck['stoi'], ck['hidden_dim'], ck['embedding_dim'],
                         seed=0, ckpt='models/bowman_snli_best.pth')
print(f"\nreal states {real.shape}")
print(f"{'alpha':>7} {'stored mean d':>14} {'recomputed mean d':>18} {'masks identical':>16} "
      f"{'min d':>8} {'max d':>8}")
bad=0
for a in (0.5,0.2,0.1,0.05,0.005):
    z = np.load(f'results/acts2k_trained_a{a}.npz')
    stored = z['acts']
    thr = np.percentile(real, 100*(1-a), axis=0)     # different API from np.quantile
    mine = (real > thr[np.newaxis,:]).T
    same = bool((stored==mine).all())
    d = mine.mean(axis=1)
    bad += (not same)
    print(f"{a:>7} {z['density'].mean():>14.5f} {d.mean():>18.5f} {str(same):>16} "
          f"{d.min():>8.5f} {d.max():>8.5f}")
print("\nRESULT:", "PASS" if is_per_unit and bad==0 else "FAIL")
