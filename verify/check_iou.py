"""Check 4: does our IoU agree with upstream's own metrics.iou to full precision?

10 (formula, neuron) pairs taken from the actual Phase B results. Our value comes from
src/corrected_metrics-adjacent arithmetic on (n_fires, n_inter); the reference comes from
upstream src/metrics.iou applied to the evaluated masks. Also checks the value the SEARCH
returned (best_iou), which is a third, independent path.
"""
import os, sys, csv, importlib.util
sys.path.insert(0, os.path.expanduser('~/projects/optimalce'))
sys.path.insert(0, 'src')
import numpy as np, torch
import real_token_masks as rtm

spec = importlib.util.spec_from_file_location(
    "up_metrics", os.path.expanduser('~/projects/optimalce/src/metrics.py'))
up = importlib.util.module_from_spec(spec); spec.loader.exec_module(up)

FEATS = os.path.expanduser('~/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats')
toks = rtm.load_tokens(FEATS, 2000)
cons = rtm.select_concepts(toks, rtm.CATEGORIES, 15, 5)
dense = rtm.build_dense(toks, cons)
idx = {f"{c}={v}": i for i,(c,v) in enumerate(cons)}

def parse(s):
    """Tiny recursive-descent parser over our rendered formula strings -> token mask."""
    s = s.strip()
    if s.startswith('(') and _matched(s):
        s = s[1:-1].strip()
    if s.startswith('NOT '):
        return ~parse(s[4:])
    for op in (' OR ', ' AND '):
        d = 0
        for i in range(len(s)-len(op)+1):
            c = s[i]
            if c == '(': d += 1
            elif c == ')': d -= 1
            if d == 0 and s[i:i+len(op)] == op:
                l, r = parse(s[:i]), parse(s[i+len(op):])
                return (l | r) if op==' OR ' else (l & r)
    return dense[idx[s]]

def _matched(s):
    d=0
    for i,c in enumerate(s):
        if c=='(': d+=1
        elif c==')':
            d-=1
            if d==0: return i==len(s)-1
    return False

rows=[r for r in csv.DictReader(open('results/beam_vs_exact_K15.csv'))
      if r['exact_formula'] and r['halted']!='time'][:10]
print(f"{'unit':>18} {'ours (n_inter/union)':>22} {'upstream metrics.iou':>22} {'search best_iou':>18} {'match':>6}")
bad=0
for r in rows:
    z = np.load(f"results/acts2k_{r['arm']}_a{r['alpha']}.npz")
    neuron = z['acts'][int(r['unit'].replace('unit',''))].astype(bool)
    F = parse(r['exact_formula'])
    inter = int((F & neuron).sum()); union = int((F | neuron).sum())
    ours = inter/union
    ref = up.iou(torch.from_numpy(F.reshape(1,-1)), torch.from_numpy(neuron.reshape(1,-1)))
    search = float(r['exact_IoU'])
    ok = (ours.hex() if hasattr(ours,'hex') else float(ours).hex()) == float(ref).hex()
    close_search = abs(ours-search) < 5e-5     # CSV stores 4dp
    if not (ok and close_search): bad+=1
    print(f"{r['arm'][:2]+' '+r['unit']+' a'+r['alpha']:>18} {ours!r:>22} {float(ref)!r:>22} "
          f"{search:>18.4f} {'OK' if ok and close_search else 'MISMATCH':>6}")
print("\nRESULT:", "PASS" if bad==0 else f"FAIL ({bad} mismatches)")
