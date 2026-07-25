# OptimalCE — NLP Extension

Extending **OptimalCE** (optimal compositional explanations of neurons) from vision to
NLP/tokens. This repo holds *our* work — experiment harness, results, research diary, the
fix as a patch, and the infra manifests — kept separate from the upstream method code so it
is portable across Nautilus namespaces and PVCs.

Upstream method (not vendored here): <https://github.com/aiea-lab/optimal-compositional-explanations>
pinned at the commit in [`UPSTREAM`](UPSTREAM) (`70805299`).

## Why this repo is standalone

The compute + data (Cityscapes, activation cache) currently live on a Nautilus PVC
(`optimalce-data`, RWO). To avoid being locked to one namespace/PVC, everything reproducible
lives here and can be re-applied to any fresh checkout:

- **Our code** (`src/`) is self-contained and imports the upstream package at runtime.
- **The fix** (`patches/`) is a diff against the pinned upstream commit — no upstream source
  is copied into this repo.
- **Infra** (`infra/`) has the pod/PVC manifests so a new namespace is a `kubectl apply` away.

## Layout

```
src/synthetic_overlap_sweep.py   experiment harness (overlap / scale / beam modes)
patches/0001-frontier-beam-fallback.patch   the fix, vs upstream 70805299
results/*.csv                    recorded experiment outputs
diary/summer_d5.md               research diary for this session
infra/{pvc,pod,pod-cpu}.yaml      Nautilus manifests
scripts/setup_pod.sh             clone upstream + apply patch + sync harness into a pod
UPSTREAM                         pinned upstream commit
```

## The result in one paragraph

Under vision's disjoint concepts the optimal search is cheap at any scale; under token-like
concept **overlap** the disjoint fast-path vanishes, admissible ceilings stop separating
candidates, and the best-first frontier balloons to OOM (wall at formula length ≥ 4). The
fix is the paper's §3.3 escape hatch — a **size-bounded beam fallback** (`MAX_FRONTIER_SIZE`
in `compositional/optimal.py`). Key finding: a size cap bounds *memory* but the beam must be
*tight* to also bound *time* — a wide cap (2000) still times out, while beam 100–500 turns a
non-terminating OOM search into 1.4–30 s with a usable explanation. See
[`diary/summer_d5.md`](diary/summer_d5.md) for the full write-up and tables.

## Reproduce (current or a new namespace)

Everything is **CPU-only** — the optimal search is Python `heapq` + estimation over tiny
boolean tensors; no GPU is needed (GPU only ever computed the vision activations, which the
synthetic harness bypasses).

```bash
# 0. (new namespace) provision storage + a CPU pod
kubectl apply -f infra/pvc.yaml
kubectl apply -f infra/pod-cpu.yaml        # optimalce-cpu; use infra/pod.yaml if you need a GPU

# 1. clone upstream onto the PVC, apply our patch, sync the harness
POD=optimalce-cpu ./scripts/setup_pod.sh

# 2. run the experiments (see modes below)
kubectl exec $POD -- bash -lc '
  cd /workspace/data/optimal-compositional-explanations &&
  PYTHONPATH=. python -u nlp_extension/synthetic_overlap_sweep.py --mode scale \
    --cap 200000 --time_budget 30 --out nlp_extension/results/scale_curve.csv'
```

### Harness modes
- `--mode overlap` — hold size fixed, sweep concept overlap (the mechanism).
- `--mode scale` — sweep K×length, contrasting a disjoint control vs the token regime (the wall).
- `--mode beam --beam_list none 100 200 500 1000 2000` — fix a hard case, sweep beam width
  (validate the fix; the tradeoff curve).

The harness stubs the upstream vision-only imports (`detectron2`, `cityscapesscripts`) via a
meta-path finder, so it runs on a bare CPU container without those installed.

## Cleanup / moving off the PVC

`kubectl delete pod optimalce-cpu` frees compute (no GPU is held by the CPU pod anyway).
Keep the PVC only for the Cityscapes data + activation cache; none of *this* repo depends on
it. To migrate: push this repo to your own GitHub, then `setup_pod.sh` against a pod in the
new namespace.
