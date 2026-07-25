#!/usr/bin/env bash
# Provision an OptimalCE-NLP working tree inside a running pod:
#   1. clone the pinned upstream method repo onto the PVC (if absent)
#   2. apply our frontier-beam-fallback patch
#   3. sync our harness into <upstream>/nlp_extension/
#
# Portable across Nautilus namespaces: set POD (and optionally NS) and run.
#   POD=optimalce-cpu ./scripts/setup_pod.sh
set -euo pipefail

POD="${POD:-optimalce-cpu}"
NS="${NS:-}"                       # optional -n <namespace>
NSARG=(); [ -n "$NS" ] && NSARG=(-n "$NS")
REPO_DIR="/workspace/data/optimal-compositional-explanations"
UPSTREAM_URL="https://github.com/aiea-lab/optimal-compositional-explanations.git"
PIN="$(cat "$(dirname "$0")/../UPSTREAM")"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "[1/3] ensure upstream clone @ $PIN"
kubectl exec ${NSARG[@]+"${NSARG[@]}"} "$POD" -- bash -lc "
  set -e
  if [ ! -d '$REPO_DIR/.git' ]; then git clone '$UPSTREAM_URL' '$REPO_DIR'; fi
  cd '$REPO_DIR' && git fetch --all -q && git checkout -q '$PIN'
"

echo "[2/3] apply frontier-beam-fallback patch (idempotent)"
kubectl cp ${NSARG[@]+"${NSARG[@]}"} "$HERE/patches/0001-frontier-beam-fallback.patch" \
  "$POD:$REPO_DIR/.optimalce_nlp.patch"
kubectl exec ${NSARG[@]+"${NSARG[@]}"} "$POD" -- bash -lc "
  cd '$REPO_DIR'
  if git apply --check .optimalce_nlp.patch 2>/dev/null; then
    git apply .optimalce_nlp.patch && echo 'patch applied'
  elif git apply --reverse --check .optimalce_nlp.patch 2>/dev/null; then
    echo 'patch already applied'
  else
    echo 'WARNING: patch does not apply cleanly against current tree'; exit 1
  fi
"

echo "[3/3] sync harness into $REPO_DIR/nlp_extension/"
kubectl exec ${NSARG[@]+"${NSARG[@]}"} "$POD" -- mkdir -p "$REPO_DIR/nlp_extension/results"
kubectl cp ${NSARG[@]+"${NSARG[@]}"} "$HERE/src/synthetic_overlap_sweep.py" \
  "$POD:$REPO_DIR/nlp_extension/synthetic_overlap_sweep.py"

echo "done. run e.g.:"
echo "  kubectl exec ${NSARG[*]:-} $POD -- bash -lc 'cd $REPO_DIR && PYTHONPATH=. python -u nlp_extension/synthetic_overlap_sweep.py --mode overlap'"
