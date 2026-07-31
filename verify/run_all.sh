#!/usr/bin/env bash
# Run every correctness check and print a PASS / FAIL / CANNOT VERIFY table.
#
# Exit code is nonzero if anything FAILED. CANNOT VERIFY does not fail the run -- it means a
# gitignored input is missing (checkpoint, activations, or the clean upstream tree), and the
# table says which file each such check needs.
#
#   ./verify/run_all.sh
#
# Env:
#   PY                    python to use (default: the compexp env, else `python3`)
#   OPTIMALCE_UPSTREAM    patched upstream tree (default ~/projects/optimalce)
#   UPSTREAM_CLEAN        clean upstream @70805299, for the patch no-op check.
#                         If unset, check 3 reports CANNOT VERIFY with the clone command.
#   ORACLE_LENGTH         3 (default) or 4. Length 4 takes ~20 min.

set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-$HOME/miniconda3/envs/compexp/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
export OPTIMALCE_UPSTREAM="${OPTIMALCE_UPSTREAM:-$HOME/projects/optimalce}"
FEATS="${FEATS:-$HOME/projects/neuron-explanations-nli/nli/data/analysis/snli_1.0_dev.feats}"
CKPT="models/bowman_snli_best.pth"
ACTS="results/acts2k_trained_a0.1.npz"
ACTS_U="results/acts2k_untrained_a0.1.npz"
ORACLE_LENGTH="${ORACLE_LENGTH:-3}"

NAMES=(); STATUS=(); NOTES=()
record() { NAMES+=("$1"); STATUS+=("$2"); NOTES+=("$3"); }

hdr() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

# A check PASSES if its script exits 0 and prints a RESULT line that is not FAIL.
run_check() {
  local name="$1" cmd="$2" need="$3" needdesc="$4"
  hdr "$name"
  for f in $need; do
    if [ ! -e "$f" ]; then
      echo "  SKIPPED: missing $f"
      record "$name" "CANNOT VERIFY" "needs $needdesc"
      return 0
    fi
  done
  local out rc
  out="$(eval "$cmd" 2>&1)"; rc=$?
  echo "$out" | tail -n 4
  if [ $rc -ne 0 ] || echo "$out" | grep -qE '^RESULT: FAIL|MISMATCH'; then
    record "$name" "FAIL" "exit $rc"
  else
    record "$name" "PASS" ""
  fi
}

echo "OptimalCE-NLP correctness audit"
"$PY" src/env_info.py >/dev/null 2>&1 || true
"$PY" -c "import sys; sys.path.insert(0,'src'); import env_info; env_info.print_banner('audit')"
echo "python: $PY"
echo "upstream (patched): $OPTIMALCE_UPSTREAM"

run_check "1. alignment (token order)" \
  "\"$PY\" verify/check_alignment.py | grep -vE '^      OK'" \
  "$FEATS" ".feats corpus (trained arm additionally needs $CKPT; it skips gracefully)"

run_check "2. padding" \
  "\"$PY\" verify/check_padding.py" \
  "$FEATS" ".feats corpus"

# Check 3 needs two trees to compare, so it is driven here rather than by one script.
hdr "3. patch is a no-op when MAX_FRONTIER_SIZE=None"
if [ -z "${UPSTREAM_CLEAN:-}" ] || [ ! -d "${UPSTREAM_CLEAN:-/nonexistent}" ]; then
  echo "  SKIPPED: no clean upstream tree."
  echo "  To enable:  git clone https://github.com/aiea-lab/optimal-compositional-explanations /tmp/upstream_clean"
  echo "              cd /tmp/upstream_clean && git checkout \$(cat $PWD/UPSTREAM)"
  echo "              UPSTREAM_CLEAN=/tmp/upstream_clean ./verify/run_all.sh"
  record "3. patch no-op" "CANNOT VERIFY" "needs UPSTREAM_CLEAN=<clean @70805299>"
elif [ ! -e "results/acts2k_trained_a0.2.npz" ]; then
  echo "  SKIPPED: missing results/acts2k_trained_a0.2.npz"
  record "3. patch no-op" "CANNOT VERIFY" "needs results/acts2k_trained_a0.2.npz"
else
  A="$(OPTIMALCE_UPSTREAM="$OPTIMALCE_UPSTREAM" "$PY" verify/check_patch_noop.py 2>&1 | grep -E 'formula|best_iou|visited|expanded|estimated|peak')"
  B="$(OPTIMALCE_UPSTREAM="$UPSTREAM_CLEAN"     "$PY" verify/check_patch_noop.py 2>&1 | grep -E 'formula|best_iou|visited|expanded|estimated|peak')"
  echo "$A" | sed 's/^/  patched: /'
  if [ "$A" = "$B" ]; then
    echo "  identical to clean upstream"; record "3. patch no-op" "PASS" ""
  else
    echo "  DIFFERS from clean upstream"; diff <(echo "$A") <(echo "$B") | sed 's/^/    /'
    record "3. patch no-op" "FAIL" "outputs differ"
  fi
fi

run_check "4. IoU vs upstream metrics.iou" \
  "\"$PY\" verify/check_iou.py" \
  "results/beam_vs_exact_K15.csv $ACTS" "results/*.npz (REPRODUCE.md step 3)"

run_check "5. masks vs raw .feats" \
  "\"$PY\" verify/check_masks.py" \
  "$FEATS" ".feats corpus"

run_check "6a. vision stubs untouched (runtime)" \
  "\"$PY\" verify/check_stubs.py" \
  "$ACTS" "results/*.npz (REPRODUCE.md step 3)"

run_check "6b. vision stubs untouched (call trace)" \
  "\"$PY\" verify/check_stub_calltrace.py" \
  "$ACTS" "results/*.npz (REPRODUCE.md step 3)"

hdr "7. checkpoint reproduces 0.7934"
if [ ! -e "$CKPT" ]; then
  echo "  SKIPPED: missing $CKPT (gitignored; see models/README.md)"
  record "7. model reproduces" "CANNOT VERIFY" "needs $CKPT + SNLI corpus"
else
  out="$("$PY" verify/check_model.py 2>&1 | grep -vE 'it/s\]')"
  echo "$out" | grep -E 'RE-EVALUATED|match to 1e-9|OOV' | sed 's/^/  /'
  if echo "$out" | grep -q 'match to 1e-9            : True'; then
    record "7. model reproduces" "PASS" ""
  else
    record "7. model reproduces" "FAIL" "dev accuracy did not match"
  fi
fi

run_check "8. binarisation is per-unit" \
  "\"$PY\" verify/check_binarise.py" \
  "$CKPT $ACTS" "$CKPT and results/*.npz"

run_check "9. brute-force oracle (length $ORACLE_LENGTH)" \
  "ORACLE_LENGTH=$ORACLE_LENGTH \"$PY\" tests/test_bruteforce_oracle.py" \
  "$ACTS $ACTS_U" "results/*.npz (REPRODUCE.md step 3)"

# ---- summary -------------------------------------------------------------------------
printf '\n\033[1m=== SUMMARY ===\033[0m\n'
fails=0; cannot=0
printf '%-42s %-15s %s\n' "CHECK" "RESULT" "NOTE"
printf '%-42s %-15s %s\n' "------------------------------------------" "---------------" "----"
for i in "${!NAMES[@]}"; do
  printf '%-42s %-15s %s\n' "${NAMES[$i]}" "${STATUS[$i]}" "${NOTES[$i]}"
  [ "${STATUS[$i]}" = "FAIL" ] && fails=$((fails+1))
  [ "${STATUS[$i]}" = "CANNOT VERIFY" ] && cannot=$((cannot+1))
done
echo
echo "passed: $(( ${#NAMES[@]} - fails - cannot ))/${#NAMES[@]}   failed: $fails   cannot verify: $cannot"
if [ $fails -gt 0 ]; then
  echo "AUDIT FAILED"; exit 1
fi
echo "AUDIT PASSED (CANNOT VERIFY entries need gitignored inputs; see the NOTE column)"
exit 0
