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
#   UPSTREAM_CLEAN        clean upstream at the pinned SHA, for the patch no-op check.
#                         If unset, it is cloned automatically into .upstream-clean/
#                         (the upstream repo is public; no credentials needed).
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

# Check 3 needs a second, CLEAN copy of upstream to compare against, so it is driven here
# rather than by one script. It self-provisions: the upstream repo is public and anonymously
# clonable, so if UPSTREAM_CLEAN is unset we clone it at the pinned SHA into a repo-local
# ignored directory. This is the load-bearing claim in the repo -- a skipped check here looks
# like avoidance, so it should only skip when the network genuinely is not there.
hdr "3. patch is a no-op when MAX_FRONTIER_SIZE=None"
PINNED="$(cat UPSTREAM)"
CLEAN_DEFAULT=".upstream-clean"
CLONE_CMD="git clone https://github.com/aiea-lab/optimal-compositional-explanations $CLEAN_DEFAULT && git -C $CLEAN_DEFAULT checkout $PINNED"

if [ -z "${UPSTREAM_CLEAN:-}" ]; then
  if [ -d "$CLEAN_DEFAULT/compositional" ]; then
    UPSTREAM_CLEAN="$CLEAN_DEFAULT"
    echo "  using existing $CLEAN_DEFAULT"
  else
    echo "  provisioning a clean upstream at $PINNED into $CLEAN_DEFAULT ..."
    rm -rf "$CLEAN_DEFAULT"
    if GIT_TERMINAL_PROMPT=0 git clone -q \
         https://github.com/aiea-lab/optimal-compositional-explanations "$CLEAN_DEFAULT" 2>/dev/null \
       && git -C "$CLEAN_DEFAULT" checkout -q "$PINNED" 2>/dev/null; then
      UPSTREAM_CLEAN="$CLEAN_DEFAULT"
      echo "  cloned and checked out $PINNED"
    else
      echo "  could not clone (no network, or the upstream repo moved)."
      echo "  Run this, then re-run this script:"
      echo
      echo "    $CLONE_CMD"
      echo
      echo "  Or point at an existing checkout:  UPSTREAM_CLEAN=/path/to/clean ./verify/run_all.sh"
      record "3. patch no-op" "CANNOT VERIFY" "clone failed; see the command printed above"
      UPSTREAM_CLEAN=""
    fi
  fi
fi

if [ -n "${UPSTREAM_CLEAN:-}" ]; then
  got="$(git -C "$UPSTREAM_CLEAN" rev-parse HEAD 2>/dev/null || echo unknown)"
  if [ "$got" != "$PINNED" ]; then
    echo "  clean tree is at $got, expected $PINNED"
    record "3. patch no-op" "FAIL" "clean tree at wrong commit"
  else
    if [ ! -e "results/acts2k_trained_a0.2.npz" ]; then
      echo "  activations absent -> comparing on the proxy neuron instead (still a valid no-op test)"
    fi
    A="$(OPTIMALCE_UPSTREAM="$OPTIMALCE_UPSTREAM" "$PY" verify/check_patch_noop.py 2>&1 | grep -E 'target|formula|best_iou|visited|expanded|estimated|peak')"
    B="$(OPTIMALCE_UPSTREAM="$UPSTREAM_CLEAN"     "$PY" verify/check_patch_noop.py 2>&1 | grep -E 'target|formula|best_iou|visited|expanded|estimated|peak')"
    echo "$A" | sed 's/^/  patched: /'
    if [ "$A" = "$B" ]; then
      echo "  identical to clean upstream @ $PINNED"; record "3. patch no-op" "PASS" ""
    else
      echo "  DIFFERS from clean upstream"; diff <(echo "$A") <(echo "$B") | sed 's/^/    /'
      record "3. patch no-op" "FAIL" "outputs differ"
    fi
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

# Numbering follows VERIFICATION.md, whose table is canonical: check 9 is "someone else can
# run this" (REPRODUCE.md 6b, prose only), and the brute-force oracle is check 10. This label
# read "9." until 2026-08-01, which is where the mis-citation of the oracle as check 9 came
# from; every .md citation already said 10.
run_check "10a. brute-force oracle (length $ORACLE_LENGTH, 3 cases)" \
  "ORACLE_LENGTH=$ORACLE_LENGTH \"$PY\" tests/test_bruteforce_oracle.py" \
  "$ACTS $ACTS_U" "results/*.npz (REPRODUCE.md step 3)"

# 10b widens 10a from 3 cases to all 27 length-3 pairs. 10a asserts search == in-grammar max
# and passes; it ran too small a sample to see that the assertion fails on 2 of the 27.
run_check "10b. in-grammar optimality, all 27 pairs" \
  "\"$PY\" tests/test_bruteforce_oracle_all27.py" \
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
