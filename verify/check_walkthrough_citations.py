"""Resolve every `file:line` citation in CODE_WALKTHROUGH.md against the current tree.

CODE_WALKTHROUGH.md was deleted in 733ff0c and restored on 2026-08-01. It claims its
citations were verified against `f1bace0`; `src/` has changed since, and upstream citations
are pinned at 70805299 and should not drift. This checker measures the drift rather than
asserting it away, and it does NOT edit any citation.

Per citation, one of four outcomes:

  MATCH        the cited line range contains the identifier the prose names
  MOVED        the identifier exists in the file, but at a different line (new line reported)
  NO_ANCHOR    no identifier could be extracted from the prose, so nothing can be checked
               automatically -- needs a human, and is reported rather than passed
  FILE_GONE    the cited file does not exist in any of the trees searched

Anchors are extracted from the prose around the citation: a backticked identifier, or a bare
identifier immediately preceding the citation's parenthesis. Path resolution tries this repo
first, then the pinned clean upstream, then the patched working upstream, then the NLI code.

Exit status is 0 unless a file is missing outright -- drift is a measurement here, not a
failure, because the point of the pass is to quantify it before anyone hand-fixes anything.

Usage::

    python verify/check_walkthrough_citations.py
"""

import collections
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(REPO, "CODE_WALKTHROUGH.md")

# Search roots, in priority order.
ROOTS = [
    ("repo", REPO),
    ("upstream-pinned", os.path.join(REPO, ".upstream-clean")),
    ("upstream-working", os.environ.get("OPTIMALCE_UPSTREAM",
                                        os.path.expanduser("~/projects/optimalce"))),
    ("nli-code", os.environ.get("NLI_CODE", os.path.expanduser(
        "~/projects/neuron-explanations-nli/nli/code"))),
]

CITE = re.compile(r"`?([A-Za-z_][A-Za-z0-9_/]*\.py):(\d+)(?:-(\d+))?`?")
IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)`")


def candidates(path):
    """Every tree in which `path` resolves, in priority order.

    The walkthrough's own path table says its `compositional/` and `utils/` citations are
    against upstream @70805299 **plus** patches/0001, and that patch inserts ~21 lines into
    optimal.py. Resolving those against the PINNED tree therefore reports spurious drift, so
    every tree is tried and the one whose cited range actually holds the anchor wins. The
    winning tree is reported, which is itself the useful output: an upstream citation that
    only resolves in the patched tree is a citation against patched line numbers.
    """
    out = []
    for name, root in ROOTS:
        if not root or not os.path.isdir(root):
            continue
        cand = os.path.join(root, path)
        if os.path.isfile(cand):
            out.append((name, cand))
            continue
        # citations often drop the package prefix, e.g. `optimal.py` for compositional/optimal.py
        for sub in ("", "compositional", "utils", "src", "verify", "tests"):
            cand = os.path.join(root, sub, os.path.basename(path))
            if os.path.isfile(cand):
                out.append((name, cand))
                break
    return out


def anchor_for(text, start, end):
    """Best identifier anchor near a citation: backticked name, else word before '('."""
    before = text[max(0, start - 180):start]
    idents = IDENT.findall(before)
    idents = [i for i in idents if not i.endswith(".py") and len(i) > 2]
    if idents:
        return idents[-1].split(".")[-1]
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]{2,})\s*\($", before.rstrip())
    if m:
        return m.group(1)
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]{2,})\s*$", before.rstrip())
    if m and m.group(1) not in ("at", "in", "of", "see", "and", "the", "line", "lines"):
        return m.group(1)
    return None


def main():
    if not os.path.isfile(DOC):
        print(f"CODE_WALKTHROUGH.md not found at {DOC}")
        return 1
    text = open(DOC).read()

    seen, results = set(), []
    for m in CITE.finditer(text):
        path, a = m.group(1), int(m.group(2))
        b = int(m.group(3)) if m.group(3) else a
        key = (m.start(), path, a, b)
        if key in seen:
            continue
        seen.add(key)
        cands = candidates(path)
        if not cands:
            results.append(("FILE_GONE", path, a, b, None, None, None))
            continue
        anchor = anchor_for(text, m.start(), m.end())
        if anchor is None:
            results.append(("NO_ANCHOR", path, a, b, cands[0][0], None, None))
            continue
        # A citation MATCHES if its range holds the anchor in ANY tree it resolves in.
        matched = None
        for tree, real in cands:
            lines = open(real, errors="replace").read().splitlines()
            lo, hi = max(1, a), min(len(lines), b)
            if lo <= len(lines) and re.search(rf"\b{re.escape(anchor)}\b",
                                              "\n".join(lines[lo - 1:hi])):
                matched = tree
                break
        if matched:
            results.append(("MATCH", path, a, b, matched, anchor, None))
            continue
        tree, real = cands[0]
        lines = open(real, errors="replace").read().splitlines()
        hits = [i + 1 for i, ln in enumerate(lines)
                if re.search(rf"\b{re.escape(anchor)}\b", ln)]
        defs = [i + 1 for i, ln in enumerate(lines)
                if re.match(rf"\s*(def|class)\s+{re.escape(anchor)}\b", ln)]
        results.append(("MOVED", path, a, b, tree, anchor, (defs or hits)[:4] or None))

    counts = collections.Counter(r[0] for r in results)
    print(f"CODE_WALKTHROUGH.md — {len(results)} file:line citations resolved\n")
    for status in ("MATCH", "MOVED", "NO_ANCHOR", "FILE_GONE"):
        print(f"  {status:<10} {counts.get(status, 0):>3}")
    print()

    by_file = collections.defaultdict(collections.Counter)
    for st, path, *_ in results:
        by_file[path][st] += 1
    print(f"  {'file':<34} {'MATCH':>6} {'MOVED':>6} {'NO_ANC':>7} {'GONE':>5}")
    for path in sorted(by_file):
        c = by_file[path]
        print(f"  {path:<34} {c['MATCH']:>6} {c['MOVED']:>6} {c['NO_ANCHOR']:>7} "
              f"{c['FILE_GONE']:>5}")

    drifted = [r for r in results if r[0] == "MOVED"]
    if drifted:
        print(f"\n  --- MOVED ({len(drifted)}): cited range no longer holds the anchor ---")
        for _, path, a, b, tree, anchor, now in drifted:
            rng = f"{a}" if a == b else f"{a}-{b}"
            print(f"    {path}:{rng:<9} [{tree}] anchor `{anchor}` now at "
                  f"{now if now else 'NOT FOUND in file'}")

    noanc = [r for r in results if r[0] == "NO_ANCHOR"]
    if noanc:
        print(f"\n  --- NO_ANCHOR ({len(noanc)}): not auto-checkable, needs a human ---")
        for _, path, a, b, tree, _, _ in noanc:
            rng = f"{a}" if a == b else f"{a}-{b}"
            print(f"    {path}:{rng} [{tree}]")

    gone = [r for r in results if r[0] == "FILE_GONE"]
    if gone:
        print(f"\n  --- FILE_GONE ({len(gone)}) ---")
        for _, path, a, b, *_ in gone:
            print(f"    {path}:{a}")

    print("\nNOTE: citations were NOT edited by this pass. Drift is reported, not fixed.")
    print("RESULT:", "PASS (no missing files)" if not gone else "FAIL (missing files)")
    return 1 if gone else 0


if __name__ == "__main__":
    raise SystemExit(main())
