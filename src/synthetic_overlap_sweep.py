"""Synthetic overlap sweep — reproduce the OptimalCE frontier explosion mechanism
without any GPU, images, or real activations.

Thesis under test (from the D3 diary): the tractability of optimal search depends on
concept masks being ~disjoint (sum_elements ~= 1 -> unique dominates). Under token-like
overlap (sum_elements > 1 -> common dominates) the disjoint fast-path stops firing and
ceilings loosen, so the best-first frontier balloons.

We hold EVERYTHING fixed except concept overlap:
  - M elements, K concepts, per-concept coverage, and the neuron target S (num_hits) are constant.
  - The only knob is `pool`: the number of distinct elements that concept assignments land in.
    Large pool  -> few collisions -> sum_elements ~= 1 -> disjoint (vision regime).
    Small pool  -> many collisions -> sum_elements  > 1 -> overlap  (token regime).

Metric: peak frontier size (heap length), plus visited/expanded/estimated nodes and time.
A cap on peak frontier prevents a real OOM while still demonstrating the balloon.
"""

import argparse
import csv
import heapq
import sys
import time

import types

import numpy as np
import scipy.sparse as sparse
import torch

# The repo's import chain pulls in vision-only deps (detectron2, cityscapesscripts)
# through utils.dataset_utils. The optimal-search code path never calls them, so we
# stub them out to import `optimal` on a bare CPU container.
import importlib.abc
import importlib.machinery


class _Poison:
    """A stubbed vision symbol. Binding it is fine; USING it raises.

    `__getattr__` cannot raise directly: `from detectron2.data import DatasetCatalog`
    resolves through it, so raising there aborts the import chain the stub exists to
    satisfy (verified -- it fails at utils/dataset_utils.py:8). Returning a value that
    raises on use gives the fail-loudly behaviour without breaking import.

    Previously this returned the builtin `object`, so a stubbed `SomeClass()` silently
    produced a plain instance instead of failing.
    """

    __slots__ = ("_where",)

    def __init__(self, where):
        object.__setattr__(self, "_where", where)

    def _die(self, how):
        raise ImportError(
            f"stubbed vision dependency actually used: {object.__getattribute__(self, '_where')} "
            f"({how}). The NLP path is not supposed to touch detectron2/cityscapesscripts; "
            f"if you are on a vision path, install the real dependency.")

    def __getattr__(self, name):
        self._die(f"attribute {name!r}")

    def __call__(self, *a, **k):
        self._die("called")

    def __iter__(self):
        self._die("iterated")

    def __getitem__(self, k):
        self._die(f"subscripted [{k!r}]")

    def __repr__(self):
        return f"<stub {object.__getattribute__(self, '_where')}>"


class _AnyModule(types.ModuleType):
    """A stub module: any attribute (or submodule) resolves to a poison object."""

    __path__ = []  # marks it as a package so submodule imports are attempted

    def __getattr__(self, name):
        return _Poison(f"{self.__name__}.{name}")


_STUB_PREFIXES = ("detectron2", "cityscapesscripts")


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Fabricate any detectron2.*/cityscapesscripts.* module on demand. The optimal-search
    code path never calls into these vision-only deps; this just lets it import."""

    def find_spec(self, fullname, path, target=None):
        root = fullname.split(".")[0]
        if root in _STUB_PREFIXES:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return _AnyModule(spec.name)

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _StubFinder())

from compositional import optimal
from utils import optimal_utils  # noqa: F401  (import parity / sanity)


# --------------------------------------------------------------------------------------
# Frontier-size instrumentation + safety cap, via a heapq shim swapped into optimal.py.
# optimal.py does `import heapq` then `heapq.heappush(frontier, ...)`, so replacing
# `optimal.heapq` lets us watch every push without editing the repo.
# --------------------------------------------------------------------------------------
class _Halt(Exception):
    """Raised to stop a runaway level. `reason` is 'cap' (frontier) or 'time'."""

    def __init__(self, reason):
        self.reason = reason


class HeapProbe:
    def __init__(self, cap, time_budget=None):
        self.cap = cap
        self.time_budget = time_budget
        self.peak = 0
        self.start = time.time()
        self._n = 0

    def heappush(self, heap, item):
        heapq.heappush(heap, item)
        n = len(heap)
        if n > self.peak:
            self.peak = n
        if n > self.cap:
            raise _Halt("cap")
        # amortize the clock check (1 in 256 pushes) to keep overhead negligible
        self._n += 1
        if self.time_budget is not None and (self._n & 0xFF) == 0:
            if time.time() - self.start > self.time_budget:
                raise _Halt("time")

    def heappop(self, heap):
        return heapq.heappop(heap)

    def heapify(self, x):
        heapq.heapify(x)
        if len(x) > self.peak:
            self.peak = len(x)

    def __getattr__(self, name):
        # Delegate anything we don't override (e.g. nsmallest) to the real heapq module.
        return getattr(heapq, name)


class StubConfig:
    """compute_optimal_explanations only touches these three accessors."""

    def __init__(self, length, M):
        self._length = length
        self._M = M

    def get_length(self):
        return self._length

    def get_mask_shape(self):
        return (1, self._M)

    def get_device(self):
        return "cpu"


def build_base(M, K, active_frac, rng):
    """Vision-like DISJOINT base: each active element belongs to exactly one concept.
    Returns dense (K, M) bool and the per-element base concept id (or -1 if inactive)."""
    n_active = int(active_frac * M)
    active_idx = rng.choice(M, size=n_active, replace=False)
    owner = np.full(M, -1, dtype=np.int64)
    owner[active_idx] = rng.integers(0, K, size=n_active)
    dense = np.zeros((K, M), dtype=bool)
    for e in active_idx:
        dense[owner[e], e] = True
    return dense, owner


def add_overlap(base_dense, owner, K, p_add, extra, rng):
    """Introduce token-like overlap: each active element, with prob p_add, joins `extra`
    additional random concepts. p_add=0 -> disjoint (vision); higher -> common dominates.
    Base memberships are preserved, so any signal defined on the base persists."""
    dense = base_dense.copy()
    active_idx = np.where(owner >= 0)[0]
    for e in active_idx:
        if rng.random() < p_add:
            others = rng.choice(K, size=min(extra, K), replace=False)
            dense[others, e] = True
    masks = [sparse.csr_matrix(dense[c].reshape(1, dense.shape[1])) for c in range(K)]
    return masks, dense


def compute_quantities(dense, M):
    """Replicates mask_utils.get_dataset_quantities: common/unique/uncoverable (1, M) bool."""
    sum_elements = dense.astype(np.int32).sum(axis=0).reshape(1, M)
    common = torch.from_numpy(sum_elements > 1)
    unique = torch.from_numpy(sum_elements == 1)
    uncoverable = torch.from_numpy(sum_elements == 0)
    return common, unique, uncoverable, sum_elements


def compute_disjoint_info(dense, K):
    """(K, K) bool: concepts a,b disjoint iff their masks never co-occur on an element."""
    d = np.ones((K, K), dtype=bool)
    for a in range(K):
        for b in range(K):
            if a == b:
                d[a, b] = False
            else:
                d[a, b] = not bool((dense[a] & dense[b]).any())
    return torch.from_numpy(d)


def run_level(*, M, K, base_dense, owner, p_add, extra, length, cap, neuron_bits, seed,
              time_budget=None, beam_cap=None):
    rng = np.random.default_rng(seed + 7)
    masks, dense = add_overlap(base_dense, owner, K, p_add, extra, rng)
    common, unique, uncoverable, sum_elements = compute_quantities(dense, M)
    masks_info = (common, unique, uncoverable)
    disjoint_info = compute_disjoint_info(dense, K)

    # Neuron target S is FIXED across levels (same seed offset), so num_hits is constant.
    bitmaps = torch.from_numpy(neuron_bits.reshape(1, M))

    # Overlap diagnostics
    active = int((sum_elements > 0).sum())
    common_frac = float((sum_elements > 1).sum()) / max(active, 1)
    mean_overlap = float(sum_elements[sum_elements > 0].mean()) if active else 0.0
    n_disjoint_pairs = int((disjoint_info.sum().item() - 0))  # off-diagonal Trues

    probe = HeapProbe(cap, time_budget=time_budget)
    optimal.heapq = probe  # swap in instrumentation
    optimal.MAX_FRONTIER_SIZE = beam_cap  # None => exact optimal; int => beam fallback
    cfg = StubConfig(length, M)

    t0 = time.time()
    halt_reason = ""
    try:
        best_label, best_iou, visited, expanded, estimated = (
            optimal.compute_optimal_explanations(
                bitmaps=bitmaps,
                masks=masks,
                masks_info=masks_info,
                disjoint_info=disjoint_info,
                config=cfg,
            )
        )
    except _Halt as h:
        halt_reason = h.reason
        best_iou, visited, expanded, estimated = (float("nan"), -1, -1, -1)
    finally:
        optimal.heapq = heapq  # restore
        optimal.MAX_FRONTIER_SIZE = None  # restore exact-optimal default
        dt = time.time() - t0
    print()  # clear the \r status line

    return {
        "K": K,
        "length": length,
        "beam_cap": beam_cap if beam_cap is not None else "-",
        "p_add": p_add,
        "mean_overlap": round(mean_overlap, 3),
        "common_frac": round(common_frac, 3),
        "disjoint_pairs": n_disjoint_pairs,
        "peak_frontier": probe.peak,
        "halted": halt_reason or "no",
        "visited": visited,
        "expanded": expanded,
        "estimated": estimated,
        "best_iou": round(best_iou, 4) if best_iou == best_iou else None,
        "time_s": round(dt, 2),
    }


def make_neuron(base_dense, M, noise, seed):
    """Fixed neuron = a 3-concept OR target over the base + small label noise. In a disjoint
    build this is (almost) perfectly explainable -> strong floor -> fast, like Cityscapes."""
    target = base_dense[0] | base_dense[1] | base_dense[2]
    nrng = np.random.default_rng(seed + 999)
    flip = nrng.random(M) < noise
    return np.where(flip, ~target, target)


def print_summary(rows):
    cols = ["K", "length", "beam_cap", "p_add", "mean_overlap", "common_frac",
            "peak_frontier", "visited", "expanded", "estimated", "best_iou", "time_s", "halted"]
    widths = {c: max(len(c), 10) for c in cols}
    print("\n=== SUMMARY ===")
    print(" ".join(f"{c:>{widths[c]}}" for c in cols))
    for r in rows:
        print(" ".join(f"{str(r.get(c, '')):>{widths[c]}}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["overlap", "scale", "beam"], default="overlap")
    ap.add_argument("--M", type=int, default=2048, help="number of elements (tokens)")
    ap.add_argument("--K", type=int, default=15, help="number of concepts")
    ap.add_argument("--active_frac", type=float, default=0.7, help="fraction of elements covered by the disjoint base")
    ap.add_argument("--extra", type=int, default=2, help="extra concepts an overlapping element joins")
    ap.add_argument("--length", type=int, default=3, help="max formula length")
    ap.add_argument("--cap", type=int, default=50000, help="peak frontier safety cap")
    ap.add_argument("--beam_cap", type=int, default=None, help="size-bounded beam fallback: max frontier nodes (None=exact optimal)")
    ap.add_argument("--time_budget", type=float, default=None, help="per-level wall-clock budget (s)")
    ap.add_argument("--noise", type=float, default=0.05, help="neuron label noise (bit flips)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p_adds", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    # scale mode: grid over sizes, each at a disjoint control (p_add=0) and a token level.
    ap.add_argument("--K_list", type=int, nargs="+", default=[15, 30, 50])
    ap.add_argument("--length_list", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--token_p_add", type=float, default=1.0, help="overlap level used as the token regime in scale/beam mode")
    # beam mode: fix one hard (K, length, token) case and sweep the beam width.
    ap.add_argument("--beam_list", type=str, nargs="+", default=["none", "200", "500", "1000", "2000"],
                    help="beam widths to sweep in beam mode; 'none' = exact optimal")
    ap.add_argument("--out", type=str, default="experiments/overlap_sweep_results.csv")
    args = ap.parse_args()

    rows = []
    if args.mode == "overlap":
        # Build the disjoint base ONCE; all overlap levels extend it so the neuron signal persists.
        base_dense, owner = build_base(args.M, args.K, args.active_frac,
                                       np.random.default_rng(args.seed))
        neuron_bits = make_neuron(base_dense, args.M, args.noise, args.seed)
        print(f"[overlap] M={args.M} K={args.K} extra={args.extra} length={args.length} "
              f"num_hits={int(neuron_bits.sum())} cap={args.cap} time_budget={args.time_budget}\n",
              flush=True)
        for p_add in args.p_adds:
            print(f"--- p_add={p_add} ---", flush=True)
            row = run_level(M=args.M, K=args.K, base_dense=base_dense, owner=owner,
                            p_add=p_add, extra=args.extra, length=args.length, cap=args.cap,
                            neuron_bits=neuron_bits, seed=args.seed, time_budget=args.time_budget,
                            beam_cap=args.beam_cap)
            rows.append(row)
            print(row, flush=True)
            print(flush=True)
    elif args.mode == "beam":  # fix a hard case, sweep beam width to show terminate/time/IoU tradeoff
        base_dense, owner = build_base(args.M, args.K, args.active_frac,
                                       np.random.default_rng(args.seed))
        neuron_bits = make_neuron(base_dense, args.M, args.noise, args.seed)
        print(f"[beam] M={args.M} K={args.K} length={args.length} p_add={args.token_p_add} "
              f"num_hits={int(neuron_bits.sum())} cap={args.cap} time_budget={args.time_budget}\n",
              flush=True)
        for bw in args.beam_list:
            beam_cap = None if bw.lower() == "none" else int(bw)
            print(f"--- beam_cap={bw} ---", flush=True)
            row = run_level(M=args.M, K=args.K, base_dense=base_dense, owner=owner,
                            p_add=args.token_p_add, extra=args.extra, length=args.length,
                            cap=args.cap, neuron_bits=neuron_bits, seed=args.seed,
                            time_budget=args.time_budget, beam_cap=beam_cap)
            rows.append(row)
            print(row, flush=True)
            print(flush=True)
    else:  # scale: sweep size, contrast disjoint control vs token regime at each size
        print(f"[scale] M={args.M} extra={args.extra} cap={args.cap} "
              f"time_budget={args.time_budget} token_p_add={args.token_p_add}\n", flush=True)
        for K in args.K_list:
            base_dense, owner = build_base(args.M, K, args.active_frac,
                                           np.random.default_rng(args.seed))
            neuron_bits = make_neuron(base_dense, args.M, args.noise, args.seed)
            for length in args.length_list:
                for p_add in (0.0, args.token_p_add):
                    tag = "disjoint" if p_add == 0.0 else "token"
                    print(f"--- K={K} length={length} {tag}(p_add={p_add}) ---", flush=True)
                    row = run_level(M=args.M, K=K, base_dense=base_dense, owner=owner,
                                    p_add=p_add, extra=args.extra, length=length, cap=args.cap,
                                    neuron_bits=neuron_bits, seed=args.seed,
                                    time_budget=args.time_budget, beam_cap=args.beam_cap)
                    rows.append(row)
                    print(row, flush=True)
                    print(flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    print_summary(rows)


if __name__ == "__main__":
    sys.exit(main())
