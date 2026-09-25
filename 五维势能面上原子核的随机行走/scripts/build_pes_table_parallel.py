# -*- coding: utf-8 -*-
"""Parallel 5D PES table builder (3QS + two-center Woods-Saxon).

The two-center single-particle diagonalization releases the GIL (numpy/scipy),
so a multiprocessing.Pool gives near-linear speedup on the embarrassingly
parallel grid evaluation.  Invalid 3QS shapes are stored as NaN.

Usage:
  python scripts/build_pes_table_parallel.py --nmax 8 --n-workers 12
"""

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pes_table import PESTable

Z, N = 92, 144

# 每个 worker 进程持有的 PES（Pool initializer 里初始化一次）
_PES = None


def _init_worker(nmax, nz_uni):
    global _PES
    from hybrid_pes import HybridMacroMicro
    from shape import Shape3QS
    _PES = HybridMacroMicro(Z, N, Nmax=nmax, nz_uni=nz_uni,
                            shape_cls=Shape3QS, lam_so_p=None)


def _eval_one(q):
    try:
        return float(_PES.energy(q))
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--nz-uni", type=int, default=48)
    ap.add_argument("--n-workers", type=int, default=12)
    ap.add_argument("--n-elong", type=int, default=13)
    ap.add_argument("--n-neck", type=int, default=10)
    ap.add_argument("--n-eta", type=int, default=13)
    ap.add_argument("--n-eps", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    grid = dict(
        elong=(0.5, 2.9, args.n_elong),
        neck=(0.1, 0.99, args.n_neck),
        eta=(-0.3, 0.3, args.n_eta),
        eps1=(-0.2, 0.4, args.n_eps),
        eps2=(-0.2, 0.4, args.n_eps),
    )

    # 轴（与 PESTable 相同约定）
    from pes_table import _make_axis
    axes = {name: _make_axis(lo, hi, n) for name, (lo, hi, n) in grid.items()}
    names = ["elong", "neck", "eta", "eps1", "eps2"]
    shape = tuple(len(axes[n]) for n in names)
    total = int(np.prod(shape))
    print(f"grid {shape} = {total} points, nmax={args.nmax}, "
          f"workers={args.n_workers}", flush=True)

    # 展平所有 q
    qs = []
    for ie, elong in enumerate(axes["elong"]):
        for in_, neck in enumerate(axes["neck"]):
            for ieta, eta in enumerate(axes["eta"]):
                for ie1, eps1 in enumerate(axes["eps1"]):
                    for ie2, eps2 in enumerate(axes["eps2"]):
                        qs.append(np.array([elong, neck, eta, eps1, eps2]))

    import multiprocessing as mp
    n_workers = min(args.n_workers, total)
    ctx = mp.get_context("spawn")
    vals = []
    with ctx.Pool(n_workers, initializer=_init_worker,
                  initargs=(args.nmax, args.nz_uni)) as pool:
        for i, v in enumerate(pool.imap(_eval_one, qs, chunksize=8), 1):
            vals.append(v)
            if i % 5000 == 0 or i == total:
                print(f"  built {i}/{total} "
                      f"({i/total*100:.0f}%)", flush=True)

    values = np.array(vals, dtype=float).reshape(shape)
    n_nan = int(np.isnan(values).sum())
    print(f"done: {n_nan}/{total} NaN ({n_nan/total*100:.1f}%)", flush=True)

    # 保存（复用 PESTable 的格式）
    table = PESTable(pes=None, grid=grid)
    table.axes = axes
    table.values = values
    table._set_interpolators()
    out = args.out
    if out is None:
        out = os.path.join(
            PROJECT_ROOT, "results", "pes_table",
            f"pes_table_U236_nmax{args.nmax}_fine.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    table.save(out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
