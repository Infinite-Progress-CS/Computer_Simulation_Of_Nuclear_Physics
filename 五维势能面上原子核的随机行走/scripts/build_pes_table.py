# -*- coding: utf-8 -*-
"""Build a regular 5D PES table for the 3QS + two-center model.

The default grid is intentionally small for smoke testing.  For a production
Moller-like table, increase the dimension counts substantially; the table is
stored on disk so subsequent random walks use only interpolation.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from hybrid_pes import CachedHybridMacroMicro
from pes_table import PESTable
from shape import Shape3QS


Z, N = 92, 144


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--nz-uni", type=int, default=48)
    ap.add_argument("--n-elong", type=int, default=5)
    ap.add_argument("--n-neck", type=int, default=5)
    ap.add_argument("--n-eta", type=int, default=5)
    ap.add_argument("--n-eps", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    grid = dict(
        elong=(0.2, 3.2, args.n_elong),
        neck=(0.05, 0.99, args.n_neck),
        eta=(-0.5, 0.5, args.n_eta),
        eps1=(-0.2, 0.4, args.n_eps),
        eps2=(-0.2, 0.4, args.n_eps),
    )

    pes = CachedHybridMacroMicro(
        Z, N,
        Nmax=args.nmax,
        nz_uni=args.nz_uni,
        shape_cls=Shape3QS,
        lam_so_p=None,
    )
    table = PESTable(pes, grid=grid).build(progress=True)

    out = args.out
    if out is None:
        out = os.path.join(
            PROJECT_ROOT, "results", "pes_table",
            f"pes_table_U236_nmax{args.nmax}.npz",
        )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    table.save(out)
    print(f"PES table saved to {out}")


if __name__ == "__main__":
    main()

