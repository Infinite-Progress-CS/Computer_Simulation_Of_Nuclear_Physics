# -*- coding: utf-8 -*-
"""Build the 3QS geometry table (Q, r_neck) for fast random-walk lookup."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from geometry_table import GeometryTable
from shape import Shape3QS


R0_LD = 1.16 * 236 ** (1.0 / 3.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-elong", type=int, default=16)
    ap.add_argument("--n-neck", type=int, default=20)
    ap.add_argument("--n-eta", type=int, default=21)
    ap.add_argument("--n-eps", type=int, default=7)
    args = ap.parse_args()

    grid = dict(
        elong=(0.2, 3.2, args.n_elong),
        neck=(0.05, 0.99, args.n_neck),
        eta=(-0.5, 0.5, args.n_eta),
        eps1=(-0.2, 0.4, args.n_eps),
        eps2=(-0.2, 0.4, args.n_eps),
    )

    shape = Shape3QS(R0_LD)
    table = GeometryTable(shape, grid=grid).build(progress=True)

    out = args.out
    if out is None:
        out = os.path.join(PROJECT_ROOT, "results", "pes_table",
                           "geometry_table_U236.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    table.save(out)
    print(f"Geometry table saved to {out}")


if __name__ == "__main__":
    main()
