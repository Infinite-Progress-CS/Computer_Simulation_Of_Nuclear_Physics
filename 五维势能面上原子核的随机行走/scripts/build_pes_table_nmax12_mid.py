# -*- coding: utf-8 -*-
"""Build a compact Nmax=12 5D PES table around the physically relevant path."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from hybrid_pes import CachedHybridMacroMicro
from pes_table import PESTable
from shape import Shape3QS


Z, N = 92, 144


def main():
    pes = CachedHybridMacroMicro(
        Z, N,
        Nmax=12,
        nz_uni=48,
        shape_cls=Shape3QS,
        lam_so_p=None,
    )
    grid = dict(
        elong=(0.5, 2.6, 3),
        neck=(0.3, 0.99, 5),
        eta=(-0.3, 0.3, 7),
        eps1=(0.0, 0.3, 3),
        eps2=(0.0, 0.3, 3),
    )
    table = PESTable(pes, grid=grid).build(progress=True)
    out = os.path.join(
        PROJECT_ROOT, "results", "pes_table",
        "pes_table_U236_nmax12_eta7.npz",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    table.save(out)
    print(f"PES table saved to {out}")


if __name__ == "__main__":
    main()
