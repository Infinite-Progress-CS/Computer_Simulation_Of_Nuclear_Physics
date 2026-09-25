# -*- coding: utf-8 -*-
"""Lazy/interpolated five-dimensional PES table for Brownian random walks."""

import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _make_axis(lo, hi, n):
    if n <= 1:
        return np.array([0.5 * (lo + hi)])
    return np.linspace(float(lo), float(hi), int(n))


class PESTable:
    """A regular 5D (elong, neck, eta, eps1, eps2) PES lookup table.

    The table is built from an arbitrary callable ``pes.energy(q)`` and can be
    saved to/loaded from an .npz file.  Invalid 3QS shapes are stored as NaN.
    Interpolation uses linear interpolation with nearest-neighbour fallback at
    NaN or outside the sampled box.

    This is deliberately independent of the microscopic model, so the
    underlying ``pes`` can later be replaced by a Moller folded-Yukawa PES
    without touching the random-walk code.
    """

    def __init__(self, pes, grid=None, table_path=None):
        self.pes = pes
        if grid is None:
            # Keep this deliberately coarse for smoke tests; increase for a
            # production Moller-like table.
            grid = dict(
                elong=(0.2, 3.2, 21),
                neck=(0.05, 0.99, 15),
                eta=(-0.5, 0.5, 21),
                eps1=(-0.2, 0.4, 13),
                eps2=(-0.2, 0.4, 13),
            )
        self.grid = grid
        self.axes = {
            name: _make_axis(lo, hi, n)
            for name, (lo, hi, n) in grid.items()
        }
        self.names = ["elong", "neck", "eta", "eps1", "eps2"]
        self.shape = tuple(len(self.axes[name]) for name in self.names)
        self.values = None
        self._interp = None
        self._nearest = None
        self.table_path = table_path

        if table_path is not None and os.path.exists(table_path):
            self.load(table_path)

    # ------------------------------------------------------------------
    def build(self, progress=True):
        values = np.full(self.shape, np.nan, dtype=float)
        total = int(np.prod(self.shape))
        done = 0
        for ie, elong in enumerate(self.axes["elong"]):
            for in_, neck in enumerate(self.axes["neck"]):
                for ieta, eta in enumerate(self.axes["eta"]):
                    for ie1, eps1 in enumerate(self.axes["eps1"]):
                        for ie2, eps2 in enumerate(self.axes["eps2"]):
                            q = np.array([elong, neck, eta, eps1, eps2])
                            try:
                                values[ie, in_, ieta, ie1, ie2] = self.pes.energy(q)
                            except Exception:
                                pass
                            done += 1
                            if progress and done % 500 == 0:
                                print(f"  PES table build {done}/{total}", flush=True)
        self.values = values
        self._set_interpolators()
        return self

    def _set_interpolators(self):
        axes = [self.axes[name] for name in self.names]
        self._interp = RegularGridInterpolator(
            axes, self.values, method="linear", bounds_error=False,
            fill_value=np.nan,
        )
        self._nearest = RegularGridInterpolator(
            axes, self.values, method="nearest", bounds_error=False,
            fill_value=np.nan,
        )
        # 快路径预计算：均匀网格线性插值（等价 RegularGridInterpolator linear，
        # 但去掉其逐点 Python 校验开销，~3-10× 更快）。行走每秒要调用数十万次，
        # 这是产额大统计的瓶颈之一。
        self._lo = np.array([ax[0] for ax in axes], dtype=float)
        self._hi = np.array([ax[-1] for ax in axes], dtype=float)
        self._n = np.array([len(ax) for ax in axes], dtype=float)
        self._inv_step = (self._n - 1.0) / (self._hi - self._lo)
        import itertools
        self._offsets = np.array(
            list(itertools.product([0, 1], repeat=len(self.names))),
            dtype=np.int64)

    def _fast_linear(self, x):
        """均匀网格 5D 线性插值（单点）。NaN 角点会传染 NaN，由 energy 里 fallback。"""
        f = (x - self._lo) * self._inv_step
        f = np.clip(f, 0.0, self._n - 1.0 - 1e-9)
        i0 = f.astype(np.int64)
        fr = f - i0
        idx = i0[None, :] + self._offsets              # (32, 5)
        w = np.where(self._offsets == 0, 1.0 - fr[None, :], fr[None, :])
        w = w.prod(axis=1)
        v = self.values[tuple(idx.T)]
        return float(np.dot(w, v))

    def _fast_nearest(self, x):
        f = (x - self._lo) * self._inv_step
        f = np.clip(f, 0.0, self._n - 1.0)
        inr = np.rint(f).astype(np.int64)
        return float(self.values[tuple(inr)])

    def energy(self, q):
        if self.values is None:
            raise RuntimeError("PESTable has no values; call build() or load() first")
        x = np.asarray(q, dtype=float)
        val = self._fast_linear(x)
        if not np.isfinite(val):
            val = self._fast_nearest(x)
        return val

    # ------------------------------------------------------------------
    def save(self, path):
        data = {"values": self.values}
        for name in self.names:
            data[name] = self.axes[name]
        np.savez_compressed(path, **data)

    def load(self, path):
        with np.load(path, allow_pickle=False) as d:
            loaded = [d[name] for name in self.names]
            self.shape = tuple(len(a) for a in loaded)
            for name, a in zip(self.names, loaded):
                self.axes[name] = a
            self.values = d["values"]
        self._set_interpolators()
        self.table_path = path
        return self

