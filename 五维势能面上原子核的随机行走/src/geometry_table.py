"""
geometry_table.py — 3QS 几何量（四极矩 Q、颈半径 r_neck）5D 网格表
====================================================================
Q(q) 和 r_neck(q) 是纯几何量（只依赖 shape.profile→build 的剖面，不依赖 PES），
但行走每步都要调它们（投影有效性判断 + 断裂判据），而 Shape3QS.build 的非对称
least_squares 占单步 ~96% 时间。预先建 5D 表 + 线性插值，把 least_squares 完全
移出行走热循环，单 walk 从 ~54s 压到 ~1s 量级。

无效形状（3QS 非对称 + 厚颈无解）存 NaN；插值保持 NaN（不做 nearest fallback），
供 brownian_yield 的投影回退判断形状有效性。
"""
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _make_axis(lo, hi, n):
    return np.linspace(float(lo), float(hi), int(n))


def _compute(q, shape, n=300):
    """一次 build 同时返回 (Q, r_neck)。Q 单位 fm^5，r_neck 单位 fm。

    等价于 quadrupole_moment 与 neck_radius_fm 的合并（共用一次 profile→build）。
    无效形状由 shape.build 抛异常，调用方 catch 后存 NaN。
    """
    z, rho = shape.profile(q, n=n)
    rho_max = rho.max()
    r_neck = rho_max
    for i in range(1, n - 1):
        if rho[i] <= rho[i - 1] and rho[i] <= rho[i + 1]:
            r_neck = min(r_neck, rho[i])
    integrand = z ** 2 * rho ** 2 - rho ** 4 / 4.0
    trapz = getattr(np, 'trapezoid', np.trapz)
    Q = float(2.0 * np.pi * trapz(integrand, z))
    return Q, float(r_neck)


class GeometryTable:
    """Q(q) 与 r_neck(q) 的 5D 规则网格表（线性插值，NaN 保持）。"""

    def __init__(self, shape, grid=None, table_path=None):
        self.shape = shape
        if grid is None:
            # 覆盖 brownian_yield 默认 q_min/q_max：
            #   elong 0.2-3.2, neck 0.05-0.99, eta -0.5-0.5, eps -0.2-0.4
            grid = dict(
                elong=(0.2, 3.2, 16),
                neck=(0.05, 0.99, 20),
                eta=(-0.5, 0.5, 21),
                eps1=(-0.2, 0.4, 7),
                eps2=(-0.2, 0.4, 7),
            )
        self.grid = grid
        self.axes = {name: _make_axis(lo, hi, n)
                     for name, (lo, hi, n) in grid.items()}
        self.names = ["elong", "neck", "eta", "eps1", "eps2"]
        self.shape_ = tuple(len(self.axes[name]) for name in self.names)
        self.Q = None
        self.Rneck = None
        self._q_interp = None
        self._r_interp = None
        self._q_near = None
        self._r_near = None
        self.table_path = table_path
        if table_path is not None and os.path.exists(table_path):
            self.load(table_path)

    # ------------------------------------------------------------------
    def build(self, progress=True):
        Q = np.full(self.shape_, np.nan)
        Rn = np.full(self.shape_, np.nan)
        total = int(np.prod(self.shape_))
        done = 0
        for ie, elong in enumerate(self.axes["elong"]):
            for in_, neck in enumerate(self.axes["neck"]):
                for ieta, eta in enumerate(self.axes["eta"]):
                    for ie1, eps1 in enumerate(self.axes["eps1"]):
                        for ie2, eps2 in enumerate(self.axes["eps2"]):
                            q = [elong, neck, eta, eps1, eps2]
                            try:
                                Q[ie, in_, ieta, ie1, ie2], \
                                    Rn[ie, in_, ieta, ie1, ie2] = _compute(q, self.shape)
                            except Exception:
                                pass
                            done += 1
                            if progress and done % 20000 == 0:
                                print(f"  geom table build {done}/{total}", flush=True)
        self.Q = Q
        self.Rneck = Rn
        self._set_interpolators()
        return self

    def _set_interpolators(self):
        axes = [self.axes[name] for name in self.names]
        self._q_interp = RegularGridInterpolator(
            axes, self.Q, method="linear", bounds_error=False, fill_value=np.nan)
        self._r_interp = RegularGridInterpolator(
            axes, self.Rneck, method="linear", bounds_error=False, fill_value=np.nan)
        self._q_near = RegularGridInterpolator(
            axes, self.Q, method="nearest", bounds_error=False, fill_value=np.nan)
        self._r_near = RegularGridInterpolator(
            axes, self.Rneck, method="nearest", bounds_error=False, fill_value=np.nan)

    # ------------------------------------------------------------------
    def is_valid(self, q):
        """形状是否可解（线性插值有限）。用于投影回退 / η 子步的有效性判断。"""
        x = np.asarray(q, dtype=float)
        return bool(np.isfinite(self._q_interp(x)[0]))

    def quadrupole(self, q):
        """Q 值（fm^5），线性插值，NaN 时退 nearest 保证有限。"""
        x = np.asarray(q, dtype=float)
        val = float(self._q_interp(x)[0])
        if not np.isfinite(val):
            val = float(self._q_near(x)[0])
        return val

    def neck_radius(self, q):
        """颈半径（fm），线性插值，NaN 时退 nearest 保证有限。"""
        x = np.asarray(q, dtype=float)
        val = float(self._r_interp(x)[0])
        if not np.isfinite(val):
            val = float(self._r_near(x)[0])
        return val

    # ------------------------------------------------------------------
    def save(self, path):
        data = {"Q": self.Q, "Rneck": self.Rneck}
        for name in self.names:
            data[name] = self.axes[name]
        np.savez_compressed(path, **data)

    def load(self, path):
        with np.load(path, allow_pickle=False) as d:
            for name in self.names:
                self.axes[name] = d[name]
            self.shape_ = tuple(len(self.axes[name]) for name in self.names)
            self.Q = d["Q"]
            self.Rneck = d["Rneck"]
        self._set_interpolators()
        self.table_path = path
        return self
