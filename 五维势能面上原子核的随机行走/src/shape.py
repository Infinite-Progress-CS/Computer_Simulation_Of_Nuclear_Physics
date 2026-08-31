"""
shape.py — 原子核形状参数化（3QS 三二次曲面）
====================================================
从原单文件搬出，零改动。轴对称形状 ρ(z)，由 5 个物理参数描述：
  q1 = elong 拉长 = 两碎片中心间距 / R0
  q2 = neck  颈部 = 中间顶点半径 ρ_v / min(a1,a2) ∈ (0,1]
  q3 = eta   质量不对称 = (M_H − M_L)/(M_H + M_L) ∈ (−1,1)
  q4 = eps1  左碎片形变（Nilsson 四极 ε_f1）
  q5 = eps2  右碎片形变（Nilsson 四极 ε_f2）
"""

import numpy as np
from scipy import integrate, optimize


def spheroid_axes(eps):
    """Nilsson 四极形变 ε → 无量纲半轴 (a, c)，体积守恒 a²c = 1。

    a = 垂直轴，c = 对称轴。ε>0 → 长椭球 (prolate, c>a)；ε<0 → 扁椭球 (oblate)。
    """
    r = (3.0 - 2.0 * eps) / (3.0 + eps)   # a/c
    return r ** (1.0 / 3.0), r ** (-2.0 / 3.0)


def spheroid_bs_bc(eps):
    """均匀带电/表面能椭球的形变函数 (B_s, B_c)，球 = 1。

    长椭球 (prolate, eps>0) 精确公式：
      e   = √(1 − r²)，r = a/c
      B_s = (1/2) r^{2/3} [1 + arcsin(e)/(e r)]
      B_c = r^{2/3} · ln((1+e)/(1−e)) / (2e)
    e→0 时用级数展开消去 0/0。扁椭球 (eps<0) 用对应 arcsinh/arctan 公式。
    """
    r = (3.0 - 2.0 * eps) / (3.0 + eps)
    r23 = r ** (2.0 / 3.0)
    if eps >= 0.0:
        e2 = 1.0 - r * r
        if e2 <= 0.0:
            return 1.0, 1.0
        e = np.sqrt(e2)
        if e < 1e-6:
            asin_e = 1.0 + e * e / 6.0
            ln_e = 1.0 + e * e / 3.0
        else:
            asin_e = np.arcsin(e) / e
            ln_e = np.log((1.0 + e) / (1.0 - e)) / (2.0 * e)
        Bs = 0.5 * r23 * (1.0 + asin_e / r)
        Bc = r23 * ln_e
    else:
        e2 = r * r - 1.0
        if e2 <= 0.0:
            return 1.0, 1.0
        e = np.sqrt(e2)
        if e < 1e-6:
            asinh_e = 1.0 - e * e / 6.0
            atan_e = 1.0 - e * e / 3.0
        else:
            asinh_e = np.arcsinh(e) / e
            atan_e = np.arctan(e) / e
        Bs = 0.5 * r23 * (1.0 + asinh_e / r)
        Bc = r23 * atan_e
    return float(Bs), float(Bc)


class ShapeSpheroid:
    """单中心椭球形状 ρ(z)（碎片形变用），接口与 Shape3QS 一致。

    profile([eps], n) → (z, ρ)，单位 fm。体积守恒（a²c = R0³）。
    """

    def __init__(self, R0=7.17):
        self.R0 = R0

    def profile(self, q, n=300):
        eps = float(np.asarray(q).ravel()[0])
        a, c = spheroid_axes(eps)
        z_d = np.linspace(-c, c, n)
        rho_d = a * np.sqrt(np.maximum(1.0 - (z_d / c) ** 2, 0.0))
        return z_d * self.R0, rho_d * self.R0


class Shape3QS:
    """轴对称形状 ρ(z)，由 5 个物理参数 (elong, neck, eta, eps1, eps2) 描述。

    左球体 + 中间二次曲面 ρ²=ρ_v²+C(z−z_v)² + 右球体，切线拼接（判别式法），
    体积守恒（∫ρ²dz = 4/3 单位球），质心居中。返回无量纲 ρ²(z)，物理半径 = R0·√ρ²。
    """

    def __init__(self, R0=7.17):
        # R0 = r0·A^{1/3}（U-236，r0=1.16 fm → ≈7.17 fm），与 liquid_drop 一致；
        # Woods-Saxon 会传入 R_ws = 1.275·A^{1/3} ≈ 7.88 fm 覆盖。
        self.R0 = R0

    def axis_ratio(self, eps):
        """Nilsson 四极形变 ε → 轴比 a/c = (3−2ε)/(3+ε)"""
        return (3.0 - 2.0 * eps) / (3.0 + eps)

    # ---------------- 对称解析解（η=0, ε1=ε2）----------------
    def _tangent_sym(self, a, c, D, rho_v):
        """返回 (z_v=0, C, z1=−w, z2=+w)；无效返回 None。"""
        Dmin = 2.0 * c * np.sqrt(max(0.0, 1.0 - rho_v ** 2 / a ** 2))
        if D <= Dmin or rho_v <= 1e-12:
            return None
        w = D / 2.0 - 2.0 * (a ** 2 - rho_v ** 2) * c ** 2 / (a ** 2 * D)
        if w <= 0:
            return None
        C = a ** 2 * (D / 2.0 - w) / (c ** 2 * w)
        return 0.0, C, -w, +w

    def build(self, q):
        elong, neck, eta, eps1, eps2 = q
        r1 = self.axis_ratio(eps1)
        r2 = self.axis_ratio(eps2)
        w1 = (1.0 - eta) / 2.0
        w2 = (1.0 + eta) / 2.0
        c1 = (w1 / r1 ** 2) ** (1.0 / 3.0); a1 = r1 * c1
        c2 = (w2 / r2 ** 2) ** (1.0 / 3.0); a2 = r2 * c2
        l1 = -elong / 2.0; l2 = +elong / 2.0
        rho_v = neck * min(a1, a2)

        # 近球极限：直接返回单位球
        if elong < 1e-6 and abs(eta) < 1e-9 and abs(eps1) < 1e-9 and abs(eps2) < 1e-9:
            return self._make(lambda z: 1.0 - z * z, -1.0, 1.0)

        g1 = a1 ** 2 / c1 ** 2
        g2 = a2 ** 2 / c2 ** 2

        # 对称解析解（用平均参数）作为数值解初值
        a_sym = (a1 + a2) / 2.0
        c_sym = (c1 + c2) / 2.0
        sym = self._tangent_sym(a_sym, c_sym, elong, rho_v)

        if abs(eta) < 1e-9 and abs(eps1 - eps2) < 1e-9:
            # 纯对称：直接用解析解（用实际 a1,c1 精确）
            exact = self._tangent_sym(a1, c1, elong, rho_v)
            if exact is None:
                raise RuntimeError(f"对称无解 q={q}")
            z_v, C, z1, z2 = exact
        else:
            # 非对称：判别式法数值求解，对称解作初值
            def resid(x):
                z_v, C = x
                A1 = C + g1; B1 = -2.0 * C * z_v - 2.0 * g1 * l1
                Cc1 = rho_v ** 2 + C * z_v ** 2 - a1 ** 2 + g1 * l1 ** 2
                D1 = B1 ** 2 - 4.0 * A1 * Cc1
                A2 = C + g2; B2 = -2.0 * C * z_v - 2.0 * g2 * l2
                Cc2 = rho_v ** 2 + C * z_v ** 2 - a2 ** 2 + g2 * l2 ** 2
                D2 = B2 ** 2 - 4.0 * A2 * Cc2
                return np.array([D1, D2])

            guesses = []
            if sym is not None:
                guesses.append([sym[0], sym[1]])
            guesses += [[0.0, 1.0], [0.0, 2.0], [0.0, 4.0], [0.0, 0.5], [0.0, -1.0]]
            sol_found = None
            for g in guesses:
                try:
                    sol = optimize.least_squares(resid, g, xtol=1e-13, ftol=1e-13,
                                                 gtol=1e-13, max_nfev=2000)
                    if sol.success and np.max(np.abs(resid(sol.x))) < 1e-7:
                        z_v, C = sol.x
                        A1 = C + g1; B1 = -2.0 * C * z_v - 2.0 * g1 * l1
                        A2 = C + g2; B2 = -2.0 * C * z_v - 2.0 * g2 * l2
                        z1 = -B1 / (2.0 * A1); z2 = -B2 / (2.0 * A2)
                        if l1 < z1 < z2 < l2:
                            sol_found = (z_v, C, z1, z2)
                            break
                except Exception:
                    continue
            if sol_found is None:
                raise RuntimeError(f"非对称无解 q={q}")
            z_v, C, z1, z2 = sol_found

        # 三段二次曲面 ρ²(z)（无量纲）
        def rho2_raw(z):
            z = np.asarray(z, dtype=float)
            p1 = a1 ** 2 * (1.0 - (z - l1) ** 2 / c1 ** 2)
            pm = rho_v ** 2 + C * (z - z_v) ** 2
            p2 = a2 ** 2 * (1.0 - (z - l2) ** 2 / c2 ** 2)
            return np.where(z < z1, p1, np.where(z < z2, pm, p2))

        return self._make(rho2_raw, l1 - c1, l2 + c2)

    def _make(self, rho2_raw, zL0, zR0):
        """体积守恒缩放 + 质心居中，返回 dict(zL,zR,rho2)。"""
        zz = np.linspace(zL0, zR0, 6000)
        rr2 = np.maximum(rho2_raw(zz), 0.0)
        V = integrate.trapezoid(rr2, zz)
        s = (4.0 / 3.0 / V) ** (1.0 / 3.0)
        zbar = integrate.trapezoid(zz * rr2, zz) / V
        shift = -zbar * s

        def rho2(z):
            zs = (np.asarray(z, dtype=float) - shift) / s
            return np.maximum(rho2_raw(zs), 0.0) * s ** 2

        return dict(zL=(zL0 - zbar) * s, zR=(zR0 - zbar) * s, rho2=rho2)

    def profile(self, q, n=300):
        """返回 (z 数组, 物理半径 ρ(z) 数组)，单位 fm。"""
        d = self.build(q)
        z_d = np.linspace(d["zL"], d["zR"], n)             # 无量纲 z（单位 R0）
        rho_d = np.sqrt(np.maximum(d["rho2"](z_d), 0.0))   # 无量纲半径
        return z_d * self.R0, rho_d * self.R0
