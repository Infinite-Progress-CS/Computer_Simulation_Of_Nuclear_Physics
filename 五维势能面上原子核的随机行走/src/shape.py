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
from scipy import integrate


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


class ShapeFunnyHills:
    """Funny-Hills 四阶剖面（平滑颈，无圆柱/胶囊假象）。

    ρ²(z) = (c²−z²)(A + B·z²/c²) + α·z·(c²−z²) ,  |z| ≤ c
      A = n²/c³ ,  B = 5(1−n²)/c³ ,  α = 8η/(3c⁴)
    体积守恒 ∫ρ²dz = 4/3（由 A+B/5 = 1/c³ 保证），α 项（奇函数）不改变体积。

    5 参数（与 Shape3QS 接口兼容）：
      q[0] = elong → c = 1 + elong/2（半长，elong=0 → 球）
      q[1] = neck  → n（腰比，1=长椭球，0=断裂）
      q[2] = eta   → α（质量不对称，η>0 → 右碎片重）
      q[3], q[4]   → eps1, eps2（碎片形变，形状中暂忽略；断裂区量子修正由
                     fragment.py 的椭球独立处理）

    与 3QS 二次中段的本质区别：中段是四次多项式，腰半径连续趋近 0，
    不会在 ρ_v=a 处退化为圆柱（C=0），故颈方向能量单调、无假势垒脊。
    """

    # 单位球参考形状：elong=0, neck=1 → ρ²=1−z²（球）。neck=0 是「花生」不是球。
    SPHERE_Q = [0.0, 1.0, 0.0, 0.0, 0.0]

    def __init__(self, R0=7.17):
        self.R0 = R0

    def build(self, q):
        elong, neck, eta, eps1, eps2 = q
        c = 1.0 + elong / 2.0
        n = float(np.clip(neck, 1e-9, 1.0))
        A = n * n / c ** 3
        B = 5.0 * (1.0 - n * n) / c ** 3
        alpha = 8.0 * eta / (3.0 * c ** 4)

        # 有效性：P(z)=A+αz+Bz²/c² 在 |z|≤c 上须 ≥0（否则 ρ²<0）
        if B > 1e-12:
            z_star = -alpha * c * c / (2.0 * B)
            z_star = float(np.clip(z_star, -c, c))
            P_min = A + alpha * z_star + B * z_star * z_star / (c * c)
        else:
            P_min = A - abs(alpha) * c
        if P_min < -1e-9:
            raise RuntimeError(f"FunnyHills 无效 q={q}")

        def rho2(z):
            z = np.asarray(z, dtype=float)
            c2mz2 = c * c - z * z
            return np.maximum(
                c2mz2 * (A + B * z * z / (c * c)) + alpha * z * c2mz2, 0.0)

        return dict(zL=-c, zR=+c, rho2=rho2)

    def profile(self, q, n=300):
        d = self.build(q)
        z_d = np.linspace(d["zL"], d["zR"], n)
        rho_d = np.sqrt(np.maximum(d["rho2"](z_d), 0.0))
        return z_d * self.R0, rho_d * self.R0


class Shape3QS:
    """轴对称形状 ρ(z)，由 5 个物理参数 (elong, neck, eta, eps1, eps2) 描述。

    左球体 + 中间二次曲面 ρ²=ρ_v²+C(z−z_v)² + 右球体，切线拼接（判别式法），
    体积守恒（∫ρ²dz = 4/3 单位球），质心居中。返回无量纲 ρ²(z)，物理半径 = R0·√ρ²。
    """

    # 单位球参考形状：3QS 在 elong=0,η=ε=0 时 build 直接返回单位球（与 neck 无关）。
    SPHERE_Q = [0.0, 0.0, 0.0, 0.0, 0.0]

    # 形状几何版本：改 build()/参数化后必须递增，否则磁盘缓存会静默复用旧壳修正。
    VERSION = "3qs_v3_neck_indep"

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

    # ---------------- 非对称解析解（η≠0 或 ε1≠ε2）----------------
    def _tangent_asym(self, a1, c1, a2, c2, l1, l2, rho_v, g1, g2):
        """非对称 3QS 切线拼接的闭式解，返回 (z_v, C, z1, z2)；无效返回 None。

        判别式法：左/中、中/右切线分别要求判别式 D1=0、D2=0。对 D1=0 可解出
        C = g1·K1 / [g1(z_v−l1)² − K1]（K1=a1²−ρ_v²），D2=0 同理；两式相减得
        z_v 的一元二次方程，故整体有闭式解，无需 least_squares（等价但更快、
        更稳，且能收敛到 least_squares 因初值不佳而漏掉的近断裂简并解）。
        """
        K1 = a1 * a1 - rho_v * rho_v
        K2 = a2 * a2 - rho_v * rho_v
        if K1 <= 0.0 or K2 <= 0.0:
            return None
        A = K1 - K2
        B = -2.0 * (K1 * l2 - K2 * l1)
        Cc = (K1 * l2 * l2 - K2 * l1 * l1) - K1 * K2 * (1.0 / g2 - 1.0 / g1)
        roots = np.roots([A, B, Cc])
        roots = roots[np.abs(roots.imag) < 1e-9].real
        for z_v in roots:
            den = g1 * (z_v - l1) ** 2 - K1
            if abs(den) < 1e-12:
                continue
            C = g1 * K1 / den
            if C <= 0.0:                # C>0 才是颈收缩（腰部内凹）
                continue
            A1 = C + g1
            B1 = -2.0 * C * z_v - 2.0 * g1 * l1
            A2 = C + g2
            B2 = -2.0 * C * z_v - 2.0 * g2 * l2
            if abs(A1) < 1e-12 or abs(A2) < 1e-12:
                continue
            z1 = -B1 / (2.0 * A1)
            z2 = -B2 / (2.0 * A2)
            if l1 < z1 < z2 < l2:
                return float(z_v), float(C), float(z1), float(z2)
        return None

    def build(self, q):
        """3QS 三二次曲面剖面（Nix/Möller 标准参数化）。

        q = (elong, neck, eta, eps1, eps2)：
          elong = 两碎片中心间距 / R0
          neck  = 中间顶点半径 ρ_v / min(a1,a2) ∈ (0,1]
          eta   = 质量不对称 (M_H−M_L)/(M_H+M_L)，η>0 → 右碎片重
          eps1/eps2 = 左右碎片 Nilsson 四极形变 ε_f1/ε_f2
        左球体 + 中间二次曲面 ρ²=ρ_v²+C(z−z_v)² + 右球体，切线拼接（判别式法），
        体积守恒（∫ρ²dz = 4/3 单位球），质心居中。返回无量纲 ρ²(z)。
        """
        elong, neck, eta, eps1, eps2 = q
        r1 = self.axis_ratio(eps1)
        r2 = self.axis_ratio(eps2)
        w1 = (1.0 - eta) / 2.0
        w2 = (1.0 + eta) / 2.0
        c1 = (w1 / r1 ** 2) ** (1.0 / 3.0); a1 = r1 * c1
        c2 = (w2 / r2 ** 2) ** (1.0 / 3.0); a2 = r2 * c2
        l1 = -elong / 2.0; l2 = +elong / 2.0
        # 颈参数化：Möller 3QS 的第五个坐标是「颈直径 d」（独立于质量不对称 η 的
        # 物理长度量），即 rho_v = d/2。这里用无量纲颈坐标 neck ∈ (0,1]，取
        # rho_v = neck · a_ref，a_ref = (1/2)^{1/3} ≈ 0.794（对称球形碎片的横向半轴，
        # 单位 R0）。neck=1 → 颈半径 = 对称碎片半径，neck→0 → 断裂。
        #
        # 重要：不能写成 rho_v = neck · min(a1,a2)。那样 η 增大 → min(a1,a2) 变小 →
        # 颈被错误地「变细」，把质量不对称与颈部收缩耦合，会在中间区制造一个
        # 假深不对称极小（液滴能量随 η 从 ~12 MeV 塌到 ~3 MeV），并让大 η 的非对称
        # 形状在厚颈区无解。颈半径必须只依赖 neck 坐标本身，与 η 无关。
        a_ref = 0.5 ** (1.0 / 3.0)
        rho_v = neck * a_ref

        # 近球极限：直接返回单位球
        if elong < 1e-6 and abs(eta) < 1e-9 and abs(eps1) < 1e-9 and abs(eps2) < 1e-9:
            return self._make(lambda z: 1.0 - z * z, -1.0, 1.0)

        g1 = a1 ** 2 / c1 ** 2
        g2 = a2 ** 2 / c2 ** 2

        if abs(eta) < 1e-9 and abs(eps1 - eps2) < 1e-9:
            # 纯对称：直接用解析解（用实际 a1,c1 精确）
            exact = self._tangent_sym(a1, c1, elong, rho_v)
            if exact is None:
                raise RuntimeError(f"对称无解 q={q}")
            z_v, C, z1, z2 = exact
        else:
            # 非对称：判别式法闭式解（二次方程），等价于原 least_squares 但 O(1)
            asym = self._tangent_asym(a1, c1, a2, c2, l1, l2, rho_v, g1, g2)
            if asym is None:
                raise RuntimeError(f"非对称无解 q={q}")
            z_v, C, z1, z2 = asym

        # 三段二次曲面 ρ²(z)（无量纲）
        def rho2_raw(z):
            z = np.asarray(z, dtype=float)
            p1 = a1 ** 2 * (1.0 - (z - l1) ** 2 / c1 ** 2)
            pm = rho_v ** 2 + C * (z - z_v) ** 2
            p2 = a2 ** 2 * (1.0 - (z - l2) ** 2 / c2 ** 2)
            return np.where(z < z1, p1, np.where(z < z2, pm, p2))

        return self._make(rho2_raw, l1 - c1, l2 + c2)

    def is_valid(self, q):
        """便宜有效性检查：只做切线拼接判别式（跳过 _make 体积积分）。

        与 build() 抛异常判据严格一致（同样的几何量 + 同样的 _tangent_sym/_tangent_asym），
        供 η 子步等只需「形状是否可解」、不需完整剖面的场景，省掉 ~0.2ms 的体积积分。
        """
        elong, neck, eta, eps1, eps2 = q
        if elong < 1e-6 and abs(eta) < 1e-9 and abs(eps1) < 1e-9 and abs(eps2) < 1e-9:
            return True
        r1 = self.axis_ratio(eps1)
        r2 = self.axis_ratio(eps2)
        w1 = (1.0 - eta) / 2.0
        w2 = (1.0 + eta) / 2.0
        c1 = (w1 / r1 ** 2) ** (1.0 / 3.0); a1 = r1 * c1
        c2 = (w2 / r2 ** 2) ** (1.0 / 3.0); a2 = r2 * c2
        l1 = -elong / 2.0; l2 = +elong / 2.0
        rho_v = neck * (0.5 ** (1.0 / 3.0))
        g1 = a1 ** 2 / c1 ** 2
        g2 = a2 ** 2 / c2 ** 2
        if abs(eta) < 1e-9 and abs(eps1 - eps2) < 1e-9:
            return self._tangent_sym(a1, c1, elong, rho_v) is not None
        return self._tangent_asym(a1, c1, a2, c2, l1, l2, rho_v, g1, g2) is not None

    def _make(self, rho2_raw, zL0, zR0):
        """体积守恒缩放 + 质心居中，返回 dict(zL,zR,rho2)。"""
        zz = np.linspace(zL0, zR0, 1000)
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
