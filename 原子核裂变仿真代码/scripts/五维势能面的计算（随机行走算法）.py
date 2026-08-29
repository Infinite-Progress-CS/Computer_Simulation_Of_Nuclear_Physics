"""
五维势能面计算 + 随机行走 + 分屏 3D 动画
====================================================
物理模型：FRLDM 宏观部分（Finite-Range Liquid-Drop Model，Möller 框架）

  形变能  ΔV(q) = E_S0·(B_s(q) − 1) + E_C0·(B_c(q) − 1)
     └ 表面项（抵抗裂变，B_s ≥ 1）        └ 库仑项（驱动裂变，B_c ≤ 1）

  表面项用有限力程 Yukawa-plus-exponential 核（Krappe-Nix-Sierk 双重曲面积分），
  取代旧的 S/S_球；有限力程平滑了颈部尖点 → 表面能随拉长增长更慢 → 势垒更低（更接近实验）。
  库仑项用尖表面，方位角解析积分（第一类完全椭圆积分 K）。

  体积守恒（不可压缩液滴）→ 体积能、对称能与形状无关，在形变能中相消，
  只留下表面能与库仑能两项的竞争，形成裂变势垒。

形状参数化：3QS（三二次曲面，标准 Nix/Möller 版）五维坐标
  q1 = elong 拉长         = 两碎片中心间距 / R0
  q2 = neck  颈部         = 中间顶点半径 ρ_v / min(a1,a2) ∈ (0,1]
  q3 = eta   质量不对称    = (M_H − M_L)/(M_H + M_L) ∈ (−1,1)
  q4 = eps1  左碎片形变    = 左碎片 Nilsson 四极形变 ε_f1
  q5 = eps2  右碎片形变    = 右碎片 Nilsson 四极形变 ε_f2
  左球体 + 中间二次曲面 ρ²=ρ_v²+C(z−z_v)² + 右球体，切线拼接（判别式法），
  体积守恒（∫ρ²dz = 4/3 单位球），质心居中。

流程：
  1. 构建五维势能面（FRLDM 宏观）
  2. Metropolis 随机行走（5 个物理参数，无效形状拒绝）
  3. 分屏动画：左 = 核形状 + 5 参数读数，右 = 3D 势能面 + 行走轨迹

作者：高涵
日期：2026-08-11（升级：2026-08-26 FRLDM + 3QS 版）
"""

import os
import numpy as np
from scipy import special, integrate, optimize
import matplotlib
matplotlib.use("Agg")   # 无界面后端，直接存文件
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 中文字体（Windows 自带微软雅黑/黑体），避免标题出现方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 输出目录 = 本脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. 原子核形状（3QS 三二次曲面参数化）
# ============================================================

class Shape3QS:
    """轴对称形状 ρ(z)，由 5 个物理参数 (elong, neck, eta, eps1, eps2) 描述。

    左球体 + 中间二次曲面 ρ²=ρ_v²+C(z−z_v)² + 右球体，切线拼接（判别式法），
    体积守恒（∫ρ²dz = 4/3 单位球），质心居中。返回无量纲 ρ²(z)，物理半径 = R0·√ρ²。
    """

    def __init__(self, R0=7.570):
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


# ============================================================
# 2. FRLDM 宏观能量（五维势能面）
# ============================================================

class FRLDMPES:
    """
    FRLDM 宏观形变能（无壳修正/对修正）：
        ΔV(q) = E_S0·(B_s − 1) + E_C0·(B_c − 1)
      B_s = 有限力程表面能比（Krappe-Nix-Sierk 双重曲面积分，核 [2−(2+s/a)e^{−s/a}]）
      B_c = 库仑能比（尖表面，方位角解析积分）
    参数（FRLDM 1995 ADNDT 59,185）：a_s=21.18466, κ_s=2.345, r0=1.16 fm, a=0.68 fm。
    """

    def __init__(self, Z, N, r0=1.16, a_s=21.18466, kappa_s=2.345, a_range=0.68,
                 nz=30, nrho=30, nsurf=40, nphi=24):
        self.Z, self.N = Z, N
        self.A = Z + N
        self.R0 = r0 * self.A ** (1.0 / 3.0)
        I = (N - Z) / self.A
        self.E_S0 = a_s * (1.0 - kappa_s * I ** 2) * self.A ** (2.0 / 3.0)
        self.E_C0 = (3.0 / 5.0) * Z * Z * 1.44 / self.R0
        self.a = a_range / self.R0          # 无量纲有限力程
        self.shape = Shape3QS(self.R0)
        self.nz, self.nrho = nz, nrho       # 库仑积分网格
        self.nsurf, self.nphi = nsurf, nphi  # 表面能积分网格
        self._sphere = self.shape.build([0.0, 0.0, 0.0, 0.0, 0.0])
        self.I_C_sphere = self._coulomb_I(self._sphere)
        self.I_S_sphere = self._surface_I(self._sphere)

    # ---- 库仑（尖表面，方位角解析 ellipk）----
    def _coulomb_I(self, d):
        nz, nrho = self.nz, self.nrho
        zL, zR, rho2 = d["zL"], d["zR"], d["rho2"]
        xz, wz = np.polynomial.legendre.leggauss(nz)
        xr, wr = np.polynomial.legendre.leggauss(nrho)
        z = zL + (xz + 1.0) * 0.5 * (zR - zL)
        wz_full = wz * (zR - zL) * 0.5
        Rz = np.sqrt(np.maximum(rho2(z), 0.0))
        rho = Rz[:, None] * (xr[None, :] + 1.0) * 0.5
        w_rho = wr[None, :] * Rz[:, None] * 0.5
        zz = (z[:, None, None, None] - z[None, None, :, None]) ** 2
        r1 = rho[:, :, None, None]
        r2 = rho[None, None, :, :]
        aa = zz + r1 ** 2 + r2 ** 2
        b = 2.0 * r1 * r2
        apb = np.maximum(aa + b, 1e-14)
        m = np.clip(2.0 * b / apb, 0.0, 1.0 - 1e-12)
        K = special.ellipk(m)
        total_w = (wz_full[:, None, None, None] * w_rho[:, :, None, None]
                   * wz_full[None, None, :, None] * w_rho[None, None, :, :])
        integrand = 2.0 * np.pi * r1 * r2 * 4.0 * K / np.sqrt(apb)
        return np.sum(total_w * integrand)

    # ---- 有限力程表面能（KNS 双重曲面积分）----
    def _surface_I(self, d):
        a = self.a
        zL, zR, rho2 = d["zL"], d["zR"], d["rho2"]
        nz, nphi = self.nsurf, self.nphi
        xz, wz = np.polynomial.legendre.leggauss(nz)
        xp, wp = np.polynomial.legendre.leggauss(nphi)
        z = zL + (xz + 1.0) * 0.5 * (zR - zL)
        wz_full = wz * (zR - zL) * 0.5
        r = np.sqrt(np.maximum(rho2(z), 0.0))
        dz = 1e-6 * (zR - zL)
        rp = np.sqrt(np.maximum(rho2(z + dz), 0.0))
        rm = np.sqrt(np.maximum(rho2(z - dz), 0.0))
        rho_prime = (rp - rm) / (2.0 * dz)

        Delta = np.pi * (xp + 1.0)
        wD = wp * np.pi
        zz = z[:, None] - z[None, :]
        rho1 = r[:, None]; rho2_ = r[None, :]
        rp1 = rho_prime[:, None]; rp2 = rho_prime[None, :]
        r1r2 = rho1 * rho2_

        total = 0.0
        for k in range(nphi):
            cosD = np.cos(Delta[k])
            s2 = zz ** 2 + rho1 ** 2 + rho2_ ** 2 - 2.0 * r1r2 * cosD
            s = np.sqrt(np.maximum(s2, 1e-24))
            K = (2.0 - (2.0 + s / a) * np.exp(-s / a)) / np.maximum(s2 ** 2, 1e-24)
            dot1 = rho1 - rho2_ * cosD - zz * rp1
            dot2 = rho1 * cosD - rho2_ - zz * rp2
            total += wD[k] * np.sum(wz_full[:, None] * wz_full[None, :]
                                    * r1r2 * dot1 * dot2 * K)
        return 2.0 * np.pi * total

    def B_s(self, q):
        return self._surface_I(self.shape.build(q)) / self.I_S_sphere

    def B_c(self, q):
        return self._coulomb_I(self.shape.build(q)) / self.I_C_sphere

    def energy_components(self, q):
        d = self.shape.build(q)
        B_s = self._surface_I(d) / self.I_S_sphere
        B_c = self._coulomb_I(d) / self.I_C_sphere
        dV_s = self.E_S0 * (B_s - 1.0)
        dV_c = self.E_C0 * (B_c - 1.0)
        return dV_s + dV_c, dV_s, dV_c, B_s, B_c

    def energy(self, q):
        return self.energy_components(q)[0]


# ============================================================
# 3. Metropolis 随机行走
# ============================================================

def metropolis_walk(pes, q0, T=6.0, n_steps=500, T_end=None, ratchet_elong=None,
                    step=np.array([0.06, 0.03, 0.03, 0.03, 0.03]),
                    q_min=None, q_max=None, seed=0):
    """
    在五维势能面上做 Metropolis 随机行走。
    p(q→q') = min(1, exp(−ΔV/T))；T 是核激发温度（MeV），T 大 → 易翻越势垒。
    无效形状（build 抛出异常，如深颈+小拉长）直接拒绝。

    T_end 给定时做退火（模拟核越过势垒后冷却、沿谷底滚落到断裂点）：
      - 若同时给出 ratchet_elong：两段式——越障前恒温 T（保证越过势垒），
        一旦 elong 越过 ratchet_elong（棘轮触发）才从 T 线性冷却到 T_end
        （沉降到断裂点）。默认 None = 恒温。
      - 若未给 ratchet_elong：全程线性退火 T → T_end。

    ratchet_elong：一旦 elong 超过该值（越过主势垒），elong 只增不减，
    模拟裂变断裂的不可逆性，强制行走沿谷底滚落到断裂点。
    （3QS 中细颈与碎片分离被解耦，产生一个假的颈部过渡势垒，棘轮用于跨越它。）

    返回：path (n+1,5)、energies (n+1,)、components (n+1,2)=[表面项,库仑项]、接受率。
    """
    rng = np.random.default_rng(seed)
    if q_min is None:
        q_min = np.array([0.1, 0.05, -0.5, -0.2, -0.2])
    if q_max is None:
        q_max = np.array([3.0, 0.99, 0.5, 0.4, 0.4])

    path = np.zeros((n_steps + 1, 5))
    energies = np.zeros(n_steps + 1)
    components = np.zeros((n_steps + 1, 2))

    q = np.array(q0, dtype=float)
    try:
        V, dV_s, dV_c, _, _ = pes.energy_components(q)
    except Exception:
        V, dV_s, dV_c = np.inf, 0.0, 0.0
    path[0], energies[0] = q, V
    components[0] = (dV_s, dV_c)
    n_accept = 0

    committed = False
    i_commit = None
    for i in range(1, n_steps + 1):
        if T_end is None:
            T_i = T
        elif ratchet_elong is not None and committed:
            # 两段式：越障前恒温 T，越障后从 T 冷却到 T_end
            frac = (i - i_commit) / max(n_steps - i_commit, 1)
            T_i = T + (T_end - T) * frac
        else:
            T_i = T + (T_end - T) * (i / n_steps)
        q_new = np.clip(q + step * rng.standard_normal(5), q_min, q_max)
        if committed:
            q_new[0] = max(q_new[0], q[0])   # 裂变不可逆：elong 只增不减
        try:
            V_new, dV_s_new, dV_c_new, _, _ = pes.energy_components(q_new)
        except Exception:
            V_new = np.inf
        dE = V_new - V
        if V_new < np.inf and (dE <= 0 or rng.random() < np.exp(-dE / T_i)):
            q, V, dV_s, dV_c = q_new, V_new, dV_s_new, dV_c_new
            n_accept += 1
        if ratchet_elong is not None and q[0] > ratchet_elong:
            if not committed:
                committed = True
                i_commit = i
        path[i], energies[i] = q, V
        components[i] = (dV_s, dV_c)
        if i % 100 == 0:
            print(f"  步数 {i}/{n_steps}  ΔV={V:+.2f} MeV  "
                  f"(表面 {dV_s:+.2f} / 库仑 {dV_c:+.2f})  接受率 {n_accept / i:.2%}")

    return path, energies, components, n_accept / n_steps


# ============================================================
# 4. 可视化
# ============================================================

def draw_shape_2d(ax, q, shape, color="#3b6ea5", alpha=0.8):
    """画原子核的 ρ(z) 填充轮廓（轴对称形状的二维剖面）"""
    z, rho = shape.profile(q)
    ax.fill_between(z, -rho, rho, color=color, alpha=alpha, linewidth=0)
    ax.plot([z.min(), z.max()], [0, 0], color="gray", lw=0.8, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_axis_off()


def neck_fraction(q, shape, nz=300):
    """断裂程度 f ∈ [0,1]：f = 1 − ρ_颈/ρ_最大，0=单核(无腰)，1=颈部断开。

    颈部 = 内部局部极小半径（腰），排除两极尖点（半径→0 的端点）。
    无内部极小（光滑长椭球/球）→ 无腰 → f=0。
    """
    z, rho = shape.profile(q, n=nz)
    rho_max = rho.max()
    r_neck = rho_max                    # 默认无腰
    for i in range(1, nz - 1):
        # 用 <= 处理颈恰落在两对称格点之间、两最近点 rho 相等的情形
        if rho[i] <= rho[i - 1] and rho[i] <= rho[i + 1]:
            r_neck = min(r_neck, rho[i])
    return float(np.clip(1.0 - r_neck / rho_max, 0.0, 1.0))


def _smooth_1d(y, win):
    """边界安全的滑动平均（不填充 0，首末帧保持真实值）"""
    cum = np.concatenate(([0.0], np.cumsum(y)))
    out = np.zeros_like(y)
    for i in range(len(y)):
        lo, hi = max(0, i - win // 2), min(len(y), i + win // 2 + 1)
        out[i] = (cum[hi] - cum[lo]) / (hi - lo)
    return out


def _smooth_path(path, win):
    """对路径的每个参数维度分别做滑动平均"""
    out = np.empty_like(path)
    for k in range(path.shape[1]):
        out[:, k] = _smooth_1d(path[:, k], win)
    return out


def shape_surface(q, shape, f, nz=80, nphi=56):
    """裂变形状的 3D 表面：对称轴沿水平 X 方向，颜色编码局部的"膨胀/收缩"。

    颜色物理意义（"气球"方案，连续渐变，不突变）：
      以 s=ρ(z)/R0 度量局部绝对粗细（球形半径 R0 为尺度），固定映射：
        - 细（两极/尖端/颈部，ρ/R0→0）→ 深红；
        - 中等（球主体，≈0.55）→ 红；
        - 粗（碎片/球赤道，≳0.75）→ 浅红。
      最终：大的碎片浅红、小的碎片深红。

    返回 (X, Y, Z, facecolors, state)。
    """
    z, rho_z = shape.profile(q, n=nz)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi)

    X = np.tile(z, (nphi, 1))
    PHI = np.tile(phi[:, None], (1, nz))
    RHO = np.tile(rho_z, (nphi, 1))
    Y = RHO * np.cos(PHI)
    Z = RHO * np.sin(PHI)

    R0 = shape.R0
    s = rho_z / R0

    c_dark = np.array([0.40, 0.04, 0.06])   # 深红（细）
    c_base = np.array([0.75, 0.18, 0.14])   # 红（基准）
    c_lite = np.array([1.00, 0.78, 0.68])   # 浅红（粗）
    s_dark, s_base, s_lite = 0.30, 0.55, 0.75

    col = np.empty((nz, 4))
    for k in range(nz):
        rr = s[k]
        if rr <= s_dark:
            col[k, :3] = c_dark
        elif rr <= s_base:
            t = (rr - s_dark) / (s_base - s_dark)
            col[k, :3] = c_dark + (c_base - c_dark) * t
        elif rr <= s_lite:
            t = (rr - s_base) / (s_lite - s_base)
            col[k, :3] = c_base + (c_lite - c_base) * t
        else:
            col[k, :3] = c_lite
        col[k, 3] = 1.0

    fc = np.tile(col, (nphi, 1, 1))

    f = float(np.clip(f, 0.0, 1.0))
    if f < 0.05:
        state = "单核"
    elif f > 0.8:
        state = "已断裂·两个碎片"
    else:
        state = "颈部收缩中"

    return X, Y, Z, fc, state


def plot_shape_schematic(pes, filename="五参数形状示意图.png"):
    """5 个参数分别单独扫描，展示各自如何改变原子核形状"""
    shape = pes.shape
    base = np.array([1.2, 0.7, 0.0, 0.1, 0.1])

    params = [
        ("elong 拉长",       0, [0.6, 0.9, 1.2, 1.6]),
        ("neck 颈部",        1, [0.95, 0.7, 0.5, 0.3]),
        ("eta 质量不对称",   2, [-0.4, -0.15, 0.15, 0.4]),
        ("eps1 左碎片形变",  3, [-0.2, 0.0, 0.2, 0.35]),
        ("eps2 右碎片形变",  4, [-0.2, 0.0, 0.2, 0.35]),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    for ax, (name, idx, vals) in zip(axes, params):
        colors = plt.cm.viridis(np.linspace(0.25, 0.9, len(vals)))
        for v, c in zip(vals, colors):
            q = base.copy()
            q[idx] = v
            try:
                draw_shape_2d(ax, q, shape, color=c, alpha=0.55)
            except Exception:
                pass
        ax.set_title(f"{name}\n值 = {vals}", fontsize=9)
    fig.suptitle("五参数下原子核形状变化（其余参数固定在基准值；颜色浅→深 = 参数值小→大）",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(BASE_DIR, filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  形状示意图已保存: {out}")


def compute_surface_slice(pes, n=34, elong_lims=(0.3, 3.0), neck_lims=(0.05, 0.99)):
    """势能面切片 V(elong, neck)，eta=eps1=eps2=0（对称裂变路径）。
    无效 (elong, neck) 组合（深颈+小拉长）记为 NaN。"""
    elong = np.linspace(*elong_lims, n)
    neck = np.linspace(*neck_lims, n)
    Q1, Q2 = np.meshgrid(elong, neck)
    V = np.full_like(Q1, np.nan)
    print(f"  正在计算势能面切片 ({n}x{n} = {n * n} 个能量点)...")
    for i in range(n):
        for j in range(n):
            try:
                V[i, j] = pes.energy([Q1[i, j], Q2[i, j], 0.0, 0.0, 0.0])
            except Exception:
                pass
    return Q1, Q2, V


def min_energy_path(pes, elong_lims=(0.3, 3.0), n_elong=55, n_neck=16):
    """确定性极小能量（绝热）裂变路径：在每个 elong 下对 neck 取极小，
    eta=eps1=eps2=0（U-236 裂变参数 x=0.711 > Businaro-Gallone 点 0.396，
    对称裂变能量最低）。这就是随机行走要逼近的"最优路径"。

    返回 (path, energies, components)。
    """
    elong = np.linspace(*elong_lims, n_elong)
    necks = np.linspace(0.05, 0.99, n_neck)
    path = np.zeros((n_elong, 5))
    energies = np.full(n_elong, np.nan)
    components = np.zeros((n_elong, 2))
    for i, el in enumerate(elong):
        best = None
        for nk in necks:
            q = [el, nk, 0.0, 0.0, 0.0]
            try:
                V, dV_s, dV_c, _, _ = pes.energy_components(q)
            except Exception:
                continue
            if best is None or V < best[0]:
                best = (V, dV_s, dV_c, q)
        if best is not None:
            path[i] = best[3]
            energies[i] = best[0]
            components[i] = (best[1], best[2])
    return path, energies, components


def plot_pes_slice(pes, path, Q1, Q2, V, min_path=None,
                   filename="势能面切片.png"):
    """2D 势能面等高线 + 随机行走轨迹（投影到 elong-neck 平面），
    可选叠加确定性极小能量路径（最优裂变路径）。"""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    Vmasked = np.ma.masked_invalid(V)
    cs = ax.contourf(Q1, Q2, Vmasked, levels=30, cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("形变能 ΔV (MeV)")
    ax.plot(path[:, 0], path[:, 1], "r-", lw=1.0, alpha=0.55,
            label="随机行走轨迹")
    ax.plot(path[0, 0], path[0, 1], "o", color="white", ms=9, mec="k", label="起点")
    ax.plot(path[-1, 0], path[-1, 1], "*", color="crimson", ms=16, label="终点")
    if min_path is not None:
        ax.plot(min_path[:, 0], min_path[:, 1], "w--", lw=2.5, alpha=0.95,
                label="极小能量路径（最优裂变路径）")
        ax.plot(min_path[:, 0], min_path[:, 1], "k--", lw=1.0, alpha=0.6)
    ax.set_xlabel("elong 拉长")
    ax.set_ylabel("neck 颈部")
    ax.set_title("五维势能面切片 (eta=eps1=eps2=0) 与随机行走轨迹")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = os.path.join(BASE_DIR, filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  势能面切片已保存: {out}")


def animate_split(path, energies, components, pes, Q1, Q2, V,
                  filename="核形状随机行走.gif", every=2, fps=8,
                  overlay_path=None, overlay_energies=None):
    """分屏动画：左上=3D 核形状(水平)，左下=5 参数读数，右=3D 势能面+轨迹。

    主轨迹沿极小能量路径（球→拉长→颈缩→断裂），做时间平滑展示平滑的裂变趋势；
    overlay_path（随机行走轨迹）+ overlay_energies 作为方法轨迹淡色叠加在势能面上。
    """
    shape = pes.shape
    R0 = pes.R0

    # 时间平滑路径、能量与能量分量（窗口 5）
    win = 5
    path = _smooth_path(path, win)
    energies = _smooth_1d(energies, win)
    components = _smooth_path(components, win)

    frames = list(range(0, len(path), every))

    # 预计算坐标范围（覆盖全程所有形状）
    z_lim = max(np.abs(shape.profile(p)[0]).max() for p in path) * 1.1
    r_lim = R0 * 1.4

    # 断裂程度 f：基于平滑后的路径
    f_smooth = np.array([neck_fraction(p, shape) for p in path])

    # 轨迹 z 坐标用实际五维能量（含 eta/eps 形变），2D 切片作参考面
    E_traj = energies

    V_valid = V[~np.isnan(V)]
    zmin = float(min(V_valid.min(), E_traj.min()))
    zmax = float(max(V_valid.max(), E_traj.max()))
    if overlay_energies is not None:
        zmax = float(max(zmax, overlay_energies.max()))

    fig = plt.figure(figsize=(19, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 3.0], height_ratios=[6.5, 0.55],
                          left=0.02, right=0.98, top=0.96, bottom=0.085,
                          wspace=0.08, hspace=0.30)
    ax_shape = fig.add_subplot(gs[0, 0], projection="3d")
    ax_surf = fig.add_subplot(gs[0, 1], projection="3d")
    ax_info = fig.add_subplot(gs[1, :])

    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=zmin, vmax=zmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_surf, shrink=0.55, pad=0.08, aspect=30)
    cbar.set_label("形变能 ΔV (MeV)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", lw=3, label="行走轨迹"),
        Line2D([0], [0], marker="o", color="lime", markersize=9, linestyle="None",
               markeredgecolor="black", markeredgewidth=1.0, label="起点"),
        Line2D([0], [0], marker="s", color="red", markersize=9, linestyle="None",
               markeredgecolor="black", markeredgewidth=1.0, label="终点"),
        Line2D([0], [0], marker="o", color="black", markersize=9, linestyle="None",
               markeredgecolor="white", markeredgewidth=1.2, label="当前点"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.50, 0.012), ncol=4, frameon=False,
               fontsize=11, handlelength=1.4, handletextpad=0.4,
               columnspacing=1.0, borderaxespad=0.0)
    fig.text(0.50, 0.062,
             "左图: 红=基准，膨胀→浅红 / 收缩→深红（断裂后 大碎片浅红·小碎片深红）    |    右图: ΔV 蓝低 → 黄高",
             ha="center", va="bottom", fontsize=10.5, color="0.25")

    def update(i):
        q = path[i]
        Vv, dV_s, dV_c = energies[i], components[i, 0], components[i, 1]

        # ---- 左上：3D 核形状（水平放置；颜色区分单核/双碎片）----
        ax_shape.clear()
        X, Y, Zs, fc, state = shape_surface(q, shape, f_smooth[i])
        ax_shape.plot_surface(X, Y, Zs, facecolors=fc, linewidth=0,
                              antialiased=True, rstride=1, cstride=1)
        ax_shape.set_xlim(-z_lim, z_lim)
        ax_shape.set_ylim(-r_lim, r_lim)
        ax_shape.set_zlim(-r_lim, r_lim)
        ax_shape.set_box_aspect((z_lim / r_lim, 1, 1))
        ax_shape.view_init(elev=15, azim=-90)
        ax_shape.dist = 8
        ax_shape.set_xlabel("z (fm) 对称轴", fontsize=9)
        ax_shape.set_ylabel("x (fm)", fontsize=9)
        ax_shape.set_zlabel("y (fm)", fontsize=9)
        ax_shape.set_title(f"核形状  Step {i}  [{state}]", fontsize=12)

        # ---- 底部：5 参数读数 ----
        ax_info.clear()
        ax_info.axis("off")
        info = (f"elong(拉长)={q[0]:+.3f}   neck(颈部)={q[1]:+.3f}   "
                f"eta(质量不对称)={q[2]:+.3f}   eps1(左碎片形变)={q[3]:+.3f}   "
                f"eps2(右碎片形变)={q[4]:+.3f}   ΔV={Vv:+.2f} MeV   "
                f"表面项 {dV_s:+.2f}   库仑项 {dV_c:+.2f}")
        ax_info.text(0.5, 0.5, info, transform=ax_info.transAxes, fontsize=13,
                     va="center", ha="center")

        # ---- 右：3D 势能面 + 轨迹 ----
        ax_surf.clear()
        Vplot = np.where(np.isnan(V), zmax + (zmax - zmin) * 0.1, V)
        ax_surf.plot_surface(Q1, Q2, Vplot, cmap="viridis", alpha=0.80,
                             linewidth=0, antialiased=True)
        traj = E_traj[:i + 1]
        ax_surf.plot(path[:i + 1, 0], path[:i + 1, 1], traj,
                     color="white", lw=5.0, alpha=0.95)
        ax_surf.plot(path[:i + 1, 0], path[:i + 1, 1], traj,
                     color="black", lw=2.2)
        if overlay_path is not None:
            ax_surf.plot(overlay_path[:, 0], overlay_path[:, 1],
                         overlay_energies, color="yellow", lw=1.0, alpha=0.45)
        ax_surf.scatter([q[0]], [q[1]], [E_traj[i]], color="black", s=60,
                        edgecolor="white", linewidth=1.2, depthshade=False)
        ax_surf.scatter([path[0, 0]], [path[0, 1]], [E_traj[0]],
                        color="lime", marker="o", s=90, edgecolor="black",
                        linewidth=1.0, depthshade=False)
        ax_surf.scatter([path[-1, 0]], [path[-1, 1]], [E_traj[-1]],
                        color="red", marker="s", s=70, edgecolor="black",
                        linewidth=1.0, depthshade=False)
        ax_surf.dist = 8
        ax_surf.set_xlabel("elong 拉长", fontsize=9)
        ax_surf.set_ylabel("neck 颈部", fontsize=9)
        ax_surf.set_zlabel("ΔV (MeV)", fontsize=9)
        ax_surf.set_xlim(float(Q1.min()), float(Q1.max()))
        ax_surf.set_ylim(float(Q2.min()), float(Q2.max()))
        ax_surf.set_zlim(zmin, zmax)
        ax_surf.view_init(elev=25, azim=-60)
        ax_surf.set_title("三维势能面 V(elong, neck) 与行走轨迹", fontsize=12)
        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    out = os.path.join(BASE_DIR, filename)
    try:
        ani.save(out, writer="pillow", fps=fps, dpi=140)
        print(f"  动画已保存: {out} ({len(frames)} 帧, {fps} fps)")
    except Exception as e:
        print(f"  (GIF 保存失败: {e})")
    plt.close(fig)


# ============================================================
# 5. 主流程
# ============================================================

if __name__ == "__main__":
    print("=" * 64)
    print("五维势能面 + 随机行走 + 分屏 3D 动画（FRLDM 有限力程液滴模型）")
    print("=" * 64)

    # 目标核素：U-236
    Z, N = 92, 144
    # 网格精度分层（重要）：库仑积分在默认 nz=nrho=30 下未收敛（断裂点误差 ~10 MeV）。
    # 随机行走用粗网格（只需相对能量），势垒/断裂点等核心数据用细网格，势能面切片用中网格。
    pes = FRLDMPES(Z, N)                                        # 粗网格：随机行走探索
    pes_fine = FRLDMPES(Z, N, nz=60, nrho=60, nsurf=64, nphi=40)  # 细网格：势垒/断裂点
    pes_mid = FRLDMPES(Z, N, nz=40, nrho=40, nsurf=64, nphi=40)   # 中网格：势能面切片
    print(f"\n核素 U-236:  R0={pes.R0:.3f} fm")
    print(f"  表面能系数 E_S0 = {pes.E_S0:.1f} MeV")
    print(f"  库仑能系数 E_C0 = {pes.E_C0:.1f} MeV")
    print(f"  裂变参数 x = E_C0/(2 E_S0) = {pes.E_C0 / (2 * pes.E_S0):.3f}")
    print(f"  (x < 1 → 存在裂变势垒，不会瞬间裂开)")

    # ---- 随机行走（方法）----
    T, n_steps, T_end = 8.0, 1000, 0.05
    q0 = np.array([0.3, 0.99, 0.0, 0.05, 0.05])  # 近球形起点（浅颈，轻微长椭球）
    print(f"\n[1] 在势能面上做 Metropolis 随机行走 (T={T} MeV 恒温越障 → 冷却到 {T_end} MeV, "
          f"{n_steps} 步)...")
    path, energies, components, accept = metropolis_walk(
        pes, q0, T=T, n_steps=n_steps, T_end=T_end, ratchet_elong=2.3, seed=1)
    print(f"  完成：接受率 = {accept:.2%}")
    print(f"  能量范围 [{energies.min():+.2f}, {energies.max():+.2f}] MeV")
    print(f"  末帧断裂程度 f = {neck_fraction(path[-1], pes.shape):.2f}")

    # ---- 极小能量路径（随机行走要逼近的"最优路径"）----
    print("\n[2] 计算极小能量（绝热）裂变路径...")
    min_path, min_E, min_C = min_energy_path(pes_fine)
    i_barrier = int(np.nanargmax(min_E))
    print(f"  裂变势垒高度 = {min_E[i_barrier]:.2f} MeV @ elong = {min_path[i_barrier, 0]:.2f}, "
          f"neck = {min_path[i_barrier, 1]:.2f}")
    print(f"  断裂点能量 = {min_E[-1]:+.2f} MeV @ elong = {min_path[-1, 0]:.2f}")

    # ---- 势能面切片（供静态图和动画共用）----
    print("\n[3] 计算势能面切片...")
    Q1, Q2, V = compute_surface_slice(pes_mid, n=34)

    # ---- 静态图 ----
    print("\n[4] 生成静态图...")
    plot_shape_schematic(pes)
    plot_pes_slice(pes, path, Q1, Q2, V, min_path=min_path)

    # ---- 动画 ----
    print("\n[5] 生成分屏 3D 动画...")
    animate_split(min_path, min_E, min_C, pes, Q1, Q2, V,
                  filename="核形状随机行走.gif", every=1, fps=6,
                  overlay_path=path, overlay_energies=energies)

    print("\n" + "=" * 64)
    print("完成。输出文件（本脚本目录下）：")
    print("  五参数形状示意图.png    —— 5 参数各自对核形状的影响")
    print("  势能面切片.png          —— 2D 势能面 + 行走轨迹 + 极小能量路径")
    print("  核形状随机行走.gif      —— 分屏动画（左形状/右势能面 + 行走轨迹叠加）")
    print("=" * 64)
