"""
woods_saxon.py — 变形 Woods-Saxon 单粒子能级（宏观-微观模型的量子部分）
====================================================================
单粒子势用"universal Woods-Saxon"（Dudek et al. 1981）：
  V(r) = V0·[1 ∓ κ·(N−Z)/A] / (1 + exp(d(r)/a))   +  V_so(r)   +  V_Coul(r, 仅质子)

  - d(r) = 到 3QS 表面的带符号距离（核内负、外正）
  - V_so  = −λ(ħ/2mc)² [∇V_cent × p]·σ        （轴对称分解为对角项 + 耦合项）
  - V_Coul = 均匀带电球近似（仅质子）

基：球对称谐振子基（好量子数 Ω=Λ+Σ、宇称 π=(−1)^N），按 (Ω, π) 分块对角化。
截断 N_max = 12。每个单粒子能级因时间反演（Kramers）2 重简并。

同时提供【球对称精确解】作校验基准：径向 Schrödinger 方程（含自旋轨道）精确数值解。
"""

import math

import numpy as np
from scipy import special
from scipy.linalg import eigh_tridiagonal

from shape import Shape3QS

# ---- 物理常数 ----
HBARC = 197.327       # MeV·fm
M_NUC = 938.918       # MeV（核子质量）
E2 = 1.44             # MeV·fm（e²）
HBAR2_2M = HBARC ** 2 / (2.0 * M_NUC)   # = 20.73 MeV·fm²（ħ²/2m）
TWO_M_HBAR2 = 2.0 * M_NUC / HBARC ** 2  # = 0.04824 fm⁻²·MeV⁻¹

# ---- universal Woods-Saxon 参数 ----
WS_DEFAULTS = dict(
    V0=49.6,        # MeV 中心势深度（取正，负号在代码里加）
    kappa=0.86,     # 同位旋不对称系数
    r0=1.275,       # fm 半径参数
    a=0.70,         # fm 弥散度
    lam_so=35.0,    # 自旋轨道强度（中子，无量纲）
    lam_so_p=None,  # 质子自旋轨道强度（None = 用 lam_so，即无同位旋依赖）
    r0_coul=1.16,   # fm 电荷半径参数
)


def _spherical_fermi(x):
    """Fermi 函数 f = 1/(1+e^x)，x = (r−R)/a。"""
    return 1.0 / (1.0 + np.exp(x))


# ======================================================================
# 球对称精确解（校验基准）
# ======================================================================

def spherical_single_particle_levels(Z, N, params=None, Rmax=20.0, h=0.02,
                                     l_max=12):
    """球对称 Woods-Saxon 径向精确解。返回 (中子能级, 质子能级)（排序，含简并说明）。

    径向方程（u = r·R）：
      −u'' + [l(l+1)/r² + 2m/ħ²(V_cent + V_so + V_coul)] u = (2mE/ħ²) u
    自旋轨道（球）：
      V_so = −λ(ħ/2mc)² (1/r)(dV_cent/dr) L·σ
           = −S (dV_cent/dr)/r · [j(j+1)−l(l+1)−3/4]
    每个 (n,l,j) 能级简并度 2j+1。
    """
    p = dict(WS_DEFAULTS)
    if params:
        p.update(params)
    A = Z + N
    I = (N - Z) / A
    R_ws = p["r0"] * A ** (1.0 / 3.0)
    R_c = p["r0_coul"] * A ** (1.0 / 3.0)
    S = p["lam_so"] * (HBARC / (2.0 * M_NUC)) ** 2   # fm²

    r = np.arange(h, Rmax + h, h)
    nr = len(r)

    def central_and_so(V0fac):
        f = _spherical_fermi((r - R_ws) / p["a"])
        Vc = -V0fac * f
        dVdr = V0fac * np.exp((r - R_ws) / p["a"]) / (
            p["a"] * (1.0 + np.exp((r - R_ws) / p["a"])) ** 2)
        return Vc, dVdr

    Vn_cent, dVn = central_and_so(p["V0"] * (1.0 - p["kappa"] * I))
    Vp_cent, dVp = central_and_so(p["V0"] * (1.0 + p["kappa"] * I))
    # 质子库仑（均匀带电球）
    Vcoul = np.where(r <= R_c,
                     (Z - 1) * E2 * (3.0 - (r / R_c) ** 2) / (2.0 * R_c),
                     (Z - 1) * E2 / r)

    def solve_for(Vcent, dV):
        levels = []
        for l in range(l_max + 1):
            for jj in (l + 0.5, l - 0.5):
                if jj < 0.5:
                    continue
                s_ls = (jj * (jj + 1) - l * (l + 1) - 0.75)   # 2·⟨L·s⟩ 已并入
                Vso = -S * (dV / r) * s_ls
                W = TWO_M_HBAR2 * (Vcent + Vso) + l * (l + 1) / r ** 2
                d = 2.0 / h ** 2 + W
                e = np.full(nr - 1, -1.0 / h ** 2)
                evals = eigh_tridiagonal(d, e, eigvals_only=True)
                for E in evals:
                    if E < 0.0:   # 束缚态（E = λ·ħ²/2m，λ<0 → E<0）
                        levels.append((E * HBAR2_2M, int(round(2 * jj + 1))))
        levels.sort(key=lambda x: x[0])
        return levels

    neutron_levels = solve_for(Vn_cent, dVn)
    proton_levels = solve_for(Vp_cent + Vcoul, dVp)   # 中心势含库仑，自旋轨道只用核梯度
    return neutron_levels, proton_levels


# ======================================================================
# 变形 Woods-Saxon（轴向谐振子基对角化）
# ======================================================================

class WoodsSaxon:
    """变形 Woods-Saxon 单粒子能级。

    用法：
        ws = WoodsSaxon(Z, N, Nmax=12)
        e_p, e_n = ws.single_particle_spectrum(q)   # 各返回排序后的"轨道"能级
    每个返回能级因时间反演（Kramers）2 重简并（配对时按 g=2 处理）。
    """

    def __init__(self, Z, N, Nmax=12, nz_gauss=40, nrho_gauss=32, nsurf=600,
                 grad_step=0.15, params=None, shape=None):
        self.Z, self.N = Z, N
        self.A = Z + N
        self.I = (N - Z) / self.A
        p = dict(WS_DEFAULTS)
        if params:
            p.update(params)
        self.V0 = p["V0"]
        self.kappa = p["kappa"]
        self.a = p["a"]                 # fm 弥散度
        self.R_ws = p["r0"] * self.A ** (1.0 / 3.0)
        self.R_c = p["r0_coul"] * self.A ** (1.0 / 3.0)
        lam_so_p = p["lam_so"] if p.get("lam_so_p") is None else p["lam_so_p"]
        self.S = p["lam_so"] * (HBARC / (2.0 * M_NUC)) ** 2       # 中子自旋轨道强度 fm²
        self.S_p = lam_so_p * (HBARC / (2.0 * M_NUC)) ** 2        # 质子自旋轨道强度 fm²
        self.Nmax = Nmax
        self.nsurf = nsurf
        self.grad_step = grad_step

        # 谐振子参数
        self.hbar_omega = 41.0 / self.A ** (1.0 / 3.0)   # MeV
        self.b = math.sqrt(HBARC ** 2 / (M_NUC * self.hbar_omega))  # fm

        self.shape = shape if shape is not None else Shape3QS(self.R_ws)

        # 谐振子基的（Ω,π）块标签：预生成 Ω 列表
        self._omega_list = [i + 0.5 for i in range(Nmax + 1)]  # 1/2,3/2,...,Nmax+1/2

        # Gauss 节点
        self._setup_gauss(nz_gauss, nrho_gauss)

        # 轴向基（z 方向 1D 谐振子 + 导数）
        self._setup_axial()

    # ---- Gauss 节点 ----
    def _setup_gauss(self, nz, nrho):
        xi, wH = np.polynomial.hermite.hermgauss(nz)      # ∫ e^{−ξ²} g dξ ≈ Σ wH g
        t, wL = np.polynomial.laguerre.laggauss(nrho)      # ∫ e^{−t} g dt ≈ Σ wL g
        self.xi, self.wH = xi, wH
        self.t, self.wL = t, wL
        self.z_g = self.b * xi                             # fm
        self.rho_g = self.b * np.sqrt(t)                   # fm（>0）
        self.nz_g, self.nrho_g = nz, nrho
        # 谐振子波函数 ψ_n(ξ) 已含 e^{−ξ²/2}，故 ∫ψ_nψ_m f dξ = Σ wH ψ_nψ_m f · e^{ξ²}
        # 径向 P 为不含 e^{−t/2} 的多项式部分，故 ∫RR' f ρdρ = Σ (0.5 wL) P P' f
        self.wH_z = wH * np.exp(xi ** 2)
        self.W2 = np.outer(self.wH_z, 0.5 * wL)

    # ---- 轴向基（z 方向）----
    def _setup_axial(self):
        nmax = self.Nmax + 1   # 多算一层供导数使用
        xi = self.xi
        H = np.zeros((nmax + 1, len(xi)))
        H[0] = 1.0
        if nmax >= 1:
            H[1] = 2.0 * xi
        for n in range(1, nmax):
            H[n + 1] = 2.0 * xi * H[n] - 2.0 * n * H[n - 1]
        Z = np.zeros_like(H)
        for n in range(nmax + 1):
            norm = 1.0 / math.sqrt(2.0 ** n * math.factorial(n) * math.sqrt(np.pi))
            Z[n] = norm * H[n] * np.exp(-0.5 * xi ** 2)
        self._Z = Z          # ψ_n(ξ)，n=0..Nmax+1
        # 导数 ψ'_n = √(n/2) ψ_{n−1} − √((n+1)/2) ψ_{n+1}
        dZ = np.zeros_like(Z)
        for n in range(nmax + 1):
            if n >= 1:
                dZ[n] += math.sqrt(n / 2.0) * Z[n - 1]
            if n <= nmax - 1:
                dZ[n] -= math.sqrt((n + 1) / 2.0) * Z[n + 1]
        self._dZ = dZ

    # ---- 径向基（对给定 Λ）----
    def _radial(self, lam):
        lam = int(lam)
        n_rho_max = (self.Nmax - lam) // 2
        if n_rho_max < 0:
            return None, None
        t = self.t
        P = np.zeros((n_rho_max + 1, len(t)))
        D = np.zeros_like(P)
        for nr in range(n_rho_max + 1):
            Ln = special.eval_genlaguerre(nr, lam, t)
            c = math.sqrt(2.0 * math.factorial(nr) / math.factorial(nr + lam))
            P[nr] = c * t ** (lam / 2.0) * Ln
            dPdt = -c * t ** (lam / 2.0) * (
                special.eval_genlaguerre(nr - 1, lam + 1, t) if nr >= 1 else 0.0)
            if lam > 0:
                dPdt += c * (lam / 2.0) * t ** (lam / 2.0 - 1.0) * Ln
            D[nr] = np.sqrt(t) * (-P[nr] + 2.0 * dPdt)
        return P, D

    # ---- 带符号距离（到 3QS 表面多段线，含分段精化）----
    def _surface(self, q):
        z, rho = self.shape.profile(q, n=self.nsurf)
        return z, rho

    def _signed_distance(self, rho_p, z_p, q):
        zs, rs = self._surface(q)
        dz = z_p[:, None] - zs[None, :]
        dr = rho_p[:, None] - rs[None, :]
        d2 = dz * dz + dr * dr
        k0 = np.argmin(d2, axis=1)
        n = self.nsurf
        kA = np.clip(k0 - 1, 0, n - 1)
        kB = np.clip(k0, 0, n - 1)
        kC = np.clip(k0 + 1, 0, n - 1)
        dA = self._seg_dist(z_p, rho_p, zs[kA], rs[kA], zs[kB], rs[kB])
        dB = self._seg_dist(z_p, rho_p, zs[kB], rs[kB], zs[kC], rs[kC])
        d = np.minimum(dA, dB)
        # 符号：核内（ρ_p < ρ(z_p)）为负
        rho_surf = np.interp(z_p, zs, rs, left=rs[0], right=rs[-1])
        sign = np.where(rho_p < rho_surf, -1.0, 1.0)
        return sign * d

    @staticmethod
    def _seg_dist(zp, rp, z1, r1, z2, r2):
        dzs = z2 - z1
        drs = r2 - r1
        L2 = dzs * dzs + drs * drs
        tt = ((zp - z1) * dzs + (rp - r1) * drs) / np.maximum(L2, 1e-30)
        tt = np.clip(tt, 0.0, 1.0)
        zc = z1 + tt * dzs
        rc = r1 + tt * drs
        return np.sqrt((zp - zc) ** 2 + (rp - rc) ** 2)

    # ---- 势场（中心势 + 梯度，在 Gauss 节点上）----
    def _potential_fields(self, q):
        zs, rs = self._surface(q)
        xi = self.xi
        t = self.t
        z = self.b * xi[:, None]            # (nz,1) fm
        rho = self.b * np.sqrt(t)[None, :]  # (1,nrho) fm
        rho_p = np.broadcast_to(rho, (len(xi), len(t)))
        z_p = np.broadcast_to(z, (len(xi), len(t)))
        # 展平求距离
        d = self._signed_distance(rho_p.ravel(), z_p.ravel(), q).reshape(rho_p.shape)

        I = self.I
        fac_n = self.V0 * (1.0 - self.kappa * I)
        fac_p = self.V0 * (1.0 + self.kappa * I)
        f = _spherical_fermi(d / self.a)
        Vn = -fac_n * f
        Vp = -fac_p * f

        # 质子库仑（均匀带电球）
        r = np.sqrt(rho ** 2 + z ** 2)
        Vcoul = np.where(r <= self.R_c,
                         (self.Z - 1) * E2 * (3.0 - (r / self.R_c) ** 2) / (2.0 * self.R_c),
                         (self.Z - 1) * E2 / r)
        Vp = Vp + Vcoul

        # 梯度：对 Vn、Vp 分别做中心差分
        gs = self.grad_step
        rho_plus = np.broadcast_to(rho + gs, rho_p.shape)
        rho_minus = np.broadcast_to(np.abs(rho - gs), rho_p.shape)

        def grad(V):
            # 直接对 V(ρ,z) 有限差分（中心差分，利用 V 偶对称反射处理 ρ−gs<0）
            dp = self._signed_distance(rho_plus.ravel(), z_p.ravel(), q).reshape(rho_p.shape)
            dm = self._signed_distance(rho_minus.ravel(), z_p.ravel(), q).reshape(rho_p.shape)
            zp_d = self._signed_distance(rho_p.ravel(), (z_p + gs).ravel(), q).reshape(rho_p.shape)
            zm_d = self._signed_distance(rho_p.ravel(), (z_p - gs).ravel(), q).reshape(rho_p.shape)
            fp = _spherical_fermi(dp / self.a)
            fm = _spherical_fermi(dm / self.a)
            fzp = _spherical_fermi(zp_d / self.a)
            fzm = _spherical_fermi(zm_d / self.a)
            return fp, fm, fzp, fzm

        # 中子
        fpn, fmn, fzpn, fzmn = grad(Vn)
        dVn_drho = -fac_n * (fpn - fmn) / (2.0 * gs)
        dVn_dz = -fac_n * (fzpn - fzmn) / (2.0 * gs)
        # 质子（中心势梯度，仅核部分——自旋轨道只取核梯度，库仑中心势单独加在 Vp 上）
        fpp, fmp, fzpp, fzmp = grad(Vp)
        dVp_drho = -fac_p * (fpp - fmp) / (2.0 * gs)
        dVp_dz = -fac_p * (fzpp - fzmp) / (2.0 * gs)

        # 自旋轨道对角项 ⟨(1/r)dV/dr⟩ = ⟨(1/ρ)∂V/∂ρ⟩：除以 ρ（= b√t）。
        # 数值校验（_test_soratio.py）：/ρ 与 dblquad 精确值一致（1.000），/√t 偏大 b 倍。
        Vn_rr = dVn_drho / np.maximum(rho, 1e-12)
        Vp_rr = dVp_drho / np.maximum(rho, 1e-12)

        return dict(Vn=Vn, Vp=Vp, dVn_drho=dVn_drho, dVn_dz=dVn_dz,
                    dVp_drho=dVp_drho, dVp_dz=dVp_dz,
                    Vn_rr=Vn_rr, Vp_rr=Vp_rr)

    # ---- 组装并对角化一个 (Ω, π) 块 ----
    def _solve_block(self, Omega, parity, fields, Vr, Vz, Vrr, S):
        """fields 字典包含 Vn/Vp 中心势；Vr/Vz/Vrr 为该核子的梯度场；S 为该核子自旋轨道强度。"""
        Lam_a = int(Omega - 0.5)
        Lam_b = int(Omega + 0.5)
        Nmax = self.Nmax
        Z = self._Z
        dZ = self._dZ

        def states_for(lam):
            lam = int(lam)
            st = []
            n_rho_max = (Nmax - lam) // 2
            for nr in range(n_rho_max + 1):
                nz_max = Nmax - 2 * nr - lam
                for nz in range(nz_max + 1):
                    if (nz + lam) % 2 == parity:   # (−1)^{nz+Λ} = π
                        st.append((nz, nr))
            return st

        st_a = states_for(Lam_a)
        st_b = states_for(Lam_b)
        n_a, n_b = len(st_a), len(st_b)
        dim = n_a + n_b
        H = np.zeros((dim, dim))

        def ho_energy(nz, nr, lam):
            N = nz + 2 * nr + lam
            return self.hbar_omega * (N + 1.5)

        Vho = 0.5 * self.hbar_omega * (
            self.t[None, :] + self.xi[:, None] ** 2)   # ½ħω0(u²+ξ²) = ½ħω0(t+ξ²)
        Vdiff = fields - Vho

        # 径向基
        Pa, Da = self._radial(Lam_a)
        Pb, Db = self._radial(Lam_b)

        # 中心 + HO 势能项（分块内对角）
        def central_matrix(st, P, lam):
            S = len(st)
            if S == 0:
                return np.zeros((0, 0))
            ZZ = np.zeros((S, S, self.nz_g))
            PP = np.zeros((S, S, self.nrho_g))
            for i, (nz, nr) in enumerate(st):
                for ip, (nzp, nrp) in enumerate(st):
                    ZZ[i, ip, :] = Z[nz, :] * Z[nzp, :]
                    PP[i, ip, :] = P[nr, :] * P[nrp, :]
            B = Vdiff * self.W2
            Hc = np.einsum('ipj,ipk,jk->ip', ZZ, PP, B)
            for i, (nz, nr) in enumerate(st):
                Hc[i, i] += ho_energy(nz, nr, lam)
            return Hc

        Ha = central_matrix(st_a, Pa, Lam_a)
        Hb = central_matrix(st_b, Pb, Lam_b)

        # 自旋轨道对角项（O_C = 2Σ ħ Λ V_ρ/ρ）
        def so_diag(st, P, lam, coef):
            S = len(st)
            if S == 0:
                return np.zeros((0, 0))
            ZZ = np.zeros((S, S, self.nz_g))
            PP = np.zeros((S, S, self.nrho_g))
            for i, (nz, nr) in enumerate(st):
                for ip, (nzp, nrp) in enumerate(st):
                    ZZ[i, ip, :] = Z[nz, :] * Z[nzp, :]
                    PP[i, ip, :] = P[nr, :] * P[nrp, :]
            B = Vrr * self.W2
            return coef * np.einsum('ipj,ipk,jk->ip', ZZ, PP, B)

        # 对角自旋轨道：V_so^diag = −S·(dV/dρ)/ρ·2ΛΣ；Σ=+1/2 → −SΛ，Σ=−1/2 → +SΛ
        Ha += so_diag(st_a, Pa, Lam_a, -S * Lam_a)
        Hb += so_diag(st_b, Pb, Lam_b, +S * Lam_b)

        # 自旋轨道耦合项（l_-σ_+ 部分，微分作用于 Λ_b 分支）：
        #   ⟨a|V_so|b⟩ = S [Λ_b⟨Vz/ρ⟩ + ⟨V_z(∂_ρ R_b)R_a⟩ − ⟨V_ρ(∂_z Z_b)Z_a⟩]
        if n_a > 0 and n_b > 0:
            ZZa = Z[[s[0] for s in st_a], :]           # (n_a, nz) 分支 a 的 z 函数
            Zb = Z[[s[0] for s in st_b], :]            # (n_b, nz) 分支 b 的 z 函数
            dZb = dZ[[s[0] for s in st_b], :]          # 分支 b 的 ∂_ξ
            Hab = np.zeros((n_a, n_b))
            ZM = np.einsum('aj,bj->abj', ZZa, Zb)      # Z_a·Z_b（用于 M 和 ∂_ρ 项）
            ZY = np.einsum('aj,bj->abj', ZZa, dZb)     # Z_a·(∂_ξ Z_b)（用于 ∂_z 项）
            # 把 (ρ,z) Gauss 权重 W2 并入势场
            MV = (Vz / np.maximum(self.rho_g[None, :], 1e-12)) * self.W2
            # X/Y 项：D_b = b²e^{t/2}∂_ρR_b、dZ = ∂_ξψ = b·∂_zZ 都多含一个 b，
            # 数值校验（_test_cross.py）证实 X/Y 比物理值大 b 倍，故除以 b。
            XV = Vz * self.W2 / self.b                 # ∂_ρ 项用 V_z
            YV = Vr * self.W2 / self.b                 # ∂_z 项用 V_ρ
            for a_i, (nz_a, nr_a) in enumerate(st_a):
                Pa_row = Pa[nr_a, :]
                for b_i, (nz_b, nr_b) in enumerate(st_b):
                    Pb_row = Pb[nr_b, :]
                    Db_row = Db[nr_b, :]
                    M = np.einsum('j,k,jk->', ZM[a_i, b_i, :], Pa_row * Pb_row, MV)
                    Xterm = np.einsum('j,k,jk->', ZM[a_i, b_i, :], Pa_row * Db_row, XV)
                    Yterm = np.einsum('j,k,jk->', ZY[a_i, b_i, :], Pa_row * Pb_row, YV)
                    Hab[a_i, b_i] = S * (Lam_b * M + Xterm - Yterm)

        # 组装
        H[:n_a, :n_a] = Ha
        H[n_a:, n_a:] = Hb
        if n_a > 0 and n_b > 0:
            H[:n_a, n_a:] = Hab
            H[n_a:, :n_a] = Hab.T
        H = 0.5 * (H + H.T)   # 显式对称化（数值边界项）
        evals = np.linalg.eigh(H)[0]
        return evals

    # ---- 单粒子能级 ----
    def single_particle_spectrum(self, q):
        """返回 (质子能级排序数组, 中子能级排序数组)。

        每个能级代表一个"轨道"，因时间反演 2 重简并（配对/Strutinsky 时按 g=2）。
        """
        fields_n = None
        fields_p = None
        p_levels, n_levels = [], []
        for Omega in self._omega_list:
            for parity in (0, 1):   # 0 = 偶宇称，1 = 奇宇称
                # 判断该 (Ω,π) 是否有态
                lam_min = int(Omega - 0.5)
                lam_max = int(Omega + 0.5)
                if lam_min < 0:
                    continue
                has = False
                for lam in (lam_min, lam_max):
                    if lam < 0:
                        continue
                    n_rho_max = (self.Nmax - lam) // 2
                    for nr in range(n_rho_max + 1):
                        nz_max = self.Nmax - 2 * nr - lam
                        for nz in range(nz_max + 1):
                            if (nz + lam) % 2 == parity:
                                has = True
                if not has:
                    continue
                # 取势场（惰性：同一 q 只算一次）
                if fields_n is None:
                    f = self._potential_fields(q)
                    fields_n = dict(Vn=f["Vn"])
                    fields_p = dict(Vp=f["Vp"])
                    Vn_r, Vn_z, Vn_rr = f["dVn_drho"], f["dVn_dz"], f["Vn_rr"]
                    Vp_r, Vp_z, Vp_rr = f["dVp_drho"], f["dVp_dz"], f["Vp_rr"]
                n_levels.extend(self._solve_block(
                    Omega, parity, fields_n["Vn"], Vn_r, Vn_z, Vn_rr, self.S))
                p_levels.extend(self._solve_block(
                    Omega, parity, fields_p["Vp"], Vp_r, Vp_z, Vp_rr, self.S_p))
        p_levels = np.sort(np.asarray(p_levels))
        n_levels = np.sort(np.asarray(n_levels))
        return p_levels, n_levels
