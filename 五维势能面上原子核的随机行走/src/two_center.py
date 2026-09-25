# -*- coding: utf-8 -*-
"""
two_center.py — 双中心（移位谐振子）Woods-Saxon 单粒子能级
============================================================
统一的非正交双中心基，取代「满核单中心 + 碎片级」两套人工插值。

基：围绕两个碎片中心 ±z_c 的移位谐振子函数（非正交），横（ρ）向沿用 Laguerre 基
R_{nρ Λ}(ρ)，纵（z）向用移位 Hermite {ψ_n((z−z_L)/b), ψ_n((z−z_R)/b)}。势仍是完整
双中心 WS 势（由 shape 的有符号距离给出，无需改 shape）。这是标准两中心壳模型（TCSM）。

分块：按 Ω=Λ+Σ 分块（不再按宇称——η≠0 时形状无反射对称，宇称不是好量子数）。每 Ω 块
含两条 Λ 分支：Λ_a=Ω−1/2（Σ=+1/2）与 Λ_b=Ω+1/2（Σ=−1/2）。求解广义本征问题
H C = S C E（scipy.linalg.eigh(H, S)）。

z0→0（两中心重合/紧凑形状）回退单中心 WoodsSaxon，保证球/紧致形状正确。
"""
import math

import numpy as np
from scipy import special
from scipy.linalg import eigh, LinAlgError

from shape import Shape3QS
from woods_saxon import (HBARC, M_NUC, WS_DEFAULTS, WoodsSaxon, potential_fields)


def trapz_weights(x):
    """均匀网格的梯形积分权重（x 单调）。"""
    w = np.full_like(x, 1.0)
    w[0] = w[-1] = 0.5
    return w * (x[-1] - x[0]) / (len(x) - 1)


def regularized_eigh(H, S, eps=1e-2):
    """Stabilize the generalized eigenvalue problem H C = S C E.

    The two-center shifted-HO basis becomes nearly linearly dependent when the
    center separation is not much larger than the oscillator length.  Truncate
    overlap eigenvectors with eigenvalue <= eps and transform to a standard
    eigenvalue problem.  This avoids spurious eigenvalues of order 1e5-1e7 MeV.
    """
    evals_S, U = np.linalg.eigh(S)
    keep = evals_S > eps
    if int(keep.sum()) == 0:
        raise LinAlgError("overlap matrix is numerically zero")
    evals_sqrt = np.sqrt(evals_S[keep])
    V = U[:, keep] / evals_sqrt[None, :]
    Ht = V.T @ H @ V
    Ht = 0.5 * (Ht + Ht.T)
    return np.linalg.eigvalsh(Ht)


def detect_centers(shape, q, nsurf=600, neck_ratio_max=0.6):
    """从形状剖面找两个碎片中心 (z_L, z_R)（fm）。无颈或颈未收缩返回 None。

    颈 = ρ(z) 的最深内部局部极小；左右瓣峰值即两个碎片中心。
    颈未充分收缩（rho_neck/rho_lobe > neck_ratio_max）时形状仍应视为单中心，
    返回 None 以回退单中心基——否则双中心基会把紧凑母核错误"碎片化"，
    壳修正假象性偏好非对称（厚颈区非对称偏好 ~5 MeV 假象）。
    """
    z, rho = shape.profile(q, n=nsurf)
    n = len(z)
    i_neck = None
    rho_neck = np.inf
    for i in range(1, n - 1):
        if rho[i] <= rho[i - 1] and rho[i] <= rho[i + 1]:
            if rho[i] < rho_neck:
                rho_neck = rho[i]
                i_neck = i
    if i_neck is None:
        return None
    if i_neck < 3 or i_neck > n - 4:
        return None
    i_L = int(np.argmax(rho[:i_neck]))
    i_R = int(np.argmax(rho[i_neck:])) + i_neck
    rho_lobe = max(rho[i_L], rho[i_R])
    if rho_lobe <= 0 or rho_neck / rho_lobe > neck_ratio_max:
        return None
    return float(z[i_L]), float(z[i_R])


class TwoCenterWoodsSaxon:
    """双中心（移位谐振子）Woods-Saxon 单粒子能级。

    用法与 WoodsSaxon 一致：
        ws = TwoCenterWoodsSaxon(Z, N, Nmax=12)
        e_p, e_n = ws.single_particle_spectrum(q)   # 排序能级，Kramers g=2
    """

    def __init__(self, Z, N, Nmax=12, nz_uni=80, nrho_gauss=32, nsurf=600,
                 grad_step=0.15, params=None, shape=None, shape_cls=Shape3QS,
                 z0_min=2.0, neck_ratio_max=0.6):
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
        self.S = p["lam_so"] * (HBARC / (2.0 * M_NUC)) ** 2       # 中子 SO fm²
        self.S_p = lam_so_p * (HBARC / (2.0 * M_NUC)) ** 2        # 质子 SO fm²
        self.Nmax = Nmax
        self.nsurf = nsurf
        self.grad_step = grad_step
        self.nz_uni = nz_uni
        self.nrho_gauss = nrho_gauss
        self.z0_min = z0_min
        self.neck_ratio_max = neck_ratio_max

        self.hbar_omega = 41.0 / self.A ** (1.0 / 3.0)   # MeV
        self.b = math.sqrt(HBARC ** 2 / (M_NUC * self.hbar_omega))  # fm

        self.shape = shape if shape is not None else shape_cls(self.R_ws)

        # 径向 Gauss-Laguerre 节点（两碎片共享 ρ 轴）
        t, wL = np.polynomial.laguerre.laggauss(nrho_gauss)
        self.t = t
        self.wL = wL
        self.rho_g = self.b * np.sqrt(t)   # fm（>0）

        self._omega_list = [i + 0.5 for i in range(Nmax + 1)]   # 1/2,3/2,...,Nmax+1/2
        self._n_ax = 2 * (Nmax + 1)         # 轴向基维度（两中心 × (Nmax+1)）

        # 单中心回退（z0→0 或紧凑形状）
        self._single = WoodsSaxon(Z, N, Nmax=Nmax, nz_gauss=40,
                                  nrho_gauss=nrho_gauss, nsurf=nsurf,
                                  grad_step=grad_step, params=params,
                                  shape=self.shape)

    # ---- 径向基（对给定 Λ，与 WoodsSaxon 相同）----
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

    # ---- 轴向移位 Hermite 基 + 重叠/动能矩阵 ----
    def _axial_basis(self, z_L, z_R, z_lo, z_hi):
        Nmax = self.Nmax
        nz_uni = self.nz_uni
        b = self.b
        z_g = np.linspace(z_lo, z_hi, nz_uni)
        w_z = trapz_weights(z_g)
        w_ax = w_z / b

        nax = 2 * (Nmax + 1)
        psi = np.zeros((nz_uni, nax))
        dpsi = np.zeros((nz_uni, nax))   # dψ/dξ（无量纲）
        for c, zc in enumerate((z_L, z_R)):
            xi = (z_g - zc) / b
            H = np.zeros((Nmax + 2, nz_uni))
            H[0] = 1.0
            H[1] = 2.0 * xi
            for n in range(1, Nmax + 1):
                H[n + 1] = 2.0 * xi * H[n] - 2.0 * n * H[n - 1]
            Z = np.zeros((Nmax + 2, nz_uni))
            for n in range(Nmax + 2):
                norm = 1.0 / math.sqrt(2.0 ** n * math.factorial(n) * math.sqrt(np.pi))
                Z[n] = norm * H[n] * np.exp(-0.5 * xi ** 2)
            dZ = np.zeros((Nmax + 2, nz_uni))
            for n in range(Nmax + 1):
                if n >= 1:
                    dZ[n] += math.sqrt(n / 2.0) * Z[n - 1]
                if n <= Nmax - 1:
                    dZ[n] -= math.sqrt((n + 1) / 2.0) * Z[n + 1]
            col0 = c * (Nmax + 1)
            for n in range(Nmax + 1):
                psi[:, col0 + n] = Z[n]
                dpsi[:, col0 + n] = dZ[n]

        S_z = np.einsum('k,ka,kb->ab', w_ax, psi, psi)
        T_z = 0.5 * self.hbar_omega * np.einsum('k,ka,kb->ab', w_ax, dpsi, dpsi)
        xi2 = (z_g / b) ** 2
        Z2 = 0.5 * self.hbar_omega * np.einsum('k,ka,kb->ab', w_ax * xi2, psi, psi)
        A_z = T_z + Z2
        return dict(psi=psi, dpsi=dpsi, z_g=z_g, w_ax=w_ax, S_z=S_z, A_z=A_z)

    # ---- 势场（均匀轴向 × Laguerre 径向网格）----
    def _fields(self, q, z_g):
        rho = self.b * np.sqrt(self.t)[None, :]          # (1, nrho)
        z = z_g[:, None]                                 # (nz_uni, 1)
        rho_p = np.broadcast_to(rho, (len(z_g), self.nrho_gauss))
        z_p = np.broadcast_to(z, (len(z_g), self.nrho_gauss))
        return potential_fields(self.shape, z_p, rho_p, q, self.a, self.V0,
                                self.kappa, self.I, self.Z, self.R_c,
                                grad_step=self.grad_step, nsurf=self.nsurf)

    # ---- 组装并对角化一个 Ω 块（广义本征问题）----
    def _solve_block(self, Omega, fields, Vr, Vz, Vrr, S, ax, W2):
        Lam_a = int(Omega - 0.5)
        Lam_b = int(Omega + 0.5)
        Nmax = self.Nmax
        nax = self._n_ax
        psi = ax['psi']
        dpsi = ax['dpsi']          # dψ/dξ（无量纲，与单中心 dZ 同约定）
        S_z = ax['S_z']
        A_z = ax['A_z']

        def states_for(lam):
            st = []
            n_rho_max = (Nmax - lam) // 2
            for nr in range(n_rho_max + 1):
                for a in range(nax):
                    nz = a % (Nmax + 1)
                    if nz + 2 * nr + lam <= Nmax:
                        st.append((a, nr))
            return st

        st_a = states_for(Lam_a)
        st_b = states_for(Lam_b)
        n_a, n_b = len(st_a), len(st_b)
        dim = n_a + n_b
        H = np.zeros((dim, dim))

        Vho = 0.5 * self.hbar_omega * (
            self.t[None, :] + (ax['z_g'][:, None] / self.b) ** 2)
        Vdiff = fields - Vho

        def _idx(st):
            return (np.array([a for a, _ in st]),
                    np.array([nr for _, nr in st]))

        def branch_matrix(st, P, lam, coef):
            """单分支 (Λ,Σ) 的 H = 中心势 + 对角自旋轨道，ZZ/PP 只算一次。"""
            m = len(st)
            if m == 0:
                return np.zeros((0, 0))
            a_idx, nr_idx = _idx(st)
            Psi = psi[:, a_idx]                      # (nz_uni, m)
            Pr = P[nr_idx, :]                        # (m, nrho)
            ZZ = np.einsum('ki,kj->ijk', Psi, Psi)   # (m, m, nz_uni)
            PP = np.einsum('ik,jk->ijk', Pr, Pr)     # (m, m, nrho)
            Hc = np.einsum('ipj,ipk,jk->ip', ZZ, PP, Vdiff * W2)
            Aterm = A_z[np.ix_(a_idx, a_idx)]
            Sterm = S_z[np.ix_(a_idx, a_idx)]
            E_rho = self.hbar_omega * (2 * nr_idx + lam + 1)
            delta = nr_idx[:, None] == nr_idx[None, :]
            Hc += (Aterm + Sterm * E_rho[:, None]) * delta
            Hso = coef * np.einsum('ipj,ipk,jk->ip', ZZ, PP, Vrr * W2)
            return Hc + Hso

        Pa, Da = self._radial(Lam_a)
        Pb, Db = self._radial(Lam_b)
        Ha = branch_matrix(st_a, Pa, Lam_a, -S * Lam_a)
        Hb = branch_matrix(st_b, Pb, Lam_b, +S * Lam_b)

        # 自旋轨道耦合项（与单中心同式，仅换移位基）
        if n_a > 0 and n_b > 0:
            a_a = np.array([s[0] for s in st_a])
            nr_a = np.array([s[1] for s in st_a])
            a_b = np.array([s[0] for s in st_b])
            nr_b = np.array([s[1] for s in st_b])
            ZZa = psi[:, a_a].T     # (n_a, nz_uni)
            Zb = psi[:, a_b].T      # (n_b, nz_uni)
            dZb = dpsi[:, a_b].T    # (n_b, nz_uni)
            ZM = np.einsum('aj,bj->abj', ZZa, Zb)     # ψ_a·ψ_b
            ZY = np.einsum('aj,bj->abj', ZZa, dZb)    # ψ_a·(∂_ξ ψ_b)
            MV = (Vz / np.maximum(self.rho_g[None, :], 1e-12)) * W2
            XV = Vz * W2 / self.b
            YV = Vr * W2 / self.b
            Pa_all = Pa[nr_a, :]   # (n_a, nrho)
            Pb_all = Pb[nr_b, :]   # (n_b, nrho)
            Db_all = Db[nr_b, :]   # (n_b, nrho)
            M = np.einsum('abj,ak,bk,jk->ab', ZM, Pa_all, Pb_all, MV)
            Xterm = np.einsum('abj,ak,bk,jk->ab', ZM, Pa_all, Db_all, XV)
            Yterm = np.einsum('abj,ak,bk,jk->ab', ZY, Pa_all, Pb_all, YV)
            Hab = S * (Lam_b * M + Xterm - Yterm)

        H[:n_a, :n_a] = Ha
        H[n_a:, n_a:] = Hb
        if n_a > 0 and n_b > 0:
            H[:n_a, n_a:] = Hab
            H[n_a:, :n_a] = Hab.T
        H = 0.5 * (H + H.T)

        # 重叠矩阵 S_full = S_z ⊗ I_ρ
        def build_S(st):
            m = len(st)
            Sblk = np.zeros((m, m))
            nr_arr = np.array([nr for _, nr in st])
            for i in range(m):
                for j in np.where(nr_arr == nr_arr[i])[0]:
                    Sblk[i, j] = S_z[st[i][0], st[j][0]]
            return Sblk

        Sfull = np.zeros((dim, dim))
        Sfull[:n_a, :n_a] = build_S(st_a)
        Sfull[n_a:, n_a:] = build_S(st_b)
        Sfull = 0.5 * (Sfull + Sfull.T)

        evals = regularized_eigh(H, Sfull)
        return evals

    # ---- 单粒子能级 ----
    def single_particle_spectrum(self, q):
        """返回 (质子能级排序数组, 中子能级排序数组)，Kramers g=2。"""
        centers = detect_centers(self.shape, q, self.nsurf, self.neck_ratio_max)
        if centers is None:
            return self._single.single_particle_spectrum(q)
        z_L, z_R = centers
        if abs(z_R - z_L) < self.z0_min * self.b:
            return self._single.single_particle_spectrum(q)

        z, _ = self.shape.profile(q, n=self.nsurf)
        margin = 4.0 * self.b
        z_lo = min(z.min(), z_L, z_R) - margin
        z_hi = max(z.max(), z_L, z_R) + margin

        ax = self._axial_basis(z_L, z_R, z_lo, z_hi)
        z_g = ax['z_g']
        W2 = np.outer(ax['w_ax'], 0.5 * self.wL)
        f = self._fields(q, z_g)
        Vn_r, Vn_z, Vn_rr = f["dVn_drho"], f["dVn_dz"], f["Vn_rr"]
        Vp_r, Vp_z, Vp_rr = f["dVp_drho"], f["dVp_dz"], f["Vp_rr"]

        p_levels, n_levels = [], []
        try:
            for Omega in self._omega_list:
                n_levels.extend(self._solve_block(
                    Omega, f["Vn"], Vn_r, Vn_z, Vn_rr, self.S, ax, W2))
                p_levels.extend(self._solve_block(
                    Omega, f["Vp"], Vp_r, Vp_z, Vp_rr, self.S_p, ax, W2))
        except LinAlgError:
            return self._single.single_particle_spectrum(q)

        p_levels = np.sort(np.asarray(p_levels))
        n_levels = np.sort(np.asarray(n_levels))
        return p_levels, n_levels
