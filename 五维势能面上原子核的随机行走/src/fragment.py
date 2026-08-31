"""
fragment.py — 碎片核 (A, Z) 的能量（断裂点模型的输入）
====================================================================
裂变断裂瞬间，每个碎片核的能量 = 球液滴能 + 形变能 + 壳修正 + 对修正：

  E_frag(A, Z, ε) = [E_表面 + E_库仑 + E_体积对称]           (球)
                   + E_S0(B_s(ε)−1) + E_C0(B_c(ε)−1)         (形变)
                   + δE_壳(ε) + δE_对(ε)                      (量子，随形变)

- 液滴部分用 FRLDM 系数解析计算，形变能由椭球 B_s/B_c 形变函数给出。
- 量子部分复用 woods_saxon.WoodsSaxon + strutinsky.shell_correction
  + pairing.pairing_correction，在椭球（β₂ 形变）下求。
- 碎片基态 = 对四极形变 ε 求最小（幻数核球、中壳核形变）。

形变 ε 用 Nilsson 四极参数：ε>0 长椭球 (prolate)。幻数核（N=82）壳修正
在 ε=0 最负、形变破坏壳 → 保持球形；中壳核（Z=40/N=56 区）在有限 ε 有
形变壳闭合 → 形变极小。这是重现轻峰（形变 Z=40 区）的关键。
"""
import numpy as np

from woods_saxon import WoodsSaxon, WS_DEFAULTS
from strutinsky import shell_correction
from pairing import pairing_correction
from shape import ShapeSpheroid, spheroid_bs_bc

E2 = 1.44          # e²（MeV·fm）


class Fragment:
    """单个碎片核 (Z, N) 的能量，带形变 ε 的壳/对修正缓存。

    每次计算按需实例化（同一个核多次出现时用 cache 复用）。
    """

    def __init__(self, Z, N, Nmax=12, gamma_fac=1.1, p=4,
                 g0_n=16.0, g0_p=16.0, pair_window=6.0,
                 a_s=21.18466, kappa_s=2.345, r0=1.16, a_sym=23.5,
                 deformed=True, lam_so_p=None, lam_so_n=None):
        self.Z, self.N = Z, N
        self.A = Z + N
        self.I = (N - Z) / self.A
        self.deformed = deformed

        # ---- 液滴系数（FRLDM 1995 ADNDT 59,185，解析，无积分）----
        self.R0 = r0 * self.A ** (1.0 / 3.0)
        self.E_S0 = a_s * (1.0 - kappa_s * self.I ** 2) * self.A ** (2.0 / 3.0)   # 表面（含表面对称）
        self.E_C0 = (3.0 / 5.0) * Z * Z * E2 / self.R0                            # 库仑自能
        self.a_sym = a_sym

        # ---- 量子部分（Woods-Saxon + Strutinsky + BCS）----
        # 碎片用椭球形状（ε=0 即球）；父核/其他用途默认 3QS。
        if deformed:
            R_ws = WS_DEFAULTS["r0"] * self.A ** (1.0 / 3.0)
            ws_shape = ShapeSpheroid(R_ws)
        else:
            ws_shape = None
        params = {}
        if lam_so_p is not None:
            params['lam_so_p'] = lam_so_p
        if lam_so_n is not None:
            params['lam_so'] = lam_so_n
        params = params or None
        self.ws = WoodsSaxon(Z, N, Nmax=Nmax, shape=ws_shape, params=params)
        self.gamma = gamma_fac * self.ws.hbar_omega
        self.p = p
        self.G_n = g0_n / self.A
        self.G_p = g0_p / self.A
        self.pair_window = pair_window
        self._quantum = {}

    # ---- 量子修正（随形变 ε），按 ε 缓存 ----
    def quantum_components(self, eps=None):
        """形变 ε 下的 (δE_壳, δE_对)（MeV）。ε=None 取 0。"""
        eps = 0.0 if eps is None else float(eps)
        if eps in self._quantum:
            return self._quantum[eps]
        q = [eps] if self.deformed else [0.0, 0.0, 0.0, 0.0, 0.0]
        e_p, e_n = self.ws.single_particle_spectrum(q)
        dE_sh_n, _ = shell_correction(e_n, self.N, self.gamma, p=self.p, degeneracy=2)
        dE_sh_p, _ = shell_correction(e_p, self.Z, self.gamma, p=self.p, degeneracy=2)
        dE_sh = dE_sh_n + dE_sh_p

        dE_p_n = pairing_correction(e_n, self.N, self.G_n,
                                    window=self.pair_window, degeneracy=2)
        dE_p_p = pairing_correction(e_p, self.Z, self.G_p,
                                    window=self.pair_window, degeneracy=2)
        dE_pair = dE_p_n + dE_p_p

        self._quantum[eps] = (dE_sh, dE_pair)
        return dE_sh, dE_pair

    def shell_correction(self, eps=None):
        return self.quantum_components(eps)[0]

    def pairing_correction(self, eps=None):
        return self.quantum_components(eps)[1]

    # ---- 液滴部分 ----
    def liquid_drop_bulk(self):
        """球液滴能 = 表面 + 库仑 + 体积对称（MeV）。"""
        return self.E_S0 + self.E_C0 + self.a_sym * self.I ** 2 * self.A

    def liquid_drop_deformation(self, eps):
        """椭球形变能 = E_S0(B_s−1) + E_C0(B_c−1)（MeV）。球=0。"""
        if eps == 0.0:
            return 0.0
        Bs, Bc = spheroid_bs_bc(eps)
        return self.E_S0 * (Bs - 1.0) + self.E_C0 * (Bc - 1.0)

    # ---- 碎片总能（随形变）----
    def fragment_energy(self, eps=None):
        """形变 ε 下的碎片总能 = 液滴(球) + 形变能 + 壳 + 对（MeV）。"""
        dE_sh, dE_pair = self.quantum_components(eps)
        return (self.liquid_drop_bulk() + self.liquid_drop_deformation(eps)
                + dE_sh + dE_pair)

    # ---- 基态（对 ε 求最小）----
    def ground_state(self, eps_list=(0.0, 0.1, 0.2, 0.3, 0.4)):
        """对四极形变 ε 求最小，返回 (E_gs, ε_gs)（MeV）。"""
        Emin, epsmin = np.inf, 0.0
        for eps in eps_list:
            E = self.fragment_energy(eps)
            if E < Emin:
                Emin, epsmin = E, eps
        return float(Emin), float(epsmin)

    def ground_state_energy(self, eps_list=(0.0, 0.1, 0.2, 0.3, 0.4)):
        """基态能（MeV）。"""
        return self.ground_state(eps_list)[0]
