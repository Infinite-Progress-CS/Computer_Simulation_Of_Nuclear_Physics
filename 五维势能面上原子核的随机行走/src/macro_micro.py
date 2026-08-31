"""
macro_micro.py — 宏观-微观模型整合
====================================================================
E(q) = E_液滴(q) + δE_壳(q) + δE_对(q)

  E_液滴 : FRLDM 有限力程液滴模型形变能（liquid_drop.py）
  δE_壳  : Strutinsky 壳修正（strutinsky.py，单粒子能级来自变形 Woods-Saxon）
  δE_对  : BCS 对修正（pairing.py）

接口与 FRLDMPES 对齐：energy_components(q) → (V_total, dV_s, dV_c, dE_sh, dE_pair)，
随机行走 / 极小能量路径 / 势能面切片代码无需改动即可切换宏微模型。
"""

import numpy as np

from liquid_drop import FRLDMPES
from woods_saxon import WoodsSaxon
from strutinsky import shell_correction
from pairing import pairing_correction


class MacroMicro:
    """宏观-微观总能量（液滴 + 壳修正 + 对修正）。

    energy_components(q) 返回 5 元组 (V_total, dV_s, dV_c, dE_sh, dE_pair)，
    与 FRLDMPES.energy_components(q) → (dV_s+dV_c, dV_s, dV_c, B_s, B_c) 的
    解包约定兼容（metropolis_walk 里 `V, dV_s, dV_c, _, _`）。
    """

    def __init__(self, Z, N, nz=60, nrho=60, nsurf=64, nphi=40,
                 Nmax=12, gamma_fac=1.1, p=4, g0_n=16.0, g0_p=16.0,
                 pair_window=6.0):
        self.Z, self.N = Z, N
        self.A = Z + N
        self.ld = FRLDMPES(Z, N, nz=nz, nrho=nrho, nsurf=nsurf, nphi=nphi)
        self.ws = WoodsSaxon(Z, N, Nmax=Nmax)
        self.shape = self.ld.shape
        self.R0 = self.ld.R0

        self.gamma = gamma_fac * self.ws.hbar_omega
        self.p = p
        self.G_n = g0_n / self.A
        self.G_p = g0_p / self.A
        self.pair_window = pair_window

        self._cache = {}

    # ---- 量子部分（壳修正 + 对修正），带缓存 ----
    def quantum_components(self, q):
        key = tuple(np.round(np.asarray(q, dtype=float), 6))
        if key in self._cache:
            return self._cache[key]

        e_p, e_n = self.ws.single_particle_spectrum(q)
        dE_sh_n, _ = shell_correction(e_n, self.N, self.gamma, p=self.p,
                                      degeneracy=2)
        dE_sh_p, _ = shell_correction(e_p, self.Z, self.gamma, p=self.p,
                                      degeneracy=2)
        dE_sh = dE_sh_n + dE_sh_p

        dE_p_n = pairing_correction(e_n, self.N, self.G_n,
                                    window=self.pair_window, degeneracy=2)
        dE_p_p = pairing_correction(e_p, self.Z, self.G_p,
                                    window=self.pair_window, degeneracy=2)
        dE_pair = dE_p_n + dE_p_p

        self._cache[key] = (dE_sh, dE_pair)
        return dE_sh, dE_pair

    # ---- 单粒子谱（供外部核对幻数 / 壳修正分量）----
    def single_particle_spectrum(self, q):
        return self.ws.single_particle_spectrum(q)

    # ---- 总能量（对齐 FRLDMPES 解包约定）----
    def energy_components(self, q):
        V_ld, dV_s, dV_c, _, _ = self.ld.energy_components(q)
        dE_sh, dE_pair = self.quantum_components(q)
        return V_ld + dE_sh + dE_pair, dV_s, dV_c, dE_sh, dE_pair

    def energy(self, q):
        return self.energy_components(q)[0]

    # ---- 便于分项对比：纯液滴能量 ----
    def liquid_drop_energy(self, q):
        return self.ld.energy(q)
