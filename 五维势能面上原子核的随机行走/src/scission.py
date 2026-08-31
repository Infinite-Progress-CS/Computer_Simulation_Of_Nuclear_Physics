"""
scission.py — 断裂点模型（Wilkins-Steinberg-Chasman 1976）
====================================================================
裂变碎片产额由断裂瞬间的势能决定：

  Y(A_L, Z_L) ∝ exp(−V_scission / T)

  V_scission = E_frag(A_L, Z_L) + E_frag(A_H, Z_H) + E_Coul
  E_Coul     = e² Z_L Z_H / d         （碎片间库仑排斥）
  d          = r0(A_L^{1/3} + A_H^{1/3}) + d_extra    （断裂距离）
"""
import numpy as np

E2 = 1.44          # e²（MeV·fm）


def scission_distance(A_L, A_H, r0=1.16, d_extra=2.0):
    """碎片质心距离 d（fm）。d_extra 为断裂点颈外延伸量。"""
    return r0 * (A_L ** (1.0 / 3.0) + A_H ** (1.0 / 3.0)) + d_extra


def coulomb_mutual(Z_L, Z_H, d):
    """碎片间库仑能 e² Z_L Z_H / d（MeV）。"""
    return E2 * Z_L * Z_H / d


def scission_energy(frag_L, frag_H, d_extra=2.0):
    """断裂点势能 V_scission = E_frag(L) + E_frag(H) + E_Coul（MeV）。

    碎片能取各自形变基态（对四极形变 ε 求最小，绝热近似）。碎片若为
    Fragment（deformed=True）用 ground_state_energy()，否则回退 fragment_energy()。
    """
    Z_L, Z_H = frag_L.Z, frag_H.Z
    A_L, A_H = frag_L.A, frag_H.A
    d = scission_distance(A_L, A_H, d_extra=d_extra)
    E_L = (frag_L.ground_state_energy() if hasattr(frag_L, 'ground_state_energy')
           else frag_L.fragment_energy())
    E_H = (frag_H.ground_state_energy() if hasattr(frag_H, 'ground_state_energy')
           else frag_H.fragment_energy())
    return E_L + E_H + coulomb_mutual(Z_L, Z_H, d)


def yield_factor(V_scission, T):
    """未归一产额因子 exp(−V/T)。"""
    return float(np.exp(-V_scission / T))


def scission_distance_deformed(A_L, A_H, eps_L, eps_H, r0=1.16, d_extra=2.0):
    """变形碎片的质心距离 d（fm）。

    长椭球（ε>0）对称轴半长 c = R0(1+2ε/3)，断裂距离 = 两碎片半长之和 + 颈延伸。
    碎片被库仑拉伸（ε 增大）→ d 增大 → 库仑能 e²Z_LZ_H/d 降低（进一步倾向拉伸），
    与壳修正（幻数核抵抗形变）竞争，给出断裂点形变。
    """
    R0_L = r0 * A_L ** (1.0 / 3.0)
    R0_H = r0 * A_H ** (1.0 / 3.0)
    return R0_L * (1.0 + 2.0 * eps_L / 3.0) + R0_H * (1.0 + 2.0 * eps_H / 3.0) + d_extra


def scission_energy_fixed_eps(frag_L, frag_H, eps, d_extra=2.0):
    """固定形变 ε（两碎片同 ε）下的断裂点势能 V_scission（MeV）。

    与 scission_energy() 的区别：碎片不在各自形变基态，而是取断裂点拉伸形变 ε，
    断裂距离随 ε 变长。用于研究断裂点形变如何洗掉 Z=50 质子幻数（Sn 双幻数）。
    """
    Z_L, Z_H = frag_L.Z, frag_H.Z
    A_L, A_H = frag_L.A, frag_H.A
    d = scission_distance_deformed(A_L, A_H, eps, eps, d_extra=d_extra)
    E_L = frag_L.fragment_energy(eps)
    E_H = frag_H.fragment_energy(eps)
    return E_L + E_H + coulomb_mutual(Z_L, Z_H, d)


def scission_energy_2d(frag_L, frag_H, eps_grid, d_extra=2.0):
    """二维最小化断裂点势能：min_{ε_L,ε_H} [E_L(ε_L)+E_H(ε_H)+e²Z_LZ_H/d(ε_L,ε_H)]。

    eps_grid 为形变列表（两碎片同网格）。返回 (V_min, ε_L_opt, ε_H_opt)。
    """
    Z_L, Z_H = frag_L.Z, frag_H.Z
    A_L, A_H = frag_L.A, frag_H.A
    best = (np.inf, 0.0, 0.0)
    for eps_L in eps_grid:
        E_L = frag_L.fragment_energy(eps_L)
        for eps_H in eps_grid:
            d = scission_distance_deformed(A_L, A_H, eps_L, eps_H, d_extra=d_extra)
            V = E_L + frag_H.fragment_energy(eps_H) + coulomb_mutual(Z_L, Z_H, d)
            if V < best[0]:
                best = (V, eps_L, eps_H)
    return best
