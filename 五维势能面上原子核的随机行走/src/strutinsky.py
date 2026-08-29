"""
strutinsky.py — Strutinsky 壳修正
====================================================================
把一组单粒子能级（每个 2 重简并，Kramers）平滑，得到平滑能级密度 g̃(ε)，
壳修正 δE_壳 = 占据能之和 − 平滑能之和。

平滑函数（曲率修正高斯）：
  f_p(x) = (1/√π) e^{−x²} · L_p^{1/2}(x²)
  g̃(ε) = Σ_i (g_i/γ) f_p((ε−ε_i)/γ)，g_i = 2

壳修正（Strutinsky 1968）：
  δE_壳 = 2 Σ_{i=1}^{N/2} ε_i − ∫^{λ̃}_{−∞} ε g̃(ε) dε
  λ̃ 由 N = ∫^{λ̃} g̃ dε 定（平滑费米能）。
"""

import numpy as np
from scipy import special


def smoothing_kernel(x, p=6):
    """曲率修正高斯 f_p(x) = (1/√π) e^{−x²} L_p^{1/2}(x²)。"""
    lag = special.eval_genlaguerre(p, 0.5, x ** 2)
    return (1.0 / np.sqrt(np.pi)) * np.exp(-x ** 2) * lag


def shell_correction(levels, n_particles, gamma, p=6, degeneracy=2):
    """计算壳修正 δE_壳（MeV）。

    参数：
      levels     : 单粒子"轨道"能级（排序数组，每个 degeneracy 重简并）
      n_particles: 核子数（中子 N 或质子 Z）
      gamma      : 平滑宽度（≈ 0.9 ħω0）
    """
    levels = np.asarray(levels, dtype=float)
    if len(levels) == 0:
        return 0.0
    n_occ = int(round(n_particles / degeneracy))   # 占据的轨道数
    if n_occ <= 0 or n_occ > len(levels):
        return 0.0
    lam_F = levels[n_occ - 1]                      # 尖锐费米能

    # 平滑能级密度 g̃(ε) 的数值网格
    # 上界必须取最高能级 levels[-1]（而非费米能 lam_F）：费米面之上的空能级
    # 同样参与平滑，截断它们会让平滑能级密度缺态、壳修正产生系统性偏差。
    eps_lo = levels[0] - 4.0 * gamma
    eps_hi = levels[-1] + 5.0 * gamma
    de = gamma / 20.0
    eps = np.arange(eps_lo, eps_hi, de)

    # g̃(ε) = Σ_i (g/γ) f_p((ε−ε_i)/γ)
    g_smooth = np.zeros_like(eps)
    for e_i in levels:
        g_smooth += (degeneracy / gamma) * smoothing_kernel((eps - e_i) / gamma, p)

    # 累积积分：N(ε) = ∫_{−∞}^{ε} g̃ dε，及 ∫ ε g̃ dε
    cum_N = np.cumsum(g_smooth) * de
    cum_E = np.cumsum(eps * g_smooth) * de

    # 求 λ̃ 使 N(λ̃) = n_particles
    idx = np.searchsorted(cum_N, n_particles)
    idx = int(np.clip(idx, 1, len(eps) - 1))
    lam_tilde = eps[idx]
    E_smooth = cum_E[idx]

    E_occ = degeneracy * np.sum(levels[:n_occ])
    return float(E_occ - E_smooth), lam_tilde


def smoothed_fermi(levels, n_particles, gamma, p=6, degeneracy=2):
    """返回平滑费米能 λ̃（供 BCS 初值用）。"""
    _, lam = shell_correction(levels, n_particles, gamma, p=p, degeneracy=degeneracy)
    return lam
