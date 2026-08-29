"""
pairing.py — BCS 对修正
====================================================================
在费米面附近窗口内解 BCS 能隙方程（能隙 Δ、化学势 λ、占据概率 v_i²），
对修正能量（相对正常态，仅窗口内能级参与）：
  δE_对 = 2 Σ_i e_i (v_i² − v_i0²) − Δ²/G

  其中 v_i0² = 1（窗口内正常态占据，e_i ≤ 费米能） / 0（空）。

关键约定：低于窗口的核芯（能量深束缚态）是惰性的、始终满占据，不参与配对，
只守恒窗口内的粒子数 n_window = n_particles − 2·idx[0]（idx[0] 为窗口首能级下标）。
配对强度 G = g0/A（中子/质子可分别调 g0）。
每个单粒子能级 2 重简并（时间反演配对对）。
"""

import numpy as np


def bcs_solve(levels, n_particles, G, window=6.0, degeneracy=2,
              max_iter=300, tol=1e-8):
    """解 BCS，返回 (Δ, λ, v_sq, idx_window)。

    在费米面 ±window MeV 窗口内的能级上解 BCS。
    """
    levels = np.asarray(levels, dtype=float)
    n_occ = int(round(n_particles / degeneracy))
    if n_occ <= 0 or n_occ > len(levels):
        return 0.0, levels[n_occ - 1] if n_occ > 0 else 0.0, None, None

    lam_F = levels[n_occ - 1]
    mask = (levels >= lam_F - window) & (levels <= lam_F + window)
    # 若窗口内能级太少则扩大窗口
    if mask.sum() < 10:
        mask = (levels >= lam_F - 2.0 * window) & (levels <= lam_F + 2.0 * window)
    idx = np.where(mask)[0]
    e = levels[idx]
    if len(e) < 2:
        return 0.0, lam_F, None, None

    # 窗口内正常态粒子数：低于窗口的核芯 2·idx[0] 个粒子惰性不参与
    n_window = n_particles - degeneracy * idx[0]
    if n_window <= 0:
        return 0.0, lam_F, None, None

    lam = lam_F
    Delta = 1.0
    v_sq = np.full_like(e, n_window / (degeneracy * len(e)))

    for _ in range(max_iter):
        d = e - lam
        E = np.sqrt(d * d + Delta * Delta)
        # 占据概率
        v_sq = 0.5 * (1.0 - d / E)
        # 窗口内粒子数
        N_calc = degeneracy * v_sq.sum()
        # 能隙方程右端：Δ = (G/2) Σ_i Δ/E_i
        gap_rhs = (G / 2.0) * np.sum(Delta / E)
        # 化学势修正（dN/dλ = Σ Δ²/E³）
        dN_dlam = degeneracy * np.sum(Delta ** 2 / E ** 3)
        if dN_dlam > 1e-14:
            lam += (n_window - N_calc) / dN_dlam
        # 能隙阻尼迭代
        Delta_new = 0.5 * Delta + 0.5 * gap_rhs
        if gap_rhs <= 1e-10:
            Delta_new = 0.0
        if abs(Delta_new - Delta) < tol and abs(N_calc - n_window) < 1e-6:
            Delta = Delta_new
            break
        Delta = Delta_new
    else:
        Delta = Delta if Delta > 1e-10 else 0.0

    return float(Delta), float(lam), v_sq, idx


def pairing_correction(levels, n_particles, G, window=6.0, degeneracy=2):
    """返回对修正能量 δE_对（MeV）。"""
    levels = np.asarray(levels, dtype=float)
    n_occ = int(round(n_particles / degeneracy))
    if n_occ <= 0 or n_occ > len(levels):
        return 0.0

    Delta, lam, v_sq, idx = bcs_solve(levels, n_particles, G, window=window,
                                      degeneracy=degeneracy)
    if Delta <= 1e-10 or idx is None:
        return 0.0

    e = levels[idx]
    lam_F = levels[n_occ - 1]
    v0_sq = (e <= lam_F).astype(float)   # 正常态占据（窗口内）
    dE = degeneracy * np.sum(e * (v_sq - v0_sq)) - Delta ** 2 / G
    return float(dE)
