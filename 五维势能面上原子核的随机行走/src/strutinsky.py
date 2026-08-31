"""
strutinsky.py — Strutinsky 壳修正（能级求和法）
====================================================================
把一组单粒子能级（每个 2 重简并，Kramers）平滑，得到平滑能级密度 g̃(ε)，
壳修正 δE_壳 = 占据能之和 − 平滑能之和。

平滑函数（曲率修正高斯）：
  f_p(x) = (1/√π) e^{−x²} · L_p^{1/2}(x²)
  g̃(ε) = Σ_i (g_i/γ) f_p((ε−ε_i)/γ)，g_i = 2

壳修正（Strutinsky 1968）：
  δE_壳 = 2 Σ_{i=1}^{N/2} ε_i − 2 Σ_i ε_i w_i
  w_i = F_p((λ̃−ε_i)/γ)   （平滑占据数，F_p 为 f_p 的累积分布函数）
  λ̃ 由 N = 2 Σ_i F_p((λ̃−ε_i)/γ) 定（平滑费米能）。

实现要点：用「能级求和」而非网格积分。δE_壳 是两个大数（~2000 MeV）
的小差（~10 MeV），网格积分（cumsum、de=γ/20）数值不稳定（de 不同结果
差 3~10 倍）。能级求和把平滑占据数写成解析 CDF 的取值，直接对能级求和，
没有小差大数问题，且对 γ 有清晰平台（Strutinsky 平台条件）。

曲率修正阶 p 取 4、宽度 γ = 1.1 ħω₀：p=6 的高阶波瓣在深幻数处给反号，
p=4 与 γ=1.1 ħω₀ 组合下 N=82 / Z=50 得负、N=72 / Z=46 得正、量级正确。
"""

import numpy as np
from scipy import special


def smoothing_kernel(x, p=4):
    """曲率修正高斯 f_p(x) = (1/√π) e^{−x²} L_p^{1/2}(x²)。∫f_p dx = 1。"""
    lag = special.eval_genlaguerre(p, 0.5, x ** 2)
    return (1.0 / np.sqrt(np.pi)) * np.exp(-x ** 2) * lag


# ----------------------------------------------------------------------
# f_p 的累积分布函数 F_p(v) = ∫_{−∞}^{v} f_p(x) dx，用细网格预计算 + 缓存。
# 归一化：f_p 是归一化密度（∫f_p dx = 1，曲率修正多项式不改变零阶矩），
# F_p(−xmax)→0、F_p(+xmax)→1。xmax=12 时 e^{−144}≈0，截断误差可忽略。
# ----------------------------------------------------------------------
_XMAX = 12.0
_NG = 240001
_XG = np.linspace(-_XMAX, _XMAX, _NG)
_F_CACHE = {}


def _cdf_table(p):
    f = smoothing_kernel(_XG, p)
    F = np.cumsum(f) * (_XG[1] - _XG[0])
    return F - F[0]          # 保证 F(−xmax)=0


def _Fp(v, p):
    """F_p(v)，向量化。v 越界时截断到 ±xmax（F_p 在边界已饱和）。"""
    if p not in _F_CACHE:
        _F_CACHE[p] = _cdf_table(p)
    F = _F_CACHE[p]
    v = np.clip(np.asarray(v, dtype=float), -_XMAX, _XMAX)
    return np.interp(v, _XG, F)


def shell_correction(levels, n_particles, gamma, p=4, degeneracy=2):
    """计算壳修正 δE_壳（MeV）。返回 (δE_壳, 平滑费米能 λ̃)。

    参数：
      levels     : 单粒子"轨道"能级（排序数组，每个 degeneracy 重简并）
      n_particles: 核子数（中子 N 或质子 Z）
      gamma      : 平滑宽度（≈ 1.1 ħω0）
      p          : 曲率修正阶（4）
      degeneracy : 每轨道简并度（2，Kramers）
    """
    levels = np.asarray(levels, dtype=float)
    if len(levels) == 0:
        return 0.0, 0.0
    n_full = n_particles // degeneracy             # 满占据轨道数
    n_extra = n_particles % degeneracy             # 部分占据轨道上的剩余粒子数
    if n_full <= 0 or n_full > len(levels):
        return 0.0, 0.0
    if n_extra > 0 and n_full >= len(levels):      # 半满轨道越界
        return 0.0, 0.0

    lam_F = levels[n_full - 1] if n_extra == 0 else levels[n_full]   # 尖锐费米能

    # 二分求平滑费米能 λ̃：degeneracy·Σ F_p((λ̃−ε_i)/γ) = n_particles
    lo, hi = lam_F - 6.0 * gamma, lam_F + 6.0 * gamma
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if degeneracy * np.sum(_Fp((mid - levels) / gamma, p)) < n_particles:
            lo = mid
        else:
            hi = mid
    lam_tilde = 0.5 * (lo + hi)

    # 平滑占据数 w_i = F_p((λ̃−ε_i)/γ)
    w = _Fp((lam_tilde - levels) / gamma, p)

    # 壳修正 = 占据能之和 − 平滑能之和（能级求和，数值稳定）。
    # 奇核正确处理半满轨道：满占据前 n_full 个轨道，半满轨道只加 n_extra 个粒子。
    E_occ = degeneracy * np.sum(levels[:n_full])
    if n_extra > 0:
        E_occ += n_extra * levels[n_full]
    E_smooth = degeneracy * np.sum(levels * w)
    return float(E_occ - E_smooth), float(lam_tilde)


def smoothed_fermi(levels, n_particles, gamma, p=4, degeneracy=2):
    """返回平滑费米能 λ̃（供 BCS 初值用）。"""
    _, lam = shell_correction(levels, n_particles, gamma, p=p, degeneracy=degeneracy)
    return lam
