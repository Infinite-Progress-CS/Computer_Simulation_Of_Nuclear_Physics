# -*- coding: utf-8 -*-
"""plot_yield_final.py — 最终产额交付图（光滑质量产额 + 电荷产额 Y(Z)）。

严格按教授方法（Möller/Randrup Brownian 形状运动）：
  1. 光滑 5D PES（Funny-Hills 剖面 + 双中心壳修正）
  2. Brownian 随机行走（Metropolis + 偏置势 V0(Q0/Q)^2）
  3. 断裂点（颈半径 ≤ 2.5 fm）采样质量不对称 η
  4. 拟合（KDE 光滑）→ 质量产额 Y(A) → 电荷产额 Y(Z)

质量产额：对 η_scission 采样做核密度估计（KDE），消除有限行走直方图的尖刺。
电荷产额：UCD 标度（Z_p = A·Z0/A0）+ 质量相关电荷极化 ΔZ(A) + 高斯电荷弥散 σ_z。

用法:
  python plot_yield_final.py --npz results/随机行走/walk_yield_funnyhills_nw80_ms400_s42.npz
"""
import os, sys, csv, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
DATA = os.path.join(ROOT, 'results', '产额分布', 'data')
OUT = os.path.join(ROOT, 'results', '产额分布', 'figures')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from frldm2020 import load_yield, mass_yield as frldm_mass_yield_raw, charge_yield as frldm_charge_yield_raw, conditional_charge

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

Z_PARENT, A_PARENT = 92, 236
FRLDM_FILE = os.path.join(DATA, 'frldm2020', 'ascii_table', 'frldm_Z092A236.dat')


# ---------------------------------------------------------------- 实验数据
def exp_charge_yield(path):
    """ENDF 累计产额 (A,Z,yield) 按 Z 求和 → 实验电荷产额 Y(Z)（归一 200%）。"""
    Yz = {}
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            z = int(row['Z'])
            Yz[z] = Yz.get(z, 0.0) + float(row['cumulative_yield_percent'])
    Zs = np.array(sorted(Yz))
    Y = np.array([Yz[z] for z in Zs])
    Y = 200.0 * Y / Y.sum()
    return Zs, Y


def exp_mass_yield(path):
    """ENDF 质量链产额 → 实验质量产额 Y(A)。"""
    A, Y = [], []
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            A.append(int(row['A']))
            Y.append(float(row['mass_chain_yield_percent']))
    A = np.array(A); Y = np.array(Y)
    Y = 200.0 * Y / Y.sum()
    return A, Y


# ---------------------------------------------------------------- 光滑质量产额
def smooth_mass_yield(eta, A_parent=236, A_lo=66, A_hi=170, bw=0.35):
    """对 η_scission 采样做 KDE → 光滑质量产额 Y(A)（Σ=200%，双峰对称）。

    A_L = A_parent·(1−|η|)/2 为轻碎片质量；重碎片 A_H = A_parent − A_L。
    Y(A) ∝ p_A(A) + p_A(A_parent − A)，p_A 为 A_L 样本的核密度。
    """
    A_L = A_parent * (1.0 - np.abs(eta)) / 2.0
    kde = gaussian_kde(A_L, bw_method=bw)
    A_grid = np.arange(A_lo, A_hi + 1, dtype=float)
    p = kde(A_grid)                      # 轻碎片质量密度
    Y = p + kde(A_parent - A_grid)       # 加上重碎片（镜像）
    Y = 200.0 * Y / Y.sum()
    return A_grid, Y


# ---------------------------------------------------------------- 电荷极化
def charge_polarization(A, dz_light=3.0, dz_heavy=1.0, A_light=95.0, A_heavy=140.0):
    """质量相关电荷极化 ΔZ(A)（经验：轻碎片 +3，重碎片 +1，线性插值）。

    ΔZ 加在 UCD 上：Z_p(A) = A·Z0/A0 + ΔZ(A)。轻碎片质子富集（+）、重碎片
    中子富集，故 ΔZ 随 A 递减。范围外线性外推并截断到 [0, 4]。
    """
    slope = (dz_heavy - dz_light) / (A_heavy - A_light)
    dz = dz_light + slope * (A - A_light)
    return np.clip(dz, 0.0, 4.0)


def mass_to_charge(A, Y_A, sigma_z=0.0, dz_func=None, z_lo=20, z_hi=76):
    """质量产额 → 电荷产额：UCD 中心 + 电荷极化 + 高斯电荷弥散。

    Y(Z) = Σ_A Y(A)·Gauss(Z; Z_p(A), sigma_z)，Z_p(A) = A·Z0/A0 + ΔZ(A)。
    """
    if dz_func is None:
        dz_func = charge_polarization
    Z = np.arange(z_lo, z_hi + 1, dtype=float)
    if sigma_z <= 0.0:
        A_of_Z = Z * (A_PARENT / Z_PARENT)
        Y_Z = np.interp(A_of_Z, A, Y_A, left=0.0, right=0.0) * (A_PARENT / Z_PARENT)
        s = Y_Z.sum()
        return Z, (200.0 * Y_Z / s if s > 0 else np.zeros_like(Y_Z))

    Y_Z = np.zeros_like(Z)
    for a, y in zip(A, Y_A):
        if y <= 0:
            continue
        zp = a * (Z_PARENT / A_PARENT)
        if dz_func is not None:
            zp += dz_func(a)
        g = np.exp(-(Z - zp) ** 2 / (2 * sigma_z ** 2))
        g /= g.sum()
        Y_Z += y * g
    s = Y_Z.sum()
    Y_Z = 200.0 * Y_Z / s if s > 0 else Y_Z
    return Z, Y_Z


def mass_to_charge_frldm(A, Y_A, Z_grid, A_grid, P, z_lo=20, z_hi=76):
    """用 FRLDM 2020 的 P(Z|A) 条件电荷分布，把计算质量产额映射为 Y(Z)。

    相比经验 UCD+极化公式，P(Z|A) 直接取自同一篇 FRLDM 二维产额数据库，
    能自然包含质子壳结构、电荷极化与电荷弥散的综合效果。
    """
    Z = np.arange(z_lo, z_hi + 1, dtype=float)
    Y_Z = np.zeros_like(Z)
    for iz, z in enumerate(Z):
        idx = int(z - Z_grid[0])
        if 0 <= idx < len(Z_grid):
            prob_A = P[idx, :]
        else:
            prob_A = np.zeros(len(A_grid), dtype=float)
        p = np.interp(A, A_grid, prob_A, left=0.0, right=0.0)
        Y_Z[iz] = float(np.sum(np.asarray(Y_A) * p))
    s = Y_Z.sum()
    if s > 0:
        Y_Z = 200.0 * Y_Z / s
    return Z, Y_Z


def peaks(x, y, lo, hi, val_lo, val_hi):
    """质量/电荷产额的轻峰、重峰、谷。"""
    light = (x >= lo) & (x < hi)
    heavy = x >= hi
    xl = int(x[light][np.argmax(y[light])]); yl = float(y[light].max())
    xh = int(x[heavy][np.argmax(y[heavy])]); yh = float(y[heavy].max())
    v = (x >= val_lo) & (x <= val_hi)
    xv = int(x[v][np.argmin(y[v])]); yv = float(y[x == xv][0])
    return xl, yl, xh, yh, xv, yv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=None,
                    help='随机行走 npz；默认取 results/随机行走/ 最新 walk_yield_funnyhills_*.npz')
    ap.add_argument('--sigma-z', type=float, default=0.5,
                    help='电荷高斯弥散 σ_z（UCD+极化模型）')
    ap.add_argument('--bw', type=float, default=0.45, help='KDE 带宽因子（Silverman 倍数）')
    ap.add_argument('--dz-light', type=float, default=3.0)
    ap.add_argument('--dz-heavy', type=float, default=1.0)
    ap.add_argument('--no-polarization', action='store_true', default=False,
                    help='禁用电荷极化（纯 UCD）')
    ap.add_argument('--frldm', default=FRLDM_FILE,
                    help='FRLDM 2020 二维产额文件路径')
    ap.add_argument('--charge-model', choices=['ucd', 'frldm'],
                    default='ucd',
                    help='Y(A)→Y(Z) 的转换模型：ucd=经验 UCD+极化；frldm=FRLDM P(Z|A)')
    args = ap.parse_args()

    if args.npz is None:
        cand = sorted(glob.glob(os.path.join(ROOT, 'results', '随机行走',
                                             'walk_yield_*.npz')))
        if not cand:
            sys.exit('找不到 npz')
        args.npz = cand[-1]
    res = np.load(args.npz)
    eta = res['eta_scission']
    n_sc = int(res['n_scission'])
    tag = os.path.basename(args.npz).replace('.npz', '')

    # ---- FRLDM 2020 公开理论基准 ----
    frldm_available = os.path.exists(args.frldm)
    if not frldm_available and args.charge_model == 'frldm':
        print(f'[warning] 找不到 FRLDM 文件 {args.frldm}，电荷转换回退到 ucd')
        args.charge_model = 'ucd'

    Af, Yf, Zf, Yzf, Zg_f, Ag_f, P = (None,) * 7
    if frldm_available:
        Af, Yf = frldm_mass_yield_raw(args.frldm, A_parent=A_PARENT)
        Zf, Yzf = frldm_charge_yield_raw(args.frldm)
        Zg_f, Ag_f, P = conditional_charge(args.frldm, A_parent=A_PARENT)

    # ---- 光滑质量产额 Y(A) ----
    Aa, Ya = smooth_mass_yield(eta, bw=args.bw)
    Ae, Ye = exp_mass_yield(os.path.join(DATA, 'ENDF_U235_mass_chain_yield.csv'))

    ml = peaks(Aa, Ya, 80, 118, 105, 135)
    mf = peaks(Af, Yf, 80, 118, 105, 135) if frldm_available else (None,) * 6
    el = peaks(Ae, Ye, 80, 118, 105, 135)
    print(f'[{tag}] 质量产额 (calc vs FRLDM2020 vs exp):')
    print(f'  calc: 轻峰 A={ml[0]} ({ml[1]:.2f}%)  重峰 A={ml[2]} ({ml[3]:.2f}%)  '
          f'谷 A={ml[4]} ({ml[5]:.4f}%)  峰谷比={(ml[1]+ml[3])/2/max(ml[5],1e-9):.0f}')
    if frldm_available:
        print(f'  FRLDM: 轻峰 A={mf[0]} ({mf[1]:.2f}%)  重峰 A={mf[2]} ({mf[3]:.2f}%)  '
              f'谷 A={mf[4]} ({mf[5]:.4f}%)  峰谷比={(mf[1]+mf[3])/2/max(mf[5],1e-9):.0f}')
    print(f'  exp : 轻峰 A={el[0]} ({el[1]:.2f}%)  重峰 A={el[2]} ({el[3]:.2f}%)  '
          f'谷 A={el[4]} ({el[5]:.4f}%)')

    # ---- 电荷产额 Y(Z) ----
    dz_func = None if args.no_polarization else (
        lambda A: charge_polarization(A, dz_light=args.dz_light,
                                      dz_heavy=args.dz_heavy))
    Zucd, Yucd = mass_to_charge(Aa, Ya, sigma_z=args.sigma_z, dz_func=dz_func)
    if args.charge_model == 'frldm' and frldm_available:
        Zc, Yc = mass_to_charge_frldm(Aa, Ya, Zg_f, Ag_f, P, z_lo=20, z_hi=76)
    else:
        Zc, Yc = Zucd, Yucd
    Ze2, Ye2 = exp_charge_yield(os.path.join(DATA, 'ENDF_U235_cumulative_yield.csv'))

    cl = peaks(Zc, Yc, 25, 48, 44, 54)
    cu = peaks(Zucd, Yucd, 25, 48, 44, 54)
    cf = peaks(Zf, Yzf, 25, 48, 44, 54) if frldm_available else (None,) * 6
    elc = peaks(Ze2, Ye2, 25, 48, 44, 54)
    pol_tag = 'off' if args.no_polarization else '+%g/+%g' % (args.dz_light, args.dz_heavy)
    print(f'  电荷产额 (model={args.charge_model}, sigma_z={args.sigma_z}, pol={pol_tag}):')
    print(f'  calc ({args.charge_model}): 轻峰 Z={cl[0]} ({cl[1]:.2f}%)  重峰 Z={cl[2]} ({cl[3]:.2f}%)  '
          f'谷 Z={cl[4]} ({cl[5]:.4f}%)  P/V={(cl[1]+cl[3])/2/max(cl[5],1e-9):.0f}')
    print(f'  calc (UCD only)  : 轻峰 Z={cu[0]} ({cu[1]:.2f}%)  重峰 Z={cu[2]} ({cu[3]:.2f}%)  '
          f'谷 Z={cu[4]} ({cu[5]:.4f}%)')
    if frldm_available:
        print(f'  FRLDM2020        : 轻峰 Z={cf[0]} ({cf[1]:.2f}%)  重峰 Z={cf[2]} ({cf[3]:.2f}%)  '
              f'谷 Z={cf[4]} ({cf[5]:.4f}%)  P/V={(cf[1]+cf[3])/2/max(cf[5],1e-9):.0f}')
    print(f'  exp : 轻峰 Z={elc[0]} ({elc[1]:.2f}%)  重峰 Z={elc[2]} ({elc[3]:.2f}%)  '
          f'谷 Z={elc[4]} ({elc[5]:.4f}%)')
    print(f'  n_scission={n_sc}  η: mean={eta.mean():+.3f} σ={eta.std():.3f} '
          f'asym(|η|>0.03)={(np.abs(eta)>0.03).mean():.0%}')

    # ---- 画图 ----
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 9.5))

    ax = axes[0]
    ax.plot(Aa, Ya, '-', lw=2.0, color='k', label='Calc. (random walk + KDE)')
    if frldm_available:
        ax.plot(Af, Yf, '--', lw=1.8, color='tab:blue', label='FRLDM 2020')
    ax.plot(Ae, Ye, '--', lw=1.6, color='tab:red', label=r'Exp. $^{235}$U(n,f)')
    ax.set_xlabel('Fragment Mass Number $A_f$')
    ax.set_ylabel('Yield $Y(A)$ (%)')
    ax.set_xlim(60, 175)
    ax.set_ylim(0, None)
    ax.set_title('Mass yield: Brownian random walk vs FRLDM vs exp')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    calc_label = 'Calc. (FRLDM $P(Z|A)$)' if args.charge_model == 'frldm' else 'Calc. (UCD + pol + Gauss)'
    ax.plot(Zc, Yc, '-', lw=2.0, color='k', label=calc_label)
    if args.charge_model == 'frldm':
        ax.plot(Zucd, Yucd, ':', lw=1.2, color='gray', label='Calc. (UCD only)')
    if frldm_available:
        ax.plot(Zf, Yzf, '--', lw=1.8, color='tab:blue', label='FRLDM 2020')
    ax.plot(Ze2, Ye2, '--', lw=1.6, color='tab:red', label=r'Exp. $^{235}$U(n,f)')
    ax.set_xlabel('Fragment Charge Number $Z_f$')
    ax.set_ylabel('Yield $Y(Z)$ (%)')
    ax.set_xlim(24, 68)
    ax.set_ylim(0, None)
    ax.set_title('Charge yield: Brownian random walk vs FRLDM vs exp')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(OUT, 'U236_质量电荷产额_计算_FRLDM_ENDF.png')
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    # ---- Möller 2011 风格的电荷产额单图 ----
    fig2, ax2 = plt.subplots(figsize=(7.0, 4.5))
    if args.charge_model == 'frldm':
        ax2.plot(Zc, Yc, '-', lw=2.0, color='k', label='Calc. (FRLDM $P(Z|A)$)')
        ax2.plot(Zucd, Yucd, ':', lw=1.2, color='gray', label='Calc. (Möller UCD scaling)')
    else:
        ax2.plot(Zucd, Yucd, '-', lw=2.0, color='k', label='Calc. (Möller UCD scaling)')
    if frldm_available:
        ax2.plot(Zf, Yzf, '--', lw=1.8, color='tab:blue', label='FRLDM 2020')
    ax2.plot(Ze2, Ye2, 'o-', ms=3, lw=1.2, color='tab:red',
             label=r'Exp. $^{235}$U(n,f)')
    ax2.set_xlabel('Fragment Charge Number $Z_f$')
    ax2.set_ylabel('Yield $Y(Z_f)$ (%)')
    ax2.set_xlim(30, 60)
    ax2.set_ylim(0, 25)
    ax2.set_title('U-236 fission-fragment charge yield')
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    out_charge = os.path.join(OUT, 'U236_电荷产额_Moller2011风格.png')
    fig2.savefig(out_charge, dpi=160)
    plt.close(fig2)

    print(f'图已写出 {out_png}')
    print(f'电荷产额单图已写出 {out_charge}')


if __name__ == '__main__':
    main()
