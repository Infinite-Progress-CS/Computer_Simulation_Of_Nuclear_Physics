# -*- coding: utf-8 -*-
"""plot_walk_yield.py — 从随机行走 npz 画质量产额 + 电荷产额（对齐 Möller 原图格式）。

Brownian 行走给出质量产额 Y(A)（断裂点 η 直方图）。电荷产额 Y(Z) 用
UCD 标度（Z_p=A·Z0/A0）+ 高斯电荷弥散展宽得到（Möller 图口径）。

用法:
  python plot_walk_yield.py --npz results/随机行走/walk_yield_funnyhills_nw80_ms400_s42.npz
"""
import os, sys, csv, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
DATA = os.path.join(ROOT, 'results', '产额分布', 'data')
OUT = os.path.join(ROOT, 'results', '产额分布', 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

Z_PARENT, A_PARENT = 92, 236


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


def mass_to_charge(A, Y_A, sigma_z=0.6, dz_pol=0.0, z_lo=20, z_hi=72):
    """质量产额 → 电荷产额：UCD 中心 + 高斯电荷弥散。

    Z_p(A) = A·Z0/A0 + dz_pol（dz_pol 为可选电荷极化修正，逐 A 加同一偏移）。
    Y(Z)  = Σ_A Y(A)·Gauss(Z; Z_p(A), sigma_z)。
    """
    Z = np.arange(z_lo, z_hi + 1, dtype=float)
    Y_Z = np.zeros_like(Z)
    for a, y in zip(A, Y_A):
        if y <= 0:
            continue
        zp = a * (Z_PARENT / A_PARENT) + dz_pol
        g = np.exp(-(Z - zp) ** 2 / (2 * sigma_z ** 2))
        g /= g.sum()
        Y_Z += y * g
    Y_Z = 200.0 * Y_Z / Y_Z.sum()
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
                    help='随机行走 npz 路径；默认取 results/随机行走/ 最新 walk_yield_funnyhills_*.npz')
    ap.add_argument('--sigma-z', type=float, default=0.6)
    ap.add_argument('--dz-pol', type=float, default=0.0,
                    help='电荷极化偏移（加在 UCD 上，逐 A）')
    args = ap.parse_args()

    if args.npz is None:
        cand = sorted(glob.glob(os.path.join(ROOT, 'results', '随机行走',
                                             'walk_yield_hybrid_*.npz')))
        if not cand:
            sys.exit('找不到 npz')
        args.npz = cand[-1]
    res = np.load(args.npz)
    A = res['A']
    Y_A = res['Y_A']
    eta = res['eta_scission']
    n_sc = int(res['n_scission'])
    tag = os.path.basename(args.npz).replace('.npz', '')

    # ---- 质量产额 Y(A) ----
    A_mask = (A >= 66) & (A <= 166)
    Aa = A[A_mask].astype(int)
    Ya = Y_A[A_mask]
    Ae, Ye = exp_mass_yield(os.path.join(DATA, 'ENDF_U235_mass_chain_yield.csv'))

    ml = peaks(Aa, Ya, 80, 118, 110, 130)
    el = peaks(Ae, Ye, 80, 118, 110, 130)
    print(f'[{tag}] 质量产额 (calc vs exp):')
    print(f'  calc: 轻峰 A={ml[0]} ({ml[1]:.2f}%)  重峰 A={ml[2]} ({ml[3]:.2f}%)  '
          f'谷 A={ml[4]} ({ml[5]:.4f}%)  峰谷比={(ml[1]+ml[3])/2/max(ml[5],1e-9):.0f}')
    print(f'  exp : 轻峰 A={el[0]} ({el[1]:.2f}%)  重峰 A={el[2]} ({el[3]:.2f}%)  '
          f'谷 A={el[4]} ({el[5]:.4f}%)')

    # ---- 电荷产额 Y(Z) ----
    Zc, Yc = mass_to_charge(Aa, Ya, sigma_z=args.sigma_z, dz_pol=args.dz_pol)
    Ze2, Ye2 = exp_charge_yield(os.path.join(DATA, 'ENDF_U235_cumulative_yield.csv'))

    cl = peaks(Zc, Yc, 25, 46, 44, 52)
    elc = peaks(Ze2, Ye2, 25, 46, 44, 52)
    print(f'  电荷产额 (sigma_z={args.sigma_z}, dz_pol={args.dz_pol:+.1f}):')
    print(f'  calc: 轻峰 Z={cl[0]} ({cl[1]:.2f}%)  重峰 Z={cl[2]} ({cl[3]:.2f}%)  '
          f'谷 Z={cl[4]} ({cl[5]:.4f}%)  P/V={(cl[1]+cl[3])/2/max(cl[5],1e-9):.0f}')
    print(f'  exp : 轻峰 Z={elc[0]} ({elc[1]:.2f}%)  重峰 Z={elc[2]} ({elc[3]:.2f}%)  '
          f'谷 Z={elc[4]} ({elc[5]:.4f}%)')
    print(f'  n_scission={n_sc}  η: mean={eta.mean():+.3f} σ={eta.std():.3f} '
          f'asym(>0.03)={(np.abs(eta)>0.03).mean():.0%}')

    # ---- 画图 ----
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 9.5))

    ax = axes[0]
    ax.plot(Aa, Ya, '-', lw=2.0, color='k', label='Calc. (random walk)')
    ax.plot(Ae, Ye, '--', lw=1.6, color='tab:red', label=r'Exp. $^{235}$U(n,f)')
    ax.set_xlabel('Fragment Mass Number $A_f$')
    ax.set_ylabel('Yield $Y(A)$ (%)')
    ax.set_xlim(60, 175)
    ax.set_ylim(0, 12)
    ax.set_title('Mass yield: Brownian random walk vs exp')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(Zc, Yc, '-', lw=2.0, color='k', label='Calc. (UCD + Gauss)')
    ax.plot(Ze2, Ye2, '--', lw=1.6, color='tab:red', label=r'Exp. $^{235}$U(n,f)')
    ax.set_xlabel('Fragment Charge Number $Z_f$')
    ax.set_ylabel('Yield $Y(Z)$ (%)')
    ax.set_xlim(20, 72)
    ax.set_ylim(0, 22)
    ax.set_title('Charge yield: Brownian random walk vs exp')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(OUT, f'{tag}_mass_charge.png')
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f'图已写出 {out_png}')


if __name__ == '__main__':
    main()
