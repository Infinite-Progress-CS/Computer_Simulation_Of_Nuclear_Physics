# -*- coding: utf-8 -*-
"""把计算的质量产额 Y(A) 与 ENDF/B-VIII.0 实验数据叠加对比。

输入
    data/ENDF_U235_mass_chain_yield.csv  实验质量链产额（U-235 热中子裂变）
    data/computed_mass_yield.csv         计算产额（calc_fragment_yield.py 输出）

输出
    output/compare_mass_yield.png        计算 vs 实验叠图（对数轴）
    output/compare_mass_yield_log.png    对数纵轴版（看对称谷）

对比量
    双峰位置（轻峰 A、重峰 A）、峰谷比、均方对数偏差（rms log-deviation）。
    注意实验是 U-235(n_th,f) → 复合核 U-236*，与计算 U-236 直接可比。
"""
import os
import sys
import csv
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
DATA_DIR = os.path.join(ROOT, 'results', '产额分布', 'data')
OUT_DIR = os.path.join(ROOT, 'results', '产额分布', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def load_csv(path, a_col, y_col):
    """读 CSV 返回 (A 数组, Y 数组)。"""
    A, Y = [], []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            A.append(float(row[a_col]))
            Y.append(float(row[y_col]))
    return np.array(A), np.array(Y)


def interp_on_grid(A_exp, Y_exp, A_calc, Y_calc):
    """把两条产额插到公共 A 网格（calc 的网格），返回对齐后的 Y。"""
    Y_exp_on_calc = np.interp(A_calc, A_exp, Y_exp, left=0.0, right=0.0)
    return Y_exp_on_calc


def rms_log_deviation(y1, y2, mask):
    """均方根对数偏差 sqrt(mean(log10(y1/y2)^2))，只在 mask 上（非零）。"""
    ratio = np.log10(y1[mask] / y2[mask])
    return float(np.sqrt(np.mean(ratio ** 2)))


def symmetrize(A, Y, A_parent=236):
    """把发射中子后的（不对称）质量链产额折叠成发射前的（对称）初级产额。

    初级（发射中子前）产额满足 Y0(A) = Y0(A_parent−A)（质量守恒）。发射中子
    只把质量向下平移（轻 ~2、重 ~2.5 个中子），故 ENDF 的 Y(A) 与 Y(A_parent−A)
    不对称（轻峰 A=95、重峰 A=134）。初级产额的一阶恢复：

        Y0(A) ≈ [Y(A) + Y(A_parent−A)] / 2

    折叠后轻峰回到 A≈96-100、重峰 A≈136-140，谷仍在 A=118。
    """
    Y0 = np.empty_like(Y)
    for i, a in enumerate(A):
        a_comp = A_parent - a
        Y0[i] = 0.5 * (Y[i] + np.interp(a_comp, A, Y, left=0.0, right=0.0))
    return Y0


def peak_valley(A, Y):
    """返回 (轻峰A, 轻峰Y, 重峰A, 重峰Y, 谷A, 谷Y)。"""
    light = A < 120
    heavy = A > 120
    A_light = int(A[light][np.argmax(Y[light])])
    A_heavy = int(A[heavy][np.argmax(Y[heavy])])
    valley = (A >= 105) & (A <= 135)
    A_val = int(A[valley][np.argmin(Y[valley])])
    return (A_light, float(Y[light].max()), A_heavy, float(Y[heavy].max()),
            A_val, float(Y[A == A_val][0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=float, default=None,
                    help='计算温度，加载 computed_mass_yield_T{T}.csv（默认自动选最新）')
    ap.add_argument('--calc', type=str, default=None, help='显式指定计算产额 CSV 路径')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    exp_path = os.path.join(DATA_DIR, 'ENDF_U235_mass_chain_yield.csv')
    if args.calc:
        calc_path = args.calc
    elif args.T is not None:
        calc_path = os.path.join(DATA_DIR, f'computed_mass_yield_T{args.T:.1f}.csv')
    else:
        # 自动选最新的 computed_mass_yield_T*.csv
        cands = sorted([f for f in os.listdir(DATA_DIR)
                        if f.startswith('computed_mass_yield_T') and f.endswith('.csv')])
        if not cands:
            sys.exit('找不到 computed_mass_yield_T*.csv，请先跑 calc_fragment_yield.py')
        calc_path = os.path.join(DATA_DIR, cands[-1])
        print(f'自动选择计算文件：{cands[-1]}')
    for p in (exp_path, calc_path):
        if not os.path.exists(p):
            sys.exit(f'缺少输入文件 {p}，请先跑 calc_fragment_yield.py')

    A_exp, Y_exp = load_csv(exp_path, 'A', 'mass_chain_yield_percent')
    A_calc, Y_calc = load_csv(calc_path, 'A', 'mass_yield_percent')

    # 发射中子后（原始）与发射前（对称化）两套实验口径
    Y_exp_sym = symmetrize(A_exp, Y_exp)

    # 对齐到计算网格
    Y_exp_on_calc = interp_on_grid(A_exp, Y_exp, A_calc, Y_calc)
    Y_sym_on_calc = interp_on_grid(A_exp, Y_exp_sym, A_calc, Y_calc)

    # 峰/谷
    pe = peak_valley(A_exp, Y_exp)          # 原始（发射中子后）
    ps = peak_valley(A_exp, Y_exp_sym)      # 对称化（发射前，初级产额）
    pc = peak_valley(A_calc, Y_calc)        # 计算（发射前）
    print('=== 双峰位置对比（口径：初级/发射前 vs 计算）===')
    print(f'  实验原始(发射中子后) : 轻峰 A={pe[0]} ({pe[1]:.2f}%)  重峰 A={pe[2]} ({pe[3]:.2f}%)  谷 A={pe[4]}')
    print(f'  实验对称化(初级)     : 轻峰 A={ps[0]} ({ps[1]:.2f}%)  重峰 A={ps[2]} ({ps[3]:.2f}%)  谷 A={ps[4]}')
    print(f'  计算(初级)           : 轻峰 A={pc[0]} ({pc[1]:.2f}%)  重峰 A={pc[2]} ({pc[3]:.2f}%)  谷 A={pc[4]}')

    # 峰谷比（用对称化实验）
    ratio_sym = (ps[1] + ps[3]) / (2.0 * ps[5]) if ps[5] > 0 else np.inf
    ratio_calc = (pc[1] + pc[3]) / (2.0 * pc[5]) if pc[5] > 0 else np.inf
    print(f'  峰谷比     实验(对称化) = {ratio_sym:.0f}   计算 = {ratio_calc:.0f}')

    # 均方对数偏差（只在高产额区 A 80~155 比较，避开边缘噪声）
    mask = (A_calc >= 80) & (A_calc <= 155) & (Y_calc > 0) & (Y_sym_on_calc > 0)
    rms = rms_log_deviation(Y_calc, Y_sym_on_calc, mask)
    print(f'  RMS log10 偏差 vs 对称化实验（A 80~155）= {rms:.3f}')

    if args.no_plot:
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 线性轴
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(A_exp, Y_exp, 'o-', ms=3, lw=1, color='gray', label='ENDF (post-n, raw)')
    ax.plot(A_exp, Y_exp_sym, 'o-', ms=3, lw=1, color='k', label='ENDF (pre-n, symmetrized)')
    ax.plot(A_calc, Y_calc, 's-', ms=3, lw=1, color='tab:blue', label='calc (scission-point)')
    ax.set_xlabel('A (mass number)')
    ax.set_ylabel('Y(A) (%)')
    ax.set_title('U-236 fission fragment mass yield: calc vs experiment')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'compare_mass_yield.png'), dpi=160)
    plt.close(fig)

    # 对数轴（看对称谷）
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(A_exp, np.clip(Y_exp, 1e-6, None), 'o-', ms=3, lw=1, color='gray',
                label='ENDF (post-n, raw)')
    ax.semilogy(A_exp, np.clip(Y_exp_sym, 1e-6, None), 'o-', ms=3, lw=1, color='k',
                label='ENDF (pre-n, symmetrized)')
    ax.semilogy(A_calc, np.clip(Y_calc, 1e-6, None), 's-', ms=3, lw=1, color='tab:blue',
                label='calc (scission-point)')
    ax.set_xlabel('A (mass number)')
    ax.set_ylabel('Y(A) (%)  (log)')
    ax.set_title('Mass yield (log scale)')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'compare_mass_yield_log.png'), dpi=160)
    plt.close(fig)

    print('对比图已写出到 output/compare_mass_yield.png 与 compare_mass_yield_log.png')


if __name__ == '__main__':
    main()
