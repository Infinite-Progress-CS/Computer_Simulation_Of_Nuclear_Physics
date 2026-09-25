# -*- coding: utf-8 -*-
"""计算 U-236 热中子裂变碎片质量产额分布 Y(A)（断裂点模型 + 电荷弥散）。

断裂点模型（Wilkins-Steinberg-Chasman 1976）：
    Y(A_L, Z_L) ∝ exp(−V_scission(A_L, Z_L) / T)
    Y(A_L)      = Σ_{Z_L} Y(A_L, Z_L)            （电荷弥散求和）
    V_scission  = E_frag(L) + E_frag(H) + E_Coul(Z_L, Z_H, d)

E_frag 用宏微模型（液滴 FRLDM 系数解析 + Strutinsky 壳修正 + BCS 对修正），
球形 Woods-Saxon 单粒子谱，能级求和法壳修正（p=4, γ=1.1 ħω₀）。

断裂点势能含 ~2000 MeV 的碎片体积能常量，产额只依赖 V−V_min，故全局减最小
V 再取指数（否则 exp(−V/T) 下溢）。

温度 T：断裂点核温度 ~1 MeV。壳修正驱动非对称（幻数核壳修正为负），
对修正的奇偶差造成 Y(A) 锯齿（mod-4 尖峰）；两者在有限 T 下被热涨落洗掉，
但对隙（Δ~1 MeV）在 T~0.5 MeV 即消失、壳结构更顽固。故只阻尼对修正
（T_d_pair≈0.5），不阻尼壳修正（T_d_sh→∞），消除锯齿的同时保住非对称峰。

势能 V_scission 现在依赖 T（壳/对修正按温度阻尼），故每个 T 单独算一遍 V
（Woods-Saxon 单粒子谱在 Fragment 内缓存，多 T 只重算阻尼乘 + 库仑，很快）。

输出
    data/computed_mass_yield_T{T}.csv    A, Y(%)
    data/computed_charge_yield.csv       Z, Y(%)
    data/most_probable_charge.csv        A, Z_p(A)
    output/mass_yield_multiT.png         多温度质量产额（含实验对比需另跑 compare 脚本）
    output/charge_yield.png
    output/shell_correction_vs_A.png

用法
    python calc_fragment_yield.py --T-list 1.0,1.5,2.0,2.5 --z-window 4
"""
import os
import sys
import time
import argparse
import csv

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)

from yield_model import YieldModel
from level_density import thermal_temperature  # noqa: F401  （文档用）

DATA_DIR = os.path.join(ROOT, 'results', '产额分布', 'data')
OUT_DIR = os.path.join(ROOT, 'results', '产额分布', 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def write_csv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def mass_yield_from_table(V_list, V_min, T):
    """从 (A,Z) 势能表算 Y(A)，Σ=200%。"""
    Y = np.array([np.exp(-(Vs - V_min) / T).sum() for _, Vs in V_list])
    return 200.0 * Y / Y.sum()


def summarize(A_arr, Y):
    """打印双峰/谷位置与峰谷比。"""
    light = A_arr < 120
    heavy = A_arr > 120
    A_light = int(A_arr[light][np.argmax(Y[light])])
    A_heavy = int(A_arr[heavy][np.argmax(Y[heavy])])
    Y_light = float(Y[light].max())
    Y_heavy = float(Y[heavy].max())
    valley = (A_arr >= 105) & (A_arr <= 135)
    A_val = int(A_arr[valley][np.argmin(Y[valley])])
    Y_val = float(Y[A_arr == A_val][0])
    ratio = (Y_light + Y_heavy) / (2.0 * Y_val) if Y_val > 0 else np.inf
    return dict(A_light=A_light, Y_light=Y_light, A_heavy=A_heavy, Y_heavy=Y_heavy,
                A_val=A_val, Y_val=Y_val, ratio=ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T-list', type=str, default='0.8,1.0,1.2',
                    help='逗号分隔的温度列表 (MeV)')
    ap.add_argument('--z-window', type=int, default=4, help='电荷弥散半宽（Z 扫描 ±window）')
    ap.add_argument('--lam-so-n', type=float, default=None, help='中子自旋轨道强度（None=默认35）')
    ap.add_argument('--lam-so-p', type=float, default=28.0,
                    help='质子自旋轨道强度。universal WS 默认 35 使球形 Z=50 幻数过强、'
                         '稳定对称碎片（Pd-118, Z=46 近 Z=50）填平对称谷（峰谷比仅 ~38）；'
                         '降到 28 削弱 Z=50、加深对称谷，峰谷比 → ~564（实验对称化 ~556），'
                         '峰位 η=0.186 不变。中子 SO 对峰谷比几乎无影响（N=82 不敏感）')
    ap.add_argument('--eps-scission', type=str, default='0.4,0.6',
                    help='固定断裂形变 eps_L,eps_H（逗号分隔，如 0.4,0.6）。'
                         '断裂点碎片被库仑拉伸到 ε_L≈0.4/ε_H≈0.6，球形幻数被洗掉、变形壳定峰位；'
                         'None=自由基态极小化（会把重碎片锁 ε≈0.1，峰错位 A_H=132）')
    ap.add_argument('--T-d-sh', type=float, default=100.0,
                    help='壳修正温度阻尼尺度 T_d_sh (MeV)，δE_sh(T)=δE_sh(0)·exp(-(T/T_d_sh)^2)。'
                         '壳修正驱动非对称，断裂点模型不阻尼（WSC 原版 T=0 壳修正），默认 100≈不阻尼')
    ap.add_argument('--T-d-pair', type=float, default=0.5,
                    help='对修正温度阻尼尺度 T_d_pair (MeV)≈0.57Δ，δE_pair(T)=δE_pair(0)·exp(-(T/T_d_pair)^2)。'
                         '对隙在 T~0.5 MeV 洗掉、消除奇偶锯齿')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()
    T_list = [float(x) for x in args.T_list.split(',')]

    t0 = time.time()
    print(f'复合核温度 = {thermal_temperature(236, 6.5):.2f} MeV（断裂点有效温度取 1.5~2.5）')
    print(f'电荷弥散 ±{args.z_window}')
    if args.lam_so_n is not None or args.lam_so_p is not None:
        print(f'自旋轨道强度 λ_so_n={args.lam_so_n} λ_so_p={args.lam_so_p}')
    eps_scission = None
    if args.eps_scission:
        eps_scission = tuple(float(x) for x in args.eps_scission.split(','))
        print(f'固定断裂形变 ε_L={eps_scission[0]}, ε_H={eps_scission[1]}（非绝热断裂点）')

    model = YieldModel(T=1.0, lam_so_n=args.lam_so_n, lam_so_p=args.lam_so_p,
                       eps_scission=eps_scission, T_d_sh=args.T_d_sh,
                       T_d_pair=args.T_d_pair)
    print(f'温度阻尼：T_d_sh={args.T_d_sh} MeV  T_d_pair={args.T_d_pair} MeV')

    # ---- 多个温度下的质量产额（每个 T 单独算 V，壳/对修正被温度阻尼）----
    print('计算全部 (A,Z) 断裂点势能（Woods-Saxon 对角化，最耗时；谱在 Fragment 内缓存，'
          '多 T 只重算阻尼乘 + 库仑）...')
    print('=== 质量产额 Y(A)（电荷弥散求和，全局归一）===')
    best_T, best_ratio = None, np.inf
    Y_by_T = {}
    for T in T_list:
        A_arr, V_list = model._all_scission_energies(z_window=args.z_window, T=T)
        V_min = min(Vs.min() for _, Vs in V_list if len(Vs))
        Y = mass_yield_from_table(V_list, V_min, T)
        Y_by_T[T] = Y
        s = summarize(A_arr, Y)
        print(f'  T={T:.1f} MeV: 轻峰 A={s["A_light"]} ({s["Y_light"]:.2f}%)  '
              f'重峰 A={s["A_heavy"]} ({s["Y_heavy"]:.2f}%)  '
              f'谷 A={s["A_val"]} ({s["Y_val"]:.4f}%)  峰谷比={s["ratio"]:.0f}')
        # 与实验峰谷比 ~600 最近者为 best
        if abs(s["ratio"] - 600) < abs(best_ratio - 600):
            best_T, best_ratio = T, s["ratio"]
        # 保存每个温度的 CSV
        write_csv(os.path.join(DATA_DIR, f'computed_mass_yield_T{T:.1f}.csv'),
                  ['A', 'mass_yield_percent'],
                  [(int(a), round(float(y), 6)) for a, y in zip(A_arr, Y)])
    print(f'  最接近实验峰谷比(~600)的温度：T={best_T:.1f} MeV (峰谷比 {best_ratio:.0f})')

    # ---- 最可几电荷 ----
    print('计算最可几电荷 Z_p(A) ...')
    _, Zp = model.most_probable_charge(z_window=args.z_window)
    write_csv(os.path.join(DATA_DIR, 'most_probable_charge.csv'),
              ['A', 'Z_most_probable'],
              [(int(a), int(z)) for a, z in zip(A_arr, Zp)])

    # ---- 电荷产额（用 best_T，从该 T 的 V 表归并）----
    print(f'计算电荷产额 Y(Z)（T={best_T:.1f}）...')
    A_arr, V_list_best = model._all_scission_energies(z_window=args.z_window, T=best_T)
    V_min_best = min(Vs.min() for _, Vs in V_list_best if len(Vs))
    Yz_map = {}
    for Zs, Vs in V_list_best:
        for Z, V in zip(Zs, Vs):
            Yz_map[Z] = Yz_map.get(Z, 0.0) + float(np.exp(-(V - V_min_best) / best_T))
    Z_arr = np.array(sorted(Yz_map))
    Yz = 200.0 * np.array([Yz_map[z] for z in Z_arr]) / sum(Yz_map.values())
    write_csv(os.path.join(DATA_DIR, 'computed_charge_yield.csv'),
              ['Z', 'charge_yield_percent'],
              [(int(z), round(float(y), 6)) for z, y in zip(Z_arr, Yz)])

    print(f'总耗时 {time.time() - t0:.0f} s')

    # ---- 绘图 ----
    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 1. 多温度质量产额
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for T in T_list:
            ax.plot(A_arr, Y_by_T[T], '-', lw=1.2, label=f'T={T:.1f} MeV')
        ax.set_xlabel('A (mass number)')
        ax.set_ylabel('Y(A) (%)')
        ax.set_title('U-236 fission fragment mass yield (multi-T)')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, 'mass_yield_multiT.png'), dpi=160)
        plt.close(fig)

        # 2. 电荷产额（best_T）
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(Z_arr, Yz, 'o-', ms=4, lw=1, color='tab:red')
        ax.set_xlabel('Z (charge number)')
        ax.set_ylabel('Y(Z) (%)')
        ax.set_title(f'Charge yield (T={best_T:.1f} MeV)')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, 'charge_yield.png'), dpi=160)
        plt.close(fig)

        # 3. 壳修正随 A（UCD 电荷）
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _, sh_L, sh_H = model.shell_correction_vs_A()
        ax.plot(A_arr, sh_L, '-', lw=1, color='tab:green', label='light frag')
        ax.plot(A_arr, sh_H, '-', lw=1, color='tab:purple', label='heavy frag')
        ax.plot(A_arr, sh_L + sh_H, '-', lw=1.5, color='k', label='sum')
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_xlabel('A (light fragment)')
        ax.set_ylabel('shell correction (MeV)')
        ax.set_title('Shell correction vs A (UCD charge)')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, 'shell_correction_vs_A.png'), dpi=160)
        plt.close(fig)

        print('图已写出到 output/')


if __name__ == '__main__':
    main()
