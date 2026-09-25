# -*- coding: utf-8 -*-
"""电荷产额 Y(Z) 分布图：断裂点模型 vs 实验（对齐原始论文格式）。

目标图（教授参考图）：
    横轴 Fragment Charge Number Zf（0~60）
    纵轴 Yield Y(Z) (%)（0~25）
    实线 = Calc.（断裂点模型）
    虚线 = Exp.（实验）
    不对称双峰
"""
import os, sys, csv
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


def exp_charge_yield(path):
    """从 ENDF 累计产额（A,Z,yield）按 Z 求和，得实验电荷产额 Y(Z)（归一 200%）。"""
    Yz = {}
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            z = int(row['Z'])
            Yz[z] = Yz.get(z, 0.0) + float(row['cumulative_yield_percent'])
    Zs = np.array(sorted(Yz))
    Y = np.array([Yz[z] for z in Zs])
    Y = 200.0 * Y / Y.sum()
    return Zs, Y


def load_calc_yz(path):
    Zs, Y = [], []
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            Zs.append(int(row['Z']))
            Y.append(float(row['charge_yield_percent']))
    return np.array(Zs), np.array(Y)


def gaussian_smooth(Z, Y, sigma):
    """对电荷产额做高斯展宽（电荷弥散），洗掉壳修正的奇偶锯齿。"""
    x = np.arange(Z.min() - 4 * sigma, Z.max() + 4 * sigma + 1)
    g = np.exp(-(x[:, None] - Z[None, :]) ** 2 / (2 * sigma ** 2))
    Ys = g @ Y
    Ys = Ys / g.sum(axis=1)
    return x, Ys


def main():
    T = 2.00
    exp_path = os.path.join(DATA, 'ENDF_U235_cumulative_yield.csv')
    calc_path = os.path.join(DATA, f'computed_charge_yield_T{T:.2f}.csv')
    if not os.path.exists(calc_path):
        calc_path = os.path.join(DATA, 'computed_charge_yield.csv')
        T = None

    Ze, Ye = exp_charge_yield(exp_path)
    Zc, Yc = load_calc_yz(calc_path)
    # 计算曲线做高斯电荷弥散展宽，洗掉壳修正的奇偶锯齿
    sigma_z = 0.5
    Zc, Yc = gaussian_smooth(Zc, Yc, sigma_z)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    # 实线 = 计算
    ax.plot(Zc, Yc, '-', lw=2.0, color='k',
            label=f'Calc. ({T:.2f} MeV)' if T else 'Calc.')
    # 虚线 = 实验
    ax.plot(Ze, Ye, '--', lw=1.6, color='tab:red', label='Exp. $^{235}$U(n,f)')

    ax.set_xlabel('Fragment Charge Number $Z_f$')
    ax.set_ylabel('Yield $Y(Z)$ (%)')
    # U-236 (Z=92) 对称劈裂中心 Z=46，双峰以此为对称轴居中
    ax.set_xlim(20, 72)
    ax.set_ylim(0, 25)
    ax.set_title('U-236 fission fragment charge yield: calc vs exp')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT, '电荷产额分布_Y(Z).png')
    fig.savefig(out, dpi=160)
    plt.close(fig)

    # 打印峰/谷
    def pv(Zs, Y):
        light = Zs < 46
        heavy = Zs > 46
        Zl = int(Zs[light][np.argmax(Y[light])]); Yl = float(Y[light].max())
        Zh = int(Zs[heavy][np.argmax(Y[heavy])]); Yh = float(Y[heavy].max())
        v = (Zs >= 44) & (Zs <= 50)
        Zv = int(Zs[v][np.argmin(Y[v])]); Yv = float(Y[Zs == Zv][0])
        return Zl, Yl, Zh, Yh, Zv, Yv
    cl = pv(Zc, Yc)
    el = pv(Ze, Ye)
    print(f'计算: 轻峰 Z={cl[0]} ({cl[1]:.2f}%)  重峰 Z={cl[2]} ({cl[3]:.2f}%)  谷 Z={cl[4]} ({cl[5]:.4f}%)')
    print(f'实验: 轻峰 Z={el[0]} ({el[1]:.2f}%)  重峰 Z={el[2]} ({el[3]:.2f}%)  谷 Z={el[4]} ({el[5]:.4f}%)')
    print(f'图已写出到 {out}')


if __name__ == '__main__':
    main()
