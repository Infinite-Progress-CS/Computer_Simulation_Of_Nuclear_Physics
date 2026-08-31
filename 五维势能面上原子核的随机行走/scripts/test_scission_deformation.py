# -*- coding: utf-8 -*-
"""诊断：断裂点形变（固定 ε / 二维最小化）如何移动双峰位置。

目的：球形/基态形变模型重峰落在 A≈132（Sn 双幻数 Z=50/N=82 被过强绑定），
实验初级重峰在 A≈136-138（仅 N=82）。假设：断裂点碎片被库仑拉伸到有限形变 ε，
脆弱的 Z=50 质子幻数先被洗掉，而 N=82 中子壳部分存活 → 重峰从 Sn(Z=50) 移到
Xe/Ba(Z=54-56)。

本脚本对少数 A 值（96,100,103,118,132,134,136,138）在电荷弥散 ±4 下：
  1. 基态形变（各自 ε 最小，即当前模型）
  2. 固定断裂形变 ε ∈ {0.10, 0.15, 0.20, 0.25}
  3. 二维最小化 (ε_L, ε_H)
计算 V_scission，输出 Y(A) 看峰落在哪。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src')
sys.path.insert(0, SRC)

from fragment import Fragment
from scission import (scission_energy, scission_energy_fixed_eps,
                      scission_energy_2d, scission_distance_deformed)
from yield_model import ucd_charge

Z_PARENT, N_PARENT = 92, 144
A_PARENT = Z_PARENT + N_PARENT
A_LIST = [90, 93, 96, 98, 100, 102, 104, 106, 108, 112, 118,
          124, 128, 132, 134, 136, 138, 140, 142, 146]
ZW = 4
EPS_GRID = (0.0, 0.1, 0.2, 0.3, 0.4)
EPS_FIXED_LIST = [0.10, 0.15, 0.20]

_cache = {}


def frag(Z, N):
    key = (Z, N)
    if key not in _cache:
        _cache[key] = Fragment(Z, N, Nmax=12, a_sym=23.5)
    return _cache[key]


def V_table(mode):
    """返回 dict A_L -> (Zs, Vs)。mode: 'gs' | (fixed,eps) | '2d'。"""
    table = {}
    for A_L in A_LIST:
        zc = ucd_charge(A_L, Z_PARENT, A_PARENT)
        Zs, Vs = [], []
        for Z_L in range(zc - ZW, zc + ZW + 1):
            N_L = A_L - Z_L
            if not (1 <= Z_L < Z_PARENT and 1 <= N_L < N_PARENT):
                continue
            fL = frag(Z_L, N_L)
            fH = frag(Z_PARENT - Z_L, N_PARENT - N_L)
            if mode == 'gs':
                V = scission_energy(fL, fH, d_extra=2.0)
            elif mode == '2d':
                V, _, _ = scission_energy_2d(fL, fH, EPS_GRID, d_extra=2.0)
            else:  # ('fixed', eps)
                V = scission_energy_fixed_eps(fL, fH, mode[1], d_extra=2.0)
            Zs.append(Z_L)
            Vs.append(V)
        table[A_L] = (np.asarray(Zs), np.asarray(Vs))
    return table


def mass_yield(table, T):
    V_min = min(Vs.min() for _, Vs in table.values())
    A = np.array(sorted(table))
    Y = np.array([np.exp(-(table[a][1] - V_min) / T).sum() for a in A])
    return A, 200.0 * Y / Y.sum()


def main():
    modes = [('gs',)] + [('fixed', e) for e in EPS_FIXED_LIST] + [('2d',)]
    tables = {}
    for m in modes:
        label = 'gs' if m[0] == 'gs' else ('2d' if m[0] == '2d' else f'eps={m[1]:.2f}')
        print(f'计算 {label} ...', flush=True)
        tables[label] = V_table(m[0] if m[0] != 'fixed' else ('fixed', m[1]))

    # 每个模式的最低 V 与最可几电荷（供诊断）
    for label, tb in tables.items():
        Amin = min(tb, key=lambda a: tb[a][1].min())
        Zs, Vs = tb[Amin]
        Zp = Zs[int(np.argmin(Vs))]
        print(f'{label:>10}: V_min A={Amin} (Z={Zp})  V={Vs.min():.1f} MeV')

    # 用两个温度看峰位
    for T in (1.0, 1.5):
        print(f'\n=== T={T:.1f} MeV 峰位 ===')
        for label, tb in tables.items():
            A, Y = mass_yield(tb, T)
            light = A < 120
            heavy = A > 120
            Al = int(A[light][np.argmax(Y[light])])
            Ah = int(A[heavy][np.argmax(Y[heavy])])
            valley = (A >= 104) & (A <= 132)
            Av = int(A[valley][np.argmin(Y[valley])])
            print(f'  {label:>10}: 轻峰 A={Al} ({Y[A==Al][0]:.2f}%)   '
                  f'重峰 A={Ah} ({Y[A==Ah][0]:.2f}%)   '
                  f'谷 A={Av} ({Y[A==Av][0]:.3f}%)')

    # 完整 Y(A) 表（T=1.5，看双峰/谷结构）
    T = 1.5
    print(f'\n=== T={T:.1f} MeV 完整 Y(A)（%）===')
    print('   A   ' + ''.join(f'{label:>10}' for label in tables))
    for a in A_LIST:
        row = f'  {a:>3}  '
        for label, tb in tables.items():
            A, Y = mass_yield(tb, T)
            y = Y[A == a][0]
            row += f'{y:>10.3f}'
        print(row)


if __name__ == '__main__':
    main()
