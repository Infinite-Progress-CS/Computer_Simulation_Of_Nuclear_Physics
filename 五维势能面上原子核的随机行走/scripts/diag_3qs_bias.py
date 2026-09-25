# -*- coding: utf-8 -*-
"""diag_3qs_bias.py — 检查 3QS 偏置势能量地形，看行走为何卡在 elong~1.0。

沿 elong 方向（固定 neck/eta/eps）算 E_ld、Q、Vb、V+Vb，以及沿 neck 方向。
判断 RMS 偏置 V0(Q0/Q)^2 是否真的把 elong 方向抹平。
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hybrid_pes import CachedHybridMacroMicro
from random_walk_yield import quadrupole_moment, neck_radius_fm
from shape import Shape3QS

Z, N = 92, 144
V0 = 60.0

pes = CachedHybridMacroMicro(Z, N, nz=40, nrho=40, nsurf=48, nphi=32,
                             Nmax=12, shape_cls=Shape3QS, lam_so_p=None)
shape = pes.shape

# 参考：基态 [1.0, 0.99, 0, 0, 0]
q_gs = np.array([1.0, 0.99, 0.0, 0.0, 0.0])
Q0 = quadrupole_moment(q_gs, shape)
print(f'基态 q={q_gs}  Q0={Q0:.1f} fm^5  E_ld={pes.liquid_drop_energy(q_gs):.2f}', flush=True)

def line(title, qs):
    print(f'\n===== {title} =====', flush=True)
    print(f'  {"elong":>5} {"neck":>5} {"E_ld":>8} {"Q":>10} {"Vb":>8} {"E_ld+Vb":>9} {"r_neck":>7}', flush=True)
    for q in qs:
        E_ld = pes.liquid_drop_energy(q)
        Q = quadrupole_moment(q, shape)
        Vb = V0 * (Q0 / max(Q, Q0 * 0.3)) ** 2
        rn = neck_radius_fm(q, shape)
        print(f'  {q[0]:5.2f} {q[1]:5.2f} {E_ld:8.2f} {Q:10.1f} {Vb:8.2f} {E_ld+Vb:9.2f} {rn:7.2f}', flush=True)

# 沿 elong（固定 neck=0.99 厚颈）
line('沿 elong 方向（neck=0.99, eta=0, eps=0）',
     [[el, 0.99, 0, 0, 0] for el in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0]])

# 沿 neck（固定 elong=1.0）
line('沿 neck 方向（elong=1.0, eta=0, eps=0）',
     [[1.0, nk, 0, 0, 0] for nk in [0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]])

# 沿 neck（固定 elong=2.0）
line('沿 neck 方向（elong=2.0, eta=0, eps=0）',
     [[2.0, nk, 0, 0, 0] for nk in [0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]])
