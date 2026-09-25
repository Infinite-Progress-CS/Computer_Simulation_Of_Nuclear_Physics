# -*- coding: utf-8 -*-
"""diag_3qs_walk_elongbias.py — 实验：偏置坐标从 Q 换成 elong，看 3QS 行走能否到断裂。

假设：RMS 偏置 V0(Q0/Q)^2 的 Q 在 3QS 下被 neck 主导，把行走拉进细颈死胡同
（elong=1.0 时 neck<0.6 无解）。换成 V0(elong_gs/elong)^2 后，细颈方向不再沾偏置
（elong 不变则 Vb 不变），行走被迫先拉长，拉长后颈自然下坡 → 断裂。
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hybrid_pes import CachedHybridMacroMicro
from random_walk_yield import neck_radius_fm, local_temperature
from shape import Shape3QS

Z, N = 92, 144
E_star = 6.54
V0 = 60.0
c0 = 2.5
n_walks = 3
max_steps = 300

pes = CachedHybridMacroMicro(Z, N, nz=40, nrho=40, nsurf=48, nphi=32,
                             Nmax=12, shape_cls=Shape3QS, lam_so_p=None)
shape = pes.shape

q0 = np.array([1.0, 0.99, 0.0, 0.0, 0.0])
V_gs = pes.energy(q0)
elong_gs = q0[0]
print(f'基态 q0={q0}  V_gs={V_gs:.2f}  elong_gs={elong_gs}', flush=True)

rng = np.random.default_rng(0)
step = np.array([0.05, 0.04, 0.03, 0.03, 0.03])
q_min = np.array([0.2, 0.05, -0.5, -0.2, -0.2])
q_max = np.array([3.2, 0.99, 0.5, 0.4, 0.4])

def bias(q):
    # 偏置只跟 elong 走：elong 拉长 → Vb 下降
    return V0 * (elong_gs / max(q[0], 0.05)) ** 2

eta_sc = []
for w in range(n_walks):
    q = q0.copy()
    V = pes.energy(q)
    Vb = bias(q)
    hit = False
    for s in range(max_steps):
        q_new = np.clip(q + step * rng.standard_normal(5), q_min, q_max)
        try:
            V_new = pes.energy(q_new)
        except Exception:
            V_new = np.inf
            e_avg = 0.5 * (q_new[3] + q_new[4])
            for cand in (np.array([q_new[0], q_new[1], 0.0, q_new[3], q_new[4]]),
                         np.array([q_new[0], q_new[1], 0.0, e_avg, e_avg]),
                         np.array([q_new[0], q_new[1], 0.0, 0.0, 0.0])):
                try:
                    V_new = pes.energy(cand); q_new = cand; break
                except Exception:
                    continue
        if V_new < np.inf:
            Vb_new = bias(q_new)
            dE = (V_new + Vb_new) - (V + Vb)
            T = local_temperature(V - V_gs, E_star, 236)
            if dE <= 0.0 or rng.random() < np.exp(-dE / max(T, 1e-6)):
                q, V, Vb = q_new, V_new, Vb_new
        rn = neck_radius_fm(q, shape)
        if s % 50 == 0 or rn <= c0:
            print(f'  w{w} s={s:3d} elong={q[0]:.2f} neck={q[1]:.2f} eta={q[2]:+.2f} '
                  f'eps=({q[3]:+.2f},{q[4]:+.2f}) V={V:+.1f} Vb={Vb:+.1f} r_neck={rn:.2f}fm',
                  flush=True)
        if rn <= c0:
            eta_sc.append(q[2])
            print(f'  >>> w{w} 断裂 s={s} eta={q[2]:+.3f}', flush=True)
            hit = True
            break
    if not hit:
        print(f'  w{w} 未断裂 末 elong={q[0]:.2f} neck={q[1]:.2f} eta={q[2]:+.2f}', flush=True)

print(f'\n断裂 {len(eta_sc)}/{n_walks}  eta={np.round(eta_sc,3) if eta_sc else "N/A"}', flush=True)
