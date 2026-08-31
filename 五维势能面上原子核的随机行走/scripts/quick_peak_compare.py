# -*- coding: utf-8 -*-
"""快速对比不同 (λ_so_n, λ_so_p) 下的双峰位置（降采样，Nmax=10）。

目的：验证 λ_so 调参能否把重峰从 A≈133(Sn/Sb) 移到 A≈136(Xe)。用粗网格
（A 88-148、z_window=3、Nmax=10）几分钟内出峰位，足够判断方向。

用法：python quick_peak_compare.py
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src')
sys.path.insert(0, SRC)

from yield_model import YieldModel


def peaks(model, z_window=3):
    A_arr, V_list = model._all_scission_energies(z_window=z_window)
    V_min = min(Vs.min() for _, Vs in V_list if len(Vs))
    Y = np.array([np.exp(-(Vs - V_min) / model.T).sum() for _, Vs in V_list])
    Y = 200.0 * Y / Y.sum()
    light = A_arr < 120
    heavy = A_arr > 120
    A_l = int(A_arr[light][np.argmax(Y[light])])
    A_h = int(A_arr[heavy][np.argmax(Y[heavy])])
    return A_l, A_h, float(Y[light].max()), float(Y[heavy].max())


def run(lam_so_n, lam_so_p):
    t0 = time.time()
    model = YieldModel(T=1.6, Nmax=10, A_min=88, A_max=148,
                       lam_so_n=lam_so_n, lam_so_p=lam_so_p)
    A_l, A_h, Y_l, Y_h = peaks(model)
    print(f'λ_so_n={lam_so_n:.0f} λ_so_p={lam_so_p:.0f}: '
          f'轻峰 A={A_l} ({Y_l:.2f}%)  重峰 A={A_h} ({Y_h:.2f}%)  '
          f'[{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    run(35.0, 35.0)   # 基线
    run(55.0, 15.0)   # 强中子 SO + 弱质子 SO（去 Z=50 幻数）
