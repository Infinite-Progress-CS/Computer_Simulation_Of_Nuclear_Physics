# -*- coding: utf-8 -*-
"""diag_bifurcation.py — 扫描 neck，打印断裂点 eta 双阱的形成过程（纯 ASCII 输出）。"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hybrid_pes import CachedHybridMacroMicro
from shape import ShapeFunnyHills

Z, N = 92, 144
A = 236
T = np.sqrt(6.54 / (A / 8.0))


def scan(neck_hi=0.7, neck_lo=0.6, elong=2.0, eps=(0.4, 0.6), n_eta=61):
    pes = CachedHybridMacroMicro(Z, N, nz=40, nrho=40, nsurf=48, nphi=32,
                                 Nmax=12, neck_hi=neck_hi, neck_lo=neck_lo,
                                 shape_cls=ShapeFunnyHills,
                                 lam_so_p=28.0, eps_scission=eps)
    etas = np.linspace(-0.4, 0.4, n_eta)
    out = []
    for nk in [0.99, 0.85, 0.80, 0.75, 0.70, 0.68, 0.66, 0.64, 0.62, 0.60,
               0.58, 0.55, 0.52, 0.50]:
        V = np.array([pes.energy([elong, nk, e, eps[0], eps[1]]) for e in etas])
        i0 = np.argmin(np.abs(etas))
        im = np.argmin(V)
        asym = V[i0] - V[im]        # 非对称比对称低多少 (>0 = 双阱)
        # 双阱判据：η=0 是否为局部极大（V[0] 高于相邻）
        barrier = V[i0] - V.min()
        # 平衡分布峰谷比
        Vrel = V - V.min()
        Y = np.exp(-Vrel / T)
        pv = Y.max() / max(Y[i0], 1e-12)
        out.append((nk, V[i0], V[im], etas[im], asym, pv))
    return out


if __name__ == '__main__':
    print("scan: elong=2.0, eps=(0.4,0.6), current window nh=0.7 nl=0.6")
    print("  neck   V(eta=0)  V(min)   eta_min   asym(>0=double-well)  P/V(exp)")
    for nk, v0, vm, em, asym, pv in scan(0.7, 0.6):
        print(f"  {nk:5.2f}  {v0:8.3f}  {vm:8.3f}  {em:+6.2f}   {asym:7.3f}              {pv:8.0f}")
