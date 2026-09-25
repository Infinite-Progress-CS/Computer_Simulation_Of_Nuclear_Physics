# -*- coding: utf-8 -*-
"""diag_spurious_min.py — 诊断行走假深极小（几何层）。

目标点 elong=1.38 neck=0.68 eps=(0.29,0.04)，E_ld 随 eta 从 +15.9 塌到 -1.8。
查 3QS 几何：碎片轴 a/c、rho_v、切线解 (z_v,C,z1,z2)、体积守恒、rho^2 最小值。
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hybrid_pes import CachedHybridMacroMicro
from shape import Shape3QS

Z, N = 92, 144
pes = CachedHybridMacroMicro(Z, N, nz=40, nrho=40, nsurf=48, nphi=32,
                             Nmax=12, shape_cls=Shape3QS, lam_so_p=None)
shape = pes.shape


def geom(q):
    elong, neck, eta, eps1, eps2 = q
    r1 = shape.axis_ratio(eps1)
    r2 = shape.axis_ratio(eps2)
    w1 = (1.0 - eta) / 2.0
    w2 = (1.0 + eta) / 2.0
    c1 = (w1 / r1 ** 2) ** (1.0 / 3.0); a1 = r1 * c1
    c2 = (w2 / r2 ** 2) ** (1.0 / 3.0); a2 = r2 * c2
    rho_v = neck * min(a1, a2)
    d = shape.build(q)
    zz = np.linspace(d["zL"], d["zR"], 2000)
    rr2 = np.maximum(d["rho2"](zz), 0.0)
    vol = np.trapz(rr2, zz)
    z, rho = shape.profile(q, n=400)
    return dict(a1=a1, a2=a2, c1=c1, c2=c2, rho_v=rho_v, zL=d["zL"], zR=d["zR"],
                vol=vol, rho2min=float(rr2.min()), rho_min_fm=float(rho.min()))


print('=== 几何 vs eta（elong=1.38 neck=0.68 eps=(0.29,0.04)）===')
print(f'{"eta":>6} {"a1":>6} {"c1":>6} {"a2":>6} {"c2":>6} {"rho_v":>7} '
      f'{"vol":>7} {"rho2min":>9} {"rho_min":>8} {"zL..zR":>14}')
for eta in [0.0, 0.03, 0.06, 0.09, 0.12, 0.15]:
    q = [1.38, 0.68, eta, 0.29, 0.04]
    try:
        g = geom(q)
        print(f'{eta:+.2f} {g["a1"]:6.3f} {g["c1"]:6.3f} {g["a2"]:6.3f} {g["c2"]:6.3f} '
              f'{g["rho_v"]:7.3f} {g["vol"]:7.4f} {g["rho2min"]:9.6f} '
              f'{g["rho_min_fm"]:8.3f} [{g["zL"]:.2f},{g["zR"]:.2f}]')
    except Exception as e:
        print(f'{eta:+.2f}  FAIL {str(e)[:50]}')

print('\n=== 对照：对称 eps=(0.2,0.2) 的几何 ===')
print(f'{"eta":>6} {"a1":>6} {"c1":>6} {"a2":>6} {"c2":>6} {"rho_v":>7} '
      f'{"vol":>7} {"rho2min":>9} {"rho_min":>8}')
for eta in [0.0, 0.06, 0.09, 0.15]:
    q = [1.38, 0.68, eta, 0.2, 0.2]
    try:
        g = geom(q)
        print(f'{eta:+.2f} {g["a1"]:6.3f} {g["c1"]:6.3f} {g["a2"]:6.3f} {g["c2"]:6.3f} '
              f'{g["rho_v"]:7.3f} {g["vol"]:7.4f} {g["rho2min"]:9.6f} '
              f'{g["rho_min_fm"]:8.3f}')
    except Exception as e:
        print(f'{eta:+.2f}  FAIL {str(e)[:50]}')
