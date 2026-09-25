# -*- coding: utf-8 -*-
"""diag_neck_ridge.py — 检查 3QS 颈脊假势垒（液滴双阱）当前是否还在。

扫描 E_ld(液滴) vs neck（快），再抽几个点算完整 E_mm（双中心基，慢），
判断固定 elong 下 neck 方向是否还有假势垒（非物理能量凸起）。

用法:
  python scripts/diag_neck_ridge.py
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hybrid_pes import CachedHybridMacroMicro
from shape import Shape3QS

Z, N = 92, 144

ELONG = [1.5, 2.0, 2.4, 3.0]
NECKS = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]

print('构建 CachedHybridMacroMicro (Shape3QS + 双中心基, Nmax=12)...', flush=True)
t0 = time.time()
pes = CachedHybridMacroMicro(Z, N, nz=40, nrho=40, nsurf=48, nphi=32,
                             Nmax=12, shape_cls=Shape3QS, lam_so_p=None)
print(f'  构建 {time.time()-t0:.1f}s  shape={type(pes.shape).__name__}  '
      f'γ={pes.gamma:.2f}  R0={pes.R0:.3f}', flush=True)

print('\n===== 液滴层 E_ld(neck) 扫描（eta=0, eps=0，快）=====', flush=True)
for elong in ELONG:
    print(f'--- elong={elong:.1f} ---', flush=True)
    for neck in NECKS:
        q = [elong, neck, 0.0, 0.0, 0.0]
        try:
            E_ld = pes.liquid_drop_energy(q)
            print(f'  neck={neck:5.2f}  E_ld={E_ld:8.3f}', flush=True)
        except Exception as e:
            print(f'  neck={neck:5.2f}  <无解: {type(e).__name__}>', flush=True)

print('\n===== 完整 E_mm(neck) 扫描（双中心基，慢，抽 elong=2.0/2.4）=====', flush=True)
for elong in [2.0, 2.4]:
    print(f'--- elong={elong:.1f} ---', flush=True)
    for neck in [0.90, 0.70, 0.60, 0.50, 0.40, 0.30, 0.15, 0.05]:
        q = [elong, neck, 0.0, 0.0, 0.0]
        try:
            t1 = time.time()
            V, dVs, dVc, dEsh, dEpair = pes.energy_components(q)
            E_ld = V - dEsh - dEpair
            print(f'  neck={neck:5.2f}  E_ld={E_ld:8.3f}  dE_sh={dEsh:8.3f}  '
                  f'dE_pair={dEpair:7.3f}  E_mm={V:8.3f}  ({time.time()-t1:.1f}s)', flush=True)
        except Exception as e:
            print(f'  neck={neck:5.2f}  <无解: {type(e).__name__}: {e}>', flush=True)

print(f'\n总耗时 {time.time()-t0:.1f}s', flush=True)
