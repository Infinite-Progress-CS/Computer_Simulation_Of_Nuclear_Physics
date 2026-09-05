# -*- coding: utf-8 -*-
"""
run_5d_walk_hybrid.py — 五维 Metropolis 随机行走（断裂区双中心壳修正）
====================================================================
与 run_5d_walk.py 相同，但 PES 用 HybridMacroMicro（断裂区碎片级双中心壳修正）。
预期：行走翻越势垒后，在断裂区（neck<0.15）沉降到非对称 η≈0.15、ε1≈0.3、
ε2≈0.1 的谷底，而非对称 η=0。

输出：results/随机行走/walk_5d_hybrid.npz。
"""

import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from hybrid_pes import CachedHybridMacroMicro
from metropolis import metropolis_walk, neck_fraction

Z, N = 92, 144   # U-236


def main():
    pes = CachedHybridMacroMicro(Z, N, nz=40, nrho=40, nsurf=48, nphi=32,
                                 Nmax=12, neck_hi=0.5, neck_lo=0.15)
    print("=" * 70)
    print("五维 Metropolis 随机行走（U-236，断裂区双中心壳修正，磁盘缓存）")
    print("=" * 70)
    print(f"  R0={pes.R0:.3f} fm   γ={pes.gamma:.2f} MeV   G={pes.G_n:.4f} MeV"
          f"   切换窗口 neck∈[{pes.neck_lo},{pes.neck_hi}]")

    q0 = np.array([1.0, 0.99, 0.0, 0.0, 0.0])   # 基态
    q_min = np.array([0.3, 0.03, -0.30, -0.20, -0.20])
    q_max = np.array([3.0, 0.99, 0.30, 0.45, 0.20])

    print(f"\n[1] 五维 Metropolis 随机行走 (T=8→1.0 退火, 1500 步, 棘轮 elong>2.2)...")
    t0 = time.time()
    path, energies, components, accept = metropolis_walk(
        pes, q0, T=8.0, n_steps=1500, T_end=1.0, ratchet_elong=2.2,
        step=np.array([0.08, 0.05, 0.06, 0.04, 0.03]),
        q_min=q_min, q_max=q_max, seed=0)
    print(f"  耗时 {time.time() - t0:.1f}s  接受率 {accept:.2%}  "
          f"缓存命中 {pes.cache_stats()[0]} / 未命中 {pes.cache_stats()[1]}")

    i_bar = int(np.argmax(energies))
    i_end = int(np.argmin(energies[i_bar:])) + i_bar
    print("\n" + "=" * 70)
    print("结果摘要")
    print("=" * 70)
    print(f"  起点      : {q0}  E={energies[0]:+.2f}")
    print(f"  势垒(峰值): E={energies[i_bar]:+.2f} MeV @ elong={path[i_bar,0]:.2f}, "
          f"neck={path[i_bar,1]:.2f}, η={path[i_bar,2]:+.2f}")
    print(f"  越障后最低: E={energies[i_end]:+.2f} MeV @ elong={path[i_end,0]:.2f}, "
          f"η={path[i_end,2]:+.2f}, ε1={path[i_end,3]:+.2f}, ε2={path[i_end,4]:+.2f}")
    print(f"  末帧      : elong={path[-1,0]:.2f}, neck={path[-1,1]:.2f}, "
          f"η={path[-1,2]:+.2f}, ε1={path[-1,3]:+.2f}, ε2={path[-1,4]:+.2f}")
    print(f"  末帧断裂度: f={neck_fraction(path[-1], pes.shape):.2f}")

    out = os.path.join(PROJECT_ROOT, "results", "随机行走")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "walk_5d_hybrid.npz"),
             path=path, energies=energies, components=components)
    print(f"\n已保存 results/随机行走/walk_5d_hybrid.npz")


if __name__ == "__main__":
    main()
