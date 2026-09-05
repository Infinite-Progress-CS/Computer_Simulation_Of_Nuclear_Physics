# -*- coding: utf-8 -*-
"""
run_5d_walk.py — 五维 Metropolis 随机行走（打开 η/ε1/ε2，磁盘缓存）
====================================================================
里程碑 2 的核心交付：在宏观-微观势能面上做真正的 5D Metropolis 随机行走，
从基态出发，翻越势垒，退火沉降到断裂点，全程 5 个参数 (elong, neck, η, ε1, ε2)
自由演化。与 run_5d.py 的确定性极小能量路径互为印证。

用 CachedMacroMicro（磁盘缓存）：min_energy_path 已把路径附近的量子点落盘，
行走探索路径邻域时大量命中缓存。行走本身是马尔可夫链（串行），单进程跑。
"""

import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from cached_pes import CachedMacroMicro
from metropolis import metropolis_walk, neck_fraction

Z, N = 92, 144   # U-236


def main():
    pes = CachedMacroMicro(Z, N, nz=40, nrho=40, nsurf=48, nphi=32, Nmax=12)
    print("=" * 70)
    print("五维 Metropolis 随机行走（U-236，宏观-微观，磁盘缓存）")
    print("=" * 70)
    print(f"  R0={pes.R0:.3f} fm   γ={pes.gamma:.2f} MeV   G={pes.G_n:.4f} MeV")

    q0 = np.array([1.0, 0.99, 0.0, 0.0, 0.0])   # 基态
    q_min = np.array([0.3, 0.03, -0.30, -0.20, -0.20])
    q_max = np.array([3.0, 0.99, 0.30, 0.45, 0.20])

    print(f"\n[1] 五维 Metropolis 随机行走 (T=8→0.05 退火, 1200 步, 棘轮 elong>2.2)...")
    t0 = time.time()
    path, energies, components, accept = metropolis_walk(
        pes, q0, T=8.0, n_steps=1200, T_end=0.05, ratchet_elong=2.2,
        step=np.array([0.08, 0.05, 0.04, 0.04, 0.03]),
        q_min=q_min, q_max=q_max, seed=0)
    print(f"  耗时 {time.time() - t0:.1f}s  接受率 {accept:.2%}  "
          f"缓存命中 {pes.cache_stats()[0]} / 未命中 {pes.cache_stats()[1]}")

    # 摘要
    i_bar = int(np.argmax(energies))
    i_end = int(np.argmin(energies[i_bar:])) + i_bar   # 越障后的最低点
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

    # 保存
    out = os.path.join(PROJECT_ROOT, "results", "随机行走")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "walk_5d.npz"),
             path=path, energies=energies, components=components)
    print(f"\n已保存 results/随机行走/walk_5d.npz")


if __name__ == "__main__":
    main()
