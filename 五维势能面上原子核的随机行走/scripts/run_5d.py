# -*- coding: utf-8 -*-
"""
run_5d.py — 5D 极小能量裂变路径（真正打开 η / ε1 / ε2 碎片形变）
====================================================================
里程碑 2 的核心诊断：之前 run_walk_asym.py 的 min_energy_path_asym 把 ε1=ε2=0
钉死，只扫 (elong, neck, η)，因此在断裂区看不到非对称裂变（重碎片 ~140 需形变）。

本脚本把碎片形变 ε1(轻)/ε2(重) 一并打开，做 5D 极小能量路径搜索：
  - Stage 1  neck 扫描（碎片坐标冻结在上一个 elong 最优，液滴主导，解耦合理）
  - Stage 2  (η × ε1 × ε2) 全耦合粗网格（捕住非对称谷的耦合最小值）
  - Stage 3  neck ±0.03 精扫
路径在 elong 上连续延拓（warm-start），Nmax=12 保证壳修正收敛，磁盘缓存共享。

输出：results/随机行走/path_5d.npz + 控制台逐点表格。
"""

import os
import sys
import time
import multiprocessing as mp

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from cached_pes import CachedMacroMicro

Z, N = 92, 144   # U-236

_PES = None


def _init_worker():
    global _PES
    _PES = CachedMacroMicro(Z, N, nz=40, nrho=40, nsurf=48, nphi=32, Nmax=12)


def _eval_full(q):
    global _PES
    try:
        V, dV_s, dV_c, dE_sh, dE_pair = _PES.energy_components(q)
        return (float(V), float(dV_s), float(dV_c), float(dE_sh), float(dE_pair))
    except Exception:
        return (float("nan"), 0.0, 0.0, 0.0, 0.0)


def min_energy_path_5d(pool, elong_lims=(0.3, 2.5), n_elong=24):
    """5D 极小能量裂变路径（耦合 neck×η 相1 + 开 ε1×ε2 相2）。

    与旧版的关键区别：相1 做耦合 (neck × η) 二维网格（旧版 neck 扫描时把 η 冻在
    上一个 elong 的 0，会漏掉「薄颈 + η>0」的非对称谷），再相2 打开 ε1×ε2。
    elong 非均匀：基态/内垒区(0.3-1.7)粗，鞍点→断裂区(1.8-2.5)细(0.05)，
    以分辨双峰结构（第二极小/外垒）。颈网格覆盖 0.03-0.99 的中间值。
    """
    elong = np.concatenate([
        np.linspace(0.3, 1.7, 8),
        np.linspace(1.8, 2.5, 15),
    ])
    neck_grid = np.array([0.03, 0.05, 0.08, 0.10, 0.15, 0.20,
                          0.30, 0.40, 0.50, 0.70, 0.85, 0.99])
    eta_grid = np.array([0.0, 0.10, 0.15, 0.19, 0.22, 0.25])
    eps1_grid = np.array([0.0, 0.15, 0.30, 0.45])
    eps2_grid = np.array([0.0, 0.10, 0.20])

    n_elong = len(elong)
    path = np.zeros((n_elong, 5))
    energies = np.full(n_elong, np.nan)
    components = np.zeros((n_elong, 2))

    print(f'{"elong":>5} {"neck":>5} {"eta":>6} {"eps1":>5} {"eps2":>5} '
          f'{"E_mm":>8} {"dE_sh":>7} {"dE_pair":>7}', flush=True)

    for i, el in enumerate(elong):
        # ---- Phase 1: 耦合 (neck × η)，ε1=ε2=0 ----
        tasks1 = [[el, nk, et, 0.0, 0.0] for nk in neck_grid for et in eta_grid]
        res1 = pool.map(_eval_full, tasks1, chunksize=4)
        best = None   # (V, [el,nk,et,e1,e2], dVs, dVc, dEsh, dEpair)
        k = 0
        for nk in neck_grid:
            for et in eta_grid:
                r = res1[k]; k += 1
                if not np.isnan(r[0]) and (best is None or r[0] < best[0]):
                    best = (r[0], [el, nk, et, 0.0, 0.0], r[1], r[2], r[3], r[4])
        if best is None:
            continue

        # ---- Phase 2: 开 ε1×ε2 于最优 (neck, η) ----
        nk, et = best[1][1], best[1][2]
        tasks2 = [[el, nk, et, e1, e2] for e1 in eps1_grid for e2 in eps2_grid]
        res2 = pool.map(_eval_full, tasks2, chunksize=4)
        k = 0
        for e1 in eps1_grid:
            for e2 in eps2_grid:
                r = res2[k]; k += 1
                if not np.isnan(r[0]) and r[0] < best[0]:
                    best = (r[0], [el, nk, et, e1, e2], r[1], r[2], r[3], r[4])

        path[i] = best[1]
        energies[i] = best[0]
        components[i] = (best[2], best[3])
        print(f'{el:5.2f} {best[1][1]:5.2f} {best[1][2]:+6.2f} {best[1][3]:+5.2f} '
              f'{best[1][4]:+5.2f} {best[0]:+8.2f} {best[4]:+7.2f} {best[5]:+7.2f}',
              flush=True)

    return path, energies, components


def main():
    n_workers = 3
    print("=" * 70)
    print("5D 极小能量裂变路径（打开 η / ε1 / ε2，Nmax=12，磁盘缓存）")
    print("=" * 70)

    t0 = time.time()
    with mp.Pool(n_workers, initializer=_init_worker) as pool:
        path, energies, components = min_energy_path_5d(pool)
    print(f"\n总耗时 {time.time() - t0:.1f}s\n")

    valid = ~np.isnan(energies)
    el = path[:, 0]
    # 基态 = 势垒前的紧致区（0.5 ≤ elong ≤ 1.3）能量最低点；势垒 = 基态之后的能量最高点（鞍点）。
    gs_mask = valid & (el >= 0.5) & (el <= 1.3)
    i_gs = int(np.nanargmin(np.where(gs_mask, energies, np.nan)))
    bar_mask = valid & (el >= el[i_gs])
    i_bar = int(np.nanargmax(np.where(bar_mask, energies, np.nan)))

    print("=" * 70)
    print("结果摘要")
    print("=" * 70)
    print(f"  基态     : {energies[i_gs]:+8.2f} MeV @ elong={path[i_gs,0]:.2f}, "
          f"neck={path[i_gs,1]:.2f}, η={path[i_gs,2]:+.2f}, "
          f"ε1={path[i_gs,3]:+.2f}, ε2={path[i_gs,4]:+.2f}")
    print(f"  势垒     : {energies[i_bar]:+8.2f} MeV @ elong={path[i_bar,0]:.2f}, "
          f"neck={path[i_bar,1]:.2f}, η={path[i_bar,2]:+.2f}")
    print(f"  势垒高度 : {energies[i_bar]-energies[i_gs]:+8.2f} MeV")
    print(f"  末点     : {energies[-1]:+8.2f} MeV @ elong={path[-1,0]:.2f}, "
          f"η={path[-1,2]:+.2f}, ε1={path[-1,3]:+.2f}, ε2={path[-1,4]:+.2f}")

    # 保存
    out = os.path.join(PROJECT_ROOT, "results", "随机行走")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "path_5d.npz"),
             path=path, energies=energies, components=components)
    print(f"\n已保存 results/随机行走/path_5d.npz")


if __name__ == "__main__":
    main()
