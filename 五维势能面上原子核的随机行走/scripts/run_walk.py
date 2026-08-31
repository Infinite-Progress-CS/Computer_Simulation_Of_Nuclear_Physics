"""
run_walk.py — 五维势能面随机行走 + 3D 动画（宏观-微观模型）
====================================================================
完整流程（与旧单文件版 main 相同，但势能面换成宏观-微观 MacroMicro）：
  1. 构建五维势能面（液滴 + Strutinsky 壳修正 + BCS 对修正）
  2. Metropolis 随机行走（5 个参数，无效形状拒绝）
  3. 极小能量（绝热）裂变路径
  4. 势能面切片（elong-neck 平面）
  5. 静态图：形状示意图 + 势能面切片
  6. 3D 分屏动画：左=核形状演化，右=势能面 + 行走轨迹

输出（本目录下）：核形状随机行走.gif、势能面切片.png、五参数形状示意图.png
"""

import os
import sys
import numpy as np

# 项目根目录 = 本脚本上一级目录；把 src/ 加入模块搜索路径，输出写进 output/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from macro_micro import MacroMicro
from metropolis import (metropolis_walk, min_energy_path, neck_fraction,
                        compute_surface_slice)
from visualize import plot_shape_schematic, plot_pes_slice, animate_split

BASE_DIR = os.path.join(PROJECT_ROOT, "results", "随机行走", "figures")
Z, N = 92, 144   # U-236


def main():
    print("=" * 66)
    print("五维势能面随机行走 + 3D 动画（宏观-微观模型，U-236）")
    print("=" * 66)

    # ---- 势能面：宏观-微观 ----
    pes = MacroMicro(Z, N)
    print(f"\n核素 U-236:  R0={pes.R0:.3f} fm")
    print(f"  液滴 E_S0={pes.ld.E_S0:.1f} MeV, E_C0={pes.ld.E_C0:.1f} MeV")
    print(f"  壳修正平滑 γ={pes.gamma:.2f} MeV, 配对 G={pes.G_n:.4f} MeV")

    # ---- [1] Metropolis 随机行走（5D）----
    q0 = np.array([0.3, 0.99, 0.0, 0.05, 0.05])   # 近球起点
    print(f"\n[1] 五维 Metropolis 随机行走 (T=8→0.05 MeV 退火, 350 步)...")
    path, energies, components, accept = metropolis_walk(
        pes, q0, T=8.0, n_steps=350, T_end=0.05, ratchet_elong=2.3, seed=1)
    print(f"  接受率 = {accept:.2%}")
    print(f"  能量范围 [{energies.min():+.2f}, {energies.max():+.2f}] MeV")
    print(f"  末帧断裂程度 f = {neck_fraction(path[-1], pes.shape):.2f}")

    # ---- [2] 极小能量（绝热）裂变路径 ----
    print("\n[2] 极小能量裂变路径...")
    min_path, min_E, min_C = min_energy_path(pes, n_elong=35, n_neck=8)
    i_bar = int(np.nanargmax(min_E))
    print(f"  势垒 = {min_E[i_bar]:.2f} MeV @ elong={min_path[i_bar, 0]:.2f}")
    print(f"  断裂点 = {min_E[-1]:+.2f} MeV @ elong={min_path[-1, 0]:.2f}")

    # ---- [3] 势能面切片 ----
    print("\n[3] 势能面切片 (elong-neck)...")
    Q1, Q2, V = compute_surface_slice(pes, n=20)

    # ---- [4] 静态图 ----
    print("\n[4] 生成静态图...")
    plot_shape_schematic(pes)
    plot_pes_slice(pes, path, Q1, Q2, V, min_path=min_path)

    # ---- [5] 3D 动画 ----
    print("\n[5] 生成 3D 分屏动画...")
    animate_split(min_path, min_E, min_C, pes, Q1, Q2, V,
                  filename="核形状随机行走.gif", every=1, fps=6,
                  overlay_path=path, overlay_energies=energies)

    print("\n" + "=" * 66)
    print("完成。输出文件（本目录下）：")
    print("  五参数形状示意图.png    —— 5 参数各自对核形状的影响")
    print("  势能面切片.png          —— 2D 势能面 + 行走轨迹 + 极小能量路径")
    print("  核形状随机行走.gif      —— 3D 分屏动画（左形状/右势能面+轨迹）")
    print("=" * 66)


if __name__ == "__main__":
    main()
