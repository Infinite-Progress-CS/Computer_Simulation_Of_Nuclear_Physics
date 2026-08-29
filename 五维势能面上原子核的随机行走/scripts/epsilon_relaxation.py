"""
epsilon_relaxation.py — ε 形变弛豫与双峰势垒
====================================================================
对称裂变（η=0, eps1=eps2=ε）下，对每个 elong 在 (ε, neck) 空间取极小，
得到含 ε 弛豫的"真实裂变路径"。由此：
  1. 定位基态（形变 ε≈0.24 长椭球，对应 β≈0.25）
  2. 从真基态测量双峰裂变势垒（内/外鞍点，目标 ~5.9 MeV）
  3. 与纯液滴路径对照（液滴基态球形、单峰势垒 ~12.7 MeV）

输出：双峰势垒图（本脚本目录下）。
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 项目根目录 = 本脚本上一级目录；把 src/ 加入模块搜索路径，输出写进 output/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from macro_micro import MacroMicro
from liquid_drop import FRLDMPES
from metropolis import min_energy_path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
BASE_DIR = os.path.join(PROJECT_ROOT, "output")

Z, N = 92, 144   # U-236


def relaxed_path(pes, elong_lims=(0.3, 3.0), n_elong=20,
                 eps_lims=(0.0, 0.6), n_eps=13, n_neck=5):
    """对每个 elong 在 (eps, neck) 取极小（η=0, eps1=eps2=ε），返回松弛裂变路径。

    返回 (elong, eps, neck, E, q_array)。
    """
    elong = np.linspace(*elong_lims, n_elong)
    eps = np.linspace(*eps_lims, n_eps)
    necks = np.linspace(0.5, 0.99, n_neck)
    E = np.full(n_elong, np.nan)
    eps_path = np.full(n_elong, np.nan)
    neck_path = np.full(n_elong, np.nan)
    q_path = np.zeros((n_elong, 5))

    for i, el in enumerate(elong):
        best = np.inf
        for ep in eps:
            for nk in necks:
                q = [el, nk, 0.0, ep, ep]
                try:
                    V = pes.energy(q)
                except Exception:
                    continue
                if V < best:
                    best = V
                    best_ep, best_nk, best_q = ep, nk, q
        if np.isfinite(best):
            E[i] = best
            eps_path[i] = best_ep
            neck_path[i] = best_nk
            q_path[i] = best_q
    return elong, eps_path, neck_path, E, q_path


def find_peaks(x, y):
    """找局部极大（平滑后），返回 [(x, y), ...]。"""
    y_s = np.convolve(y, np.ones(3) / 3.0, mode="same")
    peaks = []
    for i in range(2, len(y_s) - 2):
        if y_s[i] >= y_s[i - 1] and y_s[i] > y_s[i + 1]:
            if y_s[i] > 0.15 * np.nanmax(y_s):
                peaks.append((x[i], y_s[i]))
    return peaks


def main():
    print("=" * 66)
    print("ε 形变弛豫与双峰势垒（U-236, 对称 η=0）")
    print("=" * 66)

    pes = MacroMicro(Z, N)

    # ---- 含 ε 弛豫的裂变路径（宏观-微观）----
    print("\n[1] 对每个 elong 在 (ε, neck) 取极小，构建松弛裂变路径...")
    elong, eps_path, neck_path, E_mm, q_path = relaxed_path(pes)

    # 基态 = 势垒前单核区（elong<1.5）的局部极小；断裂点深谷（碎片幻数壳）不算基态
    i_split = int(np.searchsorted(elong, 1.5))
    i_gs = i_split + int(np.nanargmin(E_mm[:i_split]))
    i_scis = int(np.nanargmin(E_mm))
    print(f"\n  基态（单核形变区，elong<1.5）:")
    print(f"    elong={elong[i_gs]:.2f}, eps={eps_path[i_gs]:+.2f} "
          f"(β≈{eps_path[i_gs]/0.946:+.2f}), neck={neck_path[i_gs]:.2f}")
    print(f"    能量 = {E_mm[i_gs]:+.2f} MeV（相对球液滴）")
    print(f"  断裂点深谷（碎片幻数壳）: elong={elong[i_scis]:.2f}, "
          f"E={E_mm[i_scis]:+.2f} MeV（比基态低 {E_mm[i_gs]-E_mm[i_scis]:.1f} MeV，"
          f"对应裂变 Q 值区，非基态）")

    # 球点（elong→0 参考）
    E_sphere = pes.energy([0.0, 0.99, 0.0, 0.0, 0.0])
    print(f"    球形点能量 = {E_sphere:+.2f} MeV → 形变稳定化 {E_mm[i_gs]-E_sphere:.2f} MeV")

    # 双峰检测
    peaks = find_peaks(elong, E_mm)
    print(f"\n  裂变势垒（从真基态测量）:")
    print(f"    基态能量 = {E_mm[i_gs]:+.2f} MeV")
    for p_e, p_v in peaks:
        print(f"    鞍点 elong={p_e:.2f}  势垒 = {p_v - E_mm[i_gs]:.2f} MeV")

    # ---- 纯液滴对照（含 ε 弛豫）----
    print("\n[2] 纯液滴对照（含 ε 弛豫）...")
    ld = FRLDMPES(Z, N, nz=60, nrho=60, nsurf=64, nphi=40)
    elong_ld, eps_ld, neck_ld, E_ld, _ = relaxed_path(ld, n_eps=5)
    i_gs_ld = int(np.nanargmin(E_ld))
    i_bar_ld = int(np.nanargmax(E_ld))
    print(f"  液滴基态: elong={elong_ld[i_gs_ld]:.2f}, eps={eps_ld[i_gs_ld]:+.2f} "
          f"(球形), E={E_ld[i_gs_ld]:+.2f}")
    print(f"  液滴势垒: {E_ld[i_bar_ld]:.2f} MeV @ elong={elong_ld[i_bar_ld]:.2f}")

    # ---- 绘制双峰势垒图 ----
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(elong, E_ld - E_ld[i_gs_ld], "b--", lw=1.6,
            label=f"纯液滴（基态球，势垒 {E_ld[i_bar_ld]-E_ld[i_gs_ld]:.2f} MeV）")
    ax.plot(elong, E_mm - E_mm[i_gs], "r-", lw=2.2,
            label="宏观-微观（含 ε 弛豫）")
    for p_e, p_v in peaks:
        ax.plot(p_e, p_v - E_mm[i_gs], "r*", ms=16, mec="k")
    ax.plot(elong[i_gs], 0.0, "go", ms=10, label="基态（形变）")
    ax.set_xlabel("elong 拉长")
    ax.set_ylabel("能量（相对基态）(MeV)")
    ax.set_title("双峰裂变势垒：纯液滴 vs 宏观-微观（U-236，含 ε 弛豫）")
    ax.axhline(0.0, color="gray", lw=0.8, alpha=0.5)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(BASE_DIR, "双峰裂变势垒.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  双峰势垒图已保存: {out}")

    # ---- 打印路径（诊断用）----
    print(f"\n  {'elong':>6} {'eps':>6} {'neck':>6} {'E(相对基态)':>12}")
    for i in range(len(elong)):
        print(f"  {elong[i]:6.2f} {eps_path[i]:+6.2f} {neck_path[i]:6.2f} "
              f"{E_mm[i]-E_mm[i_gs]:+12.2f}")

    print("\n" + "=" * 66)
    print("ε 形变弛豫完成。")
    print("=" * 66)


if __name__ == "__main__":
    main()
