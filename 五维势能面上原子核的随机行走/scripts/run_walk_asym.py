"""
run_walk_asym.py — 非对称裂变路径 + 3D 动画（宏观-微观模型，U-236）
====================================================================
在 run_walk.py 基础上，把"极小能量路径"从对称（η=0, 碎片等质量）改成
**非对称**（每个 elong 在 η×neck 空间取极小，碎片 ~140/~96，η>0 更稳），
势能面切片也做 η 松弛（V(elong,neck)=min_η V）。

性能：单次 Woods-Saxon 对角化 ~4s。液滴库仑积分需 nz=nrho=60 细网格（峰值
~0.8GB/进程），机器空闲内存有限，故只用 3 个 worker 并行，且 worker 进程不
import matplotlib（延迟到主进程绘图），把内存峰值压在 ~2.7GB。

输出（本目录下）：
  核形状非对称裂变.gif   —— 3D 分屏动画（左形状/右势能面+轨迹，碎片一大一小）
  非对称势能面切片.png   —— 2D 势能面（η 松弛）+ 行走轨迹 + 非对称极小能量路径
  五参数形状示意图.png   —— 5 参数各自对核形状的影响（复用）
"""

import os
import sys
import numpy as np
import multiprocessing as mp

# 项目根目录 = 本脚本上一级目录；把 src/ 加入模块搜索路径，输出写进 output/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from macro_micro import MacroMicro
from metropolis import metropolis_walk, neck_fraction

BASE_DIR = os.path.join(PROJECT_ROOT, "results", "随机行走", "figures")
Z, N = 92, 144   # U-236


# ======================================================================
# 多进程 worker（Windows spawn：模块顶层函数 + initializer 构造实例）
# 注意：本模块顶部只 import numpy/scipy 依赖，不 import matplotlib，
#       这样每个 worker 进程省掉 matplotlib+PIL 的 DLL 加载（~300MB）。
# ======================================================================
_PES = None


def _init_worker():
    global _PES
    _PES = MacroMicro(Z, N)


def _eval_full(q):
    """返回 (V, dV_s, dV_c, dE_sh, dE_pair)；无效形状返回 (nan,0,0,0,0)。"""
    global _PES
    try:
        V, dV_s, dV_c, dE_sh, dE_pair = _PES.energy_components(q)
        return (float(V), float(dV_s), float(dV_c), float(dE_sh), float(dE_pair))
    except Exception:
        return (float("nan"), 0.0, 0.0, 0.0, 0.0)


def _smoothstep(t, t0, t1):
    """0→1 三次平滑阶跃：t<t0 为 0，t>t1 为 1，中间 C1 连续。"""
    x = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_demo_path(pool, n_way=110):
    """平滑物理裂变演示路径：球 → 拉长 → 颈缩 → 非对称 → 完全分离成两个碎片。

    各参数沿时间 t∈[0,1] 平滑演化（物理设定，非能量极小搜索）：
      elong 0.40→2.50→3.00   neck 0.99→0.15→0.001   η 0→0.19   ε1(轻) 0→0.40   ε2(重) 0→0.10
    主裂变 t=0→0.70：颈缩到物理断裂颈 0.15（颈半径 ~0.7 fm）、elong 到 2.5；
    漂离 t=0.70→0.90：颈停在 0.15、两碎片小幅漂离到 elong 3.0；
    掐断 t=0.90→1.0：颈快速缩到 ~0.001，最后一帧连接处收成一个点（断口）。
    针尖只出现在最后几帧的断口，中间阶段不长时间停留为细针。ε1=左/轻碎片形变（大）、ε2=右/重碎片形变（小），η≈0.19。
    返回 (path, energies, components)，path 每行 [elong,neck,eta,eps1,eps2]，
    components 每行 [dV_s, dV_c, dE_sh, dE_pair]。
    """
    t = np.linspace(0.0, 1.0, n_way)
    # 颈：主裂变 0.99→0.15（物理颈），末段(0.90→1)快速掐断到 ~0.001——最后一帧连接处收成一个点
    neck = 0.99 - 0.84 * _smoothstep(t, 0.10, 0.70) - 0.149 * _smoothstep(t, 0.90, 1.0)
    # 拉长：0→0.70 到断裂点 elong≈2.5；0.70→0.90 两碎片小幅漂离到 elong≈3.0（不抻成针）
    elong = 0.40 + 2.10 * _smoothstep(t, 0.0, 0.70) + 0.50 * _smoothstep(t, 0.70, 0.90)
    eta = 0.19 * _smoothstep(t, 0.25, 0.60)   # 断裂前长出质量不对称 η→0.19
    eps1 = 0.40 * _smoothstep(t, 0.40, 0.70)  # 轻碎片形变 ε1→0.40
    eps2 = 0.10 * _smoothstep(t, 0.45, 0.70)  # 重碎片近球形 ε2→0.10

    path = np.column_stack([elong, neck, eta, eps1, eps2])
    print(f"  演示路径 {n_way} 点（球→完全分离，η→0.19，ε1→0.40，ε2→0.10）")
    tasks = [list(p) for p in path]
    res = pool.map(_eval_full, tasks, chunksize=4)
    energies = np.array([r[0] for r in res])
    components = np.array([[r[1], r[2], r[3], r[4]] for r in res])
    n_bad = int(np.isnan(energies).sum())
    if n_bad:
        print(f"  ⚠ {n_bad} 个路径点能量无效(NaN)")
    return path, energies, components


# ======================================================================
# 非对称路径 / 切片（并行）
# ======================================================================
def min_energy_path_asym(pool, elong_lims=(0.5, 3.0), n_elong=18, eps_fix=0.0):
    """非对称裂变极小能量路径（两阶段层次化搜索），ε1=ε2=eps_fix。

    |η| 镜像等价（η 与 −η 只交换左右碎片），故只扫 η≥0。
    neck 网格覆盖到 0.15（真正断裂）；阶段1 粗网格定位，阶段2 在粗最优
    (η*, neck*) 附近 ±0.05 细扫，最终 η/neck 分辨率 ~0.05，路径光滑单调。
    返回 (path, energies, components)，path 第 3 列 = 最优 η。
    """
    elong = np.linspace(*elong_lims, n_elong)
    eta_c = np.array([0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    # neck 必须覆盖到 0.04（近断裂）：非对称裂变只在薄 neck(<0.1) 出现，
    # 0.15 太粗会漏掉非对称谷，误判成对称。
    neck_c = np.array([0.04, 0.10, 0.20, 0.40, 0.70, 0.99])
    d_step = 0.05

    # ---- 阶段1：粗网格 ----
    tasks1 = [[el, nk, et, eps_fix, eps_fix]
              for el in elong for nk in neck_c for et in eta_c]
    print(f"  阶段1 粗网格: {len(tasks1)} 点 "
          f"({n_elong} elong × {len(neck_c)} neck × {len(eta_c)} η)")
    res1 = pool.map(_eval_full, tasks1, chunksize=4)

    n_per = len(neck_c) * len(eta_c)
    coarse = []   # 每个 elong 的粗最优 (V, eta, neck)
    for i in range(n_elong):
        best = None
        k = i * n_per
        for nk in neck_c:
            for et in eta_c:
                V = res1[k][0]
                k += 1
                if not np.isnan(V) and (best is None or V < best[0]):
                    best = (V, et, nk)
        coarse.append(best)

    # ---- 阶段2：粗最优附近细扫 ----
    tasks2 = []
    for i, el in enumerate(elong):
        if coarse[i] is None:
            continue
        et, nk = coarse[i][1], coarse[i][2]
        for de in (-d_step, 0.0, d_step):
            for dn in (-d_step, 0.0, d_step):
                tasks2.append([el, float(np.clip(nk + dn, 0.03, 0.99)),
                               float(np.clip(et + de, 0.0, 0.5)), eps_fix, eps_fix])
    print(f"  阶段2 细网格: {len(tasks2)} 点")
    res2 = pool.map(_eval_full, tasks2, chunksize=4)

    # ---- 汇总（阶段1 最优 + 阶段2 候选取最小）----
    path = np.zeros((n_elong, 5))
    energies = np.full(n_elong, np.nan)
    components = np.zeros((n_elong, 2))
    k2 = 0
    for i, el in enumerate(elong):
        if coarse[i] is None:
            continue
        # 阶段1 最优（含分量）
        best = None
        k1 = i * n_per
        for nk in neck_c:
            for et in eta_c:
                V, dV_s, dV_c, _, _ = res1[k1]
                k1 += 1
                if not np.isnan(V) and (best is None or V < best[0]):
                    best = (V, et, nk, dV_s, dV_c)
        # 阶段2 候选（±0.05 细扫）
        et0, nk0 = best[1], best[2]
        for de in (-d_step, 0.0, d_step):
            for dn in (-d_step, 0.0, d_step):
                V, dV_s, dV_c, _, _ = res2[k2]
                k2 += 1
                e2 = float(np.clip(et0 + de, 0.0, 0.5))
                n2 = float(np.clip(nk0 + dn, 0.03, 0.99))
                if not np.isnan(V) and V < best[0]:
                    best = (V, e2, n2, dV_s, dV_c)
        path[i] = [el, best[2], best[1], eps_fix, eps_fix]
        energies[i] = best[0]
        components[i] = (best[3], best[4])
    return path, energies, components


def compute_surface_slice_asym(pool, n=12, elong_lims=(0.3, 3.0),
                               neck_lims=(0.05, 0.99),
                               eta_vals=(0.0, 0.15, 0.25), eps_fix=0.0):
    """非对称势能面切片 V(elong, neck) = min_η V(elong, neck, η)。"""
    elong = np.linspace(*elong_lims, n)
    neck = np.linspace(*neck_lims, n)
    Q1, Q2 = np.meshgrid(elong, neck)
    tasks = []
    for i in range(n):
        for j in range(n):
            for eta in eta_vals:
                tasks.append([Q1[i, j], Q2[i, j], eta, eps_fix, eps_fix])

    print(f"  [{len(tasks)} 个能量点 = {n}x{n} × {len(eta_vals)} η]")
    res = pool.map(_eval_full, tasks, chunksize=4)

    V = np.full((n, n), np.nan)
    k = 0
    for i in range(n):
        for j in range(n):
            seg = res[k:k + len(eta_vals)]
            k += len(eta_vals)
            vals = [s[0] for s in seg if not np.isnan(s[0])]
            if vals:
                V[i, j] = min(vals)
    return Q1, Q2, V


def main():
    # matplotlib 延迟到主进程导入（worker 进程不加载，省内存）
    from visualize import plot_shape_schematic, plot_pes_slice, animate_split

    print("=" * 66)
    print("非对称裂变路径 + 3D 动画（宏观-微观模型，U-236）")
    print("=" * 66)

    n_workers = 3
    print(f"\n启动 {n_workers} 个并行进程（单次对角化 ~2-4s；机器空闲内存有限，"
          f"worker 数受限）...")

    with mp.Pool(n_workers, initializer=_init_worker) as pool:
        # ---- [1] 平滑裂变演示路径（球 → 完全分离，非对称 + 碎片形变）----
        print("\n[1] 平滑裂变演示路径...")
        path, energies, components = build_demo_path(pool, n_way=100)

        # ---- [2] 势能面切片（延展到完全分离范围，含尾段漂离区）----
        print("\n[2] 势能面切片（每点对 η 取极小）...")
        Q1, Q2, V = compute_surface_slice_asym(
            pool, n=16, elong_lims=(0.3, 6.0), neck_lims=(0.001, 0.99),
            eta_vals=(0.0, 0.19))

    # 势垒 / 断裂点
    i_bar = int(np.nanargmax(energies))
    i_scis = int(np.nanargmin(energies))
    print(f"\n  势垒 = {energies[i_bar]:.2f} MeV @ elong={path[i_bar, 0]:.2f}, "
          f"η={path[i_bar, 2]:+.2f}")
    print(f"  断裂点 = {energies[i_scis]:+.2f} MeV @ elong={path[i_scis, 0]:.2f}, "
          f"η={path[i_scis, 2]:+.2f}")
    print(f"  η  : {path[0,2]:+.2f} → {path[-1,2]:+.2f}   "
          f"ε1 : {path[0,3]:+.2f} → {path[-1,3]:+.2f}   "
          f"ε2 : {path[0,4]:+.2f} → {path[-1,4]:+.2f}")

    # ---- [3] 静态图 ----
    print("\n[3] 生成静态图...")
    pes = MacroMicro(Z, N)   # 主进程自己的实例（并行池已关闭）
    plot_shape_schematic(pes)
    plot_pes_slice(pes, path, Q1, Q2, V, min_path=None,
                   filename="非对称势能面切片.png",
                   title="五维势能面切片（每点对 η 取极小）+ 非对称裂变路径")

    # ---- [4] 3D 动画 ----
    print("\n[4] 生成 3D 分屏动画...")
    animate_split(path, energies, components, pes, Q1, Q2, V,
                  filename="核形状非对称裂变.gif", every=1, fps=5, smooth=False,
                  vmin=-25.0, vmax=40.0)

    print("\n" + "=" * 66)
    print("完成。输出文件（本目录下）：")
    print("  五参数形状示意图.png      —— 5 参数各自对核形状的影响")
    print("  非对称势能面切片.png      —— 2D 势能面（η 松弛）+ 非对称裂变路径")
    print("  核形状非对称裂变.gif      —— 3D 分屏动画（球→拉长→颈缩→非对称→完全分离）")
    print("=" * 66)


if __name__ == "__main__":
    main()
