# -*- coding: utf-8 -*-
"""animate_walk_tc.py — 真实随机行走轨迹 + 双中心 WS 势能面 → 3D 分屏裂变动画。

与 run_walk_asym.py（8月31日老半成品）的区别：
  * 势能面由双中心 Woods-Saxon 实算（basis='two_center'，宽域 elong 0.3-6.0、
    neck 0.02-0.99，ε 在碎片基态形变上弛豫），势垒 ~4-7 MeV（物理值，对齐
    Möller 5.03），而非单中心 Nmax=12 的 17.77 MeV 假势垒。
  * 轨迹是 brownian_yield 记录的真实 Brownian/Metropolis 行走（record_trajectory=True），
    而非手搓平滑"演示路径"。断裂处的质量不对称 η 是行走自然演化出来的。
  * 断裂处两个碎片大小不等（η≈0.19~0.27），呈现真实非对称裂变。

输出（output/）：
  核形状随机行走_双中心.gif   —— 3D 分屏动画（左 3D 形状 / 右 双中心势能面+真实轨迹）
  双中心势能面切片_真实行走.png —— 2D 等高线 V(elong,neck)=min_{η,ε1,ε2} + 真实轨迹
"""
import os
import sys
import argparse

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import multiprocessing as mp

from macro_micro import MacroMicro
from shape import Shape3QS
from metropolis import neck_fraction
from visualize import shape_surface, _prepare_pes, BASE_DIR

Z, N = 92, 144
R0_LD = 1.16 * 236 ** (1.0 / 3.0)

TRAJ = os.path.join(PROJECT_ROOT, "results", "随机行走",
                    "walk_yield_pes_table_U236_nmax8_fine_nw300_ms600_s888.npz")


# ---- 多进程 worker（Windows spawn：模块顶层函数 + initializer 构造双中心模型）----
_PES = None


def _init_worker():
    global _PES
    _PES = MacroMicro(Z, N, Nmax=8, nz_uni=48, basis='two_center')


def _eval(q):
    global _PES
    try:
        return float(_PES.energy(q))
    except Exception:
        return float("nan")


def load_trajectory(npz_path, walk_idx=None, eta_target=0.186):
    d = np.load(npz_path, allow_pickle=False)
    traj = d["trajectories"]
    E = d["trajectory_energy"]
    steps = d["trajectory_steps"]
    scis = d["trajectory_scission"]
    q0 = d["q0"]

    if walk_idx is None:
        # 取断裂行走中 η 最接近 Möller 峰 η≈0.186 的一条（非对称又不过分极端）
        cands = [(abs(traj[w, steps[w], 2] - eta_target), w, int(steps[w]),
                  float(traj[w, steps[w], 2]))
                 for w in range(len(scis)) if scis[w] and abs(traj[w, steps[w], 2]) > 0.03]
        cands.sort()
        _, walk_idx, s_end, eta = cands[0]
    else:
        s_end = int(steps[walk_idx]) if scis[walk_idx] else traj.shape[1] - 1
        eta = float(traj[walk_idx, s_end, 2])

    n = s_end + 1
    path = traj[walk_idx, :n].copy()
    energies = E[walk_idx, :n].copy()
    A_L = 236 * (1.0 - abs(eta)) / 2.0
    print(f"轨迹 walk={walk_idx}  步数={s_end}  断裂 η={eta:+.3f}  "
          f"A_L={A_L:.0f}/{236 - A_L:.0f}  能量 {energies[0]:+.2f}→{energies[-1]:+.2f} MeV")
    return path, energies, q0


def extend_post_scission(pool, path, energies, n_tail=24, neck_end=0.10, elong_gain=0.5):
    """断裂点之后追加「碎片漂离」尾段：η 冻结、ε 弛豫→0，颈收细成断口，elong 继续增大。

    随机行走在断裂点（颈半径 c0=2.5 fm ≈ neck≈0.44）终止——这是产额记录的
    物理断裂点，此时两碎片仍由细颈连接。断裂之后质量不对称 η 冻结，碎片形变 ε
    随退激弛豫到分离基态（近球形），只剩几何分离（颈收成断口、碎片漂离）。尾段
    只收到 neck=0.10（颈半径 ~0.8 fm，视觉上已完全断开），不追到势能面最下边缘。
    能量用双中心模型实算，贴住势能面谷底。
    """
    def _ss(t):
        x = np.clip(t, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    q0 = path[-1].copy()
    t = np.linspace(0.0, 1.0, n_tail)
    s = _ss(t)
    tail = np.empty((n_tail, 5))
    tail[:, 0] = q0[0] + elong_gain * s          # elong 继续增大（碎片漂离）
    tail[:, 1] = q0[1] + (neck_end - q0[1]) * s  # 颈 0.44 → neck_end（收成断口）
    tail[:, 2] = q0[2]                           # η 冻结（碎片已形成）
    tail[:, 3] = q0[3] * (1.0 - s)               # ε1 弛豫→0（碎片退激到球形基态）
    tail[:, 4] = q0[4] * (1.0 - s)               # ε2 弛豫→0
    res = pool.map(_eval, [list(r) for r in tail], chunksize=4)
    E_tail = np.array(res, dtype=float)
    E_tail = np.where(np.isnan(E_tail), energies[-1], E_tail)  # NaN 回退断裂点能量
    path2 = np.vstack([path, tail])
    energies2 = np.concatenate([energies, E_tail])
    print(f"  断裂后追加 {n_tail} 帧碎片漂离尾段 "
          f"(neck {q0[1]:.2f}→{neck_end:.2f}, elong {q0[0]:.2f}→{q0[0] + elong_gain:.2f}, "
          f"η 冻结·ε→0，能量 {energies[-1]:+.1f}→{E_tail[-1]:+.1f} MeV)")
    return path2, energies2


def compute_slice_2d(pool, n=16, elong_lims=(0.3, 6.0), neck_lims=(0.02, 0.99),
                     eta_vals=(0.0, 0.19), eps_pairs=((0.0, 0.0), (0.2, 0.1), (0.4, 0.0))):
    """V(elong, neck) = min_{η,ε1,ε2} V，双中心模型实算（宽域、细网格、ε 弛豫）。

    取代旧窄 PES 表（elong 0.5-2.9、neck 0.1-0.99），域覆盖到完全分离
    （elong 6.0、neck 0.02）。ε 在 {(0,0),(0.2,0.1),(0.4,0)} 上弛豫：紧凑区
    球形 ε=0 最低，断裂/分离区重碎片 ε1=0.4、轻碎片球形 ε2=0 最低（实测
    断裂点 min ε=(0.4,0) → -10.0 MeV，轨迹 ε=(0.396,0.089) → -9.6 MeV，
    二者几乎重合，断裂点落到面上）。
    """
    el = np.linspace(*elong_lims, n)
    nk = np.linspace(*neck_lims, n)
    Q1, Q2 = np.meshgrid(el, nk)
    tasks = [[Q1[i, j], Q2[i, j], eta, eps[0], eps[1]]
             for i in range(n) for j in range(n)
             for eta in eta_vals for eps in eps_pairs]
    n_comb = len(eta_vals) * len(eps_pairs)
    print(f"  [势能面切片 {n}x{n} × {len(eta_vals)}η × {len(eps_pairs)}ε = {len(tasks)} 点，双中心实算]")
    res = pool.map(_eval, tasks, chunksize=4)
    V = np.full((n, n), np.nan)
    k = 0
    for i in range(n):
        for j in range(n):
            seg = res[k:k + n_comb]
            k += n_comb
            vals = [v for v in seg if not np.isnan(v)]
            if vals:
                V[i, j] = min(vals)
    return Q1, Q2, V


def plot_slice_2d(shape, path, Q1, Q2, V, filename, scis_idx=None,
                  vmin=-14.0, vmax=12.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if scis_idx is None:
        scis_idx = len(path) - 1

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    Vp, vmin, vmax = _prepare_pes(V, vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 25)
    cf = ax.contourf(Q1, Q2, Vp, levels=levels, cmap="viridis", extend="both")
    cs = ax.contour(Q1, Q2, Vp, levels=levels[::4], colors="0.35",
                    linewidths=0.4, alpha=0.55)
    ax.clabel(cs, fmt="%.0f", fontsize=6.5, inline=True)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("形变能 ΔV (MeV)", fontsize=11)
    ax.plot(path[:, 0], path[:, 1], color="white", lw=4.0, alpha=0.9, zorder=3)
    ax.plot(path[:, 0], path[:, 1], color="crimson", lw=1.4, alpha=0.95,
            zorder=4, label="真实随机行走轨迹")
    ax.plot(path[0, 0], path[0, 1], "o", color="white", ms=11, mec="k", mew=1.3,
            zorder=5, label="起点（近球）")
    ax.plot(path[scis_idx, 0], path[scis_idx, 1], "*", color="gold", ms=20,
            mec="k", mew=0.9, zorder=5, label="断裂点（非对称分离）")
    ax.set_xlabel("elong 拉长", fontsize=12)
    ax.set_ylabel("neck 颈部", fontsize=12)
    ax.set_title("双中心 WS 五维势能面切片 V(elong,neck)=min$_{η,ε}$ + 真实行走",
                 fontsize=12)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()
    out = os.path.join(BASE_DIR, filename)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  势能面切片已保存: {out}")


def _select_frames(n_scis, n_tail, n_transition=60, n_wait_target=40):
    """非均匀抽帧：势垒等待区粗采样、越障/断裂/分离区全帧。

    真实 Brownian 行走 ~90% 步数耗在鞍点附近热涨落（elong 0.5-1.8、neck 0.8-0.99，
    见 Kramers 越障等待），只在最后几十步破障到断裂。动画按物理故事抽帧，
    等待区压缩到 ~n_wait_target 帧，避免在势能面顶部拖沓。
    """
    wait_end = max(0, n_scis - n_transition)
    wait_stride = max(1, (wait_end + 1) // n_wait_target)
    idx = list(range(0, wait_end, wait_stride))
    idx += list(range(wait_end, n_scis + 1))                 # 越障+断裂全帧
    idx += list(range(n_scis + 1, n_scis + 1 + n_tail))      # 漂离尾段全帧
    return idx


def animate_walk(path, energies, shape, Q1, Q2, V,
                 filename="核形状随机行走_双中心.gif", every=1, fps=6,
                 scis_idx=None, vmin=-14.0, vmax=12.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    R0 = shape.R0

    if scis_idx is None:
        scis_idx = len(path) - 1
        frames = list(range(0, len(path), every))
    else:
        n_tail = len(path) - 1 - scis_idx
        frames = _select_frames(scis_idx, n_tail)

    z_lim = max(np.abs(shape.profile(p)[0]).max() for p in path) * 1.1
    r_lim = R0 * 1.4
    f_arr = np.array([neck_fraction(p, shape) for p in path])

    Vp, vmin, vmax = _prepare_pes(V, vmin=vmin, vmax=vmax)

    # 轨迹 z 用真实能量（裁剪到显示量程），如实展示行走在 η/ε 方向的热涨落
    E_traj = np.clip(energies, vmin, vmax)
    zmin, zmax = vmin, vmax

    fig = plt.figure(figsize=(19, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.9], height_ratios=[6.5, 0.55],
                          left=0.02, right=0.98, top=0.96, bottom=0.085,
                          wspace=0.08, hspace=0.30)
    ax_shape = fig.add_subplot(gs[0, 0], projection="3d")
    ax_surf = fig.add_subplot(gs[0, 1], projection="3d")
    ax_info = fig.add_subplot(gs[1, :])

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=zmin, vmax=zmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_surf, shrink=0.55, pad=0.08, aspect=30)
    cbar.set_label("形变能 ΔV (MeV)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    legend_handles = [
        Line2D([0], [0], color="black", lw=3, label="行走轨迹"),
        Line2D([0], [0], marker="o", color="lime", markersize=9, linestyle="None",
               markeredgecolor="black", markeredgewidth=1.0, label="起点"),
        Line2D([0], [0], marker="s", color="red", markersize=9, linestyle="None",
               markeredgecolor="black", markeredgewidth=1.0, label="断裂点"),
        Line2D([0], [0], marker="o", color="black", markersize=9, linestyle="None",
               markeredgecolor="white", markeredgewidth=1.2, label="当前点"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.50, 0.012), ncol=4, frameon=False,
               fontsize=11, handlelength=1.4, handletextpad=0.4,
               columnspacing=1.0, borderaxespad=0.0)
    fig.text(0.50, 0.062,
             "左图: 红=基准，膨胀→浅红 / 收缩→深红（断裂后 大碎片浅红·小碎片深红）    |    右图: ΔV 蓝低 → 黄高",
             ha="center", va="bottom", fontsize=10.5, color="0.25")

    def update(i):
        q = path[i]
        Vv = energies[i]

        ax_shape.clear()
        surfaces, state = shape_surface(q, shape, f_arr[i])
        for X, Y, Zs, fc in surfaces:
            ax_shape.plot_surface(X, Y, Zs, facecolors=fc, linewidth=0,
                                  antialiased=True, rstride=1, cstride=1)
        ax_shape.set_xlim(-z_lim, z_lim)
        ax_shape.set_ylim(-r_lim, r_lim)
        ax_shape.set_zlim(-r_lim, r_lim)
        ax_shape.set_box_aspect((z_lim / r_lim, 1, 1))
        ax_shape.view_init(elev=15, azim=-90)
        ax_shape.dist = 8
        ax_shape.set_xlabel("z (fm) 对称轴", fontsize=9)
        ax_shape.set_ylabel("x (fm)", fontsize=9)
        ax_shape.set_zlabel("y (fm)", fontsize=9)
        ax_shape.set_title(f"核形状  Step {i}  [{state}]", fontsize=12)

        ax_info.clear()
        ax_info.axis("off")
        info = (f"elong={q[0]:+.3f}   neck={q[1]:+.3f}   η={q[2]:+.3f}   "
                f"ε1={q[3]:+.3f}   ε2={q[4]:+.3f}\n"
                f"ΔV={Vv:+.2f} MeV（双中心 WS 壳修正 + 液滴 + 对修正）")
        ax_info.text(0.5, 0.5, info, transform=ax_info.transAxes, fontsize=13,
                     va="center", ha="center", linespacing=1.5)

        ax_surf.clear()
        ax_surf.plot_surface(Q1, Q2, Vp, cmap="viridis", alpha=0.55,
                             linewidth=0, antialiased=True)
        ax_surf.plot(path[:i + 1, 0], path[:i + 1, 1], E_traj[:i + 1],
                     color="white", lw=7.0, alpha=0.95)
        ax_surf.plot(path[:i + 1, 0], path[:i + 1, 1], E_traj[:i + 1],
                     color="black", lw=3.0)
        ax_surf.scatter([q[0]], [q[1]], [E_traj[i]], color="black", s=90,
                        edgecolor="white", linewidth=1.4, depthshade=False)
        ax_surf.scatter([path[0, 0]], [path[0, 1]], [E_traj[0]],
                        color="lime", marker="o", s=90, edgecolor="black",
                        linewidth=1.0, depthshade=False)
        ax_surf.scatter([path[scis_idx, 0]], [path[scis_idx, 1]],
                        [E_traj[scis_idx]], color="red", marker="s", s=90,
                        edgecolor="black", linewidth=1.0, depthshade=False)
        ax_surf.dist = 8
        ax_surf.set_xlabel("elong 拉长", fontsize=9)
        ax_surf.set_ylabel("neck 颈部", fontsize=9)
        ax_surf.set_zlabel("ΔV (MeV)", fontsize=9)
        ax_surf.set_xlim(float(Q1.min()), float(Q1.max()))
        ax_surf.set_ylim(float(Q2.min()), float(Q2.max()))
        ax_surf.set_zlim(zmin, zmax)
        ax_surf.view_init(elev=25, azim=-60)
        ax_surf.set_title("双中心五维势能面 V(elong,neck) 与真实行走轨迹", fontsize=12)
        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    out = os.path.join(BASE_DIR, filename)
    try:
        ani.save(out, writer="pillow", fps=fps, dpi=140)
        print(f"  动画已保存: {out} ({len(frames)} 帧, {fps} fps)")
    except Exception as e:
        print(f"  (GIF 保存失败: {e})")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=TRAJ, help="记录轨迹的 walk npz")
    ap.add_argument("--walk", type=int, default=None,
                    help="指定行走索引；默认取最不对称的断裂行走")
    ap.add_argument("--every", type=int, default=None,
                    help="抽帧间隔；默认自动压到 ~240 帧")
    ap.add_argument("--fps", type=float, default=6)
    ap.add_argument("--vmin", type=float, default=-50.0)
    ap.add_argument("--vmax", type=float, default=10.0)
    args = ap.parse_args()

    print("=" * 66)
    print("真实随机行走 + 双中心 WS 势能面 → 3D 分屏裂变动画（U-236）")
    print("=" * 66)

    shape = Shape3QS(R0_LD)

    print("\n[1] 加载真实行走轨迹...")
    path, energies_npz, q0 = load_trajectory(args.npz, walk_idx=args.walk)
    q_scis = path[-1].copy()
    scis_idx = len(path) - 1

    n_workers = 3
    print(f"  启动 {n_workers} 个并行进程（双中心 WS 实算，单点 ~1.8s）...")
    with mp.Pool(n_workers, initializer=_init_worker) as pool:
        print("  重算轨迹能量（双中心 WS，与势能面同模型，消除断裂处 npz→实算跳变）...")
        energies = np.array(pool.map(_eval, [list(p) for p in path], chunksize=4),
                            dtype=float)
        energies = np.where(np.isnan(energies), energies_npz, energies)
        E_scis = energies[-1]

        path, energies = extend_post_scission(pool, path, energies)

        print("\n[2] 双中心势能面切片 V(elong,neck)=min_{η,ε} ...")
        Q1, Q2, V = compute_slice_2d(pool)

    i_bar = int(np.argmax(energies))
    print(f"  势垒 = {energies[i_bar]:.2f} MeV @ elong={path[i_bar,0]:.2f}, "
          f"neck={path[i_bar,1]:.2f}, η={path[i_bar,2]:+.2f}")
    print(f"  断裂点 = {E_scis:+.2f} MeV @ elong={q_scis[0]:.2f}, "
          f"η={q_scis[2]:+.2f}（neck={q_scis[1]:.2f}）")

    print("\n[3] 生成静态图...")
    plot_slice_2d(shape, path, Q1, Q2, V, "双中心势能面切片_真实行走.png",
                  scis_idx=scis_idx, vmin=args.vmin, vmax=args.vmax)

    print("\n[4] 生成 3D 分屏动画...")
    every = args.every or max(1, len(path) // 160)
    animate_walk(path, energies, shape, Q1, Q2, V,
                 filename="核形状随机行走_双中心.gif", every=every, fps=args.fps,
                 scis_idx=scis_idx, vmin=args.vmin, vmax=args.vmax)

    print("\n" + "=" * 66)
    print("完成。输出文件（output/）：")
    print("  双中心势能面切片_真实行走.png  —— 双中心 PES 切片 + 真实轨迹")
    print("  核形状随机行走_双中心.gif      —— 3D 分屏动画（真实行走 + 非对称断裂）")
    print("=" * 66)


if __name__ == "__main__":
    main()
