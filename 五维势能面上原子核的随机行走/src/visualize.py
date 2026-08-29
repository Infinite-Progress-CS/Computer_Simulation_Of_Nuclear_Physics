"""
visualize.py — 绘图 / 动画
====================================================================
从单文件版本原样搬出：核形状示意图、势能面切片、分屏 3D 动画。
pes 需暴露 .shape（Shape3QS）与 .R0；neck_fraction 来自 metropolis.py。
"""

import os
import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")   # 无界面后端，直接存文件
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from metropolis import neck_fraction

# 中文字体（Windows 自带微软雅黑/黑体），避免标题出现方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 输出目录 = 项目根目录下的 output/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "output")


def draw_shape_2d(ax, q, shape, color="#3b6ea5", alpha=0.8):
    """画原子核的 ρ(z) 填充轮廓（轴对称形状的二维剖面）"""
    z, rho = shape.profile(q)
    ax.fill_between(z, -rho, rho, color=color, alpha=alpha, linewidth=0)
    ax.plot([z.min(), z.max()], [0, 0], color="gray", lw=0.8, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _smooth_1d(y, win):
    """边界安全的滑动平均（不填充 0，首末帧保持真实值）"""
    cum = np.concatenate(([0.0], np.cumsum(y)))
    out = np.zeros_like(y)
    for i in range(len(y)):
        lo, hi = max(0, i - win // 2), min(len(y), i + win // 2 + 1)
        out[i] = (cum[hi] - cum[lo]) / (hi - lo)
    return out


def _smooth_path(path, win):
    """对路径的每个参数维度分别做滑动平均"""
    out = np.empty_like(path)
    for k in range(path.shape[1]):
        out[:, k] = _smooth_1d(path[:, k], win)
    return out


def _prepare_pes(V, vmin=-25.0, vmax=40.0, sigma=0.7):
    """势能面数据预处理：裁剪到物理量程 + 高斯平滑。

    消除三种"丑"的成因：
      1. NaN → 填 vmax（不再凸起成"平台/悬崖"）；
      2. 深谷（薄颈处 -80 MeV 的过束缚）→ 裁到 vmin，颜色集中在势垒/浅谷；
      3. 粗网格晶面 / η 包络折痕 → 高斯平滑。
    返回 (Vp, vmin, vmax)，Vp 用于 contourf / plot_surface。
    """
    Vp = np.where(np.isnan(V), vmax, np.clip(V, vmin, vmax))
    if sigma > 0 and Vp.ndim == 2 and min(Vp.shape) > 1:
        Vp = ndimage.gaussian_filter(Vp, sigma=sigma, mode="nearest")
    return Vp, vmin, vmax


def shape_surface(q, shape, f, nz=90, nphi=56):
    """裂变形状的 3D 表面：对称轴沿水平 X 方向，颜色编码局部的"膨胀/收缩"。

    颜色物理意义（"气球"方案，连续渐变，不突变）：
      以 s=ρ(z)/R0 度量局部绝对粗细（球形半径 R0 为尺度），固定映射：
        - 细（两极/尖端/颈部，ρ/R0→0）→ 深红；
        - 中等（球主体，≈0.55）→ 红；
        - 粗（碎片/球赤道，≳0.75）→ 浅红。
      最终：大的碎片浅红、小的碎片深红。

    断裂后（颈极小处 f>0.7）在颈部拆成两个独立碎片表面，视觉上呈现"两个核"。
    返回 (surfaces, state)，surfaces 是 [(X, Y, Z, facecolors), ...]（1 或 2 个）。
    """
    z, rho_z = shape.profile(q, n=nz)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi)
    R0 = shape.R0

    # 找内部颈部极小（排除两极端点半径→0）
    i_neck = None
    for i in range(1, nz - 1):
        if rho_z[i] <= rho_z[i - 1] and rho_z[i] <= rho_z[i + 1]:
            if i_neck is None or rho_z[i] < rho_z[i_neck]:
                i_neck = i
    rho_max = rho_z.max()
    split = (i_neck is not None) and (1.0 - rho_z[i_neck] / rho_max) > 0.7

    c_dark = np.array([0.40, 0.04, 0.06])   # 深红（细）
    c_base = np.array([0.75, 0.18, 0.14])   # 红（基准）
    c_lite = np.array([1.00, 0.78, 0.68])   # 浅红（粗）
    s_dark, s_base, s_lite = 0.30, 0.55, 0.75

    def _color(rho):
        s = rho / R0
        col = np.empty((len(rho), 4))
        for k in range(len(rho)):
            rr = s[k]
            if rr <= s_dark:
                col[k, :3] = c_dark
            elif rr <= s_base:
                t = (rr - s_dark) / (s_base - s_dark)
                col[k, :3] = c_dark + (c_base - c_dark) * t
            elif rr <= s_lite:
                t = (rr - s_base) / (s_lite - s_base)
                col[k, :3] = c_base + (c_lite - c_base) * t
            else:
                col[k, :3] = c_lite
            col[k, 3] = 1.0
        return np.tile(col, (nphi, 1, 1))

    def _surface(zz, rho):
        X = np.tile(zz, (nphi, 1))
        PHI = np.tile(phi[:, None], (1, len(zz)))
        RHO = np.tile(rho, (nphi, 1))
        Y = RHO * np.cos(PHI)
        Zs = RHO * np.sin(PHI)
        return X, Y, Zs, _color(rho)

    if split:
        surfaces = [_surface(z[:i_neck + 1], rho_z[:i_neck + 1]),
                    _surface(z[i_neck:], rho_z[i_neck:])]
    else:
        surfaces = [_surface(z, rho_z)]

    f = float(np.clip(f, 0.0, 1.0))
    if f < 0.05:
        state = "单核"
    elif f > 0.8:
        state = "已断裂·两个碎片"
    else:
        state = "颈部收缩中"

    return surfaces, state


def plot_shape_schematic(pes, filename="五参数形状示意图.png"):
    """5 个参数分别单独扫描，展示各自如何改变原子核形状"""
    shape = pes.shape
    base = np.array([1.2, 0.7, 0.0, 0.1, 0.1])

    params = [
        ("elong 拉长",       0, [0.6, 0.9, 1.2, 1.6]),
        ("neck 颈部",        1, [0.95, 0.7, 0.5, 0.3]),
        ("eta 质量不对称",   2, [-0.4, -0.15, 0.15, 0.4]),
        ("eps1 左碎片形变",  3, [-0.2, 0.0, 0.2, 0.35]),
        ("eps2 右碎片形变",  4, [-0.2, 0.0, 0.2, 0.35]),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    for ax, (name, idx, vals) in zip(axes, params):
        colors = plt.cm.viridis(np.linspace(0.25, 0.9, len(vals)))
        for v, c in zip(vals, colors):
            q = base.copy()
            q[idx] = v
            try:
                draw_shape_2d(ax, q, shape, color=c, alpha=0.55)
            except Exception:
                pass
        ax.set_title(f"{name}\n值 = {vals}", fontsize=9)
    fig.suptitle("五参数下原子核形状变化（其余参数固定在基准值；颜色浅→深 = 参数值小→大）",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(BASE_DIR, filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  形状示意图已保存: {out}")


def plot_pes_slice(pes, path, Q1, Q2, V, min_path=None,
                   filename="势能面切片.png",
                   title="五维势能面切片 (eta 松弛) 与裂变路径"):
    """2D 势能面等高线 + 裂变路径（投影到 elong-neck 平面），
    可选叠加确定性极小能量路径（最优裂变路径）。"""
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    Vp, vmin, vmax = _prepare_pes(V)
    levels = np.linspace(vmin, vmax, 24)
    cf = ax.contourf(Q1, Q2, Vp, levels=levels, cmap="viridis", extend="both")
    # 稀疏等值线 + 数值标注，读出势垒/谷底量级
    cs = ax.contour(Q1, Q2, Vp, levels=levels[::4], colors="0.35",
                    linewidths=0.4, alpha=0.55)
    ax.clabel(cs, fmt="%.0f", fontsize=6.5, inline=True)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("形变能 ΔV (MeV)", fontsize=11)
    # 裂变路径（白描边 + 红主体，醒目）
    ax.plot(path[:, 0], path[:, 1], color="white", lw=4.0, alpha=0.9, zorder=3)
    ax.plot(path[:, 0], path[:, 1], color="crimson", lw=1.8, alpha=0.95,
            zorder=4, label="非对称裂变路径")
    ax.plot(path[0, 0], path[0, 1], "o", color="white", ms=11, mec="k", mew=1.3,
            zorder=5, label="起点（近球）")
    ax.plot(path[-1, 0], path[-1, 1], "*", color="gold", ms=20, mec="k", mew=0.9,
            zorder=5, label="终点（分离）")
    if min_path is not None:
        ax.plot(min_path[:, 0], min_path[:, 1], "w--", lw=2.0, alpha=0.85,
                label="极小能量路径")
    ax.set_xlabel("elong 拉长", fontsize=12)
    ax.set_ylabel("neck 颈部", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()
    out = os.path.join(BASE_DIR, filename)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  势能面切片已保存: {out}")


def animate_split(path, energies, components, pes, Q1, Q2, V,
                  filename="核形状随机行走.gif", every=1, fps=5,
                  smooth=False, overlay_path=None, overlay_energies=None,
                  vmin=-25.0, vmax=40.0):
    """分屏动画：左上=3D 核形状(水平)，左下=5 参数读数，右=3D 势能面+轨迹。

    主轨迹沿裂变路径（球→拉长→颈缩→非对称→完全分离成两个碎片）。
    演示路径本身已平滑，故默认不做时间平滑；能量极小路径可 smooth=True。
    overlay_path（随机行走轨迹）+ overlay_energies 作为方法轨迹淡色叠加。
    """
    shape = pes.shape
    R0 = pes.R0

    # 时间平滑（默认关闭；窗口 5）
    if smooth:
        win = 5
        path = _smooth_path(path, win)
        energies = _smooth_1d(energies, win)
        components = _smooth_path(components, win)

    frames = list(range(0, len(path), every))

    # 预计算坐标范围（覆盖全程所有形状）
    z_lim = max(np.abs(shape.profile(p)[0]).max() for p in path) * 1.1
    r_lim = R0 * 1.4

    # 断裂程度 f：基于平滑后的路径
    f_smooth = np.array([neck_fraction(p, shape) for p in path])

    # 势能面裁剪到物理量程 + 平滑（消除 NaN 悬崖 / 深谷 / 晶面）
    Vp, vmin, vmax = _prepare_pes(V, vmin=vmin, vmax=vmax)

    # 轨迹投影到势能面上（z = 面在该 (elong,neck) 处的能量）+ 略抬升 2 MeV，
    # 使轨迹始终贴着面走、清晰可见，既不悬空也不被面遮挡。
    elong_axis = np.asarray(Q1[0, :], dtype=float)
    neck_axis = np.asarray(Q2[:, 0], dtype=float)
    interp = RegularGridInterpolator((neck_axis, elong_axis), Vp,
                                     bounds_error=False, fill_value=vmin)
    E_traj = interp(np.column_stack([path[:, 1], path[:, 0]])) + 2.0
    E_traj = np.clip(E_traj, vmin, vmax)
    if overlay_energies is not None:
        overlay_energies = np.clip(overlay_energies, vmin, vmax)
    zmin, zmax = vmin, vmax

    fig = plt.figure(figsize=(19, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.9], height_ratios=[6.5, 0.55],
                          left=0.02, right=0.98, top=0.96, bottom=0.085,
                          wspace=0.08, hspace=0.30)
    ax_shape = fig.add_subplot(gs[0, 0], projection="3d")
    ax_surf = fig.add_subplot(gs[0, 1], projection="3d")
    ax_info = fig.add_subplot(gs[1, :])

    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=zmin, vmax=zmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_surf, shrink=0.55, pad=0.08, aspect=30)
    cbar.set_label("形变能 ΔV (MeV)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    legend_handles = [
        Line2D([0], [0], color="black", lw=3, label="行走轨迹"),
        Line2D([0], [0], marker="o", color="lime", markersize=9, linestyle="None",
               markeredgecolor="black", markeredgewidth=1.0, label="起点"),
        Line2D([0], [0], marker="s", color="red", markersize=9, linestyle="None",
               markeredgecolor="black", markeredgewidth=1.0, label="终点"),
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
        dV_s, dV_c, dE_sh, dE_pair = components[i]

        # ---- 左上：3D 核形状（水平放置；断裂后拆成两个独立碎片）----
        ax_shape.clear()
        surfaces, state = shape_surface(q, shape, f_smooth[i])
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

        # ---- 底部：5 参数读数 ----
        ax_info.clear()
        ax_info.axis("off")
        info = (f"elong={q[0]:+.3f}   neck={q[1]:+.3f}   η={q[2]:+.3f}   "
                f"ε1={q[3]:+.3f}   ε2={q[4]:+.3f}\n"
                f"ΔV={Vv:+.2f} MeV  =  表面 {dV_s:+.2f}  +  库仑 {dV_c:+.2f}  "
                f"+  壳修正 {dE_sh:+.2f}  +  对修正 {dE_pair:+.2f}")
        ax_info.text(0.5, 0.5, info, transform=ax_info.transAxes, fontsize=13,
                     va="center", ha="center", linespacing=1.5)

        # ---- 右：3D 势能面 + 轨迹 ----
        ax_surf.clear()
        ax_surf.plot_surface(Q1, Q2, Vp, cmap="viridis", alpha=0.55,
                             linewidth=0, antialiased=True)
        traj = E_traj[:i + 1]
        ax_surf.plot(path[:i + 1, 0], path[:i + 1, 1], traj,
                     color="white", lw=7.0, alpha=0.95)
        ax_surf.plot(path[:i + 1, 0], path[:i + 1, 1], traj,
                     color="black", lw=3.0)
        if overlay_path is not None:
            ax_surf.plot(overlay_path[:, 0], overlay_path[:, 1],
                         overlay_energies, color="yellow", lw=1.0, alpha=0.45)
        ax_surf.scatter([q[0]], [q[1]], [E_traj[i]], color="black", s=90,
                        edgecolor="white", linewidth=1.4, depthshade=False)
        ax_surf.scatter([path[0, 0]], [path[0, 1]], [E_traj[0]],
                        color="lime", marker="o", s=90, edgecolor="black",
                        linewidth=1.0, depthshade=False)
        ax_surf.scatter([path[-1, 0]], [path[-1, 1]], [E_traj[-1]],
                        color="red", marker="s", s=70, edgecolor="black",
                        linewidth=1.0, depthshade=False)
        ax_surf.dist = 8
        ax_surf.set_xlabel("elong 拉长", fontsize=9)
        ax_surf.set_ylabel("neck 颈部", fontsize=9)
        ax_surf.set_zlabel("ΔV (MeV)", fontsize=9)
        ax_surf.set_xlim(float(Q1.min()), float(Q1.max()))
        ax_surf.set_ylim(float(Q2.min()), float(Q2.max()))
        ax_surf.set_zlim(zmin, zmax)
        ax_surf.view_init(elev=25, azim=-60)
        ax_surf.set_title("三维势能面 V(elong, neck) 与行走轨迹", fontsize=12)
        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    out = os.path.join(BASE_DIR, filename)
    try:
        ani.save(out, writer="pillow", fps=fps, dpi=140)
        print(f"  动画已保存: {out} ({len(frames)} 帧, {fps} fps)")
    except Exception as e:
        print(f"  (GIF 保存失败: {e})")
    plt.close(fig)
