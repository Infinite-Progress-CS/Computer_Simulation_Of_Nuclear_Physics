# -*- coding: utf-8 -*-
"""plot_walk_trajectory_fit.py — 多次随机行走轨迹的采样与光滑拟合。

输入：由 run_walk_yield_table.py --record-trajectory 保存的 npz。
输出：
  U236_随机行走轨迹_能量拟合.png
  U236_随机行走轨迹_五参数拟合.png

方法：
  1. 取所有成功到达断裂的行走轨迹；
  2. 按 elong 分箱，统计每个箱内 E 以及 neck/eta/eps1/eps2 的中位数与四分位带；
  3. 对中位数再做样条光滑，得到一条代表性裂变路径。
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

RES = os.path.join(PROJECT_ROOT, "results", "随机行走")
FIG = os.path.join(RES, "figures")

PARAM_NAMES = ["neck", "eta", "eps1", "eps2"]
PARAM_LABELS = ["neck 颈", "η 质量不对称", "ε1 轻碎片形变", "ε2 重碎片形变"]


def load_latest(npz_path=None):
    if npz_path is not None:
        return np.load(npz_path)
    cand = sorted(glob.glob(os.path.join(RES, "walk_yield_*.npz")),
                  key=os.path.getmtime)
    if not cand:
        raise FileNotFoundError("找不到 walk_yield_*.npz")
    return np.load(cand[-1])


def valid_trajectory_mask(z):
    """返回成功断裂轨迹的布尔掩码；若成功轨迹太少则退化为所有已记录轨迹。"""
    if "trajectory_scission" not in z.files:
        raise ValueError("npz 没有记录逐步轨迹，请先加 --record-trajectory 重跑行走")
    sc = z["trajectory_scission"].astype(bool)
    if sc.sum() >= 5:
        return sc
    return np.ones(len(sc), dtype=bool)


def bin_curve(x, y, lo, hi, nbins=48, min_count=3):
    """按 x 分箱，返回每个箱中心、中位数、25/75 分位和样本数。"""
    bins = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    med = np.full(nbins, np.nan)
    q25 = np.full(nbins, np.nan)
    q75 = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)
    for i in range(nbins):
        m = (x >= bins[i]) & (x < bins[i + 1])
        counts[i] = int(m.sum())
        if counts[i] >= min_count:
            med[i] = float(np.median(y[m]))
            q25[i] = float(np.percentile(y[m], 25))
            q75[i] = float(np.percentile(y[m], 75))
    return centers, med, q25, q75, counts


def smooth_curve(x, y, s=0.01):
    """对分箱中位数做样条光滑，返回细网格上的 (x_fine, y_fine)。"""
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 4:
        return x, y
    order = min(3, len(x) - 1)
    spl = UnivariateSpline(x, y, k=order, s=s)
    xf = np.linspace(x[0], x[-1], 500)
    return xf, spl(xf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="含逐步轨迹的 walk_yield_*.npz")
    ap.add_argument("--elo", type=float, default=0.4)
    ap.add_argument("--ehi", type=float, default=3.2)
    ap.add_argument("--nbins", type=int, default=48)
    args = ap.parse_args()

    z = load_latest(args.npz)
    print("输入文件：", z.fid.name if hasattr(z, "fid") else "(npz)")

    traj = z["trajectories"]
    E = z["trajectory_energy"]
    steps = z["trajectory_steps"]
    mask = valid_trajectory_mask(z)

    elong_all = []
    E_all = []
    curves = {}
    for w in np.where(mask)[0]:
        n = int(steps[w])
        if n <= 0:
            continue
        elong = traj[w, :n + 1, 0]
        elong_all.append(elong)
        E_all.append(E[w, :n + 1])
        for k in range(4):
            curves.setdefault(k, []).append(traj[w, :n + 1, k + 1])

    if not elong_all:
        raise RuntimeError("没有可用于拟合的轨迹")

    elong = np.concatenate(elong_all)
    E_val = np.concatenate(E_all)

    # ---------- 图 1：能量随 elong 的轨迹云 + 中位数带 + 光滑代表曲线 ----------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(elong, E_val, s=1, color="0.72", alpha=0.18, lw=0,
               label="逐轨迹采样点")
    c, med, q25, q75, cnt = bin_curve(elong, E_val, args.elo, args.ehi, args.nbins)
    valid = np.isfinite(med)
    ax.fill_between(c[valid], q25[valid], q75[valid],
                    color="tab:red", alpha=0.18, lw=0, label="25%–75% 带")
    ax.plot(c[valid], med[valid], "o", ms=3, color="tab:blue",
            label="分箱中位数")
    xf, yf = smooth_curve(c[valid], med[valid])
    ax.plot(xf, yf, "-", color="crimson", lw=2.2, label="光滑拟合轨迹")
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("elong 拉长", fontsize=12)
    ax.set_ylabel("宏观-微观形变能 E (MeV)", fontsize=12)
    ax.set_title("U-236 五维势能面上多次随机行走轨迹拟合（E-elong）", fontsize=12)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out1 = os.path.join(FIG, "U236_随机行走轨迹_能量拟合.png")
    fig.savefig(out1, dpi=160)
    plt.close(fig)
    print(f"已保存 {out1}")

    # ---------- 图 2：neck / eta / eps1 / eps2 随 elong 的轨迹带 ----------
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    for k, ax in enumerate(axes):
        y = np.concatenate(curves[k])
        c, med, q25, q75, cnt = bin_curve(elong, y, args.elo, args.ehi, args.nbins)
        valid = np.isfinite(med)
        ax.scatter(elong, y, s=1, color="0.75", alpha=0.12, lw=0)
        ax.fill_between(c[valid], q25[valid], q75[valid],
                        color="tab:blue", alpha=0.15, lw=0)
        ax.plot(c[valid], med[valid], "o", ms=2.5, color="tab:blue")
        xf, yf = smooth_curve(c[valid], med[valid])
        ax.plot(xf, yf, "-", color="crimson", lw=2.0)
        ax.axhline(0, color="0.6", lw=0.5, ls=":")
        ax.set_ylabel(PARAM_LABELS[k], fontsize=11)
    axes[-1].set_xlabel("elong 拉长", fontsize=12)
    fig.suptitle("U-236 随机行走五参数演化（中位数与四分位带）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out2 = os.path.join(FIG, "U236_随机行走轨迹_五参数拟合.png")
    fig.savefig(out2, dpi=160)
    plt.close(fig)
    print(f"已保存 {out2}")

    n_used = int(mask.sum())
    print(f"拟合使用轨迹：{n_used}/{len(mask)}，成功断裂：{int(z['trajectory_scission'].astype(bool).sum())}/{len(mask)}")


if __name__ == "__main__":
    main()
