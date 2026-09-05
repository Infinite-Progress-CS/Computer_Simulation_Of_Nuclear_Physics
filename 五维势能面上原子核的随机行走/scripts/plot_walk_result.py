# -*- coding: utf-8 -*-
"""
plot_walk_result.py — 5D Metropolis 随机行走结果可视化
========================================================
对比随机行走轨迹与确定性极小能量路径：
  - 图1: E(elong) 剖面 —— 行走（散点）+ 行走能量包络 + 极小能量路径
  - 图2: 行走过程中 neck / η / ε1 / ε2 随 elong 的演化
直接读 results/随机行走/walk_5d.npz 与 path_5d.npz，不重算势能面。
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RES = os.path.join(PROJECT_ROOT, "results", "随机行走")
FIG = os.path.join(RES, "figures")


def load(name):
    d = np.load(os.path.join(RES, name))
    return d["path"], d["energies"]


def min_envelope(elong, energies, lo=0.8, hi=3.0, nb=44):
    """把行走轨迹按 elong 分箱，取每箱最低能量，作为行走探索到的下包络。"""
    bins = np.linspace(lo, hi, nb)
    env = np.full(nb - 1, np.nan)
    for b in range(nb - 1):
        m = (elong >= bins[b]) & (elong < bins[b + 1])
        if m.any():
            env[b] = energies[m].min()
    return 0.5 * (bins[:-1] + bins[1:]), env


def main():
    os.makedirs(FIG, exist_ok=True)
    wp, we = load("walk_5d.npz")
    mp, me = load("path_5d.npz")

    # ---- 图1: E(elong) 剖面 ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(wp[:, 0], we, "-", color="0.80", lw=0.7, alpha=0.7,
            label="随机行走（逐步能量）")
    elc, env = min_envelope(wp[:, 0], we)
    valid = ~np.isnan(env)
    ax.plot(elc[valid], env[valid], "-o", color="crimson", lw=1.6, ms=3.5,
            label="随机行走能量包络（每 elong 最低）")
    mvalid = ~np.isnan(me)
    ax.plot(mp[mvalid, 0], me[mvalid], "--", color="royalblue", lw=2.2,
            label="确定性极小能量路径")
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("elong 拉长", fontsize=12)
    ax.set_ylabel("宏观-微观形变能 E (MeV)", fontsize=12)
    ax.set_title("5D Metropolis 随机行走 vs 确定性极小能量路径 (U-236)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out1 = os.path.join(FIG, "walk_vs_path_energy.png")
    fig.savefig(out1, dpi=160)
    plt.close(fig)
    print(f"已保存 {out1}")

    # ---- 图2: neck/η/ε1/ε2 演化 ----
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    labels = ["neck 颈", "η 质量不对称", "ε1 轻碎片形变", "ε2 重碎片形变"]
    for k, ax in enumerate(axes):
        ax.plot(wp[:, 0], wp[:, k + 1], "-", color="0.55", lw=0.7, alpha=0.8)
        ax.axhline(0, color="0.7", lw=0.5, ls=":")
        ax.set_ylabel(labels[k], fontsize=11)
    axes[-1].set_xlabel("elong 拉长", fontsize=12)
    fig.suptitle("随机行走 5 参数演化（颈在 elong~2.3 塌缩断裂）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out2 = os.path.join(FIG, "walk_5d_evolution.png")
    fig.savefig(out2, dpi=160)
    plt.close(fig)
    print(f"已保存 {out2}")

    # ---- 数值摘要 ----
    i_bar = int(np.argmax(we))
    i_end = int(np.argmin(we[i_bar:])) + i_bar
    print("\n行走数值摘要")
    print(f"  起点      E={we[0]:+.2f} @ elong={wp[0,0]:.2f}")
    print(f"  能量峰值  E={we[i_bar]:+.2f} @ elong={wp[i_bar,0]:.2f}, "
          f"neck={wp[i_bar,1]:.2f}, η={wp[i_bar,2]:+.2f}")
    print(f"  峰值后最低 E={we[i_end]:+.2f} @ elong={wp[i_end,0]:.2f}, "
          f"neck={wp[i_end,1]:.2f}, η={wp[i_end,2]:+.2f}, "
          f"ε1={wp[i_end,3]:+.2f}, ε2={wp[i_end,4]:+.2f}")
    print(f"  末帧      elong={wp[-1,0]:.2f}, neck={wp[-1,1]:.2f}, "
          f"η={wp[-1,2]:+.2f}, ε1={wp[-1,3]:+.2f}, ε2={wp[-1,4]:+.2f}, "
          f"E={we[-1]:+.2f}")


if __name__ == "__main__":
    main()
