# -*- coding: utf-8 -*-
"""
run_walk_yield_hybrid.py — 5D Brownian 形状运动产额（断裂区双中心壳修正）
====================================================================
用 HybridMacroMicro PES + Randrup-Möller 偏置势（V_bias=V0(Q0/Q)²）做 5D
随机行走产额。偏置势沿四极矩方向把核从基态推向断裂，突破局部 Metropolis 行走
卡在连通对称形状（假 +17 MeV 内势垒）的鸡生蛋问题。

断裂点冻结 η → 质量产额 Y(A)；电荷产额 Y(Z) 用 UCD 标度。

输出：results/随机行走/walk_yield_hybrid.npz + csv。
"""
import os
import sys
import time
import argparse

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from hybrid_pes import CachedHybridMacroMicro
from random_walk_yield import brownian_yield, quadrupole_moment, find_ground_state

Z, N = 92, 144   # U-236


def mass_yield_peaks(A, Y):
    """质量产额双峰/谷。A 为质量数数组，Y 为产额（Σ=200%）。"""
    light = A < 120
    heavy = A > 120
    A_light = int(A[light][np.argmax(Y[light])])
    A_heavy = int(A[heavy][np.argmax(Y[heavy])])
    Y_light = float(Y[light].max())
    Y_heavy = float(Y[heavy].max())
    valley = (A >= 105) & (A <= 135)
    A_val = int(A[valley][np.argmin(Y[valley])])
    Y_val = float(Y[A == A_val][0])
    ratio = (Y_light + Y_heavy) / (2.0 * Y_val) if Y_val > 0 else np.inf
    return dict(A_light=A_light, Y_light=Y_light, A_heavy=A_heavy, Y_heavy=Y_heavy,
                A_val=A_val, Y_val=Y_val, ratio=ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-walks', type=int, default=40)
    ap.add_argument('--max-steps', type=int, default=400)
    ap.add_argument('--step-elong', type=float, default=0.05)
    ap.add_argument('--step-neck', type=float, default=0.04)
    ap.add_argument('--step-eta', type=float, default=0.03)
    ap.add_argument('--step-eps', type=float, default=0.03)
    ap.add_argument('--E-star', type=float, default=6.54)
    ap.add_argument('--V0', type=float, default=60.0)
    ap.add_argument('--c0', type=float, default=2.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--q0', type=str, default=None,
                    help='逗号分隔的起点 q=[elong,neck,eta,eps1,eps2]，默认自动找基态')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    step = np.array([args.step_elong, args.step_neck, args.step_eta,
                     args.step_eps, args.step_eps])

    print("=" * 70)
    print("5D Brownian 形状运动产额（断裂区双中心壳修正 + RMS 偏置势）")
    print("=" * 70)

    pes = CachedHybridMacroMicro(Z, N, nz=40, nrho=40, nsurf=48, nphi=32,
                                 Nmax=12, neck_hi=0.5, neck_lo=0.15)
    print(f"  R0={pes.R0:.3f} fm   γ={pes.gamma:.2f} MeV   "
          f"切换窗口 neck∈[{pes.neck_lo},{pes.neck_hi}]")
    shape = pes.shape

    # 起点 + 基态
    if args.q0 is not None:
        q0 = np.array([float(x) for x in args.q0.split(',')])
        V_gs = pes.energy(q0)
        print(f"  起点(给定) : {q0}  E={V_gs:+.2f}")
    else:
        t0 = time.time()
        q0, V_gs = find_ground_state(pes)
        print(f"  基态(粗扫) : {q0}  E={V_gs:+.2f}  ({time.time()-t0:.0f}s)")
    Q0 = quadrupole_moment(q0, shape)
    print(f"  基态四极矩 Q0 = {Q0:.2f} fm^5  偏置势 V0={args.V0} MeV")

    print(f"\n[1] Brownian 行走产额 (E*={args.E_star} MeV, V0={args.V0}, "
          f"c0={args.c0} fm, {args.n_walks} walks × {args.max_steps} steps)...")
    t0 = time.time()
    res = brownian_yield(
        pes, shape, A_parent=236, Z_parent=92, E_star=args.E_star,
        V0=args.V0, c0=args.c0, n_walks=args.n_walks, max_steps=args.max_steps,
        q0=q0, step=step, seed=args.seed, verbose=True)
    dt = time.time() - t0

    eta = res["eta_scission"]
    n_sc = res["n_scission"]
    print("\n" + "=" * 70)
    print("结果摘要")
    print("=" * 70)
    print(f"  成功断裂 {n_sc}/{args.n_walks}  接受率 {res['acceptance']:.2%}  "
          f"耗时 {dt:.0f}s")
    if n_sc > 0:
        asym = float((np.abs(eta) > 0.03).mean())
        print(f"  η_scission: 均值={eta.mean():+.3f}  σ={eta.std():.3f}  "
              f"非对称占比(|η|>0.03)={asym:.0%}")
        print(f"  η 直方图 (10 bins):")
        h, edges = np.histogram(eta, bins=10, range=(-0.5, 0.5))
        for i in range(10):
            bar = '#' * int(40 * h[i] / max(h.max(), 1))
            print(f"    η∈[{edges[i]:+.1f},{edges[i+1]:+.1f}) {h[i]:3d} {bar}")

    pk = mass_yield_peaks(res["A"], res["Y_A"])
    print(f"  质量产额 Y(A): 轻峰 A={pk['A_light']} ({pk['Y_light']:.2f}%)  "
          f"重峰 A={pk['A_heavy']} ({pk['Y_heavy']:.2f}%)  "
          f"谷 A={pk['A_val']} ({pk['Y_val']:.4f}%)  峰谷比={pk['ratio']:.0f}")

    out = os.path.join(PROJECT_ROOT, "results", "随机行走")
    os.makedirs(out, exist_ok=True)
    tag = f"nw{args.n_walks}_ms{args.max_steps}_s{args.seed}"
    np.savez(os.path.join(out, f"walk_yield_hybrid_{tag}.npz"),
             A=res["A"], Y_A=res["Y_A"], eta_scission=eta,
             n_scission=n_sc, acceptance=res["acceptance"],
             V_gs=V_gs, q0=q0, Q0=Q0)
    with open(os.path.join(out, f"walk_yield_hybrid_{tag}.csv"), "w",
              encoding="utf-8-sig") as f:
        f.write("A,mass_yield_percent\n")
        for a, y in zip(res["A"], res["Y_A"]):
            if y > 0:
                f.write(f"{int(a)},{y:.6f}\n")
    print(f"\n已保存 results/随机行走/walk_yield_hybrid_{tag}.npz / .csv")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        A, Y = res["A"], res["Y_A"]
        axes[0].plot(A, Y, '-', lw=2, color='tab:blue')
        axes[0].plot([pk['A_light'], pk['A_heavy']],
                     [pk['Y_light'], pk['Y_heavy']], 'v', ms=9, color='tab:red')
        axes[0].set_ylabel('Y(A) (%)')
        axes[0].set_title('Brownian 行走质量产额（双中心壳修正）')
        axes[0].grid(alpha=0.3)
        axes[1].semilogy(A, np.clip(Y, 1e-6, None), '-', lw=2, color='tab:blue')
        axes[1].set_xlabel('A (mass number)')
        axes[1].set_ylabel('Y(A) (%) (log)')
        axes[1].grid(alpha=0.3, which='both')
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"walk_yield_hybrid_{tag}.png"), dpi=160)
        plt.close(fig)
        print(f"图已保存 results/随机行走/walk_yield_hybrid_{tag}.png")


if __name__ == "__main__":
    main()
