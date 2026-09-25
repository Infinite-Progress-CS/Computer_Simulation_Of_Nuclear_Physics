# -*- coding: utf-8 -*-
"""Brownian random walk on an interpolated 5D PES table."""

import argparse
import glob
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pes_table import PESTable
from random_walk_yield import brownian_yield, quadrupole_moment, find_ground_state
from shape import Shape3QS


Z, N = 92, 144
R0_LD = 1.16 * 236 ** (1.0 / 3.0)


def mass_yield_peaks(A, Y):
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
    return A_light, Y_light, A_heavy, Y_heavy, A_val, Y_val, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=None)
    ap.add_argument("--n-walks", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--E-star", type=float, default=6.54)
    ap.add_argument("--V0", type=float, default=15.0)
    ap.add_argument("--c0", type=float, default=2.5)
    ap.add_argument("--n-eta-sub", type=int, default=0,
                    help="每形状步的 η-only Metropolis 子步数（时间尺度分离，加深谷）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--q0", default=None)
    ap.add_argument("--geom", default=None,
                    help="几何表路径（Q/颈半径插值，默认自动找 geometry_table_U236.npz）")
    ap.add_argument("--record-trajectory", action="store_true",
                    help="保存每次行走的逐步轨迹，用于多轨迹采样/拟合")
    ap.add_argument("--record-max-walks", type=int, default=200,
                    help="最多记录前多少条轨迹")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    table_path = args.table
    if table_path is None:
        # 生产表固定为 nmax8_fine（排除 smoke/BUGGY/q0grid 等诊断/对照表）
        preferred = os.path.join(PROJECT_ROOT, "results", "pes_table",
                                 "pes_table_U236_nmax8_fine.npz")
        if os.path.exists(preferred):
            table_path = preferred
        else:
            candidates = sorted(glob.glob(
                os.path.join(PROJECT_ROOT, "results", "pes_table",
                             "pes_table_U236_*.npz")))
            if not candidates:
                sys.exit("No PES table found. Run scripts/build_pes_table.py first.")
            table_path = candidates[-1]

    table = PESTable(pes=None, table_path=table_path)
    shape = Shape3QS(R0_LD)

    # shape.build 非对称分支已改为闭式解（二次方程），无需 least_squares / 几何表
    geom = None

    if args.q0 is not None:
        q0 = np.array([float(x) for x in args.q0.split(",")])
        V_gs = table.energy(q0)
    else:
        # 合法紧凑起点：elong 0.5 在 neck=0.8 处 D<Dmin（对称无解），须用 neck≈1
        # （rho_v≈a 的「肚」形状，C<0 仍可解）才能起步。
        q0 = np.array([0.5, 0.99, 0.0, 0.3, 0.3])
        V_gs = table.energy(q0)
    Q0 = quadrupole_moment(q0, shape)

    print(f"PES table : {table_path}")
    print(f"start     : {q0}  E={V_gs:+.3f}  Q0={Q0:.1f} fm^5")
    print(f"walk      : n={args.n_walks} max_steps={args.max_steps} "
          f"V0={args.V0} c0={args.c0} E*={args.E_star} n_eta_sub={args.n_eta_sub}")

    res = brownian_yield(
        table, shape,
        A_parent=236, Z_parent=92,
        E_star=args.E_star, V0=args.V0, c0=args.c0,
        n_walks=args.n_walks, max_steps=args.max_steps,
        n_eta_sub=args.n_eta_sub,
        q0=q0, seed=args.seed, verbose=True,
        record_trajectory=args.record_trajectory,
        record_max_walks=args.record_max_walks,
        geom=geom,
    )

    eta = res["eta_scission"]
    pk = mass_yield_peaks(res["A"], res["Y_A"])
    print("\nResults")
    print(f"  scission : {res['n_scission']}/{args.n_walks}")
    if len(eta):
        print(f"  eta      : mean={eta.mean():+.3f} sigma={eta.std():.3f} "
              f"asym={(np.abs(eta) > 0.03).mean():.0%}")
    print(f"  mass     : light A={pk[0]} heavy A={pk[2]} valley A={pk[4]} "
          f"P/V={pk[6]:.0f}")

    outdir = os.path.join(PROJECT_ROOT, "results", "随机行走")
    os.makedirs(outdir, exist_ok=True)
    table_tag = os.path.splitext(os.path.basename(table_path))[0]
    tag = f"{table_tag}_nw{args.n_walks}_ms{args.max_steps}_s{args.seed}"
    save_dict = dict(A=res["A"], Y_A=res["Y_A"], eta_scission=eta,
                     n_scission=res["n_scission"], acceptance=res["acceptance"],
                     V_gs=V_gs, q0=q0, Q0=Q0)
    if args.record_trajectory:
        save_dict.update(
            trajectories=res["trajectories"],
            trajectory_energy=res["trajectory_energy"],
            trajectory_steps=res["trajectory_steps"],
            trajectory_scission=res["trajectory_scission"])
    np.savez(os.path.join(outdir, f"walk_yield_{tag}.npz"), **save_dict)
    print(f"Saved results/随机行走/walk_yield_{tag}.npz")


if __name__ == "__main__":
    main()
