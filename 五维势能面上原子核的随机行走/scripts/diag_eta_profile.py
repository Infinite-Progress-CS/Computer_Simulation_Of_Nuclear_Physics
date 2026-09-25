# -*- coding: utf-8 -*-
"""diag_eta_profile.py — 断裂点 PES 的 η 剖面诊断。

在断裂构型 (elong≈2.0, neck≈0.5, eps1/eps2 固定) 上扫 η，输出：
  V(η) 总能量、δE_壳(η)、exp(−V/T) 期望产额分布 → 判断谷深是 PES 结构还是行走未弛豫。

用法:
  python scripts/diag_eta_profile.py --elong 2.0 --neck 0.5 --eps 0.4,0.6
"""
import os, sys, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hybrid_pes import CachedHybridMacroMicro
from shape import ShapeFunnyHills

Z, N = 92, 144
A = 236
T = np.sqrt(6.54 / (A / 8.0))   # ≈0.47 MeV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--elong', type=float, default=2.0)
    ap.add_argument('--neck', type=float, default=0.5)
    ap.add_argument('--eps', type=str, default='0.4,0.6')
    ap.add_argument('--Nmax', type=int, default=12)
    ap.add_argument('--lam-so-p', type=float, default=28.0)
    ap.add_argument('--neck-hi', type=float, default=0.7)
    ap.add_argument('--neck-lo', type=float, default=0.6)
    ap.add_argument('--n-eta', type=int, default=61)
    args = ap.parse_args()

    eps = tuple(float(x) for x in args.eps.split(','))
    pes = CachedHybridMacroMicro(Z, N, nz=40, nrho=40, nsurf=48, nphi=32,
                                 Nmax=args.Nmax, neck_hi=args.neck_hi,
                                 neck_lo=args.neck_lo,
                                 shape_cls=ShapeFunnyHills,
                                 lam_so_p=args.lam_so_p, eps_scission=eps)

    etas = np.linspace(-0.45, 0.45, args.n_eta)
    V = np.empty_like(etas)
    dsh = np.empty_like(etas)
    for i, eta in enumerate(etas):
        q = [args.elong, args.neck, eta, eps[0], eps[1]]
        V[i] = pes.energy(q)
        # 壳修正：断裂区碎片级
        sh, _ = pes._fragment_quantum(np.asarray(q, dtype=float))
        dsh[i] = sh

    # 期望产额（scission-point 口径，仅作参考）
    Vrel = V - V.min()
    Y = np.exp(-Vrel / T)

    print(f"断裂构型: elong={args.elong} neck={args.neck} eps={eps}")
    print(f"  T = {T:.3f} MeV")
    print(f"  V(η) 范围: {V.min():.3f} ~ {V.max():.3f} MeV  (谷深 {V.max()-V.min():.2f} MeV)")
    # 对称 vs 非对称
    i0 = np.argmin(np.abs(etas))
    im = np.argmin(V)
    print(f"  V(η=0) = {V[i0]:.3f}   V(η={etas[im]:+.3f}) = {V[im]:.3f}")
    print(f"  非对称比对称低 {V[i0]-V[im]:.3f} MeV")
    # 期望峰谷比
    y0 = Y[i0]
    ym = Y[im]
    print(f"  期望 P/V (scission-point exp(-V/T)) = {ym/y0:.0f}")
    # 期望 σ
    w = Y / Y.sum()
    mu = (etas * w).sum()
    sig = np.sqrt((w * (etas - mu) ** 2).sum())
    print(f"  期望 η 分布 σ = {sig:.3f} (行走实测 σ≈0.154)")

    print("\n  η        V(η)       δE_壳(η)   exp(-ΔV/T)")
    for i in range(args.n_eta):
        if i % 5 == 0 or abs(etas[i]) < 0.03:
            print(f"  {etas[i]:+.2f}   {V[i]:8.3f}   {dsh[i]:8.3f}   {Y[i]/ym:6.3f}")


if __name__ == '__main__':
    main()
