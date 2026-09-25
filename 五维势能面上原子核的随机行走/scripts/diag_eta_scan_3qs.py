# -*- coding: utf-8 -*-
"""diag_eta_scan_3qs.py — 3QS + 双中心基的 η 不对称偏好诊断。

在鞍点→断裂区的 (elong, neck) 网格上扫 η，分解 E_ld / dE_sh / dE_pair，
量不对称偏好 Δ = E(η=0) − E(η=0.19)。Δ>0 表示非对称更稳（谷深的方向）。

用法:
  python scripts/diag_eta_scan_3qs.py --eps 0.3,0.1
"""
import os, sys, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hybrid_pes import CachedHybridMacroMicro
from shape import Shape3QS

Z, N = 92, 144
ETA_LIST = [0.0, 0.05, 0.10, 0.15, 0.19, 0.22, 0.25]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=str, default='0.3,0.1',
                    help='碎片形变 eps1,eps2（轻,重）')
    ap.add_argument('--lam-so-p', type=float, default=None)
    ap.add_argument('--Nmax', type=int, default=12)
    args = ap.parse_args()

    eps1, eps2 = (float(x) for x in args.eps.split(','))
    pes = CachedHybridMacroMicro(Z, N, nz=40, nrho=40, nsurf=48, nphi=32,
                                 Nmax=args.Nmax, shape_cls=Shape3QS,
                                 lam_so_p=args.lam_so_p)
    print(f'3QS + 双中心 WS 基 (Nmax={args.Nmax}, lam_so_p={args.lam_so_p}), '
          f'eps=({eps1},{eps2})', flush=True)

    grid = [(1.6, 0.70), (1.8, 0.60), (2.0, 0.55), (2.2, 0.50), (2.4, 0.45)]
    hdr = (f'{"elong":>5} {"neck":>5} | ' +
           ' '.join(f'{e:+.2f}' for e in ETA_LIST) + ' | Δ(0→.19)')
    print(hdr, flush=True)
    for (el, nk) in grid:
        Vs, LDs, SHs = [], [], []
        ok = True
        for eta in ETA_LIST:
            q = [el, nk, eta, eps1, eps2]
            try:
                V, dVs, dVc, dEsh, dEpair = pes.energy_components(q)
                Vs.append(V)
                LDs.append(dVs + dVc)
                SHs.append(dEsh)
            except Exception as e:
                Vs.append(np.nan)
                LDs.append(np.nan)
                SHs.append(np.nan)
                ok = False
        v_str = ' '.join(f'{v:+.1f}' if not np.isnan(v) else ' ERR ' for v in Vs)
        d0 = Vs[0]
        d19 = Vs[ETA_LIST.index(0.19)]
        gap = d0 - d19 if (ok and not np.isnan(d0) and not np.isnan(d19)) else np.nan
        print(f'{el:5.1f} {nk:5.2f} | {v_str} | {gap:+.2f}', flush=True)

    # 分解表：在断裂点 (elong=2.2, neck=0.5) 细分 η
    print(f'\n分解 @ elong=2.2 neck=0.50 eps=({eps1},{eps2}):', flush=True)
    print(f'{"eta":>5} {"E_ld":>8} {"dE_sh":>8} {"dE_pair":>8} {"E_mm":>8} '
          f'{"ΔE_mm":>8}', flush=True)
    base = None
    for eta in ETA_LIST:
        q = [2.2, 0.50, eta, eps1, eps2]
        try:
            V, dVs, dVc, dEsh, dEpair = pes.energy_components(q)
            E_ld = dVs + dVc
            if eta == 0.0:
                base = V
            rel = V - base if base is not None else np.nan
            print(f'{eta:+.2f} {E_ld:8.2f} {dEsh:8.2f} {dEpair:8.2f} {V:8.2f} '
                  f'{rel:8.2f}', flush=True)
        except Exception as e:
            print(f'{eta:+.2f}  FAIL {str(e)[:40]}', flush=True)


if __name__ == '__main__':
    main()
