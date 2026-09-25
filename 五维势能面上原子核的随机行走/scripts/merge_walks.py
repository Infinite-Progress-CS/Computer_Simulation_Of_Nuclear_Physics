# -*- coding: utf-8 -*-
"""合并多个随机行走 npz 的 eta_scission，供 plot_yield_final.py 做大统计产额。"""
import os, sys, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'results', '随机行走')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pattern', nargs='?',
                    default='walk_yield_pes_table_U236_nmax8_fine_nw1000_ms600_s30*.npz',
                    help='glob 模式（相对 results/随机行走/）')
    ap.add_argument('--out', default=None, help='输出 npz 名（默认 merged_<pattern>）')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(OUT, args.pattern)))
    if not files:
        sys.exit(f'no npz match: {args.pattern}')
    etas, n_sc, accepts = [], 0, []
    for f in files:
        d = np.load(f)
        etas.append(d['eta_scission'])
        n_sc += int(d['n_scission'])
        if 'acceptance' in d:
            accepts.append(float(d['acceptance']))
    eta = np.concatenate(etas)
    if args.out is None:
        args.out = f"merged_{os.path.splitext(os.path.basename(files[0]))[0][:30]}_x{len(files)}.npz"
    out_path = os.path.join(OUT, args.out)
    np.savez(out_path, eta_scission=eta, n_scission=n_sc,
             acceptance=float(np.mean(accepts)) if accepts else 0.0,
             n_files=len(files))
    print(f'merged {len(files)} files -> {out_path}')
    print(f'  n_scission={n_sc}  eta: mean={eta.mean():+.3f} sigma={eta.std():.3f} '
          f'asym(|eta|>0.03)={(np.abs(eta)>0.03).mean():.0%}  sym(|eta|<0.03)={(np.abs(eta)<0.03).mean():.2%}')
    print(f'  files: {[os.path.basename(f) for f in files]}')


if __name__ == '__main__':
    main()
