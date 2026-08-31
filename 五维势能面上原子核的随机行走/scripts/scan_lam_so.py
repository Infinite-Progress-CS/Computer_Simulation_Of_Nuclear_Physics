# -*- coding: utf-8 -*-
"""扫描 (λ_so_n, λ_so_p) 看 N=82 与 Z=50 壳随形变的相对韧性。

目标：找 λ_so_n（中子，↑让 N=82 更抗形变）/ λ_so_p（质子，↓让 Z=50 更快溶解），
使重碎片最稳同位素从 Sn-132(A=132, 双幻数) 移到 Xe/Ba(A=136-138, 仅 N=82)。

用法：python scan_lam_so.py <lam_so_n> <lam_so_p>
一次跑一个组合（每次进程小，避免 OOM）。
"""
import sys
import os
import gc
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src')
sys.path.insert(0, SRC)

from fragment import Fragment

# 重碎片（N=82 同位素）+ 轻碎片（形变区）
NUCLEI = [
    ('Sn-132', 50, 82),
    ('Sb-133', 51, 82),
    ('Te-134', 52, 82),
    ('I-135', 53, 82),
    ('Xe-136', 54, 82),
    ('Cs-137', 55, 82),
    ('Ba-138', 56, 82),
    ('La-139', 57, 82),
]
LIGHT = [
    ('Zr-100', 40, 60),
    ('Mo-102', 42, 60),
    ('Mo-104', 42, 62),
]
EPS = (0.0, 0.1, 0.2, 0.3)


def shell_at(Z, N, lam_so_n, lam_so_p, eps):
    f = Fragment(Z, N, Nmax=12, a_sym=23.5, lam_so_n=lam_so_n, lam_so_p=lam_so_p)
    return f.shell_correction(eps)


def main():
    lam_so_n = float(sys.argv[1]) if len(sys.argv) > 1 else 35.0
    lam_so_p = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0
    print(f'===== λ_so_n={lam_so_n:.0f}  λ_so_p={lam_so_p:.0f} =====')
    hdr = '  ' + ''.join(f'{e:>8.1f}' for e in EPS)
    print('重碎片 N=82 同位素 δE_shell(ε) [MeV]:' + hdr)
    for name, Z, N in NUCLEI:
        row = f'{name:>7}'
        for eps in EPS:
            row += f'{shell_at(Z, N, lam_so_n, lam_so_p, eps):>8.1f}'
        print(row)
        gc.collect()
    print('轻碎片（形变区）:')
    for name, Z, N in LIGHT:
        row = f'{name:>7}'
        for eps in EPS:
            row += f'{shell_at(Z, N, lam_so_n, lam_so_p, eps):>8.1f}'
        print(row)
        gc.collect()


if __name__ == '__main__':
    main()
