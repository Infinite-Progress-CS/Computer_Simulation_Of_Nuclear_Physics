# -*- coding: utf-8 -*-
"""扫描质子自旋轨道强度 λ_so_p，看 N=82 与 Z=50 壳随形变的相对韧性。

目标：找到 λ_so_p 使 Z=50 质子幻数（Sn-132 双幻数的质子半）随形变快速溶解、
而 N=82 中子壳存活——从而重峰从 Sn-132(A=132) 移到 Xe/Ba(A=136-138, 仅 N=82)。

打印 δE_shell(ε) 表：行=核，列=ε，对每个 λ_so_p。
"""
import sys
import os
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


def shell_at(Z, N, lam_so_p, eps):
    f = Fragment(Z, N, Nmax=12, a_sym=23.5, lam_so_p=lam_so_p)
    return f.shell_correction(eps)


def main():
    for lam_so_p in (35.0, 30.0, 25.0, 20.0, 15.0):
        print(f'\n===== λ_so_p = {lam_so_p:.0f}（λ_so_n=35）=====')
        print('重碎片 N=82 同位素：δE_shell(ε) [MeV]')
        hdr = '  ' + ''.join(f'{e:>8.1f}' for e in EPS)
        print(hdr)
        for name, Z, N in NUCLEI:
            row = f'{name:>7}'
            for eps in EPS:
                row += f'{shell_at(Z, N, lam_so_p, eps):>8.1f}'
            print(row)
        print('轻碎片（形变区）：')
        for name, Z, N in LIGHT:
            row = f'{name:>7}'
            for eps in EPS:
                row += f'{shell_at(Z, N, lam_so_p, eps):>8.1f}'
            print(row)


if __name__ == '__main__':
    main()
