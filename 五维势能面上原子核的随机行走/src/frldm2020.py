# -*- coding: utf-8 -*-
"""frldm2020.py — 读取 Mumpower/Jaffke/Verriere/Randrup PRC 101, 054607 (2020)
公开发布的 FRLDM 裂变碎片产额 ASCII 数据。

数据文件格式为三列：
    Z   A     Y(Z,A)
其中 Y(Z,A) 是初级碎片产额（质量与电荷联合分布），一个文件的总和归一化为 2，
即两个碎片的产额各按 100% 计。

这里提供：
    load_yield        -> (Z, A, Y) 原始联合产额
    mass_yield        -> (A, Y_A(%))  质量产额，归一 200%
    charge_yield      -> (Z, Y_Z(%))  电荷产额，归一 200%
    conditional_charge-> (Z_grid, A_grid, P[Z|A]) 每个 A 上归一化的条件电荷分布
"""

import numpy as np


def load_yield(path):
    """读取 FRLDM 2020 ASCII 产额文件，返回 (Z, A, Y)。Y 单位同原文。"""
    Z, A, Y = [], [], []
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 3:
            Z.append(int(parts[0]))
            A.append(int(parts[1]))
            Y.append(float(parts[2]))
    return np.asarray(Z, dtype=int), np.asarray(A, dtype=int), np.asarray(Y, dtype=float)


def mass_yield(path, A_parent=236, A_lo=60, A_hi=176):
    """质量产额 Y(A)，百分比，总和归一 200%。"""
    Z, A, Y = load_yield(path)
    A_grid = np.arange(A_lo, A_hi + 1, dtype=int)
    Y_A = np.array([Y[A == a].sum() for a in A_grid], dtype=float)
    Y_A *= 100.0
    return A_grid, Y_A


def charge_yield(path, Z_lo=20, Z_hi=92):
    """电荷产额 Y(Z)，百分比，总和归一 200%。"""
    Z, A, Y = load_yield(path)
    Z_grid = np.arange(Z_lo, Z_hi + 1, dtype=int)
    Y_Z = np.array([Y[Z == z].sum() for z in Z_grid], dtype=float)
    Y_Z *= 100.0
    return Z_grid, Y_Z


def conditional_charge(path, A_parent=236, Z_lo=20, Z_hi=92, A_lo=60, A_hi=176):
    """由 FRLDM 二维产额构造 P(Z|A)。

    返回 (Z_grid, A_grid, P)，P[iZ, iA] 在固定 A 上对 Z 归一（总和=1）。
    对于个别没有记录数据的 A，该列全部为 0，调用方应做回退。
    """
    Z, A, Y = load_yield(path)
    Z_grid = np.arange(Z_lo, Z_hi + 1, dtype=int)
    A_grid = np.arange(A_lo, A_hi + 1, dtype=int)
    P = np.zeros((len(Z_grid), len(A_grid)), dtype=float)
    for iz, z in enumerate(Z_grid):
        mask_z = Z == z
        if not mask_z.any():
            continue
        for ia, a in enumerate(A_grid):
            mask = mask_z & (A == a)
            if mask.any():
                P[iz, ia] = Y[mask].sum()
    # 每个 A 上归一；若某列无数据，保持 0，调用方负责回退到 UCD。
    col_sum = P.sum(axis=0)
    valid = col_sum > 0
    P[:, valid] /= col_sum[None, valid]
    return Z_grid, A_grid, P
