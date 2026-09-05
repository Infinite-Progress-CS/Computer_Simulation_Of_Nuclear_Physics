# -*- coding: utf-8 -*-
"""
cached_pes.py — 磁盘缓存包装 MacroMicro
====================================================================
量子部分（Strutinsky 壳修正 + BCS 对修正）单点 ~2-4s，是 5D 行走的瓶颈。
把 quantum_components(q) 的结果落盘，多进程 / 多次运行共享，避免重复对角化。

实现用「一键一文件」：每个 5D 坐标 q → 一个 .npz 小文件，原子写（临时文件 +
os.replace）。无共享数据库、无锁、无死锁，天然多进程安全（此前 SQLite 版在
worker 死亡 / 并发 commit 时卡死）。缓存目录按模型配置 (Z,N,Nmax,gamma,p) 分目录，
避免不同 Nmax / 平滑参数串库导致结果错误。
"""

import hashlib
import os

import numpy as np

from macro_micro import MacroMicro


def _q_key(q):
    return ",".join(f"{float(x):.4f}" for x in q)


class DiskCache:
    """一文件一键磁盘缓存：quantum(q) -> (dE_sh, dE_pair)。"""

    def __init__(self, dirpath):
        self.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)

    def _path(self, q_key):
        h = hashlib.md5(q_key.encode("utf-8")).hexdigest()
        return os.path.join(self.dirpath, h + ".npy")

    def get(self, q_key):
        p = self._path(q_key)
        if os.path.exists(p):
            try:
                arr = np.load(p)
                return float(arr[0]), float(arr[1])
            except Exception:
                return None
        return None

    def put(self, q_key, dE_sh, dE_pair):
        p = self._path(q_key)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            np.save(f, np.array([float(dE_sh), float(dE_pair)]))
        os.replace(tmp, p)


class CachedMacroMicro(MacroMicro):
    """MacroMicro + 磁盘缓存。quantum_components 先查盘，未命中才对角化并写盘。"""

    def __init__(self, Z, N, cache_dir=None, **kw):
        super().__init__(Z, N, **kw)
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "cache")
        Nmax = kw.get("Nmax", 12)
        dname = f"quantum_Z{Z}_N{N}_Nmax{Nmax}_g{self.gamma:.2f}_p{self.p}"
        self._disk = DiskCache(os.path.join(cache_dir, dname))
        self._n_hit = 0
        self._n_miss = 0

    def quantum_components(self, q):
        key = _q_key(q)
        hit = self._disk.get(key)
        if hit is not None:
            self._n_hit += 1
            return hit[0], hit[1]
        self._n_miss += 1
        dE_sh, dE_pair = super().quantum_components(q)
        self._disk.put(key, dE_sh, dE_pair)
        return dE_sh, dE_pair

    def cache_stats(self):
        return self._n_hit, self._n_miss
