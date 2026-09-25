"""
hybrid_pes.py — 统一双中心壳修正 PES
====================================================================
用统一的非正交双中心（移位谐振子）Woods-Saxon 基取代「满核单中心 + 碎片级」两套
人工插值。双中心基平滑描述 球 → 鞍点 → 双碎片，颈方向无假势垒，壳修正的不对称
偏好显著增强（鞍点区 ~5 MeV，单中心仅 ~1.5 MeV），解决产额谷太浅的结构性问题。

保留 CachedHybridMacroMicro 磁盘缓存外壳（dname 标签区分双中心基）。
"""
import os

from macro_micro import MacroMicro
from cached_pes import DiskCache, _q_key


class HybridMacroMicro(MacroMicro):
    """统一双中心基宏观-微观 PES（basis='two_center'）。

    neck_hi/neck_lo/r_hi/r_lo/eps_scission 保留仅为向后兼容（旧断裂区切换已删除）。
    """

    def __init__(self, Z, N, neck_hi=0.5, neck_lo=0.15, r_hi=None, r_lo=None,
                 eps_scission=(0.4, 0.6), **kw):
        kw['basis'] = 'two_center'
        super().__init__(Z, N, **kw)


class CachedHybridMacroMicro(HybridMacroMicro):
    """HybridMacroMicro + 磁盘缓存（dname 用 tc_ 前缀，与旧 hybrid_ 缓存隔离）。"""

    def __init__(self, Z, N, cache_dir=None, **kw):
        super().__init__(Z, N, **kw)
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "cache")
        Nmax = kw.get("Nmax", 12)
        shape_name = type(self.shape).__name__
        shape_ver = getattr(type(self.shape), "VERSION", "")
        dname = (f"tc_Z{Z}_N{N}_Nmax{Nmax}_g{self.gamma:.2f}_p{self.p}"
                 f"_lsp{self.lam_so_p}_{shape_name}{shape_ver}")
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
