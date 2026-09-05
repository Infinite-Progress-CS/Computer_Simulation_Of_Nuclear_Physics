"""
hybrid_pes.py — 双中心壳修正（断裂区切换到碎片级量子修正）
====================================================================
根因（2026-09-04 诊断 _diag_frag_asym.py）：满核单中心 HO 基（Nmax=12）在断颈区
解不开双中心结构，壳修正不对称偏好只有 −1.24 MeV（碎片级应为 −10.57 MeV），
液滴不对称罚 +2.35 MeV 压过 → 对称裂变。

修复：颈薄时（neck ≤ neck_lo）用碎片级壳/对修正（两碎片各自独立 Woods-Saxon，
双中心正确），颈厚时（neck ≥ neck_hi）用满核单中心修正，中间线性插值。
液滴部分始终用满核 FRLDM（断颈形状也正确）。

碎片分解（η>0 → 右侧重碎片，与 shape.py 的 w2=(1+η)/2、eps2 一致）：
  A_L = round(A(1−η)/2)（轻，形变 eps1）；A_H = A−A_L（重，形变 eps2）
  Z 取 UCD：Z_L = round(Z·A_L/A)。
"""
import os

import numpy as np

from macro_micro import MacroMicro
from fragment import Fragment
from cached_pes import DiskCache, _q_key


class HybridMacroMicro(MacroMicro):
    """满核液滴 + 断裂区碎片级量子修正的宏观-微观 PES。

    neck_hi / neck_lo：切换窗口。neck ≥ neck_hi 用满核单中心（势垒区可靠）；
    neck ≤ neck_lo 用碎片级（断颈区正确）；中间线性插值（平滑过渡，避免 PES 断裂）。
    """

    def __init__(self, Z, N, neck_hi=0.5, neck_lo=0.15, **kw):
        super().__init__(Z, N, **kw)
        self.neck_hi = neck_hi
        self.neck_lo = neck_lo
        self._frag_cache = {}

    def _fragment(self, Zf, Nf):
        key = (Zf, Nf)
        if key not in self._frag_cache:
            gamma_fac = self.gamma / self.ws.hbar_omega
            self._frag_cache[key] = Fragment(
                Zf, Nf, Nmax=self.ws.Nmax, gamma_fac=gamma_fac, p=self.p,
                g0_n=self.G_n * self.A, g0_p=self.G_p * self.A,
                pair_window=self.pair_window)
        return self._frag_cache[key]

    def _fragment_quantum(self, q):
        eta, eps1, eps2 = q[2], q[3], q[4]
        A, Z = self.A, self.Z
        A_L = int(round(A * (1.0 - eta) / 2.0))
        A_L = int(np.clip(A_L, 1, A - 1))
        A_H = A - A_L
        Z_L = int(round(Z * A_L / A))
        Z_L = int(np.clip(Z_L, 1, Z - 1))
        Z_H = Z - Z_L
        N_L = A_L - Z_L
        N_H = A_H - Z_H
        fL = self._fragment(Z_L, N_L)
        fH = self._fragment(Z_H, N_H)
        dE_sh = fL.shell_correction(eps1) + fH.shell_correction(eps2)
        dE_pair = fL.pairing_correction(eps1) + fH.pairing_correction(eps2)
        return dE_sh, dE_pair

    def quantum_components(self, q):
        neck = float(q[1])
        if neck >= self.neck_hi:
            return super().quantum_components(q)          # 满核单中心
        if neck <= self.neck_lo:
            sh_frag, _ = self._fragment_quantum(q)
            return sh_frag, 0.0                           # 断裂区对修正坍缩
        sh_full, pair_full = super().quantum_components(q)
        sh_frag, _ = self._fragment_quantum(q)
        w = (self.neck_hi - neck) / (self.neck_hi - self.neck_lo)
        # 壳修正：满核 → 碎片级双中心；对修正：满核 → 0（断裂区奇偶/热激发使配对坍缩）
        return (1.0 - w) * sh_full + w * sh_frag, (1.0 - w) * pair_full


class CachedHybridMacroMicro(HybridMacroMicro):
    """HybridMacroMicro + 磁盘缓存（缓存目录与纯满核版隔离，避免串库）。"""

    def __init__(self, Z, N, cache_dir=None, **kw):
        super().__init__(Z, N, **kw)
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "cache")
        Nmax = kw.get("Nmax", 12)
        dname = (f"hybrid_Z{Z}_N{N}_Nmax{Nmax}_g{self.gamma:.2f}_p{self.p}"
                 f"_nh{self.neck_hi}_nl{self.neck_lo}")
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
