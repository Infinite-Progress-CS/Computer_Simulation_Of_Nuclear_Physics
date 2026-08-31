"""
yield_model.py — 遍历碎片质量，得到产额分布 Y(A)、Y(Z)
====================================================================
断裂点模型（Wilkins-Steinberg-Chasman 1976）：

  Y(A_L, Z_L) ∝ exp(−V_scission(A_L, Z_L) / T)
  Y(A_L)      = Σ_{Z_L} Y(A_L, Z_L)            （电荷弥散求和）

  V_scission = E_frag(L) + E_frag(H) + E_Coul(Z_L, Z_H, d)

对 U-236（Z=92, N=144）热中子裂变，轻碎片 (A_L, Z_L)，重碎片
(A_H = 236−A_L, Z_H = 92−Z_L)。

电荷弥散是重现双峰的关键：UCD 固定电荷 Z_L = round(Z/A·A_L) 会把碎片
钉在中壳（壳修正为正），得不到双峰；必须让每个 A 在 UCD 附近扫 Z，
使碎片能到达幻数（重碎片 N=82、轻碎片 N=50 / Z=40），壳修正为负、产额增大。
"""
import numpy as np

from fragment import Fragment
from scission import scission_energy


def ucd_charge(A_L, Z_parent=92, A_parent=236):
    """未变电荷密度：Z_L = round(Z/A · A_L)。"""
    return int(round(Z_parent * A_L / A_parent))


class YieldModel:
    """断裂点产额模型：给定温度 T，输出 Y(A)、Y(Z)、壳修正 δE_shell(A)。

    温度 T 取断裂点核温度（~1 MeV），非复合核温度（~0.47 MeV）：断裂时
    碎片总激发能 ~20−25 MeV 分摊到 a≈A/8，T ≈ 1.0−1.3 MeV。T 是首要可调参数，
    决定双峰/谷比。
    """

    def __init__(self, Z_parent=92, N_parent=144, T=1.0, d_extra=2.0,
                 A_min=70, A_max=166, Nmax=12, a_sym=23.5, lam_so_p=None,
                 lam_so_n=None):
        self.Z_parent, self.N_parent = Z_parent, N_parent
        self.A_parent = Z_parent + N_parent
        self.T = T
        self.d_extra = d_extra
        self.A_min, self.A_max = A_min, A_max
        self.Nmax = Nmax
        self.a_sym = a_sym
        self.lam_so_p = lam_so_p
        self.lam_so_n = lam_so_n
        self._frag_cache = {}

    def _fragment(self, Z, N):
        """带缓存的碎片核 (Z, N)。"""
        key = (Z, N, self.lam_so_p, self.lam_so_n)
        if key not in self._frag_cache:
            self._frag_cache[key] = Fragment(Z, N, Nmax=self.Nmax,
                                             a_sym=self.a_sym,
                                             lam_so_p=self.lam_so_p,
                                             lam_so_n=self.lam_so_n)
        return self._frag_cache[key]

    def scission_energy(self, Z_L, N_L):
        """给定轻碎片 (Z_L, N_L)，返回断裂点势能 V_scission（MeV）。

        重碎片为互补 (Z_H=Z_p−Z_L, N_H=N_p−N_L)。
        """
        A_L = Z_L + N_L
        Z_H = self.Z_parent - Z_L
        A_H = self.A_parent - A_L
        N_H = A_H - Z_H
        frag_L = self._fragment(Z_L, N_L)
        frag_H = self._fragment(Z_H, N_H)
        return scission_energy(frag_L, frag_H, d_extra=self.d_extra)

    def scission_energy_for_A(self, A_L, Z_L=None):
        """给定轻碎片质量 A_L（默认 UCD 定 Z），返回 V_scission（MeV）。"""
        if Z_L is None:
            Z_L = ucd_charge(A_L, self.Z_parent, self.A_parent)
        return self.scission_energy(Z_L, A_L - Z_L)

    # ---- 全部 (A,Z) 断裂点势能（供产额/电荷产额共用，避免重复对角化）----
    def _all_scission_energies(self, z_window=4):
        """返回 (A 数组, [(Z 数组, V 数组) per A])，V 为断裂点势能（MeV）。

        对每个轻碎片质量 A_L，在 UCD ± z_window 内扫描 Z_L，记录 V_scission。
        """
        A_arr = np.arange(self.A_min, self.A_max + 1)
        V_list = []
        for A_L in A_arr:
            zc = ucd_charge(A_L, self.Z_parent, self.A_parent)
            Zs, Vs = [], []
            for Z_L in range(zc - z_window, zc + z_window + 1):
                N_L = A_L - Z_L
                if not (1 <= Z_L < self.Z_parent and 1 <= N_L < self.N_parent):
                    continue
                Zs.append(Z_L)
                Vs.append(self.scission_energy(Z_L, N_L))
            V_list.append((np.asarray(Zs, dtype=int), np.asarray(Vs, dtype=float)))
        return A_arr, V_list

    # ---- 质量产额（电荷弥散求和）----
    def mass_yield(self, z_window=4):
        """返回 (A 数组, 质量产额 Y(A) 百分数数组)，ΣY = 200%。

        Y(A_L) ∝ Σ_{Z_L} exp(−V(A_L,Z_L)/T)，Z_L 在 UCD ± z_window 内扫描。
        归一化时全局减最小 V（断裂势能含 ~2000 MeV 的碎片体积能常量，直接
        exp(−V/T) 会下溢；减常量不影响产额形状）。
        """
        A_arr, V_list = self._all_scission_energies(z_window)
        V_min = min(Vs.min() for _, Vs in V_list if len(Vs))
        Y = np.array([np.exp(-(Vs - V_min) / self.T).sum() for _, Vs in V_list])
        Y = 200.0 * Y / Y.sum()
        return A_arr, Y

    def mass_yield_ucd(self):
        """UCD 固定电荷（无弥散）的质量产额，供诊断（快，但无双峰）。"""
        A_arr = np.arange(self.A_min, self.A_max + 1)
        V = np.array([self.scission_energy_for_A(int(A_L)) for A_L in A_arr])
        Y = np.exp(-(V - V.min()) / self.T)
        Y = 200.0 * Y / Y.sum()
        return A_arr, Y

    # ---- 最可几电荷（对每个 A 取 V 最小的 Z）----
    def most_probable_charge(self, z_window=4):
        """返回 (A 数组, Z_p(A))：每个 A 使 V_scission 最小的轻碎片 Z。"""
        A_arr = np.arange(self.A_min, self.A_max + 1)
        Zp = np.zeros_like(A_arr, dtype=int)
        for i, A_L in enumerate(A_arr):
            zc = ucd_charge(A_L, self.Z_parent, self.A_parent)
            best_z, best_v = zc, np.inf
            for Z_L in range(zc - z_window, zc + z_window + 1):
                N_L = A_L - Z_L
                if not (1 <= Z_L < self.Z_parent and 1 <= N_L < self.N_parent):
                    continue
                v = self.scission_energy(Z_L, N_L)
                if v < best_v:
                    best_v, best_z = v, Z_L
            Zp[i] = best_z
        return A_arr, Zp

    # ---- 壳修正随 A（诊断双峰来源）----
    def shell_correction_vs_A(self):
        """返回 (A 数组, 轻碎片 δE_shell, 重碎片 δE_shell)（MeV），UCD 电荷。"""
        A_arr = np.arange(self.A_min, self.A_max + 1)
        sh_L = np.zeros_like(A_arr, dtype=float)
        sh_H = np.zeros_like(A_arr, dtype=float)
        for i, A_L in enumerate(A_arr):
            Z_L = ucd_charge(int(A_L), self.Z_parent, self.A_parent)
            A_H = self.A_parent - int(A_L)
            Z_H = self.Z_parent - Z_L
            sh_L[i] = self._fragment(Z_L, int(A_L) - Z_L).shell_correction()
            sh_H[i] = self._fragment(Z_H, A_H - Z_H).shell_correction()
        return A_arr, sh_L, sh_H

    # ---- 电荷产额（对每个 Z 归并质量产额）----
    def charge_yield(self, z_window=4):
        """返回 (Z 数组, 电荷产额 Y(Z) 百分数数组)：Y(Z) ∝ Σ_A Y(A,Z)。"""
        A_arr, V_list = self._all_scission_energies(z_window)
        V_min = min(Vs.min() for _, Vs in V_list if len(Vs))
        Yz = {}
        for Zs, Vs in V_list:
            for Z, V in zip(Zs, Vs):
                Yz[Z] = Yz.get(Z, 0.0) + float(np.exp(-(V - V_min) / self.T))
        Z_arr = np.array(sorted(Yz))
        Y = np.array([Yz[z] for z in Z_arr])
        Y = 200.0 * Y / Y.sum()
        return Z_arr, Y
