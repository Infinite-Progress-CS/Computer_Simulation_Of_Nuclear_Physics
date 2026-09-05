"""
liquid_drop.py — FRLDM 宏观能量（有限力程液滴模型，五维势能面）
====================================================
从原单文件搬出，保留分层网格。宏观形变能（无壳修正/对修正）：
    ΔV(q) = E_S0·(B_s − 1) + E_C0·(B_c − 1)
  B_s = 有限力程表面能比（Krappe-Nix-Sierk 双重曲面积分，核 [2−(2+s/a)e^{−s/a}]）
  B_c = 库仑能比（尖表面，方位角解析积分）

参数（FRLDM 1995 ADNDT 59,185）：a_s=21.18466, κ_s=2.345, r0=1.16 fm, a=0.68 fm。
"""

import numpy as np
from scipy import special

from shape import Shape3QS


class FRLDMPES:
    """
    FRLDM 宏观形变能（无壳修正/对修正）：
        ΔV(q) = E_S0·(B_s − 1) + E_C0·(B_c − 1)
    """

    def __init__(self, Z, N, r0=1.16, a_s=21.18466, kappa_s=2.345, a_range=0.68,
                 nz=48, nrho=48, nsurf=56, nphi=36, richardson=True):
        self.Z, self.N = Z, N
        self.A = Z + N
        self.R0 = r0 * self.A ** (1.0 / 3.0)
        I = (N - Z) / self.A
        self.E_S0 = a_s * (1.0 - kappa_s * I ** 2) * self.A ** (2.0 / 3.0)
        self.E_C0 = (3.0 / 5.0) * Z * Z * 1.44 / self.R0
        self.a = a_range / self.R0          # 无量纲有限力程
        self.shape = Shape3QS(self.R0)
        self.nz, self.nrho = nz, nrho       # 库仑积分网格（细）
        self.nsurf, self.nphi = nsurf, nphi  # 表面能积分网格
        self.richardson = richardson
        self._sphere = self.shape.build([0.0, 0.0, 0.0, 0.0, 0.0])
        self.I_C_sphere = self._coulomb_I_conv(self._sphere)
        self.I_S_sphere = self._surface_I(self._sphere)

    # ---- 库仑（尖表面，方位角解析 ellipk）----
    def _coulomb_I(self, d, nz=None, nrho=None):
        """四重库仑积分 I_C = Σ_{i,j,k,l} w_i w_j w_k w_l · 2π r1 r2 · 4K/√(aa+b)。

        分块累加（沿外层 z 索引 i 循环）：峰值内存从 O(nz²·nρ²) 降到 O(nz·nρ²)，
        允许细网格（48×48 及以上）而不会爆 4D 数组内存。
        """
        if nz is None:
            nz = self.nz
        if nrho is None:
            nrho = self.nrho
        zL, zR, rho2 = d["zL"], d["zR"], d["rho2"]
        xz, wz = np.polynomial.legendre.leggauss(nz)
        xr, wr = np.polynomial.legendre.leggauss(nrho)
        z = zL + (xz + 1.0) * 0.5 * (zR - zL)
        wz_full = wz * (zR - zL) * 0.5
        Rz = np.sqrt(np.maximum(rho2(z), 0.0))
        rho = Rz[:, None] * (xr[None, :] + 1.0) * 0.5
        w_rho = wr[None, :] * Rz[:, None] * 0.5
        rho_sq = rho ** 2

        total = 0.0
        for i in range(nz):
            wi = wz_full[i]
            ri = rho[i]                       # (nrho,)
            ri2 = ri ** 2
            zz_i = (z[i] - z) ** 2            # (nz,)
            # aa[j,k,l] = (zi−zk)² + ri[j]² + rho[k,l]²  → (nrho, nz, nrho)
            aa = (zz_i[None, :, None] + ri2[:, None, None]
                  + rho_sq[None, :, :])
            b = 2.0 * ri[:, None, None] * rho[None, :, :]     # (nrho, nz, nrho)
            apb = np.maximum(aa + b, 1e-14)
            m = np.clip(2.0 * b / apb, 0.0, 1.0 - 1e-12)
            K = special.ellipk(m)
            integrand = (2.0 * np.pi * ri[:, None, None]
                         * rho[None, :, :] * 4.0 * K / np.sqrt(apb))
            total_w = (w_rho[i][:, None, None] * wz_full[None, :, None]
                       * w_rho[None, :, :])
            total += wi * np.sum(total_w * integrand)
        return total

    def _coulomb_I_conv(self, d):
        """Richardson 外推的库仑积分，消除 4 重积分的对数奇点慢收敛。

        ellipk(m) 在 m→1（两体积元重合）处有对数奇点，Gauss-Legendre 误差 ~1/n²
        （实测 n=32/48/64 误差 +2.99/+1.32/+0.73 MeV）。用 n_hi 与 n_lo=round(2/3·n_hi)
        两套网格按 1/n² 外推，误差压到 ~0.02 MeV，代价仅 +20%。
        """
        if not self.richardson:
            return self._coulomb_I(d)
        n_hi = self.nz
        n_lo = max(20, int(round(n_hi * 2.0 / 3.0)))
        I_hi = self._coulomb_I(d, n_hi, n_hi)
        I_lo = self._coulomb_I(d, n_lo, n_lo)
        r = (n_hi / n_lo) ** 2
        return I_hi + (I_hi - I_lo) / (r - 1.0)

    # ---- 有限力程表面能（KNS 双重曲面积分）----
    def _surface_I(self, d):
        a = self.a
        zL, zR, rho2 = d["zL"], d["zR"], d["rho2"]
        nz, nphi = self.nsurf, self.nphi
        xz, wz = np.polynomial.legendre.leggauss(nz)
        xp, wp = np.polynomial.legendre.leggauss(nphi)
        z = zL + (xz + 1.0) * 0.5 * (zR - zL)
        wz_full = wz * (zR - zL) * 0.5
        r = np.sqrt(np.maximum(rho2(z), 0.0))
        dz = 1e-6 * (zR - zL)
        rp = np.sqrt(np.maximum(rho2(z + dz), 0.0))
        rm = np.sqrt(np.maximum(rho2(z - dz), 0.0))
        rho_prime = (rp - rm) / (2.0 * dz)

        Delta = np.pi * (xp + 1.0)
        wD = wp * np.pi
        zz = z[:, None] - z[None, :]
        rho1 = r[:, None]; rho2_ = r[None, :]
        rp1 = rho_prime[:, None]; rp2 = rho_prime[None, :]
        r1r2 = rho1 * rho2_

        total = 0.0
        for k in range(nphi):
            cosD = np.cos(Delta[k])
            s2 = zz ** 2 + rho1 ** 2 + rho2_ ** 2 - 2.0 * r1r2 * cosD
            s = np.sqrt(np.maximum(s2, 1e-24))
            K = (2.0 - (2.0 + s / a) * np.exp(-s / a)) / np.maximum(s2 ** 2, 1e-24)
            dot1 = rho1 - rho2_ * cosD - zz * rp1
            dot2 = rho1 * cosD - rho2_ - zz * rp2
            total += wD[k] * np.sum(wz_full[:, None] * wz_full[None, :]
                                    * r1r2 * dot1 * dot2 * K)
        return 2.0 * np.pi * total

    def B_s(self, q):
        return self._surface_I(self.shape.build(q)) / self.I_S_sphere

    def B_c(self, q):
        return self._coulomb_I_conv(self.shape.build(q)) / self.I_C_sphere

    def energy_components(self, q):
        d = self.shape.build(q)
        B_s = self._surface_I(d) / self.I_S_sphere
        B_c = self._coulomb_I_conv(d) / self.I_C_sphere
        dV_s = self.E_S0 * (B_s - 1.0)
        dV_c = self.E_C0 * (B_c - 1.0)
        return dV_s + dV_c, dV_s, dV_c, B_s, B_c

    def energy(self, q):
        return self.energy_components(q)[0]
