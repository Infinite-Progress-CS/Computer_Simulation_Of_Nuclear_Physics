"""
level_density.py — 费米气体核温度
====================================================================
断裂点模型的统计因子 exp(−V/T) 需要核温度 T。费米气体模型（Bethe 1936）：

  T = √(E* / a),   a ≈ A/8 MeV⁻¹

  E* : 激发能（MeV）
  a  : 能级密度参数（MeV⁻¹）
"""
import numpy as np

# U-236 热中子裂变：中子结合能 ≈ 6.5 MeV，复合核 U-236 激发能 E* ≈ 6.5 MeV
THERMAL_EXCITATION = 6.5   # MeV


def level_density_parameter(A):
    """能级密度参数 a ≈ A/8（MeV⁻¹）。"""
    return A / 8.0


def nuclear_temperature(E_star, A):
    """核温度 T = √(E*/a)（MeV）。E* 激发能，A 质量数。"""
    if E_star <= 0.0:
        return 0.0
    return float(np.sqrt(E_star / level_density_parameter(A)))


def thermal_temperature(A=236, E_star=THERMAL_EXCITATION):
    """热中子裂变的默认核温度（U-236: T ≈ 0.47 MeV）。"""
    return nuclear_temperature(E_star, A)
