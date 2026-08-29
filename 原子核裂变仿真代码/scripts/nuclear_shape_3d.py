"""
核形状计算模块：5参数 (Q,c,α,ε₁,ε₂) → 3D 核表面网格。

物理模型：
  核表面在柱坐标系 (ρ, φ, z) 中定义为轮廓函数 ρ(z)，绕 z 轴旋转。
  体积守恒：V = 4π/3 R₀³，其中 R₀ = r₀·A^{1/3}。

参数说明：
  Q  ∈ [0.0, 1.5]  — 伸长量，Q↑ → 核拉长
  c  ∈ [0.0, 1.0]  — 颈部参数，c↓ → z=0处掐腰（裂变！）
  α  ∈ [-1.0, 1.0] — 质量不对称度，α>0 → 右半更大
  ε₁ ∈ [0.0, 1.0]  — 左碎片形变
  ε₂ ∈ [0.0, 1.0]  — 右碎片形变

作者：高涵
日期：2026-08-12
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


# ============================================================
# 物理常量
# ============================================================
R0_PARAM = 1.20       # fm, 核半径参数
DEFAULT_A = 238        # 默认质量数
DEFAULT_R0 = R0_PARAM * DEFAULT_A ** (1.0 / 3.0)  # ≈ 7.44 fm


@dataclass
class ShapeParams:
    """五维形变参数。"""
    Q: float = 0.3      # 伸长量
    c: float = 0.6      # 颈部参数
    alpha: float = 0.0  # 不对称度
    eps1: float = 0.0   # 左碎片形变
    eps2: float = 0.0   # 右碎片形变

    def to_array(self) -> np.ndarray:
        return np.array([self.Q, self.c, self.alpha, self.eps1, self.eps2])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ShapeParams':
        return cls(arr[0], arr[1], arr[2], arr[3], arr[4])


# ============================================================
# 核心：轮廓函数 ρ(z)
# ============================================================

def nuclear_profile(z: np.ndarray,
                    Q: float, c: float, alpha: float,
                    eps1: float, eps2: float,
                    R0: float = DEFAULT_R0,
                    correct_volume: bool = True) -> np.ndarray:
    """
    计算核表面在柱坐标系下的轮廓 ρ(z)。

    物理模型：
      1. 基态球形经 Q 伸长 → 体积守恒的椭球
      2. 颈部因子在 z=0 处收缩 → 模拟颈部形成
      3. 不对称因子偏移 → 左右碎片不等大
      4. 碎片形变 → 两端额外变形

    Parameters
    ----------
    z : ndarray  轴向坐标网格 [fm]
    Q, c, alpha, eps1, eps2 : 五维形变参数
    R0 : 等效球形半径 [fm]
    correct_volume : 是否迭代修正体积守恒

    Returns
    -------
    rho : ndarray  径向坐标 ρ(z) [fm]，≥0
    """
    # ── 1. 基椭球：体积守恒的伸长 ──
    # 球 → 椭球：半长轴 a = R0·(1+Q)^{2/3}，半短轴 b = R0·(1+Q)^{-1/3}
    elongation_factor = 1.0 + 1.5 * Q  # 映射 Q∈[0,1.5] → 伸长 1~3.25
    a_semi = R0 * elongation_factor     # z 半轴
    b_semi = R0 / np.sqrt(elongation_factor)  # 径向半轴

    # ── 2. 不对称度 → 中心偏移 ──
    # α ∈ [-1, 1] → 右半比例 = (1+α)/2 ∈ [0, 1]
    z_shift = alpha * a_semi * 0.4    # 有效 z 零点偏移
    z_eff = z - z_shift

    # ── 3. 基椭球轮廓 ──
    # ρ²/b² + z²/a² = 1  →  ρ(z) = b·√(1 - z²/a²)
    inside = np.abs(z_eff) < a_semi
    rho_sph = np.zeros_like(z)
    rho_sph[inside] = b_semi * np.sqrt(np.maximum(0, 1.0 - (z_eff[inside] / a_semi) ** 2))

    # ── 4. 颈部调制 ──
    # c 越小 → z=0 处越收缩
    neck_width = a_semi * 0.18  # 颈部区域宽度
    neck_factor = 1.0 - (1.0 - c) * np.exp(-z_eff ** 2 / (2.0 * neck_width ** 2))

    # ── 5. 碎片形变 ──
    # ε₁ 影响 z<0，ε₂ 影响 z≥0
    frag_factor = np.ones_like(z)
    left_mask = z_eff < 0
    right_mask = ~left_mask

    # 碎片形变使端点附近的半径改变
    # ε>0 → 端部更鼓（类长椭球碎裂），ε接近0 → 球形碎片
    z_norm = z_eff / a_semi  # ∈ [-1, 1]
    frag_factor[left_mask] += eps1 * (1.0 - z_norm[left_mask] ** 2) * 0.3
    frag_factor[right_mask] += eps2 * (1.0 - z_norm[right_mask] ** 2) * 0.3

    # ── 6. 组合 ──
    rho = rho_sph * neck_factor * frag_factor
    rho = np.maximum(rho, 0.0)

    # ── 7. 体积守恒修正 ──
    if correct_volume and np.any(rho > 0):
        # 当前体积（绕 z 轴旋转积分）
        dz = z[1] - z[0] if len(z) > 1 else 0.1
        current_volume = np.pi * np.sum(rho ** 2) * dz
        target_volume = 4.0 / 3.0 * np.pi * R0 ** 3

        if current_volume > 0:
            scale = np.sqrt(target_volume / current_volume)
            rho *= scale

    return rho


# ============================================================
# 3D 表面网格生成
# ============================================================

def generate_surface_mesh(Q: float, c: float, alpha: float,
                          eps1: float, eps2: float,
                          R0: float = DEFAULT_R0,
                          n_z: int = 80, n_phi: int = 60,
                          max_z_scale: float = 2.5) -> Tuple[np.ndarray, ...]:
    """
    生成核表面 3D 网格 (X, Y, Z, colors)。

    Parameters
    ----------
    n_z : z 方向采样点数
    n_phi : 方位角采样点数
    max_z_scale : z 范围 = max_z_scale × R0

    Returns
    -------
    X, Y, Z : 3D 坐标网格
    rho_values : 径向坐标（用于着色）
    """
    # z 网格 — 覆盖整个核的范围
    elongation = 1.0 + 1.5 * Q
    z_max = R0 * elongation * max_z_scale * 0.8
    z = np.linspace(-z_max, z_max, n_z)

    # 计算轮廓
    rho = nuclear_profile(z, Q, c, alpha, eps1, eps2, R0, correct_volume=True)

    # 去除零边
    mask = rho > 1e-6
    if not np.any(mask):
        # 返回一个微小球体
        z_center = np.linspace(-0.1, 0.1, n_z)
        rho_center = np.sqrt(np.maximum(0, 0.01 - z_center ** 2))
        mask = rho_center > 1e-6
        z_used = z_center[mask]
        rho_used = rho_center[mask]
    else:
        z_used = z[mask]
        rho_used = rho[mask]

    # 绕 z 轴旋转生成 3D 网格
    phi = np.linspace(0, 2.0 * np.pi, n_phi)

    Z = np.outer(z_used, np.ones(n_phi))
    RHO = np.outer(rho_used, np.ones(n_phi))
    PHI = np.outer(np.ones(len(z_used)), phi)

    X = RHO * np.cos(PHI)
    Y = RHO * np.sin(PHI)

    return X, Y, Z, rho_used, z_used


def compute_shape_metrics(Q: float, c: float, alpha: float,
                          eps1: float, eps2: float,
                          R0: float = DEFAULT_R0) -> dict:
    """
    计算核形状的物理诊断量。
    """
    elongation = 1.0 + 1.5 * Q
    z_max = R0 * elongation * 1.2
    z = np.linspace(-z_max, z_max, 200)
    rho = nuclear_profile(z, Q, c, alpha, eps1, eps2, R0, correct_volume=True)

    # 体积
    dz = z[1] - z[0]
    volume = np.pi * np.sum(rho ** 2) * dz
    volume_target = 4.0 / 3.0 * np.pi * R0 ** 3

    # 颈部半径（z=0处）
    z0_idx = np.argmin(np.abs(z))
    neck_radius = float(rho[z0_idx])

    # 最大半径
    max_radius = float(np.max(rho))
    max_radius_z = float(z[np.argmax(rho)])

    # 总长度（rho>0 的 z 范围）
    nonzero = rho > 1e-6
    if np.any(nonzero):
        z_nonzero = z[nonzero]
        total_length = float(z_nonzero[-1] - z_nonzero[0])
    else:
        total_length = 0.0

    # 质心（一阶矩）
    mass_dist = rho ** 2  # ∝ 每个 z 截面的质量
    if np.sum(mass_dist) > 0:
        com = float(np.sum(z * mass_dist) / np.sum(mass_dist))
    else:
        com = 0.0

    return {
        'volume': float(volume),
        'volume_target': float(volume_target),
        'volume_ratio': float(volume / volume_target),
        'neck_radius': neck_radius,
        'max_radius': max_radius,
        'max_radius_z': max_radius_z,
        'total_length': total_length,
        'center_of_mass': com,
        'neck_to_max_ratio': float(neck_radius / max_radius) if max_radius > 0 else 1.0,
    }


# ============================================================
# 快速诊断
# ============================================================

if __name__ == "__main__":
    # 测试几个典型参数组合
    test_cases = [
        ("球形 (Q=0,c=1)", 0.0, 1.0, 0.0, 0.0, 0.0),
        ("鞍点 (Q=0.3,c=0.6)", 0.3, 0.6, 0.0, 0.0, 0.0),
        ("预断点 (Q=0.8,c=0.2)", 0.8, 0.2, 0.0, 0.0, 0.0),
        ("不对称 (Q=0.3,c=0.6,α=0.5)", 0.3, 0.6, 0.5, 0.0, 0.0),
        ("碎片形变 (Q=0.8,c=0.3,ε₁=0.5,ε₂=0.2)", 0.8, 0.3, 0.0, 0.5, 0.2),
    ]

    for name, Q, c, alpha, eps1, eps2 in test_cases:
        m = compute_shape_metrics(Q, c, alpha, eps1, eps2)
        print(f"\n{name}:")
        print(f"  体积比: {m['volume_ratio']:.3f}  "
              f"颈/最大半径比: {m['neck_to_max_ratio']:.3f}  "
              f"全长: {m['total_length']:.1f} fm  "
              f"质心偏移: {m['center_of_mass']:.1f} fm")
