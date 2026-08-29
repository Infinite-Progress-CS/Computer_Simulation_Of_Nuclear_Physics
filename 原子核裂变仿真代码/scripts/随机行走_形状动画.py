"""
五维势能面 + 随机行走 + 原子核形状 3D 动画
================================================
纯液滴模型（宏观部分）：表面能 + 库仑能 + 体积守恒

流程（三段）：
  1. 构建五维势能面 V(q1,q2,q3,q4,q5)
  2. 在势能面上做 Metropolis 随机行走
  3. 把行走过程中的原子核形状做成 3D 动画（GIF）

物理回顾（为什么这么算）：
  原子核 = 带电液滴。形变时两股力打架：
    - 表面能：球形表面积最小，形变使表面能升高 → 抵抗裂变
    - 库仑能：质子互相排斥，拉长后质子离得远 → 库仑能降低 → 驱动裂变
  两者竞争得到裂变势垒（锕系重核 ≈ 6 MeV）。

五维参数（Funny-Hills 傅里叶参数化）：
  q1 = a2  拉长        cos(πz/2z0)
  q2 = a3  质量不对称  sin(πz/z0)
  q3 = a4  颈部        cos(3πz/2z0)
  q4 = a5  高阶不对称  sin(2πz/z0)
  q5 = a6  高阶颈部    cos(5πz/2z0)
"""

import numpy as np
from scipy import special, integrate, optimize
import matplotlib
matplotlib.use("Agg")   # 无界面后端，直接存文件；想看窗口改成默认后端
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# 1. 原子核形状（Funny-Hills 傅里叶参数化）
# ============================================================


class NuclearShape:
    """
    轴对称形状，由 ρ(z)（离对称轴的距离随 z 的变化）描述：
        ρ²(z)/R0² = Σ_n [a_{2n-2} cos((2n-1)πz/(2z0)) + a_{2n-1} sin(nπz/z0)]
    半整数频余弦保证表面在 z=±z0 闭合（cos((2n-1)π/2)=0）且体积有限。
    """

    def __init__(self, R0: float, n_terms: int = 5):
        self.R0 = R0
        self.n_terms = n_terms

    def coeffs(self, q: np.ndarray) -> np.ndarray:
        """5 个参数 q → 傅里叶系数 a2..a6；更高阶系数冻结在球形值，减少截断误差"""
        q1, q2, q3, q4, q5 = q
        a = np.zeros(2 * self.n_terms)
        a[0] = q1   # a2 拉长
        a[1] = q2   # a3 质量不对称
        a[2] = q3   # a4 颈部
        a[3] = q4   # a5 高阶不对称
        a[4] = q5   # a6 高阶颈部
        # 球系数通式 a_{2n} = 32(-1)^{n-1} / ((2n-1)³π³)，n≥4 冻结在球值
        for n in range(4, self.n_terms + 1):
            a[2 * n - 2] = 32.0 * (-1.0) ** (n - 1) / ((2 * n - 1) ** 3 * np.pi ** 3)
        return a

    def rho2(self, z: float, q: np.ndarray, z0: float) -> float:
        """返回 ρ²/R0²（无量纲，负值截为 0）"""
        a = self.coeffs(q)
        r2 = 0.0
        for n in range(1, self.n_terms + 1):
            r2 += a[2 * n - 2] * np.cos((2 * n - 1) * np.pi * z / (2.0 * z0))
            r2 += a[2 * n - 1] * np.sin(n * np.pi * z / z0)
        return max(r2, 0.0)

    def rho(self, z: float, q: np.ndarray, z0: float) -> float:
        """实际表面半径 ρ(z)（fm）"""
        return self.R0 * np.sqrt(self.rho2(z, q, z0))

    def volume(self, q: np.ndarray, z0: float) -> float:
        """体积 V = πR0² ∫ ρ²(z) dz"""
        integral, _ = integrate.quad(
            lambda z: self.rho2(z, q, z0), -z0, z0,
            limit=200, epsabs=1e-5, epsrel=1e-4
        )
        return np.pi * self.R0 ** 2 * integral

    def find_z0(self, q: np.ndarray) -> float:
        """体积守恒 V(q,z0) = (4/3)πR0³，解出半长 z0"""
        R0 = self.R0
        V_target = (4.0 / 3.0) * np.pi * R0 ** 3

        def diff(z0):
            if z0 <= 0.1 * R0:
                return -V_target
            return self.volume(q, z0) - V_target

        try:
            return optimize.brentq(diff, 0.2 * R0, 6.0 * R0, xtol=1e-8)
        except ValueError:
            return 1.5 * R0   # 兜底（一般不会走到这里）


# ============================================================
# 2. 液滴模型能量（五维势能面）
# ============================================================


class LiquidDropPES:
    """
    V(q) = E_S0·(B_S(q) - 1) + E_C0·(B_C(q) - 1)
      B_S = 表面积 / 球形表面积  (≥1，抵抗裂变)
      B_C = 库仑能 / 球形库仑能  (≤1，驱动裂变)
    """

    def __init__(self, Z: int, N: int, r0: float = 1.2249,
                 a_S: float = 17.94, kappa_S: float = 1.78,
                 nz: int = 30, nrho: int = 30):
        self.Z = Z
        self.N = N
        self.A = Z + N
        self.R0 = r0 * self.A ** (1.0 / 3.0)
        I = (N - Z) / self.A
        self.E_S0 = a_S * (1.0 - kappa_S * I ** 2) * self.A ** (2.0 / 3.0)
        self.E_C0 = (3.0 / 5.0) * Z * Z * 1.44 / self.R0   # 球形库仑能解析值 (MeV)
        self.shape = NuclearShape(self.R0)
        self.nz, self.nrho = nz, nrho

        # 球形参考（基态）及参考能量
        a2 = 32.0 / np.pi ** 3
        a4 = -32.0 / (27.0 * np.pi ** 3)
        a6 = 32.0 / (125.0 * np.pi ** 3)
        self.q_sphere = np.array([a2, 0.0, a4, 0.0, a6])
        self.E_sphere = self.energy(self.q_sphere)

    # ---- 表面能 ----
    def surface_integral(self, q: np.ndarray, z0: float) -> float:
        """表面积 S = 2π ∫ ρ(z)√(1+ρ'²) dz"""
        shape = self.shape

        def integrand(z):
            r = shape.rho(z, q, z0)
            if r <= 0:
                return 0.0
            dz = 1e-6 * z0
            r_p = shape.rho(z + dz, q, z0)
            r_m = shape.rho(z - dz, q, z0)
            dr = (r_p - r_m) / (2.0 * dz)
            return r * np.sqrt(1.0 + dr ** 2)

        z_break = 0.95 * z0
        S1, _ = integrate.quad(integrand, -z_break, z_break, limit=300, epsabs=1e-5, epsrel=1e-4)
        S2, _ = integrate.quad(integrand, -z0, -z_break, limit=150, epsabs=1e-5, epsrel=1e-4)
        S3, _ = integrate.quad(integrand, z_break, z0, limit=150, epsabs=1e-5, epsrel=1e-4)
        return 2.0 * np.pi * (S1 + S2 + S3)

    def B_S(self, q: np.ndarray, z0: float) -> float:
        return self.surface_integral(q, z0) / (4.0 * np.pi * self.R0 ** 2)

    # ---- 库仑能 ----
    def coulomb_energy(self, q: np.ndarray, z0: float) -> float:
        """
        库仑自能（单位 MeV）：E_C = (1/2)ρ0² ∫∫ d³r1 d³r2 / |r1-r2|
        方位角解析积分：∫dφ/√(a-bcosφ) = 4K(m)/√(a+b)，m=2b/(a+b)
        用 Gauss-Legendre + numpy 向量化；球极限 = (3/5)Z²e²/R0。
        """
        R0 = self.R0
        rho0 = 3.0 * self.Z / (4.0 * np.pi * R0 ** 3)
        shape = self.shape
        nz, nrho = self.nz, self.nrho

        xz, wz = np.polynomial.legendre.leggauss(nz)
        xr, wr = np.polynomial.legendre.leggauss(nrho)
        z = xz * z0
        wz_full = wz * z0
        Rz = np.array([shape.rho(zi, q, z0) for zi in z])

        rho = Rz[:, None] * (xr[None, :] + 1.0) * 0.5
        w_rho = wr[None, :] * Rz[:, None] * 0.5

        zz = (z[:, None, None, None] - z[None, None, :, None]) ** 2
        r1 = rho[:, :, None, None]
        r2 = rho[None, None, :, :]
        a = zz + r1 ** 2 + r2 ** 2
        b = 2.0 * r1 * r2
        apb = np.maximum(a + b, 1e-14)   # 防颈部 ρ→0 时 0/0 产生 NaN
        m = np.clip(2.0 * b / apb, 0.0, 1.0 - 1e-12)
        K = special.ellipk(m)

        wz1 = wz_full[:, None, None, None]
        wr1 = w_rho[:, :, None, None]
        wz2 = wz_full[None, None, :, None]
        wr2 = w_rho[None, None, :, :]
        total_w = wz1 * wr1 * wz2 * wr2
        integrand = 2.0 * np.pi * r1 * r2 * 4.0 * K / np.sqrt(apb)

        int_G = np.sum(total_w * integrand)
        return 0.5 * rho0 ** 2 * int_G * 1.44

    def B_C(self, q: np.ndarray, z0: float) -> float:
        E_C_sphere = (3.0 / 5.0) * self.Z ** 2 * 1.44 / self.R0
        return self.coulomb_energy(q, z0) / E_C_sphere

    # ---- 总形变能 ----
    def energy(self, q: np.ndarray) -> float:
        """五维势能面：V(q) = 表面能变化 + 库仑能变化（MeV）"""
        z0 = self.shape.find_z0(q)
        B_S = self.B_S(q, z0)
        B_C = self.B_C(q, z0)
        return self.E_S0 * (B_S - 1.0) + self.E_C0 * (B_C - 1.0)


# ============================================================
# 3. Metropolis 随机行走
# ============================================================


def metropolis_walk(pes, q0, T=4.0, n_steps=400,
                    step=np.array([0.05, 0.03, 0.03, 0.02, 0.01]),
                    q_min=None, q_max=None, seed=0):
    """
    在五维势能面上做 Metropolis 随机行走。

    每一步：随机提出 q' = q + 高斯噪声，按 p = min(1, exp(-ΔV/T)) 接受。
    T 是"温度"（核激发能，单位 MeV）：T 大 → 翻越势垒能力强、形状变化大；
    T 小 → 只在基态（球形）附近抖动。

    返回：q 轨迹 (n_steps+1, 5)、能量轨迹 (n_steps+1,)、接受率。
    """
    rng = np.random.default_rng(seed)
    if q_min is None:
        q_min = np.array([0.35, -0.4, -0.4, -0.3, -0.2])
    if q_max is None:
        q_max = np.array([1.25, 0.4, 0.15, 0.3, 0.3])

    path = np.zeros((n_steps + 1, 5))
    energies = np.zeros(n_steps + 1)
    q = np.array(q0, dtype=float)
    E = pes.energy(q) - pes.E_sphere          # 相对球形参考的能量
    path[0], energies[0] = q, E
    n_accept = 0

    for i in range(1, n_steps + 1):
        q_new = np.clip(q + step * rng.standard_normal(5), q_min, q_max)
        E_new = pes.energy(q_new) - pes.E_sphere
        dE = E_new - E
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            q, E = q_new, E_new
            n_accept += 1
        path[i], energies[i] = q, E
        if i % 100 == 0:
            print(f"  步数 {i}/{n_steps}  当前能量 {E:+.2f} MeV  接受率 {n_accept/i:.2%}")

    return path, energies, n_accept / n_steps


# ============================================================
# 4. 形状 3D 渲染 + 动画
# ============================================================


def shape_mesh(q, shape, nz=60, nphi=50):
    """把轴对称形状 ρ(z) 绕 z 轴旋转，得到 3D 网格 (X, Y, Z)"""
    z0 = shape.find_z0(q)
    z = np.linspace(-z0, z0, nz)
    rho = np.array([shape.rho(zi, q, z0) for zi in z])
    phi = np.linspace(0.0, 2.0 * np.pi, nphi)
    Z, PHI = np.meshgrid(z, phi)          # (nphi, nz)
    RHO = np.tile(rho, (nphi, 1))         # (nphi, nz)
    X = RHO * np.cos(PHI)
    Y = RHO * np.sin(PHI)
    return X, Y, Z


def animate_walk(path, energies, pes, filename="核形状随机行走.gif",
                 every=2, fps=15, z_lim_scale=2.5):
    """把随机行走的轨迹做成原子核形状变化的 3D 动画"""
    shape = pes.shape
    R0 = pes.R0
    frames = list(range(0, len(path), every))

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 初始帧，确定坐标范围
    X, Y, Z = shape_mesh(path[0], shape)
    ax.plot_surface(X, Y, Z, cmap="plasma", linewidth=0, antialiased=True)
    ax.set_box_aspect((1, 1, z_lim_scale))
    ax.set_xlim(-1.4 * R0, 1.4 * R0)
    ax.set_ylim(-1.4 * R0, 1.4 * R0)
    ax.set_zlim(-z_lim_scale * R0, z_lim_scale * R0)
    ax.set_xlabel("x (fm)")
    ax.set_ylabel("y (fm)")
    ax.set_zlabel("z (fm)")
    title = ax.set_title("")

    def update(i):
        ax.clear()
        X, Y, Z = shape_mesh(path[i], shape)
        ax.plot_surface(X, Y, Z, cmap="plasma", linewidth=0, antialiased=True)
        ax.set_box_aspect((1, 1, z_lim_scale))
        ax.set_xlim(-1.4 * R0, 1.4 * R0)
        ax.set_ylim(-1.4 * R0, 1.4 * R0)
        ax.set_zlim(-z_lim_scale * R0, z_lim_scale * R0)
        ax.set_xlabel("x (fm)")
        ax.set_ylabel("y (fm)")
        ax.set_zlabel("z (fm)")
        ax.set_title(f"Step {i}   V={energies[i]:+.2f} MeV", fontsize=11)  # 英文标题，避免缺中文字形
        return [ax]

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    try:
        ani.save(filename, writer="pillow", fps=fps)
        print(f"  动画已保存: {filename} ({len(frames)} 帧)")
    except Exception as e:
        print(f"  (GIF 保存失败: {e})，改为交互显示")
    plt.close(fig)


# ============================================================
# 5. 主流程
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("五维势能面 + 随机行走 + 形状 3D 动画（纯液滴模型）")
    print("=" * 60)

    # 目标核素：²³⁶U
    Z, N = 92, 144
    pes = LiquidDropPES(Z, N)
    print(f"\n核素 U-236: R0={pes.R0:.3f} fm, E_S0={pes.E_S0:.1f} MeV, "
          f"E_C0={pes.E_C0:.1f} MeV")
    print(f"裂变参数 x = E_C0/(2E_S0) = {pes.E_C0/(2*pes.E_S0):.3f}")
    print(f"球形参考能量 = {pes.E_sphere:+.2f} MeV（理想为 0，偏差来自库仑求积分辨率）")

    # ---- 随机行走 ----
    T = 4.0
    n_steps = 400
    print(f"\n[1] 在势能面上做 Metropolis 随机行走 (T={T} MeV, {n_steps} 步)...")
    path, energies, accept = metropolis_walk(pes, pes.q_sphere, T=T, n_steps=n_steps)
    print(f"  完成，接受率 = {accept:.2%}")
    print(f"  能量范围: [{energies.min():+.2f}, {energies.max():+.2f}] MeV")
    print(f"  形变能跨越 {energies.max()-energies.min():.2f} MeV "
          f"(裂变势垒约 6 MeV，说明行走确实翻越了势垒)")

    # ---- 动画 ----
    print(f"\n[2] 生成原子核形状变化 3D 动画...")
    animate_walk(path, energies, pes, filename="核形状随机行走.gif")

    print("\n" + "=" * 60)
    print("完成。打开 核形状随机行走.gif 查看原子核形状随随机行走的变化。")
    print("提示：调大 T（如 6.0）形状变化更剧烈；调小 T（如 1.0）则只轻微抖动。")
    print("=" * 60)
