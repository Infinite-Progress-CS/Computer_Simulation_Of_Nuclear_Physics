"""
裂变过程 3D 动画：五维形变空间中随机行走的核形状演化。

功能：
  1. 运行随机行走（Metropolis）获取形变轨迹
  2. 将每步的 (Q,c,α,ε₁,ε₂) 映射为 3D 核表面
  3. 四面板同步动画：
     - 3D 核形状（φ 方向旋转视角）
     - 势能下降曲线 + 当前标记
     - 五参数柱状图
     - 形状诊断量（颈/最大半径比、质心偏移）

用法：
  python fission_animation_3d.py                    # 默认参数
  python fission_animation_3d.py --method langevin  # 朗之万动力学
  python fission_animation_3d.py --n-steps 300      # 自定义步数

作者：高涵
日期：2026-08-12
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，加速渲染
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import animation
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.gridspec as gridspec

# ── CJK 字体配置 ──
_CJK_FONT = None
for _font_name in ['Noto Sans SC', 'SimHei', 'Microsoft YaHei', 'STXihei']:
    try:
        _test = fm.findfont(_font_name, fallback_to_default=False)
        if _test:
            _CJK_FONT = _font_name
            break
    except Exception:
        continue
if _CJK_FONT:
    plt.rcParams['font.family'] = _CJK_FONT
    plt.rcParams['axes.unicode_minus'] = False
    print(f'[Font] Using: {_CJK_FONT}')
else:
    print('[Font] No CJK font found, using default')

# 导入本目录下的模块
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

from nuclear_shape_3d import (
    nuclear_profile, generate_surface_mesh, compute_shape_metrics,
    DEFAULT_R0, DEFAULT_A, R0_PARAM, ShapeParams
)


# ============================================================
# 势能面（与仿真代码一致）
# ============================================================

def potential_5d(Q, c, alpha, eps1, eps2):
    """五维势能面：谷底在 Q=0.8, c=0.4, alpha=0, eps1=0, eps2=0"""
    Q0, c0, a0, e10, e20 = 0.8, 0.4, 0.0, 0.0, 0.0
    V = 2.0 * ((Q - Q0)**2 + (c - c0)**2 + (alpha - a0)**2
               + (eps1 - e10)**2 + (eps2 - e20)**2)
    V += 0.5 * np.exp(-((Q - 0.3)**2 + (c - 0.6)**2) / 0.05)
    return V


# ============================================================
# 随机行走（Metropolis）
# ============================================================

class RandomWalkSimulator:
    """在五维势能面上执行 Metropolis 随机行走，记录完整轨迹。"""

    def __init__(self, temperature=1.0, step_size=0.4,
                 max_steps=500, target_V=0.08):
        self.T = temperature
        self.step_size = step_size
        self.max_steps = max_steps
        self.target_V = target_V

        self.low_bounds = np.array([0.0, 0.0, -1.0, 0.0, 0.0])
        self.high_bounds = np.array([1.5, 1.0, 1.0, 1.0, 1.0])
        self.start_state = np.array([0.3, 0.6, 0.0, 0.0, 0.0])

    def run(self, seed=42) -> dict:
        """执行一次随机行走，返回完整轨迹。"""
        rng = np.random.default_rng(seed)
        state = self.start_state.copy()
        V = potential_5d(*state)
        trajectory = [state.copy()]
        potentials = [V]
        arrived = False
        accepted_count = 0

        for i in range(1, self.max_steps + 1):
            # 随机试探
            action = rng.uniform(-self.step_size, self.step_size, 5)
            new_state = np.clip(state + action, self.low_bounds, self.high_bounds)
            new_V = potential_5d(*new_state)

            # Metropolis 判据
            dV = new_V - V
            if dV < 0 or rng.random() < np.exp(-dV / self.T):
                state = new_state
                V = new_V
                accepted_count += 1

            trajectory.append(state.copy())
            potentials.append(V)

            if V < self.target_V:
                arrived = True
                break

        return {
            'trajectory': np.array(trajectory),
            'potentials': np.array(potentials),
            'arrived': arrived,
            'n_steps': len(trajectory),
            'acceptance_rate': accepted_count / i,
            'final_potential': potentials[-1],
        }


# ============================================================
# 朗之万动力学模拟器
# ============================================================

class LangevinSimulator:
    """
    五维势能面上的过阻尼朗之万动力学。

    离散化方程:
        β(t+dt) = β(t) - (dt/γ)·∇V + √(2kT·dt/γ)·𝒩(0,1)

    参数说明:
        T=0.05: 低温 → 梯度驱动为主，80%成功率
        T=0.1:  中温 → 涨落与梯度竞争，40%成功率
        T=0.5:  高温 → 噪声淹没梯度（论文对比用）
    """

    def __init__(self, friction=0.3, temperature=0.05, dt=0.01,
                 max_steps=300, target_V=0.01):
        self.gamma = friction
        self.T = temperature
        self.dt = dt
        self.max_steps = max_steps
        self.target_V = target_V

        self.low_bounds = np.array([0.0, 0.0, -1.0, 0.0, 0.0])
        self.high_bounds = np.array([1.5, 1.0, 1.0, 1.0, 1.0])
        self.start_state = np.array([0.3, 0.6, 0.0, 0.0, 0.0])

        # 预计算系数
        self.drift_coef = dt / friction
        self.noise_coef = np.sqrt(2.0 * temperature * dt / friction)

    def _gradient(self, state):
        """数值梯度 grad = -∇V（指向下坡）"""
        grad = np.zeros(5)
        delta = 1e-4
        for i in range(5):
            plus = state.copy(); minus = state.copy()
            plus[i] += delta; minus[i] -= delta
            grad[i] = -(potential_5d(*plus) - potential_5d(*minus)) / (2.0 * delta)
        return grad

    def run(self, seed=42) -> dict:
        rng = np.random.default_rng(seed)
        state = self.start_state.copy()
        V = potential_5d(*state)
        trajectory = [state.copy()]
        potentials = [V]
        arrived = False

        for step in range(1, self.max_steps + 1):
            grad = self._gradient(state)
            drift = self.drift_coef * grad
            diffusion = self.noise_coef * rng.normal(0, 1, 5)
            state = np.clip(state + drift + diffusion,
                            self.low_bounds, self.high_bounds)
            V = potential_5d(*state)
            trajectory.append(state.copy())
            potentials.append(V)

            if V < self.target_V:
                arrived = True
                break

        return {
            'trajectory': np.array(trajectory),
            'potentials': np.array(potentials),
            'arrived': arrived,
            'n_steps': len(trajectory),
            'acceptance_rate': 1.0,
            'final_potential': potentials[-1],
        }


# ============================================================
# DRL (SAC) 模拟器
# ============================================================

class DRLSimulator:
    """
    用 SAC (Soft Actor-Critic) 在五维势能面上学习裂变路径。

    训练一个 SAC 智能体后，用 deterministic 策略跑出一条轨迹。
    """

    def __init__(self, total_timesteps=30000, max_steps=200,
                 target_V=0.08, learning_rate=3e-4, seed=42):
        self.total_timesteps = total_timesteps
        self.max_steps = max_steps
        self.target_V = target_V
        self.learning_rate = learning_rate
        self.seed = seed

    def run(self, seed=42) -> dict:
        """训练 SAC 并跑出一条最优轨迹。"""
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_checker import check_env

        # ── 内置环境 ──
        low_bounds = np.array([0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high_bounds = np.array([1.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        start_state = np.array([0.3, 0.6, 0.0, 0.0, 0.0], dtype=np.float32)
        target_V = self.target_V
        max_steps = self.max_steps

        class _FissionEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(
                    low=low_bounds, high=high_bounds, shape=(5,), dtype=np.float32)
                self.action_space = spaces.Box(
                    low=-0.1, high=0.1, shape=(5,), dtype=np.float32)
                self.state = None
                self.steps = 0
                self.prev_V = None

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.state = start_state.copy()
                self.steps = 0
                self.prev_V = potential_5d(*self.state)
                return self.state.copy(), {}

            def step(self, action):
                self.state = np.clip(self.state + action, low_bounds, high_bounds)
                self.steps += 1
                V = potential_5d(*self.state)
                delta_V = self.prev_V - V
                reward = float(delta_V) - 0.01 * V
                self.prev_V = V
                terminated = bool(V < target_V)
                truncated = bool(self.steps >= max_steps)
                return (self.state.copy(), reward, terminated, truncated,
                        {'potential': V, 'delta_V': delta_V})

        env = _FissionEnv()
        check_env(env, warn=False)

        # ── 训练 SAC ──
        model = SAC(
            'MlpPolicy', env,
            learning_rate=self.learning_rate,
            buffer_size=10000,
            learning_starts=500,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            seed=seed,
            verbose=0,
        )
        model.learn(total_timesteps=self.total_timesteps,
                    progress_bar=False)

        # ── 确定性推演 ──
        obs, _ = env.reset(seed=seed)
        trajectory = [obs.copy()]
        potentials = [potential_5d(*obs)]
        arrived = False

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            trajectory.append(obs.copy())
            potentials.append(info['potential'])
            if terminated or truncated:
                arrived = terminated
                break

        return {
            'trajectory': np.array(trajectory),
            'potentials': np.array(potentials),
            'arrived': arrived,
            'n_steps': len(trajectory),
            'acceptance_rate': 1.0,
            'final_potential': potentials[-1],
        }


# ============================================================
# 3D 裂变动画
# ============================================================

class FissionAnimation:
    """
    四面板裂变过程动画。

    Layout (2×2):
      ┌──────────────┬──────────────┐
      │  3D 核形状    │  势能曲线     │
      │  (Axes3D)    │  (2D line)   │
      ├──────────────┼──────────────┤
      │  五参数       │  形状诊断     │
      │  柱状图       │  颈比/质心    │
      └──────────────┴──────────────┘
    """

    def __init__(self, trajectory: np.ndarray, potentials: np.ndarray,
                 R0: float = DEFAULT_R0,
                 n_phi: int = 50, n_z: int = 60,
                 elev: float = 20.0, azim_start: float = -60.0):
        """
        Parameters
        ----------
        trajectory : ndarray (N, 5)  每步的 (Q,c,α,ε₁,ε₂)
        potentials : ndarray (N,)    每步的势能
        R0 : 等效球半径 [fm]
        n_phi, n_z : 形状网格分辨率
        elev, azim_start : 3D 视角
        """
        self.traj = trajectory
        self.potentials = potentials
        self.n_frames = len(trajectory)
        self.R0 = R0
        self.n_phi = n_phi
        self.n_z = n_z
        self.elev = elev
        self.azim_start = azim_start

        # 形状诊断量缓存
        self._shape_metrics_cache = {}

        # 创建画布
        self.fig = plt.figure(figsize=(16, 12), dpi=100)
        self._setup_axes()

    def _setup_axes(self):
        """初始化四个子图。"""
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               width_ratios=[1.0, 0.8],
                               height_ratios=[1.0, 0.7],
                               hspace=0.35, wspace=0.30)

        # 左上：3D 核形状
        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        self.ax_3d.set_facecolor('#1a1a2e')

        # 右上：势能曲线
        self.ax_energy = self.fig.add_subplot(gs[0, 1])
        self.ax_energy.set_facecolor('#f8f8f8')

        # 左下：五参数柱状图
        self.ax_params = self.fig.add_subplot(gs[1, 0])
        self.ax_params.set_facecolor('#f8f8f8')

        # 右下：形状诊断
        self.ax_diag = self.fig.add_subplot(gs[1, 1])
        self.ax_diag.set_facecolor('#f8f8f8')

        self.fig.patch.set_facecolor('white')

    def _get_shape_metrics(self, i: int) -> dict:
        """获取第 i 帧的形状诊断量（带缓存）。"""
        if i not in self._shape_metrics_cache:
            Q, c, alpha, eps1, eps2 = self.traj[i]
            self._shape_metrics_cache[i] = compute_shape_metrics(
                Q, c, alpha, eps1, eps2, self.R0)
        return self._shape_metrics_cache[i]

    def _render_3d_shape(self, i: int):
        """渲染第 i 帧的 3D 核形状。"""
        self.ax_3d.clear()

        Q, c, alpha, eps1, eps2 = self.traj[i]
        X, Y, Z, rho_vals, z_vals = generate_surface_mesh(
            Q, c, alpha, eps1, eps2, self.R0, self.n_z, self.n_phi)

        # 表面着色：用径向坐标 ρ 映射颜色（颈部蓝色→赤道红色）
        rho_on_surface = np.sqrt(X**2 + Y**2)
        norm = colors.Normalize(vmin=0, vmax=self.R0 * 1.2)

        self.ax_3d.plot_surface(
            X, Y, Z,
            facecolors=cm.plasma(norm(rho_on_surface)),
            alpha=0.92, shade=True,
            rstride=1, cstride=1,
            antialiased=True, linewidth=0)

        # 坐标轴与标签
        limit = self.R0 * 2.5
        self.ax_3d.set_xlim(-limit, limit)
        self.ax_3d.set_ylim(-limit, limit)
        self.ax_3d.set_zlim(-limit, limit)
        self.ax_3d.set_xlabel('x [fm]', fontsize=9, labelpad=2)
        self.ax_3d.set_ylabel('y [fm]', fontsize=9, labelpad=2)
        self.ax_3d.set_zlabel('z (裂变轴) [fm]', fontsize=9, labelpad=2)

        # 标题：当前步数 + 势能
        V = self.potentials[i]
        neck_ratio = self._get_shape_metrics(i)['neck_to_max_ratio']
        self.ax_3d.set_title(
            f'Step {i+1}/{self.n_frames}  |  '
            f'V = {V:.3f}  |  neck/max = {neck_ratio:.2f}',
            fontsize=11, fontweight='bold', pad=8)

        # 旋转视角（随时间缓慢旋转）
        azim = self.azim_start + i * 0.8
        self.ax_3d.view_init(elev=self.elev, azim=azim)

        # 网格
        self.ax_3d.xaxis.pane.fill = False
        self.ax_3d.yaxis.pane.fill = False
        self.ax_3d.zaxis.pane.fill = False

    def _render_energy_curve(self, i: int):
        """渲染势能下降曲线 + 当前位置标记。"""
        self.ax_energy.clear()
        steps = np.arange(len(self.potentials))

        # 历史轨迹
        self.ax_energy.plot(steps[:i+1], self.potentials[:i+1],
                            'b-', linewidth=2.0, alpha=0.8, label='V(path)')
        # 后续（灰色虚线）
        if i < len(steps) - 1:
            self.ax_energy.plot(steps[i:], self.potentials[i:],
                                'gray', linewidth=0.8, alpha=0.3, linestyle='--')

        # 当前位置标记
        self.ax_energy.plot(i, self.potentials[i], 'ro', markersize=8,
                            zorder=5, markeredgecolor='darkred', markeredgewidth=1.5)

        # 目标线
        self.ax_energy.axhline(y=0.08, color='green', linestyle=':',
                               linewidth=1.5, alpha=0.7, label='Target (V=0.08)')
        # 起点线
        self.ax_energy.axhline(y=self.potentials[0], color='orange', linestyle='--',
                               linewidth=1.0, alpha=0.5, label=f'Start (V={self.potentials[0]:.2f})')

        self.ax_energy.set_xlabel('Step', fontsize=10)
        self.ax_energy.set_ylabel('Potential V', fontsize=10)
        self.ax_energy.set_title('Potential Energy Along Path', fontsize=11, fontweight='bold')
        self.ax_energy.legend(fontsize=8, loc='upper right')
        self.ax_energy.set_xlim(0, len(steps))
        self.ax_energy.set_ylim(-0.1, max(self.potentials) * 1.15)
        self.ax_energy.grid(True, alpha=0.3, linestyle='--')

    def _render_param_bars(self, i: int):
        """渲染五参数柱状图。"""
        self.ax_params.clear()
        Q, c, alpha, eps1, eps2 = self.traj[i]

        labels = ['Q\n(elongation)', 'c\n(neck)', 'alpha\n(asymmetry)',
                  'eps1\n(L-frag)', 'eps2\n(R-frag)']
        values = [Q, c, alpha, eps1, eps2]
        bounds_low = [0.0, 0.0, -1.0, 0.0, 0.0]
        bounds_high = [1.5, 1.0, 1.0, 1.0, 1.0]

        x = np.arange(len(labels))
        colors_bar = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

        # 归一化到 [0,1] 显示
        values_norm = [(v - lo) / (hi - lo) if hi > lo else 0.5
                       for v, lo, hi in zip(values, bounds_low, bounds_high)]

        bars = self.ax_params.bar(x, values_norm, color=colors_bar, alpha=0.85,
                                  edgecolor='#333333', linewidth=1.2)

        # 在柱上标注实际值
        for j, (bar, val) in enumerate(zip(bars, values)):
            self.ax_params.text(bar.get_x() + bar.get_width()/2.,
                                bar.get_height() + 0.03,
                                f'{val:.2f}',
                                ha='center', va='bottom', fontsize=9, fontweight='bold')

        self.ax_params.set_xticks(x)
        self.ax_params.set_xticklabels(labels, fontsize=9)
        self.ax_params.set_ylim(0, 1.25)
        self.ax_params.set_ylabel('Normalized Value', fontsize=10)
        self.ax_params.set_title('Deformation Parameters', fontsize=11, fontweight='bold')
        self.ax_params.grid(True, alpha=0.3, axis='y', linestyle='--')

    def _render_diagnostics(self, i: int):
        """渲染形状诊断量变化曲线。"""
        self.ax_diag.clear()

        # 计算所有帧的颈/最大半径比和质心偏移（每 5 帧采样以加速）
        sample_step = max(1, self.n_frames // 100)
        sampled_indices = list(range(0, self.n_frames, sample_step))
        if self.n_frames - 1 not in sampled_indices:
            sampled_indices.append(self.n_frames - 1)

        neck_ratios = []
        com_shifts = []
        for idx in sampled_indices:
            m = self._get_shape_metrics(idx)
            neck_ratios.append(m['neck_to_max_ratio'])
            com_shifts.append(abs(m['center_of_mass']))

        sampled_indices = np.array(sampled_indices)
        neck_ratios = np.array(neck_ratios)
        com_shifts = np.array(com_shifts)

        # 颈/最大半径比
        self.ax_diag.plot(sampled_indices, neck_ratios, 'b-', linewidth=2.0,
                          alpha=0.7, label='Neck / Max Radius')

        # 质心偏移
        self.ax_diag.plot(sampled_indices, com_shifts, 'r-', linewidth=2.0,
                          alpha=0.7, label='|COM shift| [fm]')

        # 当前位置竖线
        self.ax_diag.axvline(x=i, color='gray', linestyle='--',
                             linewidth=1.0, alpha=0.5)

        # 裂变标记：neck/max < 0.3 视为接近断点
        self.ax_diag.axhline(y=0.3, color='orange', linestyle=':',
                             linewidth=1.0, alpha=0.6, label='Pre-scission (0.3)')

        self.ax_diag.set_xlabel('Step', fontsize=10)
        self.ax_diag.set_ylabel('Value', fontsize=10)
        self.ax_diag.set_title('Shape Diagnostics', fontsize=11, fontweight='bold')
        self.ax_diag.legend(fontsize=8, loc='best')
        self.ax_diag.set_xlim(0, self.n_frames)
        self.ax_diag.set_ylim(-0.05, 1.3)
        self.ax_diag.grid(True, alpha=0.3, linestyle='--')

    def _update_frame(self, i: int):
        """更新第 i 帧的所有面板。"""
        # 进度打印
        if i % 20 == 0:
            print(f'\r  Rendering frame {i+1}/{self.n_frames}...', end='', flush=True)

        self._render_3d_shape(i)
        self._render_energy_curve(i)
        self._render_param_bars(i)
        self._render_diagnostics(i)

        # 整体标题
        arrived = self.potentials[i] < 0.08
        status = 'ARRIVED! (Fission)' if arrived else 'Evolving...'
        color = '#27ae60' if arrived else '#2c3e50'
        self.fig.suptitle(
            f'Nuclear Fission Path — 5D Deformation Space Random Walk  |  '
            f'{status}',
            fontsize=14, fontweight='bold', color=color, y=0.98)

        return []

    def render(self, output_path: str = 'fission_3d_animation.mp4',
               fps: int = 8, dpi: int = 120,
               frame_step: int = 1,
               max_frames: int = None):
        """
        渲染并保存动画。

        Parameters
        ----------
        output_path : 输出文件路径 (.mp4 或 .gif)
        fps : 帧率
        dpi : 分辨率
        frame_step : 帧采样间隔（>1 可加速渲染）
        max_frames : 最大帧数（None = 全部）
        """
        # 帧范围
        total = min(self.n_frames, max_frames) if max_frames else self.n_frames
        frame_indices = list(range(0, total, frame_step))

        print(f'\nRendering {len(frame_indices)} frames to {output_path}...')
        print(f'  Trajectory: {self.n_frames} steps, '
              f'final V = {self.potentials[-1]:.4f}, '
              f'arrived = {self.potentials[-1] < 0.08}')
        t0 = time.time()

        ani = animation.FuncAnimation(
            self.fig, self._update_frame,
            frames=frame_indices,
            interval=1000 // fps,
            blit=False, repeat=False)

        # 写入文件
        ext = os.path.splitext(output_path)[1].lower()
        if ext == '.gif':
            ani.save(output_path, writer='pillow', fps=fps, dpi=dpi)
        else:
            # mp4
            try:
                ani.save(output_path, writer='ffmpeg', fps=fps, dpi=dpi,
                         extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            except Exception:
                # 降级到 pillow（逐帧保存为 GIF）
                gif_path = output_path.replace('.mp4', '.gif')
                print('  ffmpeg not available, falling back to GIF...')
                ani.save(gif_path, writer='pillow', fps=fps, dpi=dpi)
                output_path = gif_path

        elapsed = time.time() - t0
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f'\n  Saved: {output_path} ({file_size:.1f} MB)')
        print(f'  Render time: {elapsed:.0f}s ({elapsed/len(frame_indices):.1f}s/frame)')

        plt.close(self.fig)
        return output_path


# ============================================================
# 静态快照（论文用图）
# ============================================================

def render_static_snapshots(trajectory: np.ndarray,
                            potentials: np.ndarray,
                            output_dir: str = 'results/snapshots'):
    """
    生成论文用的静态快照：选取几个关键时刻展示 3D 核形状。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 关键时刻
    key_moments = {
        '01_start': 0,
        '02_early': min(len(trajectory) // 4, len(trajectory) - 1),
        '03_mid': min(len(trajectory) // 2, len(trajectory) - 1),
        '04_near_scission': min(len(trajectory) * 3 // 4, len(trajectory) - 1),
        '05_final': len(trajectory) - 1,
    }

    for name, idx in key_moments.items():
        Q, c, alpha, eps1, eps2 = trajectory[idx]
        X, Y, Z, rho_vals, z_vals = generate_surface_mesh(
            Q, c, alpha, eps1, eps2, DEFAULT_R0, n_z=80, n_phi=70)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        rho_on_surface = np.sqrt(X**2 + Y**2)
        norm = colors.Normalize(vmin=0, vmax=DEFAULT_R0 * 1.2)
        ax.plot_surface(X, Y, Z, facecolors=cm.plasma(norm(rho_on_surface)),
                        alpha=0.9, shade=True, linewidth=0, antialiased=True)

        limit = DEFAULT_R0 * 2.5
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.set_xlabel('x [fm]')
        ax.set_ylabel('y [fm]')
        ax.set_zlabel('z [fm]')
        ax.set_title(f'{name}: Step {idx+1}, V={potentials[idx]:.3f}\n'
                     f'Q={Q:.2f} c={c:.2f} a={alpha:.2f} e1={eps1:.2f} e2={eps2:.2f}',
                     fontsize=12)
        ax.view_init(elev=20, azim=-50)

        path = os.path.join(output_dir, f'{name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')

    print(f'\nAll snapshots saved to {output_dir}/')


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='核裂变 3D 动画')
    parser.add_argument('--method', type=str, default='metropolis',
                        choices=['metropolis', 'langevin', 'sac'],
                        help='路径搜索方法')
    parser.add_argument('--n-steps', type=int, default=500,
                        help='最大步数')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Metropolis 温度')
    parser.add_argument('--step-size', type=float, default=0.4,
                        help='Metropolis 步长')
    parser.add_argument('--fps', type=int, default=8,
                        help='动画帧率')
    parser.add_argument('--frame-step', type=int, default=1,
                        help='帧采样间隔')
    parser.add_argument('--max-frames', type=int, default=200,
                        help='最大动画帧数')
    parser.add_argument('--output', type=str, default='fission_3d_animation.mp4',
                        help='输出文件路径')
    parser.add_argument('--snapshots', action='store_true',
                        help='同时生成静态快照')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()

    print("=" * 60)
    print("核裂变 3D 可视化")
    print("=" * 60)

    # ── 1. 执行模拟 ──
    print(f"\n[1] Running {args.method} simulation...")
    if args.method == 'metropolis':
        sim = RandomWalkSimulator(
            temperature=args.temperature,
            step_size=args.step_size,
            max_steps=args.n_steps)
        result = sim.run(seed=args.seed)
    elif args.method == 'langevin':
        sim = LangevinSimulator(
            friction=0.3, temperature=0.05, dt=0.01,
            max_steps=args.n_steps)
        result = sim.run(seed=args.seed)
    elif args.method == 'sac':
        sim = DRLSimulator(
            total_timesteps=30000, max_steps=args.n_steps,
            seed=args.seed)
        result = sim.run(seed=args.seed)
        # SAC 通常几步就到，降低渲染负载
        args.fps = max(args.fps, 10)
        if not args.frame_step:
            args.frame_step = 1
    else:
        raise ValueError(f"Unknown method: {args.method}")

    trajectory = result['trajectory']
    potentials = result['potentials']

    print(f"    Steps: {result['n_steps']}")
    print(f"    Arrived: {result['arrived']}")
    print(f"    Final V: {result['final_potential']:.4f}")
    print(f"    Acceptance: {result['acceptance_rate']:.1%}")

    # ── 2. 保存轨迹数据 ──
    os.makedirs('results', exist_ok=True)
    np.savez('results/trajectory_for_animation.npz',
             trajectory=trajectory,
             potentials=potentials,
             arrived=result['arrived'],
             method=args.method)

    # ── 3. 渲染动画 ──
    print(f"\n[2] Rendering 3D animation...")
    # SAC 轨迹通常较短，降低渲染负载
    render_dpi = 72 if args.method == 'sac' else 100
    max_frames = min(args.max_frames, 60) if args.method == 'sac' else args.max_frames
    anim = FissionAnimation(trajectory, potentials,
                            n_phi=45, n_z=55,
                            elev=22.0, azim_start=-60.0)
    anim.render(
        output_path=args.output,
        fps=args.fps,
        dpi=render_dpi,
        frame_step=args.frame_step,
        max_frames=max_frames)

    # ── 4. 静态快照 ──
    if args.snapshots:
        print(f"\n[3] Generating static snapshots...")
        render_static_snapshots(trajectory, potentials)

    print("\nDone!")
