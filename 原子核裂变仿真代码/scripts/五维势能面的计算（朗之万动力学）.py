"""
五维势能面朗之万动力学采样

作者：高涵
日期：2026-08-10
功能：在五维形变空间中使用朗之万动力学进行路径采样
      输出：路径轨迹、势能变化、统计结果
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os


# 1. 五维势能面
def potential_5d(Q, c, alpha, eps1, eps2):
    """
    五维势能面函数
    谷底位置：Q=0.8, c=0.4, alpha=0, eps1=0, eps2=0
    """
    Q0, c0, a0, e10, e20 = 0.8, 0.4, 0.0, 0.0, 0.0
    # 抛物面
    V = 2.0 * ((Q - Q0)**2 + (c - c0)**2 + (alpha - a0)**2 + (eps1 - e10)**2 + (eps2 - e20)**2)
    # 局部山包
    V += 0.5 * np.exp(-((Q - 0.3)**2 + (c - 0.6)**2) / 0.05)
    return V


# 2. 朗之万动力学采样器
class LangevinSampler:
    """
    五维势能面上的朗之万动力学采样

    参数：
        gamma: 摩擦系数
        T: 温度
        dt: 时间步长
        max_steps: 最大步数
        target_V: 目标势能阈值

    物理模型：
        dq = (grad_V / gamma) * dt + sqrt(2*T*dt/gamma) * noise
        （过阻尼朗之万方程）
    """
    def __init__(self, gamma=1.0, T=0.5, dt=0.01,
                 max_steps=200, target_V=0.01):
        self.gamma = gamma
        self.T = T
        self.dt = dt
        self.max_steps = max_steps
        self.target_V = target_V

        # 物理边界
        self.low_bounds = np.array([0.0, 0.0, -1.0, 0.0, 0.0])
        self.high_bounds = np.array([1.5, 1.0, 1.0, 1.0, 1.0])

    def _compute_gradient(self, state):
        """计算势能梯度（数值微分）"""
        grad = np.zeros(5)
        delta = 1e-4
        for i in range(5):
            q_plus = state.copy()
            q_plus[i] += delta
            q_minus = state.copy()
            q_minus[i] -= delta
            grad[i] = -(potential_5d(*q_plus) - potential_5d(*q_minus)) / (2 * delta)
        return grad

    def _langevin_step(self, state):
        """执行一步朗之万更新"""
        # 1. 计算驱动力（grad = -∇V，指向下坡方向）
        grad = self._compute_gradient(state)

        # 2. 漂移项：-(dt/γ)·∇V  （确定性下坡）
        drift = self.dt * grad / self.gamma

        # 3. 扩散项：√(2kT·dt/γ) · 𝒩(0,1)  （涨落-耗散定理）
        diffusion = np.sqrt(2.0 * self.T * self.dt / self.gamma) * np.random.normal(0, 1, 5)

        # 4. 过阻尼朗之万更新：β(t+dt) = β(t) + drift + diffusion
        new_state = state + drift + diffusion

        # 5. 边界约束
        new_state = np.clip(new_state, self.low_bounds, self.high_bounds)

        return new_state

    def sample(self, n_runs=10):
        """执行多次朗之万采样"""
        results = []

        for run in range(n_runs):
            # 重置到起点
            state = np.array([0.3, 0.6, 0.0, 0.0, 0.0])
            start_V = potential_5d(*state)
            trajectory = [state.copy()]
            potentials = [start_V]
            steps = 0
            success = False

            for _ in range(self.max_steps):
                state = self._langevin_step(state)
                V = potential_5d(*state)
                trajectory.append(state.copy())
                potentials.append(V)
                steps += 1

                if V < self.target_V:
                    success = True
                    break

            results.append({
                'trajectory': np.array(trajectory),
                'potentials': np.array(potentials),
                'final_potential': potentials[-1],
                'steps': steps,
                'success': success,
                'potential_reduction': start_V - potentials[-1]
            })
            """
            将结果用字典结构储存起来
            每次运行结束时，这段代码会把所有结果打包成一个字典，然后放入 results 列表
            """

        return results


# 3. 评估与可视化
def compute_metrics(results):
    """计算统计指标"""
    successes = [r['success'] for r in results]
    """
    等价于：
    successes = []
    for r in results:
    successes.append(r['success'])
    r['success'] 的意思是：从字典 r 中取出键为 'success' 对应的值
    """
    steps = [r['steps'] for r in results]
    final_potentials = [r['final_potential'] for r in results]
    reductions = [r['potential_reduction'] for r in results]

    return {
        'success_rate': np.mean(successes),
        'avg_steps': np.mean(steps),
        'std_steps': np.std(steps),
        'avg_final_potential': np.mean(final_potentials),
        'std_final_potential': np.std(final_potentials),
        'avg_reduction': np.mean(reductions),
        'std_reduction': np.std(reductions),
        'n_runs': len(results)
    }
    #结果返回一个字典

def plot_langevin_results(results, save_path=None):
    """绘制朗之万采样结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 取第一次运行的轨迹
    traj = results[0]['trajectory']
    pots = results[0]['potentials']

    # --- 路径图 (Q-c平面) ---
    ax = axes[0, 0]
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1.5, alpha=0.7, label='朗之万路径')
    ax.plot(0.8, 0.4, 'g*', markersize=12, label='谷底')
    ax.plot(0.3, 0.6, 'ro', markersize=8, label='起点')
    ax.set_xlabel('伸长量 Q')
    ax.set_ylabel('颈部参数 c')
    ax.set_title('朗之万路径 (Q-c投影)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 势能下降曲线 ---
    ax = axes[0, 1]
    ax.plot(pots, 'b-', linewidth=1.5)
    ax.axhline(y=0.01, color='r', linestyle='--', label='目标阈值')
    ax.set_xlabel('步数')
    ax.set_ylabel('势能 V')
    ax.set_title('势能变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 步数分布 ---
    ax = axes[1, 0]
    all_steps = [r['steps'] for r in results]
    ax.hist(all_steps, bins=15, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(all_steps), color='red', linestyle='--', label=f'平均: {np.mean(all_steps):.1f}')
    ax.set_xlabel('步数')
    ax.set_ylabel('频次')
    ax.set_title('到达谷底的步数分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 最终势能分布 ---
    ax = axes[1, 1]
    all_final = [r['final_potential'] for r in results]
    ax.hist(all_final, bins=15, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(all_final), color='red', linestyle='--', label=f'平均: {np.mean(all_final):.3f}')
    ax.set_xlabel('最终势能 V')
    ax.set_ylabel('频次')
    ax.set_title('最终势能分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图片已保存: {save_path}")
    else:
        plt.show()


def print_results(metrics):
    """打印统计结果"""
    print("\n" + "=" * 50)
    print("朗之万动力学采样结果")
    print("=" * 50)
    print(f"  运行次数: {metrics['n_runs']}")
    print(f"  成功率: {metrics['success_rate']:.2f}")
    print(f"  平均步数: {metrics['avg_steps']:.1f} ± {metrics['std_steps']:.1f}")
    print(f"  最终势能: {metrics['avg_final_potential']:.3f} ± {metrics['std_final_potential']:.3f}")
    print(f"  势能降低: {metrics['avg_reduction']:.3f} ± {metrics['std_reduction']:.3f}")


# 4. 主程序入口
if __name__ == "__main__":
    # 创建结果目录
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/data", exist_ok=True)

    print("=" * 60)
    print("五维势能面朗之万动力学采样")
    print("=" * 60)

    # 创建采样器
    sampler = LangevinSampler(
        gamma=0.3,       # 降低摩擦 → 梯度漂移增强
        T=0.05,           # 降低温度 → 热涨落减弱
        dt=0.01,
        max_steps=300,    # 增加最大步数
        target_V=0.01
    )

    # 执行采样
    n_runs = 50
    print(f"\n运行 {n_runs} 次采样...")
    results = sampler.sample(n_runs=n_runs)

    # 计算指标
    metrics = compute_metrics(results)
    print_results(metrics)

    # 保存数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(f"results/data/langevin_results_{timestamp}.npy", results, allow_pickle=True)

    # 绘图
    plot_langevin_results(results, save_path=f"results/plots/langevin_{timestamp}.png")

    print(f"\n数据已保存至 results/data/")
    print("朗之万采样完成！")


