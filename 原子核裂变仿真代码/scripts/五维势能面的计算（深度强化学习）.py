import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import os



# 1. 五维势能面函数
def potential_5d(Q, c, alpha, eps1, eps2):
    """
    五维势能面：一个带“谷底”的抛物面 + 一个小山包
    真实情况下，这里应该从 Möller 数据中插值
    """
    # 谷底位置在 Q=0.8, c=0.4, alpha=0, eps1=0, eps2=0
    Q0, c0, a0, e10, e20 = 0.8, 0.4, 0.0, 0.0, 0.0
    # 抛物面 + 一个小山包（模拟位垒）
    V = 2.0 * ((Q - Q0)**2 + (c - c0)**2 + (alpha - a0)**2 + (eps1 - e10)**2 + (eps2 - e20)**2)     #距离平方和
    """
    乘以 2.0： 这是一个刚度系数，用来控制这个碗的“陡峭程度”。系数越大，碗壁越陡，智能体“滑”向碗底的速度就越快。
    这个部分创造了一个“能量最低点”在谷底的、处处光滑的碗状地形。
    """
    # 加一个小山包在 Q=0.3, c=0.6 处
    V += 0.5 * np.exp(-((Q - 0.3)**2 + (c - 0.6)**2) / 0.05)
    """
    np.exp(...) 是什么？ 这是高斯函数。它在中心点取最大值 1，然后随着距离增加，值迅速衰减到0。它的作用是在一个局部区域“鼓起来”一个包。
    除以 0.05 是为什么？ 这是宽度控制参数。分母越小，这个包的宽度越窄，峰越尖锐；分母越大，峰越平缓。这里 0.05 是一个非常小的数，意味着这个包窄而高，像一个陡峭的小山头。
    乘以 0.5 是为什么？ 这是高度控制参数。它把峰顶的高度限制在 0.5，让这个山包不至于太高，以免完全阻挡智能体下山。
    这个部分在平滑的碗壁上制造了一个局部障碍，增加智能体找到谷底的难度。
    """
    """
    高斯函数（Gaussian function）是一个形如“钟形”的平滑函数，在中心点取最大值，然后向两侧平滑地衰减到零
    它的标准形式是：f(x)=np.exp(-x**2)
    这个形状在自然界的各种随机现象中反复出现，从粒子扩散到测量误差，都离不开它
    """
    return V


# 2. 五维核形变环境
class Fission5DEnv(gym.Env):
    """
    五维核形变路径规划环境
    状态：五维形变参数 (Q, c, alpha, eps1, eps2)
    动作：五维形变增量 (dQ, dc, dalpha, deps1, deps2)
    目标：走到势能面的谷底（即能量最低处）
    终止：到达谷底附近 或 超过最大步数
    """
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # --- 1. 状态空间：五维形变参数 ---
        # 五个参数的物理边界
        self.low_bounds = np.array([0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
        self.high_bounds = np.array([1.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=self.low_bounds,
            high=self.high_bounds,
            shape=(5,),
            dtype=np.float32
        )

        # --- 2. 动作空间：五维形变增量 ---
        # 每一步可以改变的幅度
        self.action_space = spaces.Box(
            low=-0.1,
            high=0.1,
            shape=(5,),
            dtype=np.float32
        )

        # --- 3. 内部状态 ---
        self.state = None          # 当前形变参数
        """
        self.state = None:
        它的作用：这是一个占位符，用来存储原子核当前的形变状态。
        它的类型：在reset()之前，它什么都不指向（None）。在reset()被调用后，它会被赋值为一个包含五个形变参数的NumPy数组，例如 np.array([0.3, 0.6, 0.0, 0.0, 0.0])。
        形象理解：这就像你正在跟踪一个移动的小球，self.state就是它当前时刻在坐标系中的精确位置。
        """
        self.steps = 0
        self.max_steps = 500
        self.target_V = 0.2     # 目标势能阈值（接近谷底）
        """
        self.target_V = 0.01
        它的作用：定义了一个成功的判定标准。当原子核的当前势能 current_V 小于这个值时，我们就认为它“到达谷底”了。
        它的意义：这相当于一个容差阈值。我们不需要势能精确地降到0，只要它足够小（比如 0.01），就可以认为任务成功了。
        形象理解：这就像在靶心周围画了一个小圆圈，只要射中的箭落在这个圆圈内，就算“命中靶心”。
        """
        # 记录每一步的势能，用于计算奖励
        self.prev_potential = None
        """
        它的作用：一个记忆变量，用来存储上一步的势能值。
        它的用途：在计算这一步的奖励时，我们需要知道“这一步和上一步相比，势能是降低了还是升高了？”这个差值（delta_V）正是通过比较current_V和self.prev_potential得到的。
        形象理解：这就像在测量高度变化，你需要记住你刚才站在多高的位置，才能知道现在是上坡了还是下坡了。
        """

    def reset(self, *, seed=None, options=None):
        """重置环境到鞍点附近"""
        # 从鞍点附近出发（Q=0.3, c=0.6, alpha=0, eps1=0, eps2=0）
        self.state = np.array([0.3, 0.6, 0.0, 0.0, 0.0], dtype=np.float32)
        self.steps = 0
        self.prev_potential = potential_5d(*self.state)
        #Python 中的“解包”操作符:把一个可迭代对象（如列表、元组、数组）拆解成单独的元素。
        return self.state, {}

    def step(self, action):
        """执行一步：更新形变，计算奖励，判断终止"""
        # --- 1. 执行动作：更新形变参数，确保不超出边界 ---
        self.state = np.clip(self.state + action, self.low_bounds, self.high_bounds)
        #np.clip的作用：“把数组里所有超出范围的值，强行‘拉’到边界上。”
        self.steps += 1

        # --- 2. 计算当前势能 ---
        current_V = potential_5d(*self.state)

        # --- 3. 计算奖励：势能降低量 = 奖励 ---
        # 如果势能降低了，给正奖励；升高了，给负奖励
        delta_V = self.prev_potential - current_V
        reward = float(delta_V)  # 势能降低越多，奖励越高

        # 额外奖励：如果势能本身很低，给予额外奖励
        reward += -0.1 * current_V

        # 保存当前势能供下一步使用
        self.prev_potential = current_V

        # --- 4. 判断终止条件 ---
        # 终止：势能足够低（到达谷底）
        terminated = bool(current_V < self.target_V)
        # 截断：超时
        truncated = bool(self.steps >= self.max_steps)

        return self.state, float(reward), terminated, truncated, {
            'potential': current_V,
            'delta_V': delta_V
        }


# 3. 测试运行
if __name__ == "__main__":
    # 创建环境
    env = Fission5DEnv()
    check_env(env, warn=True)
    print("环境检查通过！")

    # 创建智能体
    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=0.003,
        buffer_size=10000,
        learning_starts=500,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
    )

    # 训练
    print("\n开始训练...")
    model.learn(total_timesteps=20000)
    print("训练完成！")

    # 测试
    obs, _ = env.reset()
    trajectory = []
    potentials = []

    for i in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(obs.copy())
        potentials.append(info['potential'])
        if terminated or truncated:
            print(f"第 {i+1} 步结束，势能 = {info['potential']:.4f}")
            break

    # 可视化：Q 和 c 的轨迹
    trajectory = np.array(trajectory)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='形变路径')
    plt.plot(0.8, 0.4, 'r*', markersize=15, label='谷底 (Q=0.8, c=0.4)')
    plt.plot(0.3, 0.6, 'go', markersize=10, label='起点 (鞍点)')
    plt.xlim(0, 1.5)
    plt.ylim(0, 1.0)
    plt.xlabel('伸长量 Q')
    plt.ylabel('颈部参数 c')
    plt.title('五维形变路径 (Q-c 投影)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(potentials, 'r-', linewidth=2)
    plt.xlabel('步数')
    plt.ylabel('势能 V')
    plt.title('势能随步数变化')
    plt.grid(True)

    plt.tight_layout()
    """
    plt.tight_layout() 是 Matplotlib 里一个自动调整子图间距的函数。
    它的作用可以总结为一句话：“让所有子图（subplot）都整整齐齐地排列，避免标题、轴标签、图例之间互相重叠或‘打架’。”
    """

    os.makedirs("results/plots", exist_ok=True)  # 如果目录不存在则自动创建
    plt.savefig("results/plots/drl_path_plot.png", dpi=150)
    print("图片已保存至 results/plots/drl_path_plot.png")

    plt.show()


