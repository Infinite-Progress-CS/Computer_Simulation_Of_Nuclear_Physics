"""
random_walk_yield.py — Brownian 形状运动产额（Randrup-Möller 2011 PRL 106,132503）
====================================================================================
在强阻尼（Smoluchowski）极限下，核形状演化等价于在 5D 势能面上的随机行走。
产额由「行走到断裂点时的质量不对称 η 直方图」给出：

  Y(A_L, A_H) ∝ （行走到断裂点的次数）

关键物理（严格照 RMS 2011 / Möller-Randrup 2015）：
  1. Metropolis 判据 P(i→i'):P(i'→i) = exp(−ΔV_biased / T)
  2. 局域温度  T² = [E* − V(q)] / aA，aA = A/(8 MeV)；V 相对基态
  3. 偏置势    V_bias = V0·(Q0/Q)²   （Q = 四极矩，加速越障，不重加权）
  4. 断裂判据  颈半径 c_neck ≤ c0 = 2.5 fm 时冻结质量不对称 η

质量不对称 η = (M_H−M_L)/(M_H+M_L) 在碎片质量数 A_f 上等距
（A_L = A_parent·(1−η)/2），故提取质量分布无需 Jacobian 修正（与文献一致）。

电荷产额用 UCD 标度：P(Z_f) = P(A_f)·A0/Z0（文献 Fig.1 口径）。
"""
import numpy as np


# ---- 物理常数 ----
E2 = 1.44          # e²（MeV·fm）


def neck_radius_fm(q, shape, n=300):
    """物理颈半径（fm）：形状剖面 ρ(z) 的内部局部极小半径（排除两极 ρ→0 端点）。

    无腰（紧凑/球）→ 无内部极小 → 返回 ρ_max（≈满半径 ~7 fm）。
    有腰（双碎片）→ 返回腰部最小半径。断裂判据：≤ c0。
    """
    z, rho = shape.profile(q, n=n)   # rho 单位 fm
    rho_max = rho.max()
    r_neck = rho_max
    for i in range(1, n - 1):
        if rho[i] <= rho[i - 1] and rho[i] <= rho[i + 1]:
            r_neck = min(r_neck, rho[i])
    return float(r_neck)


def quadrupole_moment(q, shape, n=300):
    """轴对称形状的四极矩 Q20 ∝ ∫ (2z² − ρ²) dV = 2π∫[z²ρ² − ρ⁴/4] dz（fm⁵）。

    球 → Q20 = 0；长椭球（prolate，沿 z 拉长）→ Q20 > 0；扁椭球（oblate）→ Q20 < 0。
    用于偏置势 V0(Q0/Q)²（Q0=基态四极矩，随拉长 Q 单调增大 → 偏置从 V0 平滑降到 0）。
    """
    z, rho = shape.profile(q, n=n)   # fm
    integrand = z ** 2 * rho ** 2 - rho ** 4 / 4.0
    return float(2.0 * np.pi * np.trapz(integrand, z))


def local_temperature(V_rel, E_star, A):
    """局域核温度 T = √(max(0, E* − V_rel)/aA)，V_rel 相对基态（MeV）。

    aA = A/(8 MeV)。返回 MeV。
    """
    aA = A / 8.0
    dE = E_star - V_rel
    if dE <= 0.0:
        return 0.0
    return float(np.sqrt(dE / aA))


def find_ground_state(pes, elong_lims=(0.3, 1.0), n_elong=8, necks=(0.9, 0.99),
                      etas=(0.0,), eps=(0.0, 0.1, 0.2, 0.25, 0.3)):
    """在紧致区域粗扫求基态（最低能量点），返回 (q_gs, V_gs)。

    用于设定起点与能量基准 E0 = E* + V_gs（局域温度）。基态对产额不敏感
    （RMS 2015：从基态/同质异能态出发产额几乎相同），故粗扫即可。
    """
    best_q, best_V = None, np.inf
    for el in np.linspace(*elong_lims, n_elong):
        for nk in necks:
            for eta in etas:
                for e in eps:
                    q = [el, nk, eta, e, e]
                    try:
                        V = pes.energy(q)
                    except Exception:
                        continue
                    if V < best_V:
                        best_V, best_q = V, q
    return np.asarray(best_q, dtype=float), float(best_V)


def _propose(q, rng, step, q_min, q_max):
    """高斯随机步 + 边界裁剪。返回 q_new。"""
    q_new = q + step * rng.standard_normal(q.shape[0])
    return np.clip(q_new, q_min, q_max)


def brownian_yield(pes, shape, A_parent=236, Z_parent=92, E_star=6.54,
                   V0=60.0, c0=2.5, n_walks=200, max_steps=600,
                   q0=None, q_min=None, q_max=None,
                   step=np.array([0.05, 0.04, 0.03, 0.03, 0.03]),
                   seed=0, verbose=False):
    """Brownian 形状运动产额：N 条行走 → 断裂点 η 直方图。

    返回 dict：
      A          : 整数质量数数组（70..166 之类）
      Y_A        : 质量产额百分数（Σ=200%）
      eta_scission: 每条成功行走到断裂点的 η 数组
      n_scission : 成功到达断裂的行走数
      acceptance : 平均接受率

    参数：
      E_star : 复合核激发能（MeV）。U-236 热中子 = 6.54（RMS Fig.1c）
      V0     : 偏置势强度（MeV）。RMS 2011 用 15，2015 用 60（差异不显著）
      c0     : 断裂颈半径（fm）。RMS 用 2.5，结果对其不敏感
    """
    rng = np.random.default_rng(seed)
    if q_min is None:
        q_min = np.array([0.2, 0.05, -0.5, -0.2, -0.2])
    if q_max is None:
        q_max = np.array([3.2, 0.99, 0.5, 0.4, 0.4])

    # 基态（起点 + 能量基准）
    if q0 is None:
        q0, V_gs = find_ground_state(pes)
    else:
        q0 = np.asarray(q0, dtype=float)
        V_gs = pes.energy(q0)
    E0 = E_star + V_gs               # 相对球液滴基准的总激发
    Q0 = max(quadrupole_moment(q0, shape), 1e-3)
    Q_floor = max(Q0 * 0.3, 1e-3)    # 防止 Q→0 时偏置发散

    A_edges = np.arange(0.5, A_parent + 0.5, 1.0)   # 0..A_parent 整数边界
    hist = np.zeros(A_parent, dtype=float)

    eta_scission = []
    n_scission = 0
    n_accept = 0
    n_evals = 0

    for w in range(n_walks):
        q = q0.copy()
        V = pes.energy(q)
        Vb = V0 * (Q0 / max(quadrupole_moment(q, shape), Q_floor)) ** 2
        for s in range(max_steps):
            q_new = _propose(q, rng, step, q_min, q_max)
            try:
                V_new = pes.energy(q_new)
            except Exception:
                # 3QS 非对称形状在小 elong（碎片未分离）无解 → 退回对称投影
                # （保留 elong/neck，η=ε1=ε2=0）再试；仍无解才拒绝。
                q_sym = q_new.copy()
                q_sym[2:] = 0.0
                try:
                    V_new = pes.energy(q_sym)
                    q_new = q_sym
                except Exception:
                    V_new = np.inf
            if V_new < np.inf:
                Vb_new = V0 * (Q0 / max(quadrupole_moment(q_new, shape), Q_floor)) ** 2
                dE = (V_new + Vb_new) - (V + Vb)
                T = local_temperature(V - V_gs, E_star, A_parent)
                if dE <= 0.0 or rng.random() < np.exp(-dE / max(T, 1e-6)):
                    q, V, Vb = q_new, V_new, Vb_new
                    n_accept += 1
                n_evals += 1

            # 断裂判据
            if neck_radius_fm(q, shape) <= c0:
                eta = float(q[2])
                A_L = A_parent * (1.0 - abs(eta)) / 2.0
                # 两个碎片各计 1（A_L 与 A_H = A_parent − A_L）
                iL = int(np.floor(A_L))
                if 0 <= iL < A_parent:
                    hist[iL] += 1.0
                    hist[A_parent - 1 - iL] += 1.0
                eta_scission.append(eta)
                n_scission += 1
                if verbose:
                    print(f"  [walk {w}/{n_walks}] 断裂 step={s}  η={eta:+.3f}  "
                          f"A_L={A_L:.0f}", flush=True)
                break
        else:
            if verbose:
                print(f"  [walk {w}/{n_walks}] 未断裂 末 elong={q[0]:.2f} "
                      f"neck={q[1]:.2f} η={q[2]:+.2f}", flush=True)

    acceptance = n_accept / max(n_evals, 1)
    # 归一化到 200%（每裂变两个碎片）
    total = hist.sum()
    Y_A = 200.0 * hist / total if total > 0 else hist

    A = np.arange(A_parent, dtype=float)   # 0..235（碎片质量数）
    return dict(A=A, Y_A=Y_A, eta_scission=np.asarray(eta_scission),
                n_scission=n_scission, acceptance=acceptance,
                V_gs=V_gs, q0=q0)


def mass_to_charge_yield(A, Y_A, Z_parent=92, A_parent=236):
    """质量产额 Y(A) → 电荷产额 Y(Z_f)（UCD 标度 P(Z_f)=P(A_f)·A0/Z0）。

    返回 (Z 数组, Y_Z 数组)。Z = A·Z0/A0，Y_Z = Y_A·A0/Z0（面积守恒 Σ=200%）。
    """
    scale = A_parent / Z_parent           # dA/dZ
    Z = A * (Z_parent / A_parent)
    Y_Z = Y_A * scale
    return Z, Y_Z
