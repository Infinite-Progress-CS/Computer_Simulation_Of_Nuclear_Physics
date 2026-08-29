"""
metropolis.py — Metropolis 随机行走 / 极小能量路径 / 势能面切片
====================================================================
从单文件版本原样搬出，pes 只需实现 energy_components(q)（返回首元素为总能量），
因此同时兼容纯液滴 FRLDMPES 与宏观-微观 MacroMicro。
"""

import numpy as np


def metropolis_walk(pes, q0, T=6.0, n_steps=500, T_end=None, ratchet_elong=None,
                    step=np.array([0.06, 0.03, 0.03, 0.03, 0.03]),
                    q_min=None, q_max=None, seed=0):
    """
    在五维势能面上做 Metropolis 随机行走。
    p(q→q') = min(1, exp(−ΔV/T))；T 是核激发温度（MeV），T 大 → 易翻越势垒。
    无效形状（build 抛出异常，如深颈+小拉长）直接拒绝。

    T_end 给定时做退火（模拟核越过势垒后冷却、沿谷底滚落到断裂点）：
      - 若同时给出 ratchet_elong：两段式——越障前恒温 T（保证越过势垒），
        一旦 elong 越过 ratchet_elong（棘轮触发）才从 T 线性冷却到 T_end
        （沉降到断裂点）。默认 None = 恒温。
      - 若未给 ratchet_elong：全程线性退火 T → T_end。

    ratchet_elong：一旦 elong 超过该值（越过主势垒），elong 只增不减，
    模拟裂变断裂的不可逆性，强制行走沿谷底滚落到断裂点。
    （3QS 中细颈与碎片分离被解耦，产生一个假的颈部过渡势垒，棘轮用于跨越它。）

    返回：path (n+1,5)、energies (n+1,)、components (n+1,2)=[表面项,库仑项]、接受率。
    """
    rng = np.random.default_rng(seed)
    if q_min is None:
        q_min = np.array([0.1, 0.05, -0.5, -0.2, -0.2])
    if q_max is None:
        q_max = np.array([3.0, 0.99, 0.5, 0.4, 0.4])

    path = np.zeros((n_steps + 1, 5))
    energies = np.zeros(n_steps + 1)
    components = np.zeros((n_steps + 1, 2))

    q = np.array(q0, dtype=float)
    try:
        V, dV_s, dV_c, _, _ = pes.energy_components(q)
    except Exception:
        V, dV_s, dV_c = np.inf, 0.0, 0.0
    path[0], energies[0] = q, V
    components[0] = (dV_s, dV_c)
    n_accept = 0

    committed = False
    i_commit = None
    for i in range(1, n_steps + 1):
        if T_end is None:
            T_i = T
        elif ratchet_elong is not None and committed:
            # 两段式：越障前恒温 T，越障后从 T 冷却到 T_end
            frac = (i - i_commit) / max(n_steps - i_commit, 1)
            T_i = T + (T_end - T) * frac
        else:
            T_i = T + (T_end - T) * (i / n_steps)
        q_new = np.clip(q + step * rng.standard_normal(5), q_min, q_max)
        if committed:
            q_new[0] = max(q_new[0], q[0])   # 裂变不可逆：elong 只增不减
        try:
            V_new, dV_s_new, dV_c_new, _, _ = pes.energy_components(q_new)
        except Exception:
            V_new = np.inf
        dE = V_new - V
        if V_new < np.inf and (dE <= 0 or rng.random() < np.exp(-dE / T_i)):
            q, V, dV_s, dV_c = q_new, V_new, dV_s_new, dV_c_new
            n_accept += 1
        if ratchet_elong is not None and q[0] > ratchet_elong:
            if not committed:
                committed = True
                i_commit = i
        path[i], energies[i] = q, V
        components[i] = (dV_s, dV_c)
        if i % 100 == 0:
            print(f"  步数 {i}/{n_steps}  ΔV={V:+.2f} MeV  "
                  f"(表面 {dV_s:+.2f} / 库仑 {dV_c:+.2f})  接受率 {n_accept / i:.2%}")

    return path, energies, components, n_accept / n_steps


def neck_fraction(q, shape, nz=300):
    """断裂程度 f ∈ [0,1]：f = 1 − ρ_颈/ρ_最大，0=单核(无腰)，1=颈部断开。

    颈部 = 内部局部极小半径（腰），排除两极尖点（半径→0 的端点）。
    无内部极小（光滑长椭球/球）→ 无腰 → f=0。
    """
    z, rho = shape.profile(q, n=nz)
    rho_max = rho.max()
    r_neck = rho_max                    # 默认无腰
    for i in range(1, nz - 1):
        # 用 <= 处理颈恰落在两对称格点之间、两最近点 rho 相等的情形
        if rho[i] <= rho[i - 1] and rho[i] <= rho[i + 1]:
            r_neck = min(r_neck, rho[i])
    return float(np.clip(1.0 - r_neck / rho_max, 0.0, 1.0))


def compute_surface_slice(pes, n=34, elong_lims=(0.3, 3.0), neck_lims=(0.05, 0.99)):
    """势能面切片 V(elong, neck)，eta=eps1=eps2=0（对称裂变路径）。
    无效 (elong, neck) 组合（深颈+小拉长）记为 NaN。"""
    elong = np.linspace(*elong_lims, n)
    neck = np.linspace(*neck_lims, n)
    Q1, Q2 = np.meshgrid(elong, neck)
    V = np.full_like(Q1, np.nan)
    print(f"  正在计算势能面切片 ({n}x{n} = {n * n} 个能量点)...")
    for i in range(n):
        for j in range(n):
            try:
                V[i, j] = pes.energy([Q1[i, j], Q2[i, j], 0.0, 0.0, 0.0])
            except Exception:
                pass
    return Q1, Q2, V


def min_energy_path(pes, elong_lims=(0.3, 3.0), n_elong=55, n_neck=16):
    """确定性极小能量（绝热）裂变路径：在每个 elong 下对 neck 取极小，
    eta=eps1=eps2=0（U-236 裂变参数 x=0.711 > Businaro-Gallone 点 0.396，
    对称裂变能量最低）。这就是随机行走要逼近的"最优路径"。

    返回 (path, energies, components)。
    """
    elong = np.linspace(*elong_lims, n_elong)
    necks = np.linspace(0.05, 0.99, n_neck)
    path = np.zeros((n_elong, 5))
    energies = np.full(n_elong, np.nan)
    components = np.zeros((n_elong, 2))
    for i, el in enumerate(elong):
        best = None
        for nk in necks:
            q = [el, nk, 0.0, 0.0, 0.0]
            try:
                V, dV_s, dV_c, _, _ = pes.energy_components(q)
            except Exception:
                continue
            if best is None or V < best[0]:
                best = (V, dV_s, dV_c, q)
        if best is not None:
            path[i] = best[3]
            energies[i] = best[0]
            components[i] = (best[1], best[2])
    return path, energies, components
