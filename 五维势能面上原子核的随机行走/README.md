# 五维势能面上原子核的随机行走（宏观-微观模型）

基于宏观-微观模型（Macroscopic–Microscopic Model）的重核裂变 / 合成最优路径预测，
目标核素 **U-236**（Z=92, N=144）。核心是把核形变势能写成液滴项加量子壳层/对修正：

```
E(q) = E_液滴(q) + δE_壳(q) + δE_对(q)
```

五维形状参数为 3QS 三二次曲面坐标 `q = (elong 拉长, neck 颈部, eta 质量不对称, eps1 左碎片形变, eps2 右碎片形变)`。
在五维势能面上做 Metropolis 随机行走、找极小能量（绝热）裂变路径，并生成核形状演化的 3D 分屏动画。

## 目录结构

```
五维势能面上原子核的随机行走/
├── src/          # 物理模块（形状 / 液滴 / 单粒子 / 壳修正 / 对修正 / 随机行走 / 绘图）
├── scripts/      # 入口脚本（4 个，各自跑一个完整流程）
├── output/       # 输出结果（静态图 + GIF 动画）
├── docs/         # 详细文档（模型说明 / 结果说明 / 公式说明）
└── README.md     # 本文件
```

## 模块一览（src/）

| 文件 | 内容 |
|------|------|
| `shape.py` | `Shape3QS` 三二次曲面形状参数化 |
| `liquid_drop.py` | `FRLDMPES` 有限力程液滴模型（表面 + 库仑） |
| `woods_saxon.py` | `WoodsSaxon` 变形 Woods-Saxon 单粒子能级 |
| `strutinsky.py` | Strutinsky 壳修正 |
| `pairing.py` | BCS 对修正 |
| `macro_micro.py` | `MacroMicro` 整合 `E = E_液滴 + δE_壳 + δE_对` |
| `metropolis.py` | Metropolis 随机行走 / 极小能量路径 / 势能面切片 |
| `visualize.py` | 核形状示意图 / 势能面切片 / 分屏 3D 动画 |

## 运行

> Windows 控制台请带 `-X utf8` 以正确处理中文输出；脚本会自动把 `src/` 加入模块搜索路径、
> 把结果写进 `output/`。

```bash
cd scripts

python -X utf8 symmetric_path_validation.py            # 二维对称路径验证（幻数壳隙 / 壳修正 / 对称势垒 / η 扫描）
python -X utf8 epsilon_relaxation.py      # ε 形变弛豫（真基态 + 双峰势垒）
python -X utf8 run_walk.py        # 五维随机行走 + 分屏动画（对称）
python -X utf8 run_walk_asym.py   # 非对称裂变路径 + 分屏动画（碎片一大一小）
```

## 输出结果（output/）

- `对称路径宏观微观势垒.png` — 纯液滴 vs 宏观-微观势垒对比（二维对称路径验证）
- `eta扫描非对称裂变.png` — 固定 elong 扫 η，验证非对称裂变（η≈0.19 更稳）
- `双峰裂变势垒.png` — ε 弛豫后的双峰势垒（ε 形变弛豫）
- `五参数形状示意图.png` — 5 个形状参数各自对核形状的影响
- `势能面切片.png` / `非对称势能面切片.png` — 2D 势能面 + 行走轨迹 + 极小能量路径
- `核形状随机行走.gif` / `核形状非对称裂变.gif` — 3D 分屏动画（左核形状 / 右势能面+轨迹）
- `裂变碎片产额分布.png` — U-236 裂变碎片质量产额分布（双峰不对称，实验特征示意）

各输出文件的含义、关键数值见 [docs/结果说明.md](docs/结果说明.md)；
模型原理、参数、各阶段判据见 [docs/模型说明.md](docs/模型说明.md)；
**代码中用到的全部数学物理公式（含推导说明、参数值、代码位置对照）见 [docs/公式说明.md](docs/公式说明.md)**；
裂变碎片产额分布（实验特征 + 壳效应解释 + 与 η≈0.19 的对应）见 [docs/产额分布说明.md](docs/产额分布说明.md)；
研究总览 + 22 篇经典基石论文 + 产额计算路线图见 [docs/文献与路线图.md](docs/文献与路线图.md)；
断裂点模型产额实验（模型/公式/文献/交付物）见 [docs/产额实验总结.md](docs/产额实验总结.md)、
结果与已知局限见 [docs/产额计算结果.md](docs/产额计算结果.md)；
**非对称裂变复现与排查（本轮实验：双中心壳修正 + 峰位偏 3 根因定位）见 [docs/非对称裂变实验总结.md](docs/非对称裂变实验总结.md)**。

---

## 相对上次汇报（静态断裂点模型）的新增内容

> 上次向教授汇报的基线 = **断裂点模型产额**（git commit `709884b`）。
> 本轮（2026-09-03 ~ 09-05）在物理模型、产额方法与代码文件上做了以下扩展。

### 一、新增物理 / 数学公式

1. **双中心壳修正**（`src/hybrid_pes.py`）——断裂区量子修正从「满核单中心」切换到「碎片级双中心」：
   ```
   δE_sh(q) = (1−w)·δE_sh_full(q) + w·δE_sh_frag(q)
   w = (neck_hi − neck) / (neck_hi − neck_lo)        # 默认 neck_hi=0.5, neck_lo=0.15
   碎片分解： A_L = round(A(1−η)/2),  A_H = A − A_L,  Z_L = round(Z·A_L/A)   # UCD
   断裂区（neck ≤ neck_lo）δE_pair → 0（配对坍缩）
   ```

2. **Brownian 形状运动产额**（`src/random_walk_yield.py`，Randrup–Möller 2011 PRL 106, 132503）：
   ```
   Metropolis 判据：  P(i→i′) : P(i′→i) = exp(−ΔV/T)
   局域温度：       T² = [E* − V(q)] / aA ,   aA = A/(8 MeV)
   偏置势：         V_bias = V0·(Q0/Q)²      # Q = 四极矩，加速越障
   断裂判据：       颈半径 c_neck ≤ c0 = 2.5 fm（冻结质量不对称 η）
   四极矩：         Q20 = 2π∫(z²ρ² − ρ⁴/4) dz
   UCD 电荷标度：   P(Z_f) = P(A_f)·A0/Z0
   ```

3. **库仑 Richardson 外推**（`src/liquid_drop.py`，修库仑 4 重积分 ~1/n² 慢收敛）：
   ```
   B_c(∞) = B_c(n_hi) + (B_c(n_hi) − B_c(n_lo)) / ((n_hi/n_lo)² − 1)
   ```

4. **高斯电荷弥散**（`scripts/plot_charge_yield.py`、`scripts/_diag_gauss_disp.py`）：
   ```
   Y(Z) = Σ_A Y(A)·exp(−(Z − Z̄(A))² / 2σ_Z²),   σ_Z = 0.75
   ```

5. **Strutinsky 奇偶交错修复**（`src/strutinsky.py`，部分占据替代银行家舍入）：
   ```
   n_full = n // degen,   n_extra = n % degen
   ```

### 二、新增模型

| 模型 | 文件 | 说明 |
|------|------|------|
| `HybridMacroMicro` | `src/hybrid_pes.py` | 满核液滴 + 断裂区碎片级双中心壳修正（复现非对称裂变） |
| `CachedHybridMacroMicro` | `src/hybrid_pes.py` | 上述 + 磁盘缓存（`DiskCache`） |
| `brownian_yield` | `src/random_walk_yield.py` | Randrup–Möller 2011 随机行走产额 |

### 三、新增 / 修改 Python 文件

**新增 src/（3 个）**：`hybrid_pes.py`、`random_walk_yield.py`、`cached_pes.py`

**修改 src/（4 个）**：
- `liquid_drop.py` — 库仑 Richardson 外推
- `metropolis.py` — 对称投影回退 + 地板棘轮 + 指数退火
- `shape.py` — 3QS 非对称形状无解修复
- `strutinsky.py` — 奇偶交错 bug 修复（部分占据）

**新增 scripts/ 入口与绘图（12 个）**：
- 5D 路径/行走：`run_5d.py`、`run_5d_hybrid.py`、`run_5d_walk.py`、`run_5d_walk_hybrid.py`、`run_walk_yield_hybrid.py`
- 绘图：`plot_walk_result.py`、`plot_walk_result_hybrid.py`、`plot_charge_yield.py`
- 数据：`parse_nfpy.py`（ENDF/IAEA 解析）、`_regen_charge.py`、`_regen_charge_scan.py`

**新增 scripts/ 诊断脚本（`_diag_*.py`，45 个）**：
```
势垒/基态   : _diag_barrier.py  _diag_barrier_fine.py  _diag_mm.py  _diag_mm_barrier.py
              _diag_gs.py  _diag_bound.py  _diag_saddle_eps.py
形状/颈     : _diag_shape.py  _diag_shape2.py  _diag_shape3.py  _diag_neck.py
              _diag_neck_eta.py  _diag_elong_map.py  _diag_spheroid_match.py
库仑/液滴   : _diag_coulomb_conv.py  _diag_richardson.py  _diag_frldm.py  _diag_kns.py
              _diag_legendre.py  _diag_legendre2.py  _diag_form.py  _diag_plateau.py
单粒子/壳   : _diag_sp.py  _diag_shell.py  _diag_shell_pb.py  _diag_shell_scale.py
              _diag_shell_breakdown.py  _diag_spherical_vs_deformed.py
              _diag_proton_shell.py  _diag_heavy_deform.py  _diag_sh_path.py
断裂/非对称 : _diag_scission.py  _diag_scission_eta.py  _diag_eta_scan.py
              _diag_eps_asym.py  _diag_frag_asym.py  _diag_hybrid.py
行走        : _diag_walk.py  _diag_bias_walk.py
电荷产额    : _diag_charge_T.py  _diag_gauss_disp.py  _diag_gauss_disp2.py
              _diag_ucd_fixed_eps.py  _diag_ucd_eps_gauss.py
测试/其它   : _bench.py  _smoke.py  _smoke2.py  _diag_calib.py
```

**新增 docs/**：`docs/非对称裂变实验总结.md`

**新增 results/**：
- `results/产额分布/` — 电荷产额数据（`computed_charge_yield_T4.50.csv` 等）与 `figures/电荷产额分布_Y(Z).png`
- `results/随机行走/` — 5D 行走轨迹（`walk_5d*.npz`）与能量对比图（`walk_vs_path_energy.png`）
