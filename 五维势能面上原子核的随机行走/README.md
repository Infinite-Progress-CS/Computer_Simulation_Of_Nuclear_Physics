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
研究总览 + 22 篇经典基石论文 + 产额计算路线图见 [docs/文献与路线图.md](docs/文献与路线图.md)。
