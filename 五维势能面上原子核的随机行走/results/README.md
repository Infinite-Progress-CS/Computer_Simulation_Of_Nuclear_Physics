# results/ 交付物目录说明

本目录集中存放两个实验的**可交付数据与图表**，数据（CSV）与图表（PNG/GIF）分开、两次实验分开。

```
results/
├── 产额分布/                  ← 本次实验：U-236 裂变碎片产额分布（断裂点模型）
│   ├── data/                  ← 处理后的数据 CSV
│   │   ├── ENDF_U235_mass_chain_yield.csv      实验质量链产额
│   │   ├── ENDF_U235_cumulative_yield.csv      实验累计产额（含 A,Z）
│   │   ├── computed_mass_yield_T{1.0~3.0}.csv  多温度计算质量产额 Y(A)
│   │   ├── computed_charge_yield.csv           计算电荷产额 Y(Z)
│   │   ├── most_probable_charge.csv            最可几电荷 Z_p(A)
│   │   └── raw/                ← 原始 ENDF 下载（.dat/.zip，共 ~53MB，可重新抓取，可删）
│   └── figures/               ← 产额图（7 张 PNG）
│       ├── mass_yield_multiT.png        多温度 Y(A) 曲线
│       ├── charge_yield.png             电荷产额 Y(Z)
│       ├── shell_correction_vs_A.png    壳修正随 A（诊断双峰来源）
│       ├── compare_mass_yield.png       计算 vs 实验叠图（线性轴）
│       ├── compare_mass_yield_log.png   计算 vs 实验叠图（对数轴）
│       ├── mass_yield.png               质量产额
│       └── 裂变碎片产额分布.png          产额分布示意图
└── 随机行走/                  ← 上一次实验：5D 随机行走里程碑 1（2D 对称验证）
    └── figures/               ← 随机行走图（8 张 PNG/GIF）
        ├── 对称路径宏观微观势垒.png
        ├── eta扫描非对称裂变.png
        ├── 双峰裂变势垒.png
        ├── 势能面切片.png
        ├── 非对称势能面切片.png
        ├── 五参数形状示意图.png
        ├── 核形状随机行走.gif
        └── 核形状非对称裂变.gif
```

## 说明

- **脚本已指向本目录**：`scripts/` 下的产额脚本（`calc_fragment_yield.py`、
  `compare_yield_experiment.py`、`fetch_endf_yield.py`）输出到 `results/产额分布/`，
  随机行走脚本（`run_walk.py`、`run_walk_asym.py`、`symmetric_path_validation.py`、
  `epsilon_relaxation.py`）输出到 `results/随机行走/figures/`。旧的 `data/`、`output/`
  工作目录已删除。
- **raw/ 可删**：`data/raw/` 里是抓取 ENDF/B-VIII 时下载的原始 `.dat` 与 `.zip`，
  可由 `fetch_endf_yield.py` 重新抓取，占 ~53MB，需要腾空间时可直接删除。
- 实验结论与调参排查见 `docs/产额计算结果.md`、`docs/产额实验总结.md`。
