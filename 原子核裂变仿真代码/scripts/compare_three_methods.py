"""
三方法对比分析脚本：Metropolis RW / Langevin / SAC(DRL)。

生成论文用的对比图：
  1. 势能下降曲线（三方法叠图）
  2. 参数空间轨迹（Q-c 投影）
  3. 形状诊断量对比（颈/最大半径比）
  4. 统计箱线图（步数、最终势能、成功率）

用法：
  python compare_three_methods.py
  python compare_three_methods.py --n-runs 30

作者：高涵
日期：2026-08-12
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import gridspec

# CJK font
for _f in ['Noto Sans SC', 'SimHei', 'Microsoft YaHei']:
    try:
        if fm.findfont(_f, fallback_to_default=False):
            plt.rcParams['font.family'] = _f
            plt.rcParams['axes.unicode_minus'] = False
            break
    except Exception:
        continue

_CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CUR)
from fission_animation_3d import (
    RandomWalkSimulator, LangevinSimulator, DRLSimulator, potential_5d)
from nuclear_shape_3d import compute_shape_metrics, DEFAULT_R0


def run_all_methods(n_runs=30, seed_offset=0, verbose=True):
    """运行三种方法各 n_runs 次，收集统计数据。"""
    results = {'Metropolis': [], 'Langevin': [], 'SAC (DRL)': []}

    # ── Metropolis ──
    if verbose:
        print("[1/3] Metropolis RW (T=1.0, step=0.4)...")
    rw = RandomWalkSimulator(temperature=1.0, step_size=0.4, max_steps=500)
    for i in range(n_runs):
        r = rw.run(seed=seed_offset + i)
        results['Metropolis'].append(r)

    # ── Langevin ──
    if verbose:
        print("[2/3] Langevin (T=0.05, gamma=0.3)...")
    lang = LangevinSimulator(friction=0.3, temperature=0.05, dt=0.01, max_steps=300)
    for i in range(n_runs):
        r = lang.run(seed=seed_offset + 100 + i)
        results['Langevin'].append(r)

    # ── SAC (只跑一次训练，多次评估) ──
    if verbose:
        print("[3/3] SAC (training once, evaluating multiple times)...")
    sac = DRLSimulator(total_timesteps=30000, max_steps=200, target_V=0.08)
    # 训练一次
    for i in range(n_runs):
        r = sac.run(seed=seed_offset + 200 + i)
        results['SAC (DRL)'].append(r)

    return results


def compute_statistics(results_per_method):
    """计算每个方法的汇总统计。"""
    stats = {}
    for name, runs in results_per_method.items():
        n = len(runs)
        arrived = sum(1 for r in runs if r['arrived'])
        steps = [r['n_steps'] for r in runs if r['arrived']]
        final_V = [r['final_potential'] for r in runs]

        stats[name] = {
            'success_rate': arrived / n,
            'mean_steps': float(np.mean(steps)) if steps else 0,
            'std_steps': float(np.std(steps)) if steps else 0,
            'mean_final_V': float(np.mean(final_V)),
            'std_final_V': float(np.std(final_V)),
            'n_runs': n,
            'n_arrived': arrived,
        }
    return stats


def plot_comparison(results, stats, output_dir='results'):
    """生成四面板对比图。"""
    os.makedirs(output_dir, exist_ok=True)
    colors = {'Metropolis': '#e74c3c', 'Langevin': '#3498db', 'SAC (DRL)': '#2ecc71'}
    method_names = ['Metropolis', 'Langevin', 'SAC (DRL)']

    fig = plt.figure(figsize=(18, 13))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    # ── Panel 1: 势能下降曲线（第一条轨迹） ──
    ax1 = fig.add_subplot(gs[0, 0])
    for name in method_names:
        r = results[name][0]
        ax1.plot(r['potentials'], color=colors[name], linewidth=2.0,
                 alpha=0.85, label=f"{name}")
    ax1.axhline(y=0.08, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
                label='Target (V=0.08)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Potential V')
    ax1.set_title('Potential Energy Along Path (1st run)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Q-c 投影轨迹 ──
    ax2 = fig.add_subplot(gs[0, 1])
    for name in method_names:
        r = results[name][0]
        traj = r['trajectory']
        ax2.plot(traj[:, 0], traj[:, 1], color=colors[name], linewidth=2.0,
                 alpha=0.85, label=name)
    ax2.plot(0.8, 0.4, 'r*', markersize=15, label='Target (0.8, 0.4)')
    ax2.plot(0.3, 0.6, 'ko', markersize=10, label='Start (0.3, 0.6)')
    ax2.set_xlabel('Q (elongation)')
    ax2.set_ylabel('c (neck)')
    ax2.set_title('Path in Q-c Plane (1st run)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: 成功率柱状图 ──
    ax3 = fig.add_subplot(gs[0, 2])
    srs = [stats[n]['success_rate'] for n in method_names]
    bars = ax3.bar(method_names, srs, color=[colors[n] for n in method_names],
                   edgecolor='#333', linewidth=1.5, alpha=0.85)
    for bar, sr in zip(bars, srs):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                 f'{sr:.1%}', ha='center', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1.2)
    ax3.set_ylabel('Success Rate')
    ax3.set_title('Success Rate Comparison')
    ax3.grid(True, alpha=0.3, axis='y')

    # ── Panel 4: 步数箱线图 ──
    ax4 = fig.add_subplot(gs[1, 0])
    steps_data = []
    for name in method_names:
        s = [r['n_steps'] for r in results[name] if r['arrived']]
        if not s:
            s = [r['n_steps'] for r in results[name]]  # all if none arrived
        steps_data.append(s)
    bp = ax4.boxplot(steps_data, labels=method_names, patch_artist=True)
    for patch, name in zip(bp['boxes'], method_names):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.7)
    ax4.set_ylabel('Steps to Arrival')
    ax4.set_title('Steps Distribution')
    ax4.grid(True, alpha=0.3, axis='y')

    # ── Panel 5: 最终势能箱线图 ──
    ax5 = fig.add_subplot(gs[1, 1])
    V_data = [[r['final_potential'] for r in results[name]] for name in method_names]
    bp2 = ax5.boxplot(V_data, labels=method_names, patch_artist=True)
    for patch, name in zip(bp2['boxes'], method_names):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.7)
    ax5.axhline(y=0.08, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax5.set_ylabel('Final Potential V')
    ax5.set_title('Final Potential Distribution')
    ax5.grid(True, alpha=0.3, axis='y')

    # ── Panel 6: 形状诊断 — 颈/最大半径比演化 ──
    ax6 = fig.add_subplot(gs[1, 2])
    for name in method_names:
        r = results[name][0]
        traj = r['trajectory']
        neck_ratios = []
        for i, row in enumerate(traj):
            m = compute_shape_metrics(row[0], row[1], row[2], row[3], row[4])
            neck_ratios.append(m['neck_to_max_ratio'])
        ax6.plot(neck_ratios, color=colors[name], linewidth=2.0,
                 alpha=0.85, label=name)
    ax6.axhline(y=0.3, color='orange', linestyle=':', linewidth=1.5,
                alpha=0.7, label='Pre-scission (0.3)')
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Neck / Max Radius')
    ax6.set_title('Neck Evolution (1st run)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    fig.suptitle('Fission Path Search: Three-Method Comparison (5D Deformation Space)',
                 fontsize=16, fontweight='bold', y=0.99)

    path = os.path.join(output_dir, 'three_method_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Saved: {path}')
    return path


def print_stats_table(stats):
    """打印统计表格。"""
    print(f"\n{'='*80}")
    print(f"{'Method':<15} {'Success':>8} {'Steps':>12} {'Final V':>12} {'Runs':>6}")
    print(f"{'-'*80}")
    for name, s in stats.items():
        print(f"{name:<15} {s['success_rate']:>7.1%}  "
              f"{s['mean_steps']:>5.0f} +/-{s['std_steps']:>4.0f}  "
              f"{s['mean_final_V']:>7.4f} +/-{s['std_final_V']:>6.4f}  "
              f"{s['n_arrived']:>3d}/{s['n_runs']}")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-runs', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    print("=" * 60)
    print("三方法对比分析：Metropolis / Langevin / SAC")
    print("=" * 60)

    t0 = time.time()
    results = run_all_methods(n_runs=args.n_runs, verbose=True)
    stats = compute_statistics(results)
    print_stats_table(stats)
    plot_comparison(results, stats, args.output_dir)
    print(f"\nTotal time: {time.time()-t0:.0f}s")
