# -*- coding: utf-8 -*-
"""从 ENDF/B-VIII.0 的 nfpy 子库文件解析 U-235(n_th, f) 累计裂变产额。

输入
    data/nfpy_9228_92-U-235.dat   ENDF-6 文件, MF=8 MT=459 (累计产额)

输出
    data/ENDF_U235_cumulative_yield.csv   每个碎片的累计产额 (A, Z, 同质异能态, 产额%, 不确定度%)
    data/ENDF_U235_mass_chain_yield.csv   质量链产额 Y(A)  (A, 产额%)

要点
    - 产额按入射能量依赖存储(LE=3), 取第一个能量点 0.0253 eV (热中子)。
    - 每个产物 4 个值 (ZAFP=1000*Z+A, FPS, Y, DY), 每行 6 个值打包, 产物跨行。
    - 质量链产额 Y(A) = 该质量数内最大 Z 同位素的累计产额(β 衰变链末端),
      不是对所有 Z 求和(那样会沿链重复累计)。
"""
import os
import re
import csv


def fnum(s):
    """ENDF 固定宽度浮点数 -> float, 支持 Fortran 指数 (如 6.413-2 = 6.413e-2)。"""
    s = s.strip()
    if not s:
        return 0.0
    m = re.match(r'^([+-]?\d+\.\d*)([+-]\d+)?$', s)
    if not m:
        return float(s)
    mant, exp = m.group(1), m.group(2)
    return float(mant + ('e' + exp if exp else ''))


def read_values(lines, start, n):
    """从 lines[start] 起读 n 个 11 字符字段(每行 6 个), 返回 list。"""
    vals, i = [], start
    while len(vals) < n:
        ln = lines[i]
        for k in range(6):
            vals.append(fnum(ln[k * 11:(k + 1) * 11]))
        i += 1
    return vals[:n], i


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, '..', 'results', '产额分布', 'data')
    src = os.path.join(data_dir, 'raw', 'nfpy_9228_92-U-235.dat')

    with open(src, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    # 1. 定位 MF=8 MT=459 的 HEAD 记录
    header = None
    for i, ln in enumerate(lines):
        if len(ln) >= 75 and ln[70:72].strip() == '8' and ln[72:75] == '459':
            header = i
            break
    assert header is not None, '未找到 MF=8 MT=459 段'

    # 2. 读能量记录 (第一个能量点)
    erec = lines[header + 1]
    energy = fnum(erec[0:11])
    nvals = int(erec[44:55])   # 值总数 = 4 * 产物数
    nfp = int(erec[55:66])     # 产物数

    # 3. 读全部值, 打包成 (ZAFP, FPS, Y, DY)
    vals, _ = read_values(lines, header + 2, nvals)
    products = [(vals[4 * p], vals[4 * p + 1], vals[4 * p + 2], vals[4 * p + 3])
                for p in range(nfp)]

    # 4. 写累计产额表 (A, Z, FPS, 产额%, 不确定度%)
    out_cum = os.path.join(data_dir, 'ENDF_U235_cumulative_yield.csv')
    with open(out_cum, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['A', 'Z', 'isomeric_state', 'cumulative_yield_percent', 'uncertainty_percent'])
        for zafp, fps, y, dy in products:
            z = int(zafp) // 1000
            a = int(zafp) % 1000
            w.writerow([a, z, int(fps), round(100 * y, 6), round(100 * dy, 6)])

    # 5. 质量链产额 Y(A) = 该质量数内最大 Z 同位素的累计产额
    y_chain = {}
    for zafp, fps, y, dy in products:
        a = int(zafp) % 1000
        y_chain[a] = max(y_chain.get(a, 0.0), y)

    out_chain = os.path.join(data_dir, 'ENDF_U235_mass_chain_yield.csv')
    with open(out_chain, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['A', 'mass_chain_yield_percent'])
        for a in sorted(y_chain):
            w.writerow([a, round(100 * y_chain[a], 6)])

    # 6. 打印验证
    total = 100 * sum(y_chain.values())
    peak = sorted(y_chain.items(), key=lambda kv: -kv[1])[:4]
    print(f'入射能量 E = {energy:g} eV (热中子)')
    print(f'产物数 = {nfp}, 质量链数 = {len(y_chain)}')
    print(f'质量链产额总和 = {total:.1f}%  (应≈200%)')
    print('产额峰值: ' + ', '.join(f'A={a} {100 * y:.2f}%' for a, y in peak))
    print(f'对称谷 A=118: {100 * y_chain.get(118, 0):.3f}%')
    print(f'已写出: {out_cum}')
    print(f'已写出: {out_chain}')


if __name__ == '__main__':
    main()
