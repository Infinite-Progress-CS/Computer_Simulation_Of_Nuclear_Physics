# -*- coding: utf-8 -*-
"""解析 ENDF/B-VIII.0 nfpy 子库文件 (MF=8 MT=459) 得到累计裂变产额 Y(A,Z)。

用法
    python parse_nfpy.py <nfpy_xxxx.dat> <输出csv前缀>
    例如: python parse_nfpy.py nfpy_9437_94-Pu-239.dat ENDF_Pu239

输出
    <前缀>_cumulative_yield.csv   每个碎片累计产额 (A, Z, 同质异能态, 产额%, 不确定度%)
    <前缀>_mass_chain_yield.csv   质量链产额 Y(A) (A, 产额%)
"""
import os
import re
import sys
import csv


def fnum(s):
    s = s.strip()
    if not s:
        return 0.0
    m = re.match(r'^([+-]?\d+\.\d*)([+-]\d+)?$', s)
    if not m:
        return float(s)
    mant, exp = m.group(1), m.group(2)
    return float(mant + ('e' + exp if exp else ''))


def read_values(lines, start, n):
    vals, i = [], start
    while len(vals) < n:
        ln = lines[i]
        for k in range(6):
            vals.append(fnum(ln[k * 11:(k + 1) * 11]))
        i += 1
    return vals[:n], i


def main():
    src = sys.argv[1]
    prefix = sys.argv[2]
    data_dir = os.path.dirname(os.path.abspath(src))

    with open(src, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    header = None
    for i, ln in enumerate(lines):
        if len(ln) >= 75 and ln[70:72].strip() == '8' and ln[72:75] == '459':
            header = i
            break
    assert header is not None, '未找到 MF=8 MT=459 段'

    erec = lines[header + 1]
    energy = fnum(erec[0:11])
    nvals = int(erec[44:55])
    nfp = int(erec[55:66])

    vals, _ = read_values(lines, header + 2, nvals)
    products = [(vals[4 * p], vals[4 * p + 1], vals[4 * p + 2], vals[4 * p + 3])
                for p in range(nfp)]

    out_cum = os.path.join(data_dir, f'{prefix}_cumulative_yield.csv')
    with open(out_cum, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['A', 'Z', 'isomeric_state', 'cumulative_yield_percent', 'uncertainty_percent'])
        for zafp, fps, y, dy in products:
            z = int(zafp) // 1000
            a = int(zafp) % 1000
            w.writerow([a, z, int(fps), round(100 * y, 6), round(100 * dy, 6)])

    y_chain = {}
    for zafp, fps, y, dy in products:
        a = int(zafp) % 1000
        y_chain[a] = max(y_chain.get(a, 0.0), y)

    out_chain = os.path.join(data_dir, f'{prefix}_mass_chain_yield.csv')
    with open(out_chain, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['A', 'mass_chain_yield_percent'])
        for a in sorted(y_chain):
            w.writerow([a, round(100 * y_chain[a], 6)])

    total = 100 * sum(y_chain.values())
    print(f'入射能量 E = {energy:g} eV (热中子)')
    print(f'产物数 = {nfp}, 质量链数 = {len(y_chain)}')
    print(f'质量链产额总和 = {total:.1f}%  (应≈200%)')
    print(f'已写出: {out_cum}')
    print(f'已写出: {out_chain}')


if __name__ == '__main__':
    main()
