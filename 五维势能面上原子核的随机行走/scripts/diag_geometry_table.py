# -*- coding: utf-8 -*-
"""Diagnose geometry table NaN structure and test symmetry-projection approximation.

Questions:
  1. Valid fraction overall and per-axis.
  2. On the symmetric subspace (eta=0, eps1=eps2), is the table fully finite?
  3. For VALID asymmetric shapes, how well do Q_sym / r_neck_sym (using eps_avg)
     approximate the true values?
  4. Timing: build a walk-relevant sample and compare table vs direct shape.build.
"""
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from geometry_table import GeometryTable
from shape import Shape3QS
from random_walk_yield import quadrupole_moment, neck_radius_fm

R0_LD = 1.16 * 236 ** (1.0 / 3.0)

shape = Shape3QS(R0_LD)
geom = GeometryTable(shape, table_path=os.path.join(
    PROJECT_ROOT, "results", "pes_table", "geometry_table_U236.npz"))

Q = geom.Q
R = geom.Rneck
print("grid shape:", Q.shape)
finite = np.isfinite(Q)
print(f"valid fraction: {finite.mean():.2%}  ({finite.sum()} / {Q.size})")

# Per-axis NaN dependence
names = geom.names
for i, nm in enumerate(names):
    ax = list(range(Q.shape[i]))
    # valid fraction along this axis (mean over all other axes)
    frac = []
    for j in ax:
        sl = [slice(None)] * len(Q.shape)
        sl[i] = j
        frac.append(finite[tuple(sl)].mean())
    print(f"  axis {nm}: valid fraction = {np.array(frac)}")

# Symmetric subspace: eta=0, eps1=eps2
ieta = int(np.argmin(np.abs(geom.axes['eta'])))
print(f"\neta axis = {geom.axes['eta']}")
print(f"eps axis = {geom.axes['eps1']}")
# diagonal eps1=eps2, eta=0
sym_valid = np.zeros((Q.shape[0], Q.shape[1], Q.shape[3]))
for ie in range(Q.shape[0]):
    for inn in range(Q.shape[1]):
        for ieps in range(Q.shape[3]):
            sym_valid[ie, inn, ieps] = finite[ie, inn, ieta, ieps, ieps]
print(f"symmetric subspace (elong x neck x eps) valid fraction: {sym_valid.mean():.2%}")

# Now: for VALID asymmetric shapes, test eps_avg approximation
rng = np.random.default_rng(0)
# Sample valid asymmetric points from the grid itself
valid_idx = np.argwhere(finite & (np.abs(geom.axes['eta'][np.argmax(np.zeros(1))]) >= 0))
# simpler: gather all valid grid points, filter to those with eta != 0 or eps1 != eps2
coords = []
for ie in range(Q.shape[0]):
    for inn in range(Q.shape[1]):
        for iet in range(Q.shape[2]):
            for ie1 in range(Q.shape[3]):
                for ie2 in range(Q.shape[4]):
                    if finite[ie, inn, iet, ie1, ie2]:
                        eta = geom.axes['eta'][iet]
                        eps1 = geom.axes['eps1'][ie1]
                        eps2 = geom.axes['eps2'][ie2]
                        if abs(eta) > 1e-9 or abs(eps1 - eps2) > 1e-9:
                            coords.append((ie, inn, iet, ie1, ie2))

print(f"\ntotal valid asymmetric grid points: {len(coords)}")
# sample up to 500 for accuracy test
if len(coords) > 500:
    idx = rng.choice(len(coords), 500, replace=False)
else:
    idx = np.arange(len(coords))

err_q = []
err_r = []
for k in idx:
    ie, inn, iet, ie1, ie2 = coords[k]
    elong = geom.axes['elong'][ie]
    neck = geom.axes['neck'][inn]
    eta = geom.axes['eta'][iet]
    eps1 = geom.axes['eps1'][ie1]
    eps2 = geom.axes['eps2'][ie2]
    q = np.array([elong, neck, eta, eps1, eps2])
    Q_true = Q[ie, inn, iet, ie1, ie2]
    R_true = R[ie, inn, iet, ie1, ie2]
    # eps_avg projection: eta=0, eps1=eps2=avg
    e_avg = 0.5 * (eps1 + eps2)
    q_sym = np.array([elong, neck, 0.0, e_avg, e_avg])
    # interpolate symmetric value from table (linear on the sym subspace)
    # find nearest grid indices
    ie_n = int(np.argmin(np.abs(geom.axes['elong'] - elong)))
    inn_n = int(np.argmin(np.abs(geom.axes['neck'] - neck)))
    ieps_n = int(np.argmin(np.abs(geom.axes['eps1'] - e_avg)))
    Q_sym = Q[ie_n, inn_n, ieta, ieps_n, ieps_n]
    R_sym = R[ie_n, inn_n, ieta, ieps_n, ieps_n]
    if np.isfinite(Q_sym):
        err_q.append(abs(Q_sym - Q_true) / abs(Q_true) * 100)
        err_r.append(abs(R_sym - R_true) / abs(R_true) * 100)
    else:
        err_q.append(np.nan)
        err_r.append(np.nan)

err_q = np.array(err_q)
err_r = np.array(err_r)
print(f"eps_avg approx: Q rel err: mean={np.nanmean(err_q):.2f}% max={np.nanmax(err_q):.2f}% (n_valid_sym={np.isfinite(err_q).sum()}/{len(err_q)})")
print(f"eps_avg approx: r rel err: mean={np.nanmean(err_r):.2f}% max={np.nanmax(err_r):.2f}%")

# absolute r error (scission at ~2.5 fm, so sub-0.1 fm error is negligible)
r_abs_err = []
for k in idx:
    ie, inn, iet, ie1, ie2 = coords[k]
    elong = geom.axes['elong'][ie]; neck = geom.axes['neck'][inn]
    eta = geom.axes['eta'][iet]; eps1 = geom.axes['eps1'][ie1]; eps2 = geom.axes['eps2'][ie2]
    R_true = R[ie, inn, iet, ie1, ie2]
    e_avg = 0.5 * (eps1 + eps2)
    ie_n = int(np.argmin(np.abs(geom.axes['elong'] - elong)))
    inn_n = int(np.argmin(np.abs(geom.axes['neck'] - neck)))
    ieps_n = int(np.argmin(np.abs(geom.axes['eps1'] - e_avg)))
    R_sym = R[ie_n, inn_n, ieta, ieps_n, ieps_n]
    if np.isfinite(R_sym):
        r_abs_err.append(abs(R_sym - R_true))
r_abs_err = np.array(r_abs_err)
print(f"eps_avg approx: r ABS err: mean={np.nanmean(r_abs_err):.3f} fm  max={np.nanmax(r_abs_err):.3f} fm")

# Timing comparison: direct shape.build vs table lookup on a sample
t0 = time.time()
n_trials = 200
for k in range(n_trials):
    kk = coords[int(idx[k])]
    q = np.array([geom.axes['elong'][kk[0]], geom.axes['neck'][kk[1]],
                  geom.axes['eta'][kk[2]], geom.axes['eps1'][kk[3]],
                  geom.axes['eps2'][kk[4]]])
    try:
        quadrupole_moment(q, shape)
    except Exception:
        pass
t1 = time.time()
print(f"\ndirect shape.build (asymmetric): {(t1-t0)/n_trials*1000:.2f} ms/query")
