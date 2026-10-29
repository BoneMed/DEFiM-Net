#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_disp_pattern.py — Diagnose how Load nodes look in the RST file.

Check if loaded nodes have prescribed displacement = u_disp exactly.
"""

import numpy as np
import sys
from pathlib import Path

if len(sys.argv) > 1:
    rst_path = Path(sys.argv[1])
else:
    # Find one
    rst_root = Path(r"J:\Metastasis_Rob")
    rsts = list(rst_root.rglob("*.rst"))
    rst_path = rsts[0] if rsts else None

if rst_path is None or not rst_path.exists():
    print("No RST found")
    sys.exit(1)

print(f"RST: {rst_path}")
from ansys.mapdl.reader import read_binary
rst = read_binary(str(rst_path))

print(f"\nTotal result sets: {rst.nsets}")
print("=" * 60)
print("Comparing FIRST vs LAST result set")
print("=" * 60)

for rnum in [0, rst.nsets - 1]:
    print(f"\n--- Result set rnum={rnum} ---")
    nnum, disp = rst.nodal_displacement(rnum=rnum)
    disp_mag = np.linalg.norm(disp[:, :3], axis=1)
    print(f"  Disp X range: [{disp[:,0].min():+.6f}, {disp[:,0].max():+.6f}]")
    print(f"  Disp Y range: [{disp[:,1].min():+.6f}, {disp[:,1].max():+.6f}]")
    print(f"  Disp Z range: [{disp[:,2].min():+.6f}, {disp[:,2].max():+.6f}]")
    print(f"  Mag p50/p95/p99/max: {np.percentile(disp_mag,50):.6f} / "
          f"{np.percentile(disp_mag,95):.6f} / "
          f"{np.percentile(disp_mag,99):.6f} / {disp_mag.max():.6f}")
    print(f"  BC candidates (mag<1e-6): {(disp_mag < 1e-6).sum()}")
    # Check if there's a "step" in disp_z (indicating prescribed value)
    z_unique_vals, z_counts = np.unique(np.round(disp[:, 2], 4), return_counts=True)
    top5 = np.argsort(z_counts)[::-1][:5]
    print(f"  Top 5 most common disp_z values:")
    for i in top5:
        print(f"    {z_unique_vals[i]:+.4f}: {z_counts[i]} nodes")

rnum = rst.nsets - 1
nnum, disp = rst.nodal_displacement(rnum=rnum)
print(f"\n=== Using rnum={rnum} for the rest of analysis ===")
print(f"Disp X range: [{disp[:,0].min():.6f}, {disp[:,0].max():.6f}]")
print(f"Disp Y range: [{disp[:,1].min():.6f}, {disp[:,1].max():.6f}]")
print(f"Disp Z range: [{disp[:,2].min():.6f}, {disp[:,2].max():.6f}]")

disp_mag = np.linalg.norm(disp[:, :3], axis=1)
print(f"\nMag range:  [{disp_mag.min():.6f}, {disp_mag.max():.6f}]")
print(f"Mag p95:    {np.percentile(disp_mag, 95):.6f}")
print(f"Mag p99:    {np.percentile(disp_mag, 99):.6f}")
print(f"Mag max:    {disp_mag.max():.6f}")

# Failure_Load.csv
failure_csv = rst_path.parent / "Failure_Load.csv"
u_disp = None
if failure_csv.exists():
    fl = np.loadtxt(failure_csv, delimiter=',', skiprows=1)
    u_disp = fl[-1, 2]
    print(f"\nu_disp from Failure_Load.csv: {u_disp:.6f}")

# Histogram of disp Z
print(f"\nDisp Z distribution:")
hist, edges = np.histogram(disp[:, 2], bins=20)
for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
    print(f"  [{lo:+.4f}, {hi:+.4f}]: {h:7d}  {'#' * min(60, h * 60 // hist.max())}")

# How many nodes have disp_z very close to u_disp?
if u_disp is not None:
    tol = abs(u_disp) * 0.02  # 2% tolerance
    matched = np.where(np.abs(disp[:, 2] - u_disp) < tol)[0]
    print(f"\nNodes with disp_z within 2% of u_disp ({u_disp:.4f}): {len(matched)}")

    # Check if these matched nodes also have small X/Y disp
    if len(matched) > 0:
        xy_mag = np.linalg.norm(disp[matched, :2], axis=1)
        print(f"  Their X/Y disp magnitude: min={xy_mag.min():.6f}, "
              f"max={xy_mag.max():.6f}, mean={xy_mag.mean():.6f}")

# Also check nodes with disp_x ~= 0 AND disp_y ~= 0 AND disp_z ~= u_disp
if u_disp is not None:
    strict = np.where(
        (np.abs(disp[:, 0]) < 1e-5) &
        (np.abs(disp[:, 1]) < 1e-5) &
        (np.abs(disp[:, 2] - u_disp) < abs(u_disp) * 0.01)
    )[0]
    print(f"\nStrict prescribed-disp nodes "
          f"(X=0, Y=0, Z=u_disp ±1%): {len(strict)}")
