#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 Patches_Tumor_Train / Patches_Tumor_Val 的分布差异。

- 逐 patch 读取, 只算 mask>0.5 区域的统计量
- S3_grid 存的是 signed_sqrt(S3*10), 反演回 MPa: S3 = sign(y) * y**2 / 10
- 按 case 聚合, 再汇总 train vs val
"""
import argparse
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def inv_signed_sqrt(y):
    return np.sign(y) * (y ** 2) / 10.0


def case_of(path):
    return path.stem.rsplit('_p', 1)[0]


def half_case_names(files, seed=42):
    case_map = defaultdict(list)
    for f in files:
        case_map[case_of(f)].append(f)
    cases = sorted(case_map.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(cases)
    return set(cases[:len(cases) // 2])


def scan_dir(patch_dir, tag, half=False, seed=42):
    files = sorted(Path(patch_dir).glob("*.pt"))
    if not files:
        print(f"[{tag}] NO FILES in {patch_dir}")
        return None
    if half:
        keep = half_case_names(files, seed=seed)
        files = [f for f in files if case_of(f) in keep]

    case_map = defaultdict(list)
    for f in files:
        case_map[case_of(f)].append(f)

    print(f"[{tag}] {len(files)} patches, {len(case_map)} cases")

    # 全局收集
    s3_samples = []       # 抽样 S3 值 (MPa)
    occ_ratios = []       # mask occupancy per patch
    phys_ch_sums = []     # (4,) phys 各通道 mask 内和
    phys_ch_nonzero = []  # (4,) phys 各通道 mask 内非零比例
    geo_mean = []         # geo (E) 均值
    per_case_s3_mean = {} # case -> mean |S3|
    per_case_s3_min = {}  # case -> min S3
    per_case_n_patches = {}

    rng = np.random.RandomState(0)

    for case, fs in case_map.items():
        case_vals = []
        for f in fs:
            d = torch.load(f, weights_only=False)
            m = (d['mask'].squeeze().float().numpy() > 0.5)
            if m.sum() < 10:
                continue
            occ_ratios.append(m.mean())

            s3 = d['S3_grid'].squeeze().float().numpy()[m]
            s3_mpa = inv_signed_sqrt(s3)
            # 抽样, 最多每 patch 500 点
            k = min(500, s3_mpa.size)
            idx = rng.choice(s3_mpa.size, k, replace=False)
            s3_samples.append(s3_mpa[idx])
            case_vals.append(s3_mpa)

            phys = d['phys_input'].float().numpy()  # (4, D, D, D)
            for c in range(phys.shape[0]):
                ch = phys[c][m]
                if len(phys_ch_sums) <= c:
                    phys_ch_sums.append([])
                    phys_ch_nonzero.append([])
                phys_ch_sums[c].append(ch.mean())
                phys_ch_nonzero[c].append((np.abs(ch) > 1e-6).mean())

            geo = d['geo_input'].squeeze().float().numpy()[m]
            geo_mean.append(geo.mean())

        if case_vals:
            cv = np.concatenate(case_vals)
            per_case_s3_mean[case] = float(np.abs(cv).mean())
            per_case_s3_min[case] = float(cv.min())
            per_case_n_patches[case] = len(fs)

    all_s3 = np.concatenate(s3_samples) if s3_samples else np.array([])

    stats = dict(
        tag=tag,
        n_patches=len(files),
        n_cases=len(case_map),
        occ_ratio_mean=float(np.mean(occ_ratios)),
        occ_ratio_std=float(np.std(occ_ratios)),
        s3_mpa_mean=float(all_s3.mean()),
        s3_mpa_std=float(all_s3.std()),
        s3_mpa_min=float(all_s3.min()),
        s3_mpa_max=float(all_s3.max()),
        s3_abs_mean=float(np.abs(all_s3).mean()),
        s3_p01=float(np.percentile(all_s3, 1)),
        s3_p05=float(np.percentile(all_s3, 5)),
        s3_p50=float(np.percentile(all_s3, 50)),
        s3_p95=float(np.percentile(all_s3, 95)),
        s3_p99=float(np.percentile(all_s3, 99)),
        geo_mean=float(np.mean(geo_mean)),
        geo_std=float(np.std(geo_mean)),
        phys_ch_mean=[float(np.mean(x)) for x in phys_ch_sums],
        phys_ch_nonzero=[float(np.mean(x)) for x in phys_ch_nonzero],
        per_case_s3_mean=per_case_s3_mean,
        per_case_s3_min=per_case_s3_min,
        per_case_n_patches=per_case_n_patches,
        s3_samples=all_s3,
    )
    return stats


def print_row(name, *vals, w=18):
    fmt = "  {:<22}" + " ".join([f"{{:>{w}.5g}}"] * len(vals))
    print(fmt.format(name, *vals))


def report(stats_list):
    tags = [s['tag'] for s in stats_list]
    print("\n" + "=" * 80)
    print("  {:<22}".format("METRIC") + " ".join([f"{t:>18}" for t in tags]))
    print("=" * 80)

    print_row("n_patches",       *[s['n_patches'] for s in stats_list])
    print_row("n_cases",         *[s['n_cases'] for s in stats_list])
    print_row("mask occ mean",   *[s['occ_ratio_mean'] for s in stats_list])
    print_row("mask occ std",    *[s['occ_ratio_std'] for s in stats_list])
    print()
    print_row("S3 MPa mean",     *[s['s3_mpa_mean'] for s in stats_list])
    print_row("S3 MPa std",      *[s['s3_mpa_std'] for s in stats_list])
    print_row("S3 MPa min",      *[s['s3_mpa_min'] for s in stats_list])
    print_row("S3 MPa max",      *[s['s3_mpa_max'] for s in stats_list])
    print_row("|S3| mean",       *[s['s3_abs_mean'] for s in stats_list])
    print_row("S3 p01",          *[s['s3_p01'] for s in stats_list])
    print_row("S3 p05",          *[s['s3_p05'] for s in stats_list])
    print_row("S3 p50",          *[s['s3_p50'] for s in stats_list])
    print_row("S3 p95",          *[s['s3_p95'] for s in stats_list])
    print_row("S3 p99",          *[s['s3_p99'] for s in stats_list])
    print()
    print_row("geo(E) mean",     *[s['geo_mean'] for s in stats_list])
    print_row("geo(E) std",      *[s['geo_std'] for s in stats_list])
    for c in range(4):
        print_row(f"phys[{c}] mean",   *[s['phys_ch_mean'][c] for s in stats_list])
    for c in range(4):
        print_row(f"phys[{c}] nonzero",*[s['phys_ch_nonzero'][c] for s in stats_list])


def save_hist(stats_list, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) S3 主体 zoom: [-0.1, 0.02]
    bins = np.linspace(-0.1, 0.02, 100)
    for s in stats_list:
        x = s['s3_samples']
        axes[0, 0].hist(x, bins=bins, alpha=0.5, density=True,
                        label=f"{s['tag']} (n={len(x)//1000}k)")
    axes[0, 0].axvline(0, color='k', lw=0.5)
    axes[0, 0].set_xlabel("S3 (MPa)"); axes[0, 0].set_ylabel("density")
    axes[0, 0].set_title("S3 distribution (main body, zoomed)")
    axes[0, 0].legend()

    # (0,1) S3 log-scale 全域, 看尾巴
    bins = np.linspace(-4, 0.5, 120)
    for s in stats_list:
        x = s['s3_samples']
        axes[0, 1].hist(x, bins=bins, alpha=0.5, density=True,
                        label=f"{s['tag']}")
    axes[0, 1].set_yscale('log')
    axes[0, 1].axvline(0, color='k', lw=0.5)
    axes[0, 1].set_xlabel("S3 (MPa)"); axes[0, 1].set_ylabel("log density")
    axes[0, 1].set_title("S3 full range (log y, see tails)")
    axes[0, 1].legend()

    # (1,0) per-case |S3| 均值
    for s in stats_list:
        vals = np.array(list(s['per_case_s3_mean'].values()))
        if len(vals) == 0:
            continue
        axes[1, 0].hist(vals, bins=20, alpha=0.5, density=True,
                        label=f"{s['tag']} ({len(vals)} cases)")
    axes[1, 0].set_xlabel("per-case mean |S3| (MPa)")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].set_title("Case-level |S3| mean")
    axes[1, 0].legend()

    # (1,1) per-case S3 最小值 (最压)
    for s in stats_list:
        vals = np.array(list(s['per_case_s3_min'].values()))
        if len(vals) == 0:
            continue
        axes[1, 1].hist(vals, bins=20, alpha=0.5, density=True,
                        label=f"{s['tag']}")
    axes[1, 1].set_xlabel("per-case S3 min (MPa)  [extreme compression]")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].set_title("Case-level S3 min (outlier severity)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"[saved] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",
                    default="H:/pythonproject/PHD_PROGRAM/GCN_Stress/Data/Patches_Tumor_Train_256_128")
    ap.add_argument("--val",
                    default="H:/pythonproject/PHD_PROGRAM/GCN_Stress/Data/Patches_Tumor_Val_256_128")
    ap.add_argument("--half", action="store_true",
                    help="Local_half 模式: 只取 50% cases (seed=42)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",
                    default="H:/pythonproject/PHD_PROGRAM/GCN_Stress/Model/DualEncoder_Local_half/dist_compare.png")
    args = ap.parse_args()

    res = []
    for tag, d in [("TRAIN", args.train), ("VAL", args.val)]:
        s = scan_dir(d, tag + ("(half)" if args.half else ""),
                     half=args.half, seed=args.seed)
        if s is not None:
            res.append(s)

    if not res:
        return

    report(res)
    save_hist(res, args.out)

    # Case-level 异常检查: 打印 val 有而 train 没有的应力量级
    if len(res) == 2:
        tr, va = res
        tr_min = min(tr['per_case_s3_min'].values())
        va_min = min(va['per_case_s3_min'].values())
        print(f"\n[extreme check] train min S3 = {tr_min:.3f} MPa, val min S3 = {va_min:.3f} MPa")
        if va_min < tr_min * 1.1:
            print(f"  !! val 有比 train 更极端的压缩应力, 超出训练分布")

        tr_mu = np.mean(list(tr['per_case_s3_mean'].values()))
        va_mu = np.mean(list(va['per_case_s3_mean'].values()))
        print(f"[case mean |S3|] train={tr_mu:.3f}  val={va_mu:.3f}  ratio val/train={va_mu/tr_mu:.2f}")


if __name__ == "__main__":
    main()
