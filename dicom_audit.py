#!/usr/bin/env python3
"""
DICOM 序列验收工具 — 扫描一个 DICOM 根目录，逐序列判断是否为
可用于(半自动)标注的「真 16-bit ORIGINAL HU」数据。

判定逻辑:
  ✅ GOOD  — BitsStored>=12, ImageType 含 ORIGINAL\\PRIMARY, 像素灰阶数高,
             HU 范围覆盖骨(>200)。这才是能做阈值/区域生长标注的真 CT。
  ⚠ WARN  — 16-bit 但属 DERIVED/SECONDARY/REFORMATTED/SCREEN SAVE 等衍生序列。
  ❌ BAD   — BitsStored==8 或灰阶数<=256，HU 信息已不可逆丢失(烤窗 8-bit 导出)。

用法:
  python dicom_audit.py <DICOM根目录> [<目录2> ...]
依赖: SimpleITK, numpy  (conda 环境 BoneMed 已具备)
"""
import sys, os, glob
import numpy as np
import SimpleITK as sitk

T = {
    "modality":   "0008|0060",
    "image_type": "0008|0008",
    "desc":       "0008|103e",
    "maker":      "0008|0070",
    "bits_alloc": "0028|0100",
    "bits_store": "0028|0101",
    "pix_repr":   "0028|0103",
    "slope":      "0028|1053",
    "intercept":  "0028|1052",
    "rtype":      "0028|1054",
}

RESET, RED, GRN, YEL, DIM, BOLD = "\033[0m", "\033[31m", "\033[32m", "\033[33m", "\033[2m", "\033[1m"


def find_series_dirs(root):
    """返回 (dir, dcm文件列表)。优先把每个含 dcm 的子目录当作一个序列。"""
    out = []
    for d, _, files in os.walk(root):
        dcms = [os.path.join(d, f) for f in files
                if f.lower().endswith((".dcm", ".ima")) or f.isdigit()]
        if dcms:
            out.append((d, sorted(dcms)))
    return sorted(out)


def meta(reader, key):
    k = T[key]
    return reader.GetMetaData(k).strip() if reader.HasMetaDataKey(k) else "?"


def classify(bits_store, image_type, n_uniq, hu_max, bone_frac):
    derived = any(w in image_type for w in
                  ("DERIVED", "SECONDARY", "REFORMATTED", "SCREEN", "VOLREN", "LOCALIZER"))
    if bits_store <= 8 or n_uniq <= 256:
        return "BAD", "8-bit / 灰阶<=256，HU 已丢失(烤窗导出)"
    if "ORIGINAL" in image_type and "PRIMARY" in image_type and not derived:
        if hu_max > 200 or bone_frac > 0.1:
            return "GOOD", "真 16-bit ORIGINAL HU，含骨密度，可标注"
        return "WARN", "ORIGINAL 但未见骨信号(可能是定位/软组织窗子集)"
    return "WARN", "16-bit 但属衍生序列(DERIVED/SECONDARY/...)"


def audit_dir(root):
    print(f"\n{BOLD}=== {root} ==={RESET}")
    series = find_series_dirs(root)
    if not series:
        print("  (未找到 DICOM 文件)")
        return
    rows = []
    for d, fs in series:
        mid = fs[len(fs) // 2]
        try:
            r = sitk.ImageFileReader()
            r.SetFileName(mid)
            r.ReadImageInformation()
            img = r.Execute()
            a = sitk.GetArrayFromImage(img).astype(np.float32)
        except Exception as e:
            print(f"  {RED}ERR{RESET} {os.path.relpath(d, root)}: {e}")
            continue
        bs = int(meta(r, "bits_store")) if meta(r, "bits_store").isdigit() else 0
        it = meta(r, "image_type")
        slope = float(meta(r, "slope")) if _isnum(meta(r, "slope")) else 1.0
        inter = float(meta(r, "intercept")) if _isnum(meta(r, "intercept")) else 0.0
        # 原始像素已含 DICOM 存储值；多数已是 HU。统一按 slope/inter 再算一次。
        hu = a * slope + inter if (slope != 1.0 or inter != 0.0) else a
        n_uniq = int(np.unique(a).size)
        hu_max = float(hu.max())
        bone = float(np.mean(hu > 200) * 100)
        verdict, why = classify(bs, it, n_uniq, hu_max, bone)
        rows.append((verdict, os.path.relpath(d, root), len(fs), bs, it,
                     meta(r, "desc"), n_uniq, hu.min(), hu_max, bone, why))

    order = {"GOOD": 0, "WARN": 1, "BAD": 2}
    rows.sort(key=lambda x: order[x[0]])
    for v, rel, n, bs, it, desc, nu, hmn, hmx, bone, why in rows:
        col = {"GOOD": GRN, "WARN": YEL, "BAD": RED}[v]
        mark = {"GOOD": "✅", "WARN": "⚠ ", "BAD": "❌"}[v]
        print(f"  {col}{mark} {v:4s}{RESET} [{rel}] n={n} bits={bs} "
              f"uniq={nu} HU=[{hmn:.0f},{hmx:.0f}] bone={bone:.2f}%")
        print(f"        {DIM}{it} | '{desc}' | {why}{RESET}")
    good = [r for r in rows if r[0] == "GOOD"]
    print(f"  {BOLD}→ 可标注序列(GOOD): {len(good)}{RESET}" +
          (f"  ⇒ 用这个: {good[0][1]}" if good else "  ⇒ 无可标注序列，需回 PACS 重导"))


def _isnum(s):
    try:
        float(s); return True
    except Exception:
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for root in sys.argv[1:]:
        audit_dir(root)
