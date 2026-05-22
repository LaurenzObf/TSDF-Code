# -*- coding: utf-8 -*-
"""
TSDF Histogram Deformation Analysis - ROI around deformation (bounding box)
(Counts on y-axis, NOT density)

- Build TSDF from Epoch1 scans (reference)
- Query SIGNED TSDF at two point sets:
    * REF points (undeformed / epoch1)
    * DEF points (deformed epoch)
- Compute and save histograms:
    A) GLOBAL (all points)
    B) ROI (Bounding box around deformation center)
  For each: linear + log-y
- For ROI:
    - Overlay plot (REF vs DEF) as COUNTS (not density), same bins/range
    - Difference histogram (DEF_counts - REF_counts)

Python 3.8
"""

import os
import time
import numpy as np
from typing import List, Tuple, Optional

import laspy

from vdbfusion.pybind.vdb_volume import VDBVolume
from dataset import Dataset

# ========================= INPUT =========================
base_dir = "/home/laurenz/MQPW"
results_root = os.path.join(base_dir, "results")
data_dir = os.path.join(base_dir, "data")
epoch_def_dir = os.path.join(base_dir, "deformed_scans")

diag_file = "Brucher_202405_P50_georef_diagnostics.txt"

# TSDF reference build (Epoch1 scans)
epoch1_scans = ["s1_f1.las", "s2_f1.las", "s3_f1.las"]

# Reference point set for histogram
ref_las_files = ["s1_f1.las", "s2_f1.las", "s3_f1.las"]

# Deformed point set for histogram
def_las_files = ["s1_f1_deformed_+10mm.las", "s2_f1_deformed_+10mm.las", "s3_f1_deformed_+10mm.las"]

voxel_size = 0.03
sdf_trunc = 4.0 * voxel_size
min_weight_query = 0.5  # weight filter for query_sdf

# --- Deformation (dome) parameters ---
center = np.array([703.383728, 375.882935, 361.387207], dtype=np.float64)
radius_m = 4.0

# --- Bounding box around deformation ---
bbox_half = np.array([radius_m, radius_m, radius_m], dtype=np.float64)
roi_min = center - bbox_half
roi_max = center + bbox_half

# Histogram settings
bins = 2000
hist_range_mm: Optional[Tuple[float, float]] = None  # If None => auto from combined percentiles (0.1..99.9)

run_name = "TSDF_Histogram_ROI_DomeBBox"
# =========================================================


def load_las_points(path: str) -> np.ndarray:
    las = laspy.read(path)
    return np.vstack([las.x, las.y, las.z]).T.astype(np.float64)


def load_points_from_files(folder: str, files: List[str], label: str) -> np.ndarray:
    P_all = []
    for fn in files:
        path = os.path.join(folder, fn)
        print(f"[INFO] load LAS ({label}): {path}")
        P = load_las_points(path)
        print(f"[INFO] points loaded ({label}): {P.shape[0]}")
        P_all.append(P)
    P = np.vstack(P_all).astype(np.float64)
    print(f"[INFO] total points ({label}): {P.shape[0]}")
    return P


def mask_bbox(P: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    return (
        (P[:, 0] >= bmin[0]) & (P[:, 0] <= bmax[0]) &
        (P[:, 1] >= bmin[1]) & (P[:, 1] <= bmax[1]) &
        (P[:, 2] >= bmin[2]) & (P[:, 2] <= bmax[2])
    )


def build_tsdf_from_epoch1(scan_files: List[str]) -> VDBVolume:
    scan_paths = [os.path.join(data_dir, fn) for fn in scan_files]
    diag_path = os.path.join(data_dir, diag_file)

    ds = Dataset(scan_txt_paths=scan_paths, diagnostics_path=diag_path)
    vol = VDBVolume(voxel_size, sdf_trunc, space_carving=True)

    print("[INFO] start TSDF Integration (Epoch 1)")
    total = 0
    for i in range(len(ds)):
        points, T_WC = ds[i]
        points = np.asarray(points, dtype=np.float64)
        total += points.shape[0]
        print(f"[INFO] integrate {i+1}/{len(ds)} ({ds.scan_names[i]}): {points.shape[0]} points")
        if points.shape[0] > 0:
            vol.integrate(points, T_WC)

    print("[INFO] integrated total points:", total)
    return vol


def summarize_mm(x_m: np.ndarray, name: str) -> str:
    x = np.asarray(x_m, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return f"[{name}] no finite values\n"

    x_mm = 1000.0 * x
    q_list = [0, 1, 5, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
    qs = np.percentile(x_mm, q_list)

    med = np.median(x_mm)
    mad = np.median(np.abs(x_mm - med))
    robust_sigma = 1.4826 * mad

    s = []
    s.append(f"[{name}] (mm)")
    s.append(f"  n={x_mm.size}")
    s.append(f"  mean={float(np.mean(x_mm)):.4f}   median={float(np.median(x_mm)):.4f}   std={float(np.std(x_mm)):.4f}")
    s.append(f"  MAD={float(mad):.4f}   robust_sigma≈{float(robust_sigma):.4f}  (via 1.4826*MAD)")
    s.append("  quantiles (mm):")
    s.append("  " + ", ".join([f"p{int(p):>4}={v:>8.3f}" if p % 1 == 0 else f"p{p:>4}={v:>8.3f}" for p, v in zip(q_list, qs)]))
    return "\n".join(s) + "\n"


def choose_range_mm(a_mm: np.ndarray, b_mm: np.ndarray) -> Tuple[float, float]:
    x = np.concatenate([a_mm, b_mm]) if (a_mm.size and b_mm.size) else (a_mm if a_mm.size else b_mm)
    if x.size == 0:
        return (-1.0, 1.0)
    lo = float(np.percentile(x, 0.1))
    hi = float(np.percentile(x, 99.9))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    return (lo, hi)


def style_axes(ax, xlabel: str, ylabel: str, title: Optional[str] = None) -> None:
    if title is not None:
        ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis="both", labelsize=20)


def save_hist_plots_counts(x_mm: np.ndarray, out_linear: str, out_logy: str, title: str, bins_: int, range_mm_: Tuple[float, float]) -> None:
    import matplotlib.pyplot as plt

    # linear
    fig = plt.figure(figsize=(12, 5), dpi=140)
    ax = fig.add_subplot(111)
    ax.hist(x_mm, bins=bins_, range=range_mm_)
    style_axes(ax, xlabel="signed TSDF (mm)", ylabel="count", title=title + " (linear)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.savefig(out_linear, bbox_inches="tight")
    plt.close(fig)

    # log-y
    fig = plt.figure(figsize=(12, 5), dpi=140)
    ax = fig.add_subplot(111)
    ax.hist(x_mm, bins=bins_, range=range_mm_, log=True)
    style_axes(ax, xlabel="signed TSDF (mm)", ylabel="count (log)", title=title + " (log-y)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.savefig(out_logy, bbox_inches="tight")
    plt.close(fig)


def hist_counts(x_mm: np.ndarray, bins_: int, range_mm_: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    h, edges = np.histogram(x_mm, bins=bins_, range=range_mm_, density=False)
    return h.astype(np.int64), edges


def save_overlay_and_diff_counts(ref_mm: np.ndarray, def_mm: np.ndarray, out_overlay: str, out_diff: str, bins_: int, range_mm_: Tuple[float, float]) -> None:
    """
    Overlay: REF vs DEF as COUNTS curves (same bins/range)
    Diff: (DEF_counts - REF_counts) bar plot
    Legend: CENTER RIGHT with fontsize 20
    """
    import matplotlib.pyplot as plt

    h_ref, edges = hist_counts(ref_mm, bins_=bins_, range_mm_=range_mm_)
    h_def, _ = hist_counts(def_mm, bins_=bins_, range_mm_=range_mm_)
    centers = 0.5 * (edges[:-1] + edges[1:])
    diff = (h_def - h_ref).astype(np.int64)
    bin_w = float(edges[1] - edges[0])

    # overlay (counts)
    fig = plt.figure(figsize=(12, 5), dpi=140)
    ax = fig.add_subplot(111)
    ax.plot(centers, h_ref, label="identische Punkte")
    ax.plot(centers, h_def, label="deformierte Punkte")
    style_axes(ax, xlabel="TSDF (mm)", ylabel="count")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.5), fontsize=20, frameon=True, handlelength=3)
    fig.savefig(out_overlay, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    # diff (counts)
    fig = plt.figure(figsize=(12, 5), dpi=140)
    ax = fig.add_subplot(111)
    ax.bar(centers, diff, width=bin_w)
    ax.axhline(0.0, linewidth=1.0)
    style_axes(ax, xlabel="signed TSDF (mm)", ylabel="count difference", title="ROI histogram difference: DEF - REF (counts)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.savefig(out_diff, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()

    run_dir = os.path.join(results_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # output paths
    stats_ref_txt = os.path.join(run_dir, "tsdf_stats_ref.txt")
    stats_def_txt = os.path.join(run_dir, "tsdf_stats_def.txt")

    # GLOBAL hist
    glob_ref_lin = os.path.join(run_dir, "hist_GLOBAL_ref_linear.png")
    glob_ref_log = os.path.join(run_dir, "hist_GLOBAL_ref_logy.png")
    glob_def_lin = os.path.join(run_dir, "hist_GLOBAL_def_linear.png")
    glob_def_log = os.path.join(run_dir, "hist_GLOBAL_def_logy.png")

    # ROI hist
    roi_ref_lin = os.path.join(run_dir, "hist_ROI_ref_linear.png")
    roi_ref_log = os.path.join(run_dir, "hist_ROI_ref_logy.png")
    roi_def_lin = os.path.join(run_dir, "hist_ROI_def_linear.png")
    roi_def_log = os.path.join(run_dir, "hist_ROI_def_logy.png")

    roi_overlay = os.path.join(run_dir, "hist_ROI_overlay_counts.png")
    roi_diff = os.path.join(run_dir, "hist_ROI_diff_counts.png")

    # 1) Build TSDF from Epoch1
    vol = build_tsdf_from_epoch1(epoch1_scans)
    print("[INFO] has query_sdf:", hasattr(vol, "query_sdf"))

    # 2) Load point sets
    P_ref = load_points_from_files(data_dir, ref_las_files, label="REF")
    P_def = load_points_from_files(epoch_def_dir, def_las_files, label="DEF")

    # 3) ROI mask
    m_ref_roi = mask_bbox(P_ref, roi_min, roi_max)
    m_def_roi = mask_bbox(P_def, roi_min, roi_max)

    P_ref_roi = P_ref[m_ref_roi]
    P_def_roi = P_def[m_def_roi]

    print("[INFO] ROI bounds:")
    print("  roi_min:", roi_min)
    print("  roi_max:", roi_max)
    print(f"[INFO] ROI points REF: {P_ref_roi.shape[0]} / {P_ref.shape[0]}  ({100.0 * P_ref_roi.shape[0] / max(P_ref.shape[0], 1):.2f}%)")
    print(f"[INFO] ROI points DEF: {P_def_roi.shape[0]} / {P_def.shape[0]}  ({100.0 * P_def_roi.shape[0] / max(P_def.shape[0], 1):.2f}%)")

    if P_ref_roi.shape[0] < 1000 or P_def_roi.shape[0] < 1000:
        print("[WARN] ROI has very few points. Consider increasing bbox_half (especially z).")

    # 4) Query TSDF (UNCLIPPED)
    print("[INFO] query TSDF at points (REF) (SIGNED, UNCLIPPED)")
    tsdf_ref = np.asarray(vol.query_sdf(P_ref, min_weight=min_weight_query), dtype=np.float32).reshape(-1)
    print("[INFO] query TSDF at points (DEF) (SIGNED, UNCLIPPED)")
    tsdf_def = np.asarray(vol.query_sdf(P_def, min_weight=min_weight_query), dtype=np.float32).reshape(-1)

    # finite filtering
    ref_f = tsdf_ref[np.isfinite(tsdf_ref)].astype(np.float64)
    def_f = tsdf_def[np.isfinite(tsdf_def)].astype(np.float64)

    ref_roi_f = tsdf_ref[m_ref_roi & np.isfinite(tsdf_ref)].astype(np.float64)
    def_roi_f = tsdf_def[m_def_roi & np.isfinite(tsdf_def)].astype(np.float64)

    print(f"[INFO] finite TSDF REF: {ref_f.size}/{tsdf_ref.size} ({ref_f.size / max(tsdf_ref.size, 1):.4f})")
    print(f"[INFO] finite TSDF DEF: {def_f.size}/{tsdf_def.size} ({def_f.size / max(tsdf_def.size, 1):.4f})")
    print(f"[INFO] finite TSDF ROI REF: {ref_roi_f.size}/{P_ref_roi.shape[0]}")
    print(f"[INFO] finite TSDF ROI DEF: {def_roi_f.size}/{P_def_roi.shape[0]}")

    # 5) stats
    s_ref = summarize_mm(ref_f, "SIGNED TSDF REF (GLOBAL)")
    s_def = summarize_mm(def_f, "SIGNED TSDF DEF (GLOBAL)")
    s_ref_roi = summarize_mm(ref_roi_f, "SIGNED TSDF REF (ROI)")
    s_def_roi = summarize_mm(def_roi_f, "SIGNED TSDF DEF (ROI)")

    print(s_ref)
    print(s_def)
    print(s_ref_roi)
    print(s_def_roi)

    with open(stats_ref_txt, "w") as f:
        f.write("### TSDF Stats REF ###\n\n")
        f.write(f"epoch1_scans: {epoch1_scans}\n")
        f.write(f"ref_files: {ref_las_files}\n")
        f.write(f"voxel_size: {voxel_size}\n")
        f.write(f"sdf_trunc: {sdf_trunc}\n")
        f.write(f"min_weight_query: {min_weight_query}\n")
        f.write(f"roi_min: {roi_min.tolist()}\n")
        f.write(f"roi_max: {roi_max.tolist()}\n\n")
        f.write(s_ref + "\n")
        f.write(s_ref_roi + "\n")

    with open(stats_def_txt, "w") as f:
        f.write("### TSDF Stats DEF ###\n\n")
        f.write(f"def_files: {def_las_files}\n")
        f.write(f"voxel_size: {voxel_size}\n")
        f.write(f"sdf_trunc: {sdf_trunc}\n")
        f.write(f"min_weight_query: {min_weight_query}\n")
        f.write(f"roi_min: {roi_min.tolist()}\n")
        f.write(f"roi_max: {roi_max.tolist()}\n\n")
        f.write(s_def + "\n")
        f.write(s_def_roi + "\n")

    # 6) histogram ranges
    ref_mm = 1000.0 * ref_f
    def_mm = 1000.0 * def_f
    ref_roi_mm = 1000.0 * ref_roi_f
    def_roi_mm = 1000.0 * def_roi_f

    if hist_range_mm is None:
        range_global = choose_range_mm(ref_mm, def_mm)
        range_roi = choose_range_mm(ref_roi_mm, def_roi_mm)
    else:
        range_global = hist_range_mm
        range_roi = hist_range_mm

    # 7) save GLOBAL histograms (ref + def) - COUNTS
    save_hist_plots_counts(ref_mm, glob_ref_lin, glob_ref_log, "GLOBAL REF", bins_=bins, range_mm_=range_global)
    save_hist_plots_counts(def_mm, glob_def_lin, glob_def_log, "GLOBAL DEF", bins_=bins, range_mm_=range_global)

    # 8) save ROI histograms (ref + def) - COUNTS
    save_hist_plots_counts(ref_roi_mm, roi_ref_lin, roi_ref_log, "ROI REF", bins_=bins, range_mm_=range_roi)
    save_hist_plots_counts(def_roi_mm, roi_def_lin, roi_def_log, "ROI DEF", bins_=bins, range_mm_=range_roi)

    # 9) ROI overlay + diff (COUNTS)
    save_overlay_and_diff_counts(ref_roi_mm, def_roi_mm, roi_overlay, roi_diff, bins_=bins, range_mm_=range_roi)

    print("[OK] outputs saved in:", run_dir)
    print("[OK] runtime: {:.2f} s".format(time.time() - t0))