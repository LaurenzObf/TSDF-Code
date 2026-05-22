# # ######################################################################################################################## 
# # # Sweeper for Verkippung
# # ########################################################################################################################
# -*- coding: utf-8 -*-
"""
TSDF Heatmap (Epoch2 vs Epoch1) - SINGLE TILT FILE (one merged Epoch2 scan)

Use-case:
- Epoch1 TSDF is built from 3 scans (fixed).
- Epoch2 is ONE merged/tilted scan file (e.g. "Brucher_tilted_10.las").
- Only ONE tilt angle (no sweep over angles).
- Max deformation ~ 4 cm  -> set colorbar clip accordingly (tsdf_clip_m = 0.04).

Per run outputs (in results/<run_name>):
- epoch2_points_colored_by_TSDF.ply
- tsdf_colorbar.png
- tsdf_signed_hist_mm_linear.png
- tsdf_signed_hist_mm_logy.png
- log.txt

Also writes a global summary TXT (tab-separated):
- results/TSDF_SWEEP_SUMMARY.txt
"""

import os
import time
import numpy as np
from typing import List, Dict, Tuple

import laspy
import open3d as o3d

from vdbfusion.pybind.vdb_volume import VDBVolume
from dataset import Dataset


# ========================= INPUT =========================
base_dir = "/home/laurenz/MQPW"
results_root = os.path.join(base_dir, "results")
data_dir = os.path.join(base_dir, "data")
epoch2_dir = os.path.join(base_dir, "deformed_scans")

diag_file = "Brucher_202405_P50_georef_diagnostics.txt"

# Epoch 1 (reference TSDF) - fixed (3 scans)
epoch1_scans = ["s1_f1_ausg3.las", "s2_f1_ausg3.las", "s3_f1_ausg3.las"]
# epoch1_scans = ["s1_f1.las", "s2_f1.las", "s3_f1.las"]

# Epoch 2: SINGLE merged/tilted scan (one file!)
epoch2_single_file = "Brucher_tilted_04_ausg.las"  # <-- set your exact filename here
# epoch2_single_file = "Brucher_tilted_04.las"  # <-- set your exact filename here

# Query settings
min_weight_query = 0.5

# Histogram config (signed TSDF in mm)
hist_bins = 600
hist_range_mm = None  # None => auto 0.1..99.9 percentiles

# Points-per-voxel export filtering
min_weight_counts = 0.0
only_active_counts = True

# Export
export_points_ply = True

# ========================= RUN GRID =========================
# voxel_size in meters (keep as you want)
SWEEP: List[float] = [
    0.02,   # example: 5 cm
    0.01,   # example: 3 cm
]

# TSDF settings
sdf_trunc_factor = 4.0      # sdf_trunc = factor * voxel_size
space_carving = True

# Visualization clip:
# Max deformation ~ 4 cm => clip should cover +/- 4 cm
tsdf_clip_m = 0.003          # 0.04 m = 4 cm

# Name prefix for folders
run_prefix = "TSDF_Heatmap_POINTS_E2vsE1_GLOBAL_TILT_SINGLE"
# =========================================================


# ------------------------- helpers -------------------------
def load_las_points(path: str) -> np.ndarray:
    las = laspy.read(path)
    return np.vstack([las.x, las.y, las.z]).T.astype(np.float64)


def build_tsdf_from_epoch1(scan_files: List[str], voxel_size: float, sdf_trunc: float) -> VDBVolume:
    scan_paths = [os.path.join(data_dir, fn) for fn in scan_files]
    diag_path = os.path.join(data_dir, diag_file)

    ds = Dataset(scan_txt_paths=scan_paths, diagnostics_path=diag_path)
    vol = VDBVolume(voxel_size, sdf_trunc, space_carving=space_carving)

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


def points_per_voxel_stats(vol: VDBVolume) -> Dict[str, float]:
    _ijk, counts = vol.export_voxels_ijk_counts(min_weight=min_weight_counts, only_active=only_active_counts)
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if counts.size == 0:
        return {"n_vox": 0, "median": np.nan, "mean": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "n_vox": int(counts.size),
        "median": float(np.median(counts)),
        "mean": float(np.mean(counts)),
        "p95": float(np.percentile(counts, 95)),
        "max": float(np.max(counts)),
    }


def tsdf_stats(tsdf_signed: np.ndarray) -> Dict[str, float]:
    x = np.asarray(tsdf_signed, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    a = x[finite]
    if a.size == 0:
        return {
            "n": int(x.size),
            "finite_frac": 0.0,
            "median": np.nan,
            "mean": np.nan,
            "p95_abs": np.nan,
            "max_abs": np.nan,
        }
    return {
        "n": int(x.size),
        "finite_frac": float(finite.mean()),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p95_abs": float(np.percentile(np.abs(a), 95)),
        "max_abs": float(np.max(np.abs(a))),
    }


def save_colorbar_signed(path: str, clip: float, title: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "blue_black_red",
        [(0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        N=256,
    )
    norm = mpl.colors.Normalize(vmin=-clip, vmax=clip)

    # thin horizontal bar
    fig = plt.figure(figsize=(6.0, 1.6), dpi=200)
    ax = fig.add_axes([0.08, 0.45, 0.84, 0.18])  # [left,bottom,width,height]
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.set_label(title)
    cb.set_ticks([-clip, 0.0, clip])
    cb.set_ticklabels([f"{-clip:.3f}", "0.000", f"{clip:.3f}"])
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def colors_signed(tsdf_signed: np.ndarray, clip: float) -> np.ndarray:
    x = np.asarray(tsdf_signed, dtype=np.float64)
    x = np.clip(x, -clip, clip)
    t = (x + clip) / (2.0 * clip)  # 0..1 where 0.5 is 0
    rgb = np.zeros((t.size, 3), dtype=np.float64)

    left = t <= 0.5
    a = (t[left] / 0.5)           # 0..1
    rgb[left, 2] = 1.0 - a        # blue -> black

    right = ~left
    b = ((t[right] - 0.5) / 0.5)  # 0..1
    rgb[right, 0] = b             # black -> red

    return rgb


def save_histograms_signed_mm(tsdf_signed: np.ndarray, out_linear: str, out_logy: str) -> Dict[str, object]:
    import matplotlib.pyplot as plt

    x = np.asarray(tsdf_signed, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    x_mm = 1000.0 * x

    if x_mm.size == 0:
        plt.figure(figsize=(8, 4), dpi=200)
        plt.title("Signed TSDF histogram (linear) - EMPTY")
        plt.savefig(out_linear)
        plt.close()

        plt.figure(figsize=(8, 4), dpi=200)
        plt.title("Signed TSDF histogram (log-y) - EMPTY")
        plt.savefig(out_logy)
        plt.close()
        return {"range_mm": None, "bins": hist_bins}

    if hist_range_mm is None:
        lo = float(np.percentile(x_mm, 0.1))
        hi = float(np.percentile(x_mm, 99.9))
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        r = (lo, hi)
    else:
        r = tuple(hist_range_mm)

    # linear
    plt.figure(figsize=(8, 4), dpi=200)
    plt.hist(x_mm, bins=hist_bins, range=r)
    plt.xlabel("signed TSDF (mm)")
    plt.ylabel("count")
    plt.title("Signed TSDF histogram (linear)")
    plt.tight_layout()
    plt.savefig(out_linear)
    plt.close()

    # log-y
    plt.figure(figsize=(8, 4), dpi=200)
    plt.hist(x_mm, bins=hist_bins, range=r, log=True)
    plt.xlabel("signed TSDF (mm)")
    plt.ylabel("count (log)")
    plt.title("Signed TSDF histogram (log-y)")
    plt.tight_layout()
    plt.savefig(out_logy)
    plt.close()

    return {"range_mm": r, "bins": hist_bins}


def write_summary_txt(rows: List[Dict[str, object]], out_txt: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_txt, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k)
                if isinstance(v, float):
                    if np.isnan(v):
                        vals.append("nan")
                    else:
                        vals.append(f"{v:.10g}")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")


# ------------------------- run -------------------------
def run_one(voxel_size: float) -> Dict[str, object]:
    sdf_trunc = sdf_trunc_factor * voxel_size

    run_name = f"{run_prefix}_vx{int(round(voxel_size*1000)):03d}mm_{os.path.splitext(epoch2_single_file)[0]}"
    run_dir = os.path.join(results_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    pts_path = os.path.join(run_dir, "epoch2_points_colored_by_TSDF.ply")
    cb_path = os.path.join(run_dir, "tsdf_colorbar.png")
    log_path = os.path.join(run_dir, "log.txt")
    hist_linear_path = os.path.join(run_dir, "tsdf_signed_hist_mm_linear.png")
    hist_logy_path = os.path.join(run_dir, "tsdf_signed_hist_mm_logy.png")

    print("\n" + "=" * 80)
    print(f"[RUN] voxel_size={voxel_size} m  sdf_trunc={sdf_trunc} m  Epoch2={epoch2_single_file}  clip={tsdf_clip_m} m")
    print(f"[RUN] out: {run_dir}")
    print("=" * 80)

    t0 = time.time()

    # 1) Build TSDF
    vol = build_tsdf_from_epoch1(epoch1_scans, voxel_size=voxel_size, sdf_trunc=sdf_trunc)
    print("has query_sdf:", hasattr(vol, "query_sdf"))

    # 1b) points-per-voxel
    ppv = points_per_voxel_stats(vol)
    if ppv["n_vox"] == 0:
        print("[WARN] No voxels exported for counts (empty). Check min_weight/only_active.")
    else:
        print(f"[INFO] voxels exported: {ppv['n_vox']}")
        print(f"[INFO] points per voxel: median={ppv['median']:.2f}, mean={ppv['mean']:.2f}, p95={ppv['p95']:.2f}, max={ppv['max']:.0f}")

    # 2) Load Epoch2 points (single file)
    epoch2_path = os.path.join(epoch2_dir, epoch2_single_file)
    if not os.path.isfile(epoch2_path):
        raise FileNotFoundError(f"Missing Epoch2 file: {epoch2_path}")

    print("[INFO] load LAS (Epoch 2):", epoch2_path)
    P_eval = load_las_points(epoch2_path)
    print("[INFO] total Epoch2 points:", P_eval.shape[0])

    # 3) Query TSDF at Epoch2 points (unclipped for stats/hist)
    print("[INFO] query TSDF at Epoch2 points (signed) [GLOBAL]")
    tsdf_signed = np.asarray(vol.query_sdf(P_eval, min_weight=min_weight_query), dtype=np.float32).reshape(-1)

    # 3b) stats
    st = tsdf_stats(tsdf_signed)
    print(f"[INFO] signed TSDF stats: median={st['median']:.6f} m, mean={st['mean']:.6f} m, "
          f"p95(|TSDF|)={st['p95_abs']:.6f} m, max(|TSDF|)={st['max_abs']:.6f} m, finite_frac={st['finite_frac']:.4f}")

    # 3c) histograms
    hist_info = save_histograms_signed_mm(tsdf_signed, hist_linear_path, hist_logy_path)
    print("[OK] histogram saved:", hist_linear_path)
    print("[OK] histogram saved:", hist_logy_path)

    # 4) Visualization export (finite only, clipped colors)
    m = np.isfinite(tsdf_signed)
    P_vis = P_eval[m]
    tsdf_vis = tsdf_signed[m]

    if P_vis.shape[0] == 0:
        raise RuntimeError("No points left after filtering to finite TSDF values.")

    colors = colors_signed(tsdf_vis, clip=tsdf_clip_m)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_vis))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if export_points_ply:
        o3d.io.write_point_cloud(pts_path, pcd)
        print("[OK] colored points saved:", pts_path)

    save_colorbar_signed(
        cb_path,
        clip=tsdf_clip_m,
        title=f"TSDF (m), clip=±{tsdf_clip_m:.3f}  (blue<0 black=0 red>0)"
    )
    print("[OK] colorbar saved:", cb_path)

    runtime_s = time.time() - t0
    print("[OK] runtime: {:.2f} s".format(runtime_s))

    # 5) Log file
    with open(log_path, "w") as f:
        f.write("### TSDF Heatmap Run (TILT SINGLE) ###\n\n")
        f.write(f"run_name: {run_name}\n")
        f.write(f"epoch1_scans: {epoch1_scans}\n")
        f.write(f"epoch2_file: {epoch2_single_file}\n")
        f.write(f"voxel_size: {voxel_size}\n")
        f.write(f"sdf_trunc: {sdf_trunc}\n")
        f.write(f"space_carving: {space_carving}\n")
        f.write(f"min_weight_query: {min_weight_query}\n")
        f.write(f"tsdf_clip_m: {tsdf_clip_m}\n\n")

        f.write("[POINTS PER VOXEL]\n")
        f.write(f"  n_vox: {ppv['n_vox']}\n")
        f.write(f"  median: {ppv['median']}\n")
        f.write(f"  mean: {ppv['mean']}\n")
        f.write(f"  p95: {ppv['p95']}\n")
        f.write(f"  max: {ppv['max']}\n\n")

        f.write("[TSDF STATS] (signed, meters; p95/max are abs)\n")
        f.write(f"  n: {st['n']}\n")
        f.write(f"  finite_frac: {st['finite_frac']}\n")
        f.write(f"  median_signed_m: {st['median']}\n")
        f.write(f"  mean_signed_m: {st['mean']}\n")
        f.write(f"  p95_abs_m: {st['p95_abs']}\n")
        f.write(f"  max_abs_m: {st['max_abs']}\n\n")

        f.write("[HISTOGRAM]\n")
        f.write(f"  bins: {hist_info.get('bins')}\n")
        f.write(f"  range_mm: {hist_info.get('range_mm')}\n")
        f.write(f"  linear_png: {hist_linear_path}\n")
        f.write(f"  logy_png: {hist_logy_path}\n\n")

        f.write("[FILES]\n")
        f.write(f"  ply: {pts_path}\n")
        f.write(f"  colorbar: {cb_path}\n")
        f.write(f"  log: {log_path}\n")
        f.write(f"\nruntime_s: {runtime_s}\n")

    print("[OK] log written:", log_path)

    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "epoch2_file": epoch2_single_file,
        "voxel_size_m": voxel_size,
        "sdf_trunc_m": sdf_trunc,
        "min_weight_query": min_weight_query,
        "tsdf_clip_m": tsdf_clip_m,
        "ppv_n_vox": ppv["n_vox"],
        "ppv_median": ppv["median"],
        "ppv_mean": ppv["mean"],
        "ppv_p95": ppv["p95"],
        "ppv_max": ppv["max"],
        "tsdf_n": st["n"],
        "tsdf_finite_frac": st["finite_frac"],
        "tsdf_median_signed_m": st["median"],
        "tsdf_mean_signed_m": st["mean"],
        "tsdf_p95_abs_m": st["p95_abs"],
        "tsdf_max_abs_m": st["max_abs"],
        "ply_path": pts_path,
        "colorbar_path": cb_path,
        "hist_linear_path": hist_linear_path,
        "hist_logy_path": hist_logy_path,
        "log_path": log_path,
    }


# ------------------------- main -------------------------
if __name__ == "__main__":
    sweep_t0 = time.time()
    summary_rows: List[Dict[str, object]] = []

    summary_txt = os.path.join(results_root, "TSDF_SWEEP_SUMMARY.txt")

    for vx in SWEEP:
        row = run_one(voxel_size=vx)
        summary_rows.append(row)
        write_summary_txt(summary_rows, summary_txt)
        print("[OK] summary updated:", summary_txt)

    print("\n[ALL DONE] total runtime: {:.2f} s".format(time.time() - sweep_t0))
    print("[ALL DONE] summary:", summary_txt)


######################################################################################################################## 
# Sweeper for Bulge and Translation
########################################################################################################################
# # -*- coding: utf-8 -*-
# """
# TSDF Heatmap Sweeper (Epoch2 vs Epoch1) - unified output per run (NO surface band)

# Runs the final grid you specified:

# - Vx 5cm: 10, 5, 3, 1 mm
# - Vx 3cm: 10, 5, 3, 1 mm
# - Vx 2cm: 3, 1 mm
# - Vx 1cm: 3, 1 mm

# Rules:
# - signed TSDF (ABS_TSDF=False)
# - legend clip: tsdf_clip = 1.5 * defo_mm / 1000
# - same truncation rule: sdf_trunc = 4 * voxel_size   (kept consistent across runs)
# - no surface band
# - epoch1 scans fixed: ["s1_f1.las","s2_f1.las","s3_f1.las"]

# Per run outputs (in results/<run_name>):
# - epoch2_points_colored_by_TSDF.ply
# - tsdf_colorbar.png
# - tsdf_signed_hist_mm_linear.png
# - tsdf_signed_hist_mm_logy.png
# - log.txt (per run inside run folder)

# Also writes a global summary TXT across all runs:
# - results/TSDF_SWEEP_SUMMARY.txt   (tab-separated)

# """

# import os
# import time
# import numpy as np
# from typing import List, Dict, Tuple

# import laspy
# import open3d as o3d

# from vdbfusion.pybind.vdb_volume import VDBVolume
# from dataset import Dataset


# # ========================= INPUT =========================
# base_dir = "/home/laurenz/MQPW"
# results_root = os.path.join(base_dir, "results")
# data_dir = os.path.join(base_dir, "data")
# epoch2_dir = os.path.join(base_dir, "deformed_scans")

# diag_file = "Brucher_202405_P50_georef_diagnostics.txt"

# # Epoch 1 (reference TSDF) - fixed
# epoch1_scans = ["s1_f1_ausg3.las", "s2_f1_ausg3.las", "s3_f1_ausg3.las"]

# # Epoch 2 naming pattern, 
# # epoch2_pattern = "{scan}_f1_ausg3_tz+{mm}mm.las"  # e.g. s2_f1_deformed_+1mm.las
# epoch2_pattern = "{scan}_f1_ausg3_tz+{mm}mm.las"  # e.g. s2_f1_deformed_+1mm.las
# epoch2_scans = ["s1", "s2", "s3"] 
# # s1_f1_tilt+1.0deg.las 

# # Query settings
# min_weight_query = 0.5

# # Histogram config (signed TSDF in mm)
# hist_bins = 600
# hist_range_mm = None  # None => auto 0.1..99.9 percentiles

# # Points-per-voxel export filtering
# min_weight_counts = 0.0
# only_active_counts = True

# # Export
# export_points_ply = True

# # ========================= SWEEP GRID =========================
# # voxel_size in meters
# SWEEP: List[Tuple[float, List[int]]] = [
#     # (0.05, [10, 5, 3, 1]),
#     # (0.03, [10, 5, 3, 1]),
#     # (0.02, [3, 1]),
#     # (0.01, [3, 1]),
#     # (0.01, [1]),
#     (0.02, [1]),
# ]

# # tsdf override aktiv!!!
# clip_factor = 1.5
# sdf_trunc_factor = 4.0          # sdf_trunc = factor * voxel_size
# space_carving = True

# # Name prefix for folders
# sweep_prefix = "TSDF_Heatmap_POINTS_translateHeight_SWEEP"
# # =========================================================


# # helpers
# def load_las_points(path: str) -> np.ndarray:
#     las = laspy.read(path)
#     return np.vstack([las.x, las.y, las.z]).T.astype(np.float64)


# def build_tsdf_from_epoch1(scan_files: List[str], voxel_size: float, sdf_trunc: float) -> VDBVolume:
#     scan_paths = [os.path.join(data_dir, fn) for fn in scan_files]
#     diag_path = os.path.join(data_dir, diag_file)

#     ds = Dataset(scan_txt_paths=scan_paths, diagnostics_path=diag_path)
#     vol = VDBVolume(voxel_size, sdf_trunc, space_carving=space_carving)

#     print("[INFO] start TSDF Integration (Epoch 1)")
#     total = 0
#     for i in range(len(ds)):
#         points, T_WC = ds[i]
#         points = np.asarray(points, dtype=np.float64)
#         total += points.shape[0]
#         print(f"[INFO] integrate {i+1}/{len(ds)} ({ds.scan_names[i]}): {points.shape[0]} points")
#         if points.shape[0] > 0:
#             vol.integrate(points, T_WC)
#     print("[INFO] integrated total points:", total)
#     return vol


# def points_per_voxel_stats(vol: VDBVolume) -> Dict[str, float]:
#     ijk, counts = vol.export_voxels_ijk_counts(min_weight=min_weight_counts, only_active=only_active_counts)
#     counts = np.asarray(counts, dtype=np.int64).reshape(-1)
#     if counts.size == 0:
#         return {"n_vox": 0, "median": np.nan, "mean": np.nan, "p95": np.nan, "max": np.nan}
#     return {
#         "n_vox": int(counts.size),
#         "median": float(np.median(counts)),
#         "mean": float(np.mean(counts)),
#         "p95": float(np.percentile(counts, 95)),
#         "max": float(np.max(counts)),
#     }


# def tsdf_stats(tsdf_signed: np.ndarray) -> Dict[str, float]:
#     x = np.asarray(tsdf_signed, dtype=np.float64).reshape(-1)
#     finite = np.isfinite(x)
#     a = x[finite]
#     if a.size == 0:
#         return {
#             "n": int(x.size),
#             "finite_frac": 0.0,
#             "median": np.nan,
#             "mean": np.nan,
#             "p95_abs": np.nan,
#             "max_abs": np.nan,
#         }
#     return {
#         "n": int(x.size),
#         "finite_frac": float(finite.mean()),
#         "median": float(np.median(a)),
#         "mean": float(np.mean(a)),
#         "p95_abs": float(np.percentile(np.abs(a), 95)),
#         "max_abs": float(np.max(np.abs(a))),
#     }


# def save_colorbar_signed(path: str, clip: float, title: str) -> None:
#     import matplotlib.pyplot as plt
#     import matplotlib as mpl

#     cmap = mpl.colors.LinearSegmentedColormap.from_list(
#         "blue_black_red",
#         [(0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
#         N=256,
#     )
#     norm = mpl.colors.Normalize(vmin=-clip, vmax=clip)

#     fig = plt.figure(figsize=(6, 1.2), dpi=200)
#     ax = fig.add_axes([0.08, 0.35, 0.84, 0.35])
#     cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
#     cb.set_label(title)
#     cb.set_ticks([-clip, 0.0, clip])
#     cb.set_ticklabels([f"{-clip:.3f}", "0.000", f"{clip:.3f}"])
#     fig.savefig(path, bbox_inches="tight")
#     plt.close(fig)


# def colors_signed(tsdf_signed: np.ndarray, clip: float) -> np.ndarray:
#     x = np.asarray(tsdf_signed, dtype=np.float64)
#     x = np.clip(x, -clip, clip)
#     t = (x + clip) / (2.0 * clip)  # 0..1 where 0.5 is 0
#     rgb = np.zeros((t.size, 3), dtype=np.float64)

#     left = t <= 0.5
#     a = (t[left] / 0.5)           # 0..1
#     rgb[left, 2] = 1.0 - a        # blue -> black

#     right = ~left
#     b = ((t[right] - 0.5) / 0.5)  # 0..1
#     rgb[right, 0] = b             # black -> red

#     return rgb


# def save_histograms_signed_mm(tsdf_signed: np.ndarray, out_linear: str, out_logy: str) -> Dict[str, object]:
#     import matplotlib.pyplot as plt

#     x = np.asarray(tsdf_signed, dtype=np.float64).reshape(-1)
#     x = x[np.isfinite(x)]
#     x_mm = 1000.0 * x

#     if x_mm.size == 0:
#         plt.figure(figsize=(8, 4), dpi=200)
#         plt.title("Signed TSDF histogram (linear) - EMPTY")
#         plt.savefig(out_linear)
#         plt.close()

#         plt.figure(figsize=(8, 4), dpi=200)
#         plt.title("Signed TSDF histogram (log-y) - EMPTY")
#         plt.savefig(out_logy)
#         plt.close()
#         return {"range_mm": None, "bins": hist_bins}

#     if hist_range_mm is None:
#         lo = float(np.percentile(x_mm, 0.1))
#         hi = float(np.percentile(x_mm, 99.9))
#         if lo == hi:
#             lo, hi = lo - 1.0, hi + 1.0
#         r = (lo, hi)
#     else:
#         r = tuple(hist_range_mm)

#     # linear
#     plt.figure(figsize=(8, 4), dpi=200)
#     plt.hist(x_mm, bins=hist_bins, range=r)
#     plt.xlabel("signed TSDF (mm)")
#     plt.ylabel("count")
#     plt.title("Signed TSDF histogram (linear)")
#     plt.tight_layout()
#     plt.savefig(out_linear)
#     plt.close()

#     # log-y
#     plt.figure(figsize=(8, 4), dpi=200)
#     plt.hist(x_mm, bins=hist_bins, range=r, log=True)
#     plt.xlabel("signed TSDF (mm)")
#     plt.ylabel("count (log)")
#     plt.title("Signed TSDF histogram (log-y)")
#     plt.tight_layout()
#     plt.savefig(out_logy)
#     plt.close()

#     return {"range_mm": r, "bins": hist_bins}


# def make_epoch2_files(mm: int) -> List[str]:
#     return [epoch2_pattern.format(scan=s, mm=mm) for s in epoch2_scans]


# def run_one(voxel_size: float, defo_mm: int) -> Dict[str, object]:
#     sdf_trunc = sdf_trunc_factor * voxel_size
#     tsdf_clip = (clip_factor * float(defo_mm)) / 1000.0  # meters
#     tsdf_clip = 0.025       # override for consistent legend across runs

#     run_name = f"{sweep_prefix}_vx{int(round(voxel_size*1000)):03d}mm_defo{defo_mm:02d}mm"
#     run_dir = os.path.join(results_root, run_name)
#     os.makedirs(run_dir, exist_ok=True)

#     pts_path = os.path.join(run_dir, "epoch2_points_colored_by_TSDF.ply")
#     cb_path = os.path.join(run_dir, "tsdf_colorbar.png")
#     log_path = os.path.join(run_dir, "log.txt")
#     hist_linear_path = os.path.join(run_dir, "tsdf_signed_hist_mm_linear.png")
#     hist_logy_path = os.path.join(run_dir, "tsdf_signed_hist_mm_logy.png")

#     print("\n" + "=" * 80)
#     print(f"[RUN] voxel_size={voxel_size} m  sdf_trunc={sdf_trunc} m  defo={defo_mm} mm  clip={tsdf_clip} m")
#     print(f"[RUN] out: {run_dir}")
#     print("=" * 80)

#     t0 = time.time()

#     # 1) Build TSDF
#     vol = build_tsdf_from_epoch1(epoch1_scans, voxel_size=voxel_size, sdf_trunc=sdf_trunc)
#     print("has query_sdf:", hasattr(vol, "query_sdf"))

#     # 1b) points-per-voxel
#     ppv = points_per_voxel_stats(vol)
#     if ppv["n_vox"] == 0:
#         print("[WARN] No voxels exported for counts (empty). Check min_weight/only_active.")
#     else:
#         print(f"[INFO] voxels exported: {ppv['n_vox']}")
#         print(f"[INFO] points per voxel: median={ppv['median']:.2f}, mean={ppv['mean']:.2f}, p95={ppv['p95']:.2f}, max={ppv['max']:.0f}")

#     # 2) Load Epoch2 points
#     epoch2_files = make_epoch2_files(defo_mm)
#     P2_all = []
#     for fn in epoch2_files:
#         path = os.path.join(epoch2_dir, fn)
#         if not os.path.isfile(path):
#             raise FileNotFoundError(f"Missing Epoch2 file: {path}")
#         print("[INFO] load LAS (Epoch 2):", path)
#         P = load_las_points(path)
#         print("[INFO] points loaded:", P.shape[0])
#         P2_all.append(P)

#     P_eval = np.vstack(P2_all).astype(np.float64)
#     print("[INFO] total Epoch2 points:", P_eval.shape[0])

#     # 3) Query TSDF at Epoch2 points (unclipped for stats/hist)
#     print("[INFO] query TSDF at Epoch2 points (signed) [GLOBAL]")
#     tsdf_signed = np.asarray(vol.query_sdf(P_eval, min_weight=min_weight_query), dtype=np.float32).reshape(-1)

#     # 3b) stats
#     st = tsdf_stats(tsdf_signed)
#     print(f"[INFO] signed TSDF stats: median={st['median']:.6f} m, mean={st['mean']:.6f} m, p95(|TSDF|)={st['p95_abs']:.6f} m, max(|TSDF|)={st['max_abs']:.6f} m")

#     # 3c) histograms
#     hist_info = save_histograms_signed_mm(tsdf_signed, hist_linear_path, hist_logy_path)
#     print("[OK] histogram saved:", hist_linear_path)
#     print("[OK] histogram saved:", hist_logy_path)

#     # 4) Visualization export (finite only, clipped colors)
#     m = np.isfinite(tsdf_signed)
#     P_vis = P_eval[m]
#     tsdf_vis = tsdf_signed[m]

#     if P_vis.shape[0] == 0:
#         raise RuntimeError("No points left after filtering to finite TSDF values.")

#     colors = colors_signed(tsdf_vis, clip=tsdf_clip)
#     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_vis))
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     if export_points_ply:
#         o3d.io.write_point_cloud(pts_path, pcd)
#         print("[OK] colored points saved:", pts_path)

#     save_colorbar_signed(cb_path, clip=tsdf_clip, title=f"TSDF (m), clip=±{tsdf_clip:.3f}  (blue<0 black=0 red>0)")
#     print("[OK] colorbar saved:", cb_path)

#     runtime_s = time.time() - t0
#     print("[OK] runtime: {:.2f} s".format(runtime_s))

#     # 5) Log file (overwrite per run folder to keep it clean)
#     with open(log_path, "w") as f:
#         f.write("### TSDF Heatmap Sweep Run ###\n\n")
#         f.write(f"run_name: {run_name}\n")
#         f.write(f"epoch1_scans: {epoch1_scans}\n")
#         f.write(f"epoch2_files: {epoch2_files}\n")
#         f.write(f"voxel_size: {voxel_size}\n")
#         f.write(f"sdf_trunc: {sdf_trunc}\n")
#         f.write(f"space_carving: {space_carving}\n")
#         f.write(f"min_weight_query: {min_weight_query}\n")
#         f.write(f"defo_mm: {defo_mm}\n")
#         f.write(f"clip_factor: {clip_factor}\n")
#         f.write(f"tsdf_clip_m: {tsdf_clip}\n\n")

#         f.write("[POINTS PER VOXEL]\n")
#         f.write(f"  n_vox: {ppv['n_vox']}\n")
#         f.write(f"  median: {ppv['median']}\n")
#         f.write(f"  mean: {ppv['mean']}\n")
#         f.write(f"  p95: {ppv['p95']}\n")
#         f.write(f"  max: {ppv['max']}\n\n")

#         f.write("[TSDF STATS] (signed, meters; p95/max are abs)\n")
#         f.write(f"  n: {st['n']}\n")
#         f.write(f"  finite_frac: {st['finite_frac']}\n")
#         f.write(f"  median_signed_m: {st['median']}\n")
#         f.write(f"  mean_signed_m: {st['mean']}\n")
#         f.write(f"  p95_abs_m: {st['p95_abs']}\n")
#         f.write(f"  max_abs_m: {st['max_abs']}\n\n")

#         f.write("[HISTOGRAM]\n")
#         f.write(f"  bins: {hist_info.get('bins')}\n")
#         f.write(f"  range_mm: {hist_info.get('range_mm')}\n")
#         f.write(f"  linear_png: {hist_linear_path}\n")
#         f.write(f"  logy_png: {hist_logy_path}\n\n")

#         f.write("[FILES]\n")
#         f.write(f"  ply: {pts_path}\n")
#         f.write(f"  colorbar: {cb_path}\n")

#         f.write(f"\nruntime_s: {runtime_s}\n")

#     print("[OK] log written:", log_path)

#     return {
#         "run_name": run_name,
#         "run_dir": run_dir,
#         "voxel_size_m": voxel_size,
#         "sdf_trunc_m": sdf_trunc,
#         "defo_mm": defo_mm,
#         "tsdf_clip_m": tsdf_clip,
#         "min_weight_query": min_weight_query,
#         "ppv_n_vox": ppv["n_vox"],
#         "ppv_median": ppv["median"],
#         "ppv_mean": ppv["mean"],
#         "ppv_p95": ppv["p95"],
#         "ppv_max": ppv["max"],
#         "tsdf_n": st["n"],
#         "tsdf_finite_frac": st["finite_frac"],
#         "tsdf_median_signed_m": st["median"],
#         "tsdf_mean_signed_m": st["mean"],
#         "tsdf_p95_abs_m": st["p95_abs"],
#         "tsdf_max_abs_m": st["max_abs"],
#         "ply_path": pts_path,
#         "colorbar_path": cb_path,
#         "hist_linear_path": hist_linear_path,
#         "hist_logy_path": hist_logy_path,
#         "log_path": log_path,
#     }


# def write_summary_txt(rows: List[Dict[str, object]], out_txt: str) -> None:
#     """
#     Writes a tab-separated TXT (TSV) summary. Safe for Excel/pandas but also readable in text form.
#     Overwrites the file each time (so it's always consistent if the sweep aborts).
#     """
#     if not rows:
#         return
#     keys = list(rows[0].keys())

#     with open(out_txt, "w") as f:
#         f.write("\t".join(keys) + "\n")
#         for r in rows:
#             vals = []
#             for k in keys:
#                 v = r.get(k)
#                 if isinstance(v, float):
#                     # keep readable precision
#                     if np.isnan(v):
#                         vals.append("nan")
#                     else:
#                         vals.append(f"{v:.10g}")
#                 else:
#                     vals.append(str(v))
#             f.write("\t".join(vals) + "\n")


# # ------------------------- main -------------------------
# if __name__ == "__main__":
#     sweep_t0 = time.time()
#     summary_rows: List[Dict[str, object]] = []

#     summary_txt = os.path.join(results_root, "TSDF_SWEEP_SUMMARY.txt")

#     for vx, defos in SWEEP:
#         for dmm in defos:
#             row = run_one(voxel_size=vx, defo_mm=dmm)
#             summary_rows.append(row)
#             write_summary_txt(summary_rows, summary_txt)
#             print("[OK] summary updated:", summary_txt)

#     print("\n[ALL DONE] total sweep runtime: {:.2f} s".format(time.time() - sweep_t0))
#     print("[ALL DONE] summary:", summary_txt)