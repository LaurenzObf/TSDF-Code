# -*- coding: utf-8 -*-
"""
TSDF Noise Study (Epoch2 vs Epoch1) - SELFTEST / GLOBAL
UNCLIPPED + NO surface_band.

- Build TSDF from Epoch1 scans (reference)
- Query signed TSDF at Epoch2 points (UNCLIPPED)
- Print stats for signed TSDF and |TSDF|
- Save histogram PNGs:
    1) signed histogram (linear y)
    2) signed histogram (log-y)
  Optional:
    3) signed histogram ZOOM (linear y)
    4) signed histogram ZOOM (log-y)

Python 3.8
"""

import os
import time
import numpy as np
from typing import List, Optional

import laspy

from vdbfusion.pybind.vdb_volume import VDBVolume
from dataset import Dataset


# ========================= INPUT =========================
base_dir = "/home/laurenz/MQPW"
results_root = os.path.join(base_dir, "results")
data_dir = os.path.join(base_dir, "data")
epoch2_dir = os.path.join(base_dir, "data")  # <-- change to deformed_scans if needed

diag_file = "Brucher_202405_P50_georef_diagnostics.txt"

# Epoch 1 (reference TSDF)
# epoch1_scans = ["s2_f1_ausg3.las"]
epoch1_scans = ["s1_f1.las", "s2_f1.las", "s3_f1.las"]

# Epoch 2 (points to evaluate)
# epoch2_las_files = ["s2_f1_ausg3.las"]
epoch2_las_files = ["s1_f1.las", "s2_f1.las", "s3_f1.las"]

voxel_size = 0.05
sdf_trunc = 4.0 * voxel_size

min_weight_query = 0.5  # unknown filter in query (keeps only voxels with enough weight)

run_name = "TSDF_Histogram_test_cm"

# Histogram settings
bins = 4000
save_zoom = True
hist_xlim_cm = 5.0
# =========================================================


def load_las_points(path: str) -> np.ndarray:
    las = laspy.read(path)
    return np.vstack([las.x, las.y, las.z]).T.astype(np.float64)


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


def summarize_cm(x_m: np.ndarray, name: str) -> str:
    x = np.asarray(x_m, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return f"[{name}] no finite values\n"

    x_cm = 100.0 * x

    q_list = [0, 1, 2.5, 5, 25, 50, 75, 95, 97.5, 99, 99.5, 99.9, 100]
    qs = np.percentile(x_cm, q_list)

    med = np.median(x_cm)
    mad = np.median(np.abs(x_cm - med))
    robust_sigma = 1.4826 * mad

    s = []
    s.append(f"[{name}] (cm)")
    s.append(f"  n={x_cm.size}")
    s.append(f"  mean={float(np.mean(x_cm)):.4f}   median={float(np.median(x_cm)):.4f}   std={float(np.std(x_cm)):.4f}")
    s.append(f"  MAD={float(mad):.4f}   robust_sigma≈{float(robust_sigma):.4f}  (via 1.4826*MAD)")
    s.append("  quantiles (cm):")
    s.append("  " + ", ".join([f"p{p:>5}={v:>8.3f}" for p, v in zip(q_list, qs)]))
    return "\n".join(s) + "\n"


def compute_central_95_interval_cm(tsdf_signed_m: np.ndarray) -> dict:
    x = np.asarray(tsdf_signed_m, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"p2_5": np.nan, "p97_5": np.nan, "median": np.nan}

    x_cm = 100.0 * x
    return {"p2_5": float(np.percentile(x_cm, 2.5)), "p97_5": float(np.percentile(x_cm, 97.5)), "median": float(np.median(x_cm))}


def style_axes(ax, xlabel: str, ylabel: str, title: Optional[str] = None) -> None:
    if title is not None:
        ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis="both", labelsize=20)


def save_histograms(tsdf_signed_m: np.ndarray, out_dir: str, base: str, xlim_cm: float = None) -> None:
    import matplotlib.pyplot as plt

    x = np.asarray(tsdf_signed_m, dtype=np.float64)
    x = x[np.isfinite(x)]
    x_cm = 100.0 * x

    # --- full range ---
    for logy in [False, True]:
        fig = plt.figure(figsize=(12, 6), dpi=120)
        ax = fig.add_subplot(111)
        ax.hist(x_cm, bins=bins)
        style_axes(ax, xlabel="TSDF (cm)", ylabel="count" + (" (log)" if logy else ""), title=f"Signed TSDF histogram (cm){' [log-y]' if logy else ''}")
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        if logy:
            ax.set_yscale("log")

        fn = f"{base}_linear.png" if not logy else f"{base}_logy.png"
        fig.savefig(os.path.join(out_dir, fn), bbox_inches="tight")
        plt.close(fig)

    # --- zoom range ---
    if xlim_cm is not None:
        for logy in [False, True]:
            fig = plt.figure(figsize=(12, 6), dpi=120)
            ax = fig.add_subplot(111)
            ax.hist(x_cm, bins=bins)
            style_axes(ax, xlabel="TSDF (cm)", ylabel="count" + (" (log)" if logy else ""), title=f"Signed TSDF histogram (cm) ZOOM ±{xlim_cm:g}{' [log-y]' if logy else ''}")
            ax.grid(True, which="both", linestyle="--", alpha=0.6)
            ax.set_xlim(-xlim_cm, xlim_cm)
            if logy:
                ax.set_yscale("log")

            fn = f"{base}_zoom{str(xlim_cm).replace('.', 'p')}cm_linear.png" if not logy else f"{base}_zoom{str(xlim_cm).replace('.', 'p')}cm_logy.png"
            fig.savefig(os.path.join(out_dir, fn), bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()

    run_dir = os.path.join(results_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    stats_txt = os.path.join(run_dir, "tsdf_stats_cm.txt")

    # 1) Build TSDF from Epoch1
    vol = build_tsdf_from_epoch1(epoch1_scans)
    print("[INFO] has query_sdf:", hasattr(vol, "query_sdf"))

    # 2) Load Epoch2 points
    P2_all = []
    for fn in epoch2_las_files:
        path = os.path.join(epoch2_dir, fn)
        print("[INFO] load LAS (Epoch 2):", path)
        P = load_las_points(path)
        print("[INFO] points loaded:", P.shape[0])
        P2_all.append(P)

    P_eval = np.vstack(P2_all).astype(np.float64)
    print("[INFO] total Epoch2 points:", P_eval.shape[0])

    # 3) Query signed TSDF (UNCLIPPED)
    print("[INFO] query TSDF at Epoch2 points (SIGNED, UNCLIPPED)")
    tsdf_signed = np.asarray(vol.query_sdf(P_eval, min_weight=min_weight_query), dtype=np.float32).reshape(-1)

    tsdf_f = tsdf_signed[np.isfinite(tsdf_signed)].astype(np.float64)
    abs_tsdf_f = np.abs(tsdf_f)

    # 4) Print + save stats
    s1 = summarize_cm(tsdf_f, "SIGNED TSDF")
    s2 = summarize_cm(abs_tsdf_f, "ABS TSDF |TSDF|")

    stats95 = compute_central_95_interval_cm(tsdf_f)
    central95 = f"[CENTRAL 95% SIGNED] p2.5={stats95['p2_5']:.4f} cm, median={stats95['median']:.4f} cm, p97.5={stats95['p97_5']:.4f} cm\n"

    med_abs = float(np.median(100.0 * abs_tsdf_f)) if abs_tsdf_f.size else float("nan")
    p95_abs = float(np.percentile(100.0 * abs_tsdf_f, 95)) if abs_tsdf_f.size else float("nan")
    p99_abs = float(np.percentile(100.0 * abs_tsdf_f, 99)) if abs_tsdf_f.size else float("nan")

    summary = f"[SUMMARY] median(|TSDF|)={med_abs:.4f} cm   p95(|TSDF|)={p95_abs:.4f} cm   p99(|TSDF|)={p99_abs:.4f} cm\n"

    print(s1)
    print(central95)
    print(s2)
    print(summary)

    with open(stats_txt, "w") as f:
        f.write("### TSDF Noise Study (UNCLIPPED, no surface band) ###\n\n")
        f.write(f"epoch1_scans: {epoch1_scans}\n")
        f.write(f"epoch2_files: {epoch2_las_files}\n")
        f.write(f"voxel_size: {voxel_size}\n")
        f.write(f"sdf_trunc: {sdf_trunc}\n")
        f.write(f"min_weight_query: {min_weight_query}\n\n")
        f.write(s1 + "\n")
        f.write(central95 + "\n")
        f.write(s2 + "\n")
        f.write(summary + "\n")

    # 5) Histograms (linear + log-y) + optional zoom
    save_histograms(tsdf_signed_m=tsdf_signed, out_dir=run_dir, base="tsdf_signed_hist_cm", xlim_cm=(hist_xlim_cm if save_zoom else None))

    print("[OK] stats saved:", stats_txt)
    print("[OK] histograms saved in:", run_dir)
    print("[OK] runtime: {:.2f} s".format(time.time() - t0))