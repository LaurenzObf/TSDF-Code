# -*- coding: utf-8 -*-
"""
TSDF Query Export (Epoch2 vs Epoch1) - LAS only

- Build TSDF aus Epoch1 (Referenz)
- Query SIGNED TSDF an Epoch2 Punkten (GLOBAL)
- Speichert alle Epoch2-Punkte als eine LAS mit:
    * tsdf_value_mm
    * soll_defo_mm (falls in der Eingabe vorhanden)
"""

import os
import time
import numpy as np
from typing import List, Optional
import re

import laspy

from vdbfusion.pybind.vdb_volume import VDBVolume
from dataset import Dataset

# ========================= INPUT =========================
base_dir = "/home/laurenz/MQPW"
results_root = os.path.join(base_dir, "results")
data_dir = os.path.join(base_dir, "data")
epoch2_dir = os.path.join(base_dir, "deformed_scans")  # <-- Epoch2 hier (oder deformed_scans)

diag_file = "Brucher_202405_P50_georef_diagnostics.txt"

# Epoch 1 (reference TSDF)
epoch1_scans = ["s1_f1.las", "s2_f1.las", "s3_f1.las"]

# Epoch 2 (points to evaluate / export)
# epoch2_las_files = ["s1_f1_deformed_+1mm.las", "s2_f1_deformed_+1mm.las", "s3_f1_deformed_+1mm.las"]
epoch2_las_files = ["s1_f1_tz+1mm.las", "s2_f1_tz+1mm.las", "s3_f1_tz+1mm.las"]

voxel_size = 0.02 # m (0.05 = 5cm, 0.02=2cm, 0.01=1cm)
sdf_trunc = 4.0 * voxel_size

min_weight_query = 2.0  # filter unknown TSDF areas

# =========================================================


def upsert_extra_dim_float32(las: laspy.LasData, name: str, values: np.ndarray) -> None:
    """
    Schreibt/überschreibt ein LAS-Extra-Feld als float32.
    Falls das Feld noch nicht existiert, wird es angelegt.
    """
    values = np.asarray(values, dtype=np.float32)
    existing_dims = set(las.point_format.dimension_names)

    if name not in existing_dims:
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))

    setattr(las, name, values)


def load_las(path: str) -> laspy.LasData:
    return laspy.read(path)


def get_optional_extra_dim(las: laspy.LasData, name: str) -> Optional[np.ndarray]:
    dims = set(las.point_format.dimension_names)
    if name not in dims:
        return None
    return np.asarray(getattr(las, name))


def merge_las_files(las_list: List[laspy.LasData]) -> laspy.LasData:
    if not las_list:
        raise RuntimeError("Keine LAS-Dateien zum Zusammenführen vorhanden.")

    first = las_list[0]
    merged = laspy.LasData(first.header)
    dim_names = list(first.point_format.dimension_names)

    total_points = int(sum(len(las.points) for las in las_list))
    merged.points = laspy.ScaleAwarePointRecord.zeros(total_points, header=merged.header)

    for name in dim_names:
        if name in ("X", "Y", "Z"):
            continue
        arrays = [np.asarray(getattr(las, name)) for las in las_list]
        setattr(merged, name, np.concatenate(arrays, axis=0))

    # Wichtig: skalierten Weltkoordinaten kopieren, nicht die rohen X/Y/Z-Integer.
    merged.x = np.concatenate([np.asarray(las.x, dtype=np.float64) for las in las_list], axis=0)
    merged.y = np.concatenate([np.asarray(las.y, dtype=np.float64) for las in las_list], axis=0)
    merged.z = np.concatenate([np.asarray(las.z, dtype=np.float64) for las in las_list], axis=0)

    return merged


def scan_prefix_from_name(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"(s\d+_f\d+)", base, flags=re.IGNORECASE)
    return m.group(1).lower() if m else base.lower()


def dataset_label(files: List[str]) -> str:
    if not files:
        return "empty"

    stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
    prefixes = [scan_prefix_from_name(f) for f in files]

    unique_prefixes = []
    for p in prefixes:
        if p not in unique_prefixes:
            unique_prefixes.append(p)
    prefix_label = "+".join(unique_prefixes)

    suffixes = []
    for stem, prefix in zip(stems, prefixes):
        suffix = stem[len(prefix):].lstrip("_") if stem.lower().startswith(prefix) else stem
        suffixes.append(suffix if suffix else "raw")

    unique_suffixes = []
    for s in suffixes:
        if s not in unique_suffixes:
            unique_suffixes.append(s)

    if len(unique_suffixes) == 1:
        suffix_label = unique_suffixes[0]
    else:
        suffix_label = "mixed"

    return f"{prefix_label}_{suffix_label}".replace("__", "_")


def build_run_name(epoch1_files: List[str], epoch2_files: List[str], voxel_size_m: float) -> str:
    def compact_label(files: List[str]) -> str:
        if not files:
            return "empty"

        stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
        prefixes = [scan_prefix_from_name(f) for f in files]

        suffixes = []
        for stem, prefix in zip(stems, prefixes):
            suffix = stem[len(prefix):].lstrip("_") if stem.lower().startswith(prefix) else stem
            suffixes.append(suffix if suffix else "raw")

        unique_suffixes = []
        for s in suffixes:
            if s not in unique_suffixes:
                unique_suffixes.append(s)

        return unique_suffixes[0] if len(unique_suffixes) == 1 else "mixed"

    ref_label = compact_label(epoch1_files)
    eval_label = compact_label(epoch2_files)
    vx_mm = int(round(voxel_size_m * 1000.0))
    return f"TSDF_QUERY_{eval_label}_vs_{ref_label}_vx{vx_mm:03d}mm"


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


if __name__ == "__main__":
    t0 = time.time()
    run_name = build_run_name(epoch1_scans, epoch2_las_files, voxel_size)

    run_dir = os.path.join(results_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    las_out_path = os.path.join(run_dir, f"{run_name}.las")

    # 1) Build TSDF from Epoch1
    vol = build_tsdf_from_epoch1(epoch1_scans)
    print("has query_sdf:", hasattr(vol, "query_sdf"))

    # --- Points-per-voxel Statistik (aus integrierten Voxeln) ---
    ijk, counts = vol.export_voxels_ijk_counts(min_weight=0.0, only_active=True)
    counts = np.asarray(counts, dtype=np.int64)

    if counts.size == 0:
        print("[WARN] No voxels exported for counts (empty). Check min_weight/only_active.")
    else:
        p50 = float(np.percentile(counts, 50))
        mean = float(np.mean(counts))
        p95 = float(np.percentile(counts, 95))
        mx = int(np.max(counts))
        print(f"[INFO] voxels exported: {counts.size}")
        print(f"[INFO] points per voxel: median={p50:.2f}, mean={mean:.2f}, p95={p95:.2f}, max={mx}")

    # 2) Load Epoch2 points and optional target deformation
    las_all = []
    P2_all = []
    soll_all = []
    has_any_soll = False

    for fn in epoch2_las_files:
        path = os.path.join(epoch2_dir, fn)
        print("[INFO] load LAS (Epoch 2):", path)
        las = load_las(path)
        P = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        print("[INFO] points loaded:", P.shape[0])

        las_all.append(las)
        P2_all.append(P)

        soll = get_optional_extra_dim(las, "soll_defo_mm")
        if soll is None:
            print(f"[WARN] Eingabe-LAS hat kein Feld 'soll_defo_mm': {path}")
            soll = np.full(P.shape[0], np.nan, dtype=np.float32)
        else:
            has_any_soll = True
            soll = np.asarray(soll, dtype=np.float32)
        soll_all.append(soll)

    P_eval = np.vstack(P2_all).astype(np.float64)
    soll_eval = np.concatenate(soll_all, axis=0).astype(np.float32)
    print("[INFO] total Epoch2 points:", P_eval.shape[0])
    if P_eval.shape[0] == 0:
        raise RuntimeError("No Epoch2 points loaded.")

    # 3) Query TSDF at Epoch2 points (signed)
    print("[INFO] query TSDF at Epoch2 points (signed) [GLOBAL]")
    tsdf_signed = np.asarray(
        vol.query_sdf(P_eval, min_weight=min_weight_query),
        dtype=np.float32,
    ).reshape(-1)

    finite = np.isfinite(tsdf_signed)
    if np.any(finite):
        med_signed = float(np.median(tsdf_signed[finite]))
        mean_signed = float(np.mean(tsdf_signed[finite]))
        p95_abs = float(np.percentile(np.abs(tsdf_signed[finite]), 95))
        print(
            f"[INFO] signed TSDF stats: median={med_signed:.6f} m, "
            f"mean={mean_signed:.6f} m, p95(|TSDF|)={p95_abs:.6f} m"
        )
    else:
        print("[WARN] All TSDF values are non-finite (check min_weight_query / TSDF coverage).")

    # 4) Merge input LAS files and add queried TSDF as extra dimension
    merged_las = merge_las_files(las_all)
    tsdf_value_mm = (1000.0 * tsdf_signed).astype(np.float32)

    upsert_extra_dim_float32(merged_las, "tsdf_value_mm", tsdf_value_mm)

    if has_any_soll:
        upsert_extra_dim_float32(merged_las, "soll_defo_mm", soll_eval)

    merged_las.write(las_out_path)
    print("[OK] LAS with TSDF saved:", las_out_path)
    print(
        f"[INFO] Feld 'tsdf_value_mm' geschrieben "
        f"(finite={int(np.count_nonzero(np.isfinite(tsdf_value_mm)))}/{tsdf_value_mm.size})"
    )
    if has_any_soll:
        print("[INFO] Feld 'soll_defo_mm' in Ausgabe-LAS mitgeführt.")
    else:
        print("[WARN] Kein Eingabe-LAS enthielt 'soll_defo_mm'; Ausgabe ohne dieses Feld.")

    print("[OK] runtime: {:.2f} s".format(time.time() - t0))
    print("[DONE]")
