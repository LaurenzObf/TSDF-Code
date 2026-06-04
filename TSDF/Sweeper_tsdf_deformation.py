# -*- coding: utf-8 -*-
"""
Unified TSDF Deformation Sweeper (beule / tz / tilt)

Per run:
- Build TSDF from Epoch1 (reference scans)
- Query signed TSDF on Epoch2 deformation scans
- Save one LAS with:
    * tsdf_value_mm
    * soll_defo_mm (if present in input LAS)

Supports:
- full sweep via sweep_values
- matrix/whitelist control via run_combinations
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import laspy
import numpy as np

from dataset import Dataset
from vdbfusion.pybind.vdb_volume import VDBVolume


# ========================= INPUT =========================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_root = os.path.join(base_dir, "results")
data_dir = os.path.join(base_dir, "data")
epoch2_dir = os.path.join(base_dir, "deformed_scans")

diag_file = "Brucher_202405_P50_georef_diagnostics.txt"
epoch1_scans = ["s1_f1.las", "s2_f1.las", "s3_f1.las"]

# Sweep axes
voxel_sizes_m = [0.05, 0.03, 0.02]
beule_mm_values = [10, 5, 3, 1]
tz_mm_values = [10, 5, 3, 1]
tilt_mgrad_values = [40, 16, 10, 4]

# If True: uses all combinations of voxel_sizes x sweep_values[type]
# If False: uses run_combinations (matrix whitelist)
use_full_grid = False

# Matrix whitelist (like your grey/white table):
# - beule/tz values in mm
# - tilt values in mgrad
run_combinations = {
    "beule": [
        (0.05, 10), (0.05, 5), (0.05, 3), (0.05, 1),
        (0.03, 5), (0.03, 3), (0.03, 1),
        (0.02, 1),
    ],
    "tz": [
        (0.05, 10), (0.05, 5), (0.05, 3), (0.05, 1),
        (0.03, 5), (0.03, 3), (0.03, 1),
        (0.02, 1),
    ],
    "tilt": [
        (0.05, 40), (0.05, 16), (0.05, 10), (0.05, 4),
        (0.03, 16), (0.03, 10), (0.03, 4),
        (0.02, 10), (0.02, 4),
    ],
}

sdf_trunc_factor = 4.0
min_weight_query = 2.0
space_carving = True

out_root_name = "TSDF_SWEEP_DEFORMATION"
# =========================================================


def upsert_extra_dim_float32(las: laspy.LasData, name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float32)
    existing_dims = set(las.point_format.dimension_names)
    if name not in existing_dims:
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))
    setattr(las, name, values)


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

    merged.x = np.concatenate([np.asarray(las.x, dtype=np.float64) for las in las_list], axis=0)
    merged.y = np.concatenate([np.asarray(las.y, dtype=np.float64) for las in las_list], axis=0)
    merged.z = np.concatenate([np.asarray(las.z, dtype=np.float64) for las in las_list], axis=0)
    return merged


def build_epoch2_file_list(deformation_type: str, value: int) -> List[str]:
    if deformation_type == "beule":
        return [
            f"s1_f1_beule_+{value}mm.las",
            f"s2_f1_beule_+{value}mm.las",
            f"s3_f1_beule_+{value}mm.las",
        ]
    if deformation_type == "tz":
        return [
            f"s1_f1_tz+{value}mm.las",
            f"s2_f1_tz+{value}mm.las",
            f"s3_f1_tz+{value}mm.las",
        ]
    if deformation_type == "tilt":
        angle_deg_str = f"{(value / 1000.0):g}"
        return [
            f"s1_f1_tilt+{angle_deg_str}deg.las",
            f"s2_f1_tilt+{angle_deg_str}deg.las",
            f"s3_f1_tilt+{angle_deg_str}deg.las",
        ]
    raise ValueError(f"Unknown deformation_type: {deformation_type}")


def value_label(deformation_type: str, value: int) -> str:
    if deformation_type == "tilt":
        return f"{value:02d}mgrad"
    return f"{value:02d}mm"


def build_tsdf(scan_files: List[str], voxel_size: float) -> VDBVolume:
    sdf_trunc = sdf_trunc_factor * voxel_size
    scan_paths = [os.path.join(data_dir, fn) for fn in scan_files]
    diag_path = os.path.join(data_dir, diag_file)
    ds = Dataset(scan_txt_paths=scan_paths, diagnostics_path=diag_path)
    vol = VDBVolume(voxel_size, sdf_trunc, space_carving=space_carving)

    total = 0
    for i in range(len(ds)):
        points, T_WC = ds[i]
        points = np.asarray(points, dtype=np.float64)
        total += points.shape[0]
        if points.shape[0] > 0:
            vol.integrate(points, T_WC)
    print(f"[INFO] TSDF built (vx={voxel_size:.3f} m), integrated points: {total}")
    return vol


def run_one(vol: VDBVolume, voxel_size: float, deformation_type: str, value: int, run_root: str) -> Dict[str, object]:
    t0 = time.time()
    epoch2_files = build_epoch2_file_list(deformation_type, value)
    epoch2_paths = [os.path.join(epoch2_dir, fn) for fn in epoch2_files]

    for p in epoch2_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing Epoch2 LAS: {p}")

    run_name = (
        f"TSDF_{deformation_type}_{value_label(deformation_type, value)}"
        f"_vx{int(round(voxel_size * 1000)):03d}mm"
    )
    run_dir = os.path.join(run_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    las_out_path = os.path.join(run_dir, f"{run_name}.las")

    las_all = []
    points_all = []
    soll_all = []
    has_any_soll = False

    for path in epoch2_paths:
        las = laspy.read(path)
        P = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        las_all.append(las)
        points_all.append(P)

        soll = get_optional_extra_dim(las, "soll_defo_mm")
        if soll is None:
            soll = np.full(P.shape[0], np.nan, dtype=np.float32)
        else:
            has_any_soll = True
            soll = np.asarray(soll, dtype=np.float32)
        soll_all.append(soll)

    P_eval = np.vstack(points_all).astype(np.float64)
    soll_eval = np.concatenate(soll_all, axis=0).astype(np.float32)

    tsdf_signed = np.asarray(
        vol.query_sdf(P_eval, min_weight=min_weight_query), dtype=np.float32
    ).reshape(-1)
    tsdf_value_mm = (1000.0 * tsdf_signed).astype(np.float32)

    merged = merge_las_files(las_all)
    upsert_extra_dim_float32(merged, "tsdf_value_mm", tsdf_value_mm)
    if has_any_soll:
        upsert_extra_dim_float32(merged, "soll_defo_mm", soll_eval)
    merged.write(las_out_path)

    finite = np.isfinite(tsdf_signed)
    vals = tsdf_signed[finite].astype(np.float64)
    if vals.size == 0:
        mean_abs_mm = np.nan
        p95_abs_mm = np.nan
    else:
        mean_abs_mm = float(np.mean(np.abs(vals)) * 1000.0)
        p95_abs_mm = float(np.percentile(np.abs(vals), 95) * 1000.0)

    runtime_s = time.time() - t0
    print(f"[OK] {run_name}: n={P_eval.shape[0]}, runtime={runtime_s:.2f}s")

    return {
        "run_name": run_name,
        "deformation_type": deformation_type,
        "value": value,
        "value_label": value_label(deformation_type, value),
        "voxel_size_m": voxel_size,
        "n_points": int(P_eval.shape[0]),
        "mean_abs_tsdf_mm": mean_abs_mm,
        "p95_abs_tsdf_mm": p95_abs_mm,
        "runtime_s": runtime_s,
        "las_path": las_out_path,
    }


def build_run_plan() -> List[Tuple[str, float, int]]:
    sweep_values = {
        "beule": beule_mm_values,
        "tz": tz_mm_values,
        "tilt": tilt_mgrad_values,
    }

    plan: List[Tuple[str, float, int]] = []
    if use_full_grid:
        for deformation_type, values in sweep_values.items():
            for vx in voxel_sizes_m:
                for val in values:
                    plan.append((deformation_type, float(vx), int(val)))
    else:
        for deformation_type, combos in run_combinations.items():
            for vx, val in combos:
                plan.append((deformation_type, float(vx), int(val)))
    return plan


def main() -> None:
    run_root = os.path.join(results_root, out_root_name)
    os.makedirs(run_root, exist_ok=True)

    plan = build_run_plan()
    print("[INFO] Start unified TSDF sweep")
    print("[INFO] mode:", "full-grid" if use_full_grid else "matrix-whitelist")
    print("[INFO] output root:", run_root)
    print("[INFO] n_runs:", len(plan))

    tsdf_cache: Dict[float, VDBVolume] = {}
    rows: List[Dict[str, object]] = []

    for deformation_type, vx, val in plan:
        if vx not in tsdf_cache:
            tsdf_cache[vx] = build_tsdf(epoch1_scans, voxel_size=vx)
        vol = tsdf_cache[vx]

        print(
            f"[RUN] type={deformation_type}, value={value_label(deformation_type, val)}, "
            f"vx={int(round(vx * 1000)):03d}mm"
        )
        row = run_one(
            vol=vol,
            voxel_size=vx,
            deformation_type=deformation_type,
            value=val,
            run_root=run_root,
        )
        rows.append(row)

    print("[DONE]")


if __name__ == "__main__":
    main()
