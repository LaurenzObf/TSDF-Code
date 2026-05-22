# -*- coding: utf-8 -*-
"""
Sweeper: TSDF query for multiple tilt magnitudes and voxel sizes.

Per (voxel_size, tilt) combination:
- Build TSDF from Epoch1 reference scans
- Query TSDF on Epoch2 tilted scans
- Export one LAS with `tsdf_value_mm` (+ optional `soll_defo_mm`)

Also writes a summary table:
- results/TSDF_SWEEP_TILT_MATRIX/summary.tsv
"""

import os
import re
import time
from typing import Dict, List, Optional

import laspy
import numpy as np

from dataset import Dataset
from vdbfusion.pybind.vdb_volume import VDBVolume


# ========================= INPUT =========================
base_dir = "/home/laurenz/MQPW"
results_root = os.path.join(base_dir, "results")
data_dir = os.path.join(base_dir, "data")
epoch2_dir = os.path.join(base_dir, "deformed_scans")

diag_file = "Brucher_202405_P50_georef_diagnostics.txt"

# Epoch1 reference
epoch1_scans = ["s1_f1.las", "s2_f1.las", "s3_f1.las"]

# Sweep axes (for validation / optional helpers)
voxel_sizes_m = [0.05, 0.03, 0.02]  # 5cm, 3cm, 2cm, 1cm
tilt_mgrad_values = [40, 16, 10, 4]       # milli-degrees

# Explicit run list (whitelist):
# Only combinations in this list are executed.
# Format: (voxel_size_m, tilt_mgrad)
# Use this to mirror your table and skip "grey" cells deliberately.
run_combinations = [
    (0.05, 40),
    (0.05, 16),
    (0.05, 10),
    (0.05, 4),
    (0.03, 16),
    (0.03, 10),
    (0.03, 4),
    (0.02, 10),
    (0.02, 4),
]

# File naming pattern in `deformed_scans` from `tilt_pc.py`
# Expected examples:
# - s1_f1_tilt+0.04deg.las
# - s2_f1_tilt+0.016deg.las
# - s3_f1_tilt+0.01deg.las
tilt_filename_template = "{scan_stem}_tilt+{angle_deg_str}deg.las"

sdf_trunc_factor = 4.0
min_weight_query = 2.0
space_carving = True

out_root_name = "TSDF_SWEEP_TILT_MATRIX"
# =========================================================


def upsert_extra_dim_float32(las: laspy.LasData, name: str, values: np.ndarray) -> None:
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

    # Copy scaled world coordinates explicitly to avoid header scale/offset mismatches.
    merged.x = np.concatenate([np.asarray(las.x, dtype=np.float64) for las in las_list], axis=0)
    merged.y = np.concatenate([np.asarray(las.y, dtype=np.float64) for las in las_list], axis=0)
    merged.z = np.concatenate([np.asarray(las.z, dtype=np.float64) for las in las_list], axis=0)
    return merged


def scan_prefix_from_name(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"(s\d+_f\d+)", base, flags=re.IGNORECASE)
    return m.group(1).lower() if m else base.lower()


def compact_eval_label(files: List[str]) -> str:
    if not files:
        return "empty"
    stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
    prefixes = [scan_prefix_from_name(f) for f in files]
    suffixes = []
    for stem, prefix in zip(stems, prefixes):
        suffix = stem[len(prefix):].lstrip("_") if stem.lower().startswith(prefix) else stem
        suffixes.append(suffix if suffix else "raw")
    uniq = []
    for s in suffixes:
        if s not in uniq:
            uniq.append(s)
    return uniq[0] if len(uniq) == 1 else "mixed"


def build_tsdf(scan_files: List[str], voxel_size: float, sdf_trunc: float) -> VDBVolume:
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


def build_epoch2_file_list(mgrad: int) -> List[str]:
    angle_deg = float(mgrad) / 1000.0
    angle_deg_str = f"{angle_deg:g}"
    stems = ["s1_f1", "s2_f1", "s3_f1"]
    return [
        tilt_filename_template.format(scan_stem=stem, angle_deg_str=angle_deg_str)
        for stem in stems
    ]


def run_one(vol: VDBVolume, voxel_size: float, mgrad: int, run_root: str) -> Dict[str, object]:
    epoch2_files = build_epoch2_file_list(mgrad)
    epoch2_paths = [os.path.join(epoch2_dir, fn) for fn in epoch2_files]
    for p in epoch2_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing Epoch2 LAS: {p}")

    run_name = f"TSDF_QUERY_tilt{int(mgrad):02d}mgrad_vx{int(round(voxel_size*1000)):03d}mm"
    run_dir = os.path.join(run_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    las_out_path = os.path.join(run_dir, f"{run_name}.las")

    las_all = []
    P_all = []
    soll_all = []
    has_any_soll = False

    for path in epoch2_paths:
        las = load_las(path)
        P = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        las_all.append(las)
        P_all.append(P)

        soll = get_optional_extra_dim(las, "soll_defo_mm")
        if soll is None:
            soll = np.full(P.shape[0], np.nan, dtype=np.float32)
        else:
            has_any_soll = True
            soll = np.asarray(soll, dtype=np.float32)
        soll_all.append(soll)

    P_eval = np.vstack(P_all).astype(np.float64)
    soll_eval = np.concatenate(soll_all, axis=0).astype(np.float32)
    tsdf_signed = np.asarray(vol.query_sdf(P_eval, min_weight=min_weight_query), dtype=np.float32).reshape(-1)
    tsdf_value_mm = (1000.0 * tsdf_signed).astype(np.float32)

    merged = merge_las_files(las_all)
    upsert_extra_dim_float32(merged, "tsdf_value_mm", tsdf_value_mm)
    if has_any_soll:
        upsert_extra_dim_float32(merged, "soll_defo_mm", soll_eval)
    merged.write(las_out_path)

    finite = np.isfinite(tsdf_signed)
    finite_vals = tsdf_signed[finite].astype(np.float64)
    if finite_vals.size == 0:
        median_mm = np.nan
        p95_abs_mm = np.nan
        mean_abs_mm = np.nan
    else:
        median_mm = float(np.median(finite_vals) * 1000.0)
        p95_abs_mm = float(np.percentile(np.abs(finite_vals), 95) * 1000.0)
        mean_abs_mm = float(np.mean(np.abs(finite_vals)) * 1000.0)

    return {
        "run_name": run_name,
        "voxel_size_m": voxel_size,
        "voxel_size_cm": voxel_size * 100.0,
        "tilt_mgrad": mgrad,
        "angle_deg": mgrad / 1000.0,
        "n_points": int(P_eval.shape[0]),
        "finite_frac": float(np.mean(finite)) if finite.size else 0.0,
        "median_tsdf_mm": median_mm,
        "p95_abs_tsdf_mm": p95_abs_mm,
        "mean_abs_tsdf_mm": mean_abs_mm,
        "out_las": las_out_path,
    }


def write_summary(rows: List[Dict[str, object]], out_path: str) -> None:
    keys = [
        "run_name",
        "voxel_size_m",
        "voxel_size_cm",
        "tilt_mgrad",
        "angle_deg",
        "n_points",
        "finite_frac",
        "median_tsdf_mm",
        "p95_abs_tsdf_mm",
        "mean_abs_tsdf_mm",
        "out_las",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r[k]
                if isinstance(v, float):
                    vals.append("nan" if np.isnan(v) else f"{v:.10g}")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")


def main() -> None:
    t0 = time.time()
    run_root = os.path.join(results_root, out_root_name)
    os.makedirs(run_root, exist_ok=True)
    summary_path = os.path.join(run_root, "summary.tsv")

    print("[INFO] Start sweep")
    print("[INFO] voxel sizes (m):", voxel_sizes_m)
    print("[INFO] tilt values (mgrad):", tilt_mgrad_values)
    print("[INFO] run combinations:", run_combinations)
    print("[INFO] output root:", run_root)

    valid_vox = set(voxel_sizes_m)
    valid_tilt = set(tilt_mgrad_values)
    for vx, mg in run_combinations:
        if vx not in valid_vox:
            raise ValueError(f"Invalid voxel size in run_combinations: {vx}")
        if mg not in valid_tilt:
            raise ValueError(f"Invalid tilt_mgrad in run_combinations: {mg}")

    rows: List[Dict[str, object]] = []
    runs_by_voxel: Dict[float, List[int]] = {}
    for vx, mg in run_combinations:
        runs_by_voxel.setdefault(vx, []).append(mg)

    for voxel_size in voxel_sizes_m:
        if voxel_size not in runs_by_voxel:
            print(f"[SKIP] vx={voxel_size:.3f}m has no selected combinations")
            continue

        sdf_trunc = sdf_trunc_factor * voxel_size
        vol = build_tsdf(epoch1_scans, voxel_size=voxel_size, sdf_trunc=sdf_trunc)

        ijk, counts = vol.export_voxels_ijk_counts(min_weight=0.0, only_active=True)
        counts = np.asarray(counts, dtype=np.int64)
        if counts.size:
            print(
                f"[INFO] vx={voxel_size:.3f}m  median_pts/vx={float(np.median(counts)):.2f} "
                f"mean={float(np.mean(counts)):.2f}"
            )

        for mgrad in runs_by_voxel[voxel_size]:
            print(f"[RUN] vx={voxel_size:.3f}m, tilt={mgrad} mgrad")
            row = run_one(vol, voxel_size=voxel_size, mgrad=mgrad, run_root=run_root)
            rows.append(row)
            print(
                "[OK] ",
                row["run_name"],
                f"p95_abs={row['p95_abs_tsdf_mm']:.3f} mm",
                f"finite={row['finite_frac']:.3f}",
            )

    write_summary(rows, summary_path)
    print("[OK] Summary:", summary_path)
    print(f"[DONE] runtime: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
