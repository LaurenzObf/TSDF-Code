import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import laspy
import numpy as np
import open3d as o3d  # type: ignore


# ========================= INPUT =========================
base_dir = Path("/home/laurenz/MQPW")
epoch2_dir = base_dir / "deformed_scans"
results_root = base_dir / "results" / "Ball pivoting"

# Radius factors for BPA (fixed set for all runs)
radius_factors = (1.0, 1.5, 3.0, 5.0, 7.0, 9.0, 11.0)

# --- Deformation spaces ---
beule_mm_values = [10, 5, 3, 1]
tz_mm_values = [10, 5, 3, 1]
tilt_mgrad_values = [40, 16, 10, 4]

# Full sweep (all values for each deformation type)
sweep_values = {
    "beule": beule_mm_values,
    "tz": tz_mm_values,
    "tilt": tilt_mgrad_values,
}

# Normal estimation parameters
normal_radius_m = 0.05
normal_max_nn = 30
# Raycasting chunk size to reduce RAM pressure on large point sets
distance_chunk_size = 1_000_000

out_root_name = "BPA_SWEEP_DEFORMATION"
# =========================================================


def build_epoch2_file_list(deformation_type: str, value: int) -> List[Path]:
    if deformation_type == "beule":
        names = [
            f"s1_f1_beule_+{value}mm.las",
            f"s2_f1_beule_+{value}mm.las",
            f"s3_f1_beule_+{value}mm.las",
        ]
    elif deformation_type == "tz":
        names = [
            f"s1_f1_tz+{value}mm.las",
            f"s2_f1_tz+{value}mm.las",
            f"s3_f1_tz+{value}mm.las",
        ]
    elif deformation_type == "tilt":
        angle_deg_str = f"{(value / 1000.0):g}"
        names = [
            f"s1_f1_tilt+{angle_deg_str}deg.las",
            f"s2_f1_tilt+{angle_deg_str}deg.las",
            f"s3_f1_tilt+{angle_deg_str}deg.las",
        ]
    else:
        raise ValueError(f"Unknown deformation_type: {deformation_type}")
    return [epoch2_dir / n for n in names]


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
        raise RuntimeError("Keine LAS-Dateien zum Zusammenfuehren vorhanden.")

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


def load_las_points(paths: List[Path]) -> Tuple[np.ndarray, List[laspy.LasData], np.ndarray, bool]:
    chunks = []
    las_all = []
    soll_all = []
    has_any_soll = False

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Eingabedatei nicht gefunden: {p}")
        las = laspy.read(str(p))
        pts = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)
        chunks.append(pts)
        las_all.append(las)

        soll = get_optional_extra_dim(las, "soll_defo_mm")
        if soll is None:
            print(f"[WARN] Eingabe-LAS hat kein Feld 'soll_defo_mm': {p}")
            soll = np.full(pts.shape[0], np.nan, dtype=np.float32)
        else:
            has_any_soll = True
            soll = np.asarray(soll, dtype=np.float32)
        soll_all.append(soll)
        print(f"[INFO] geladen: {p.name} ({pts.shape[0]} Punkte)")

    points = np.vstack(chunks).astype(np.float64)
    soll = np.concatenate(soll_all, axis=0).astype(np.float32)
    return points, las_all, soll, has_any_soll


def ensure_normals(pcd: o3d.geometry.PointCloud) -> None:
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius_m, max_nn=normal_max_nn
            )
        )


def value_label(deformation_type: str, value: int) -> str:
    if deformation_type == "tilt":
        return f"{value:02d}mgrad"
    return f"{value:02d}mm"


def run_one(deformation_type: str, value: int, run_root: Path) -> Dict[str, object]:
    t0 = time.time()
    input_paths = build_epoch2_file_list(deformation_type, value)
    points, las_all, soll_defo_mm, has_any_soll = load_las_points(input_paths)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    ensure_normals(pcd)

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = float(np.mean(distances))
    radii = [avg_dist * f for f in radius_factors]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    mesh.compute_vertex_normals()

    # Option A: unsigned point-to-mesh distance as ist_defo_mm
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)
    n = points.shape[0]
    d_m = np.empty(n, dtype=np.float32)
    for s in range(0, n, distance_chunk_size):
        e = min(s + distance_chunk_size, n)
        q = o3d.core.Tensor(points[s:e].astype(np.float32), dtype=o3d.core.Dtype.Float32)
        d_m[s:e] = scene.compute_distance(q).numpy().astype(np.float32, copy=False)
    ist_defo_mm = (1000.0 * d_m).astype(np.float32, copy=False)

    run_name = f"BPA_{deformation_type}_{value_label(deformation_type, value)}"
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    las_path = run_dir / f"{run_name}.las"

    merged_las = merge_las_files(las_all)
    upsert_extra_dim_float32(merged_las, "ist_defo_mm", ist_defo_mm)
    if has_any_soll:
        upsert_extra_dim_float32(merged_las, "soll_defo_mm", soll_defo_mm)
        print("[INFO] Feld 'soll_defo_mm' in Ausgabe-LAS mitgefuehrt.")
    else:
        print("[WARN] Kein Eingabe-LAS enthielt 'soll_defo_mm'; Ausgabe ohne dieses Feld.")
    merged_las.write(las_path)

    runtime_s = time.time() - t0
    n_verts = int(np.asarray(mesh.vertices).shape[0])
    n_tris = int(np.asarray(mesh.triangles).shape[0])

    print(
        f"[OK] {run_name}: verts={n_verts}, tris={n_tris}, "
        f"avg_dist={avg_dist:.6f}, runtime={runtime_s:.2f}s, LAS={las_path.name}"
    )
    return {
        "run_name": run_name,
        "deformation_type": deformation_type,
        "value": value,
        "value_label": value_label(deformation_type, value),
        "avg_nn_dist_m": avg_dist,
        "radius_factors": ",".join([str(x) for x in radius_factors]),
        "n_points": int(points.shape[0]),
        "n_vertices": n_verts,
        "n_triangles": n_tris,
        "mean_ist_defo_mm": float(np.nanmean(ist_defo_mm)),
        "runtime_s": runtime_s,
        "las_path": str(las_path),
    }


def write_summary(rows: List[Dict[str, object]], out_path: Path) -> None:
    keys = [
        "run_name",
        "deformation_type",
        "value",
        "value_label",
        "avg_nn_dist_m",
        "radius_factors",
        "n_points",
        "n_vertices",
        "n_triangles",
        "mean_ist_defo_mm",
        "runtime_s",
        "las_path",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r[k]
                if isinstance(v, float):
                    vals.append(f"{v:.10g}")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")


def validate_sweep_values() -> None:
    valid_vals = {
        "beule": set(beule_mm_values),
        "tz": set(tz_mm_values),
        "tilt": set(tilt_mgrad_values),
    }
    for deformation_type, values in sweep_values.items():
        if deformation_type not in valid_vals:
            raise ValueError(f"Unknown key in sweep_values: {deformation_type}")
        for value in values:
            if value not in valid_vals[deformation_type]:
                raise ValueError(
                    f"Invalid value in sweep_values[{deformation_type}]: {value}"
                )


def main() -> None:
    run_root = results_root / out_root_name
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "summary.tsv"

    validate_sweep_values()

    print("[INFO] Start BPA sweep")
    print("[INFO] sweep_values:", sweep_values)
    print("[INFO] output root:", run_root)

    rows: List[Dict[str, object]] = []
    for deformation_type in ("beule", "tz", "tilt"):
        values = sweep_values.get(deformation_type, [])
        if not values:
            continue
        print(f"[INFO] deformation_type={deformation_type}, n_runs={len(values)}")
        for value in values:
            print(
                f"[RUN] type={deformation_type}, value={value_label(deformation_type, value)}"
            )
            row = run_one(
                deformation_type=deformation_type,
                value=value,
                run_root=run_root,
            )
            rows.append(row)

    write_summary(rows, summary_path)
    print("[OK] Summary:", summary_path)
    print("[DONE]")


if __name__ == "__main__":
    main()
