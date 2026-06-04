import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import laspy
import numpy as np
import open3d as o3d  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TSDF_DIR = PROJECT_ROOT / "TSDF"
if str(TSDF_DIR) not in sys.path:
    sys.path.insert(0, str(TSDF_DIR))

from dataset import Dataset

# ========================= INPUT =========================
base_dir = PROJECT_ROOT
data_dir = base_dir / "data"
epoch2_dir = base_dir / "deformed_scans"
results_root = base_dir / "results" / "PSR"
diag_file = "Brucher_202405_P50_georef_diagnostics.txt"
epoch1_scans = ["s1_f1.las", "s2_f1.las", "s3_f1.las"]

# Octree depth rows
depth_values = [12, 13, 14]

# --- Deformation spaces ---
beule_mm_values = [10, 5, 3, 1]
tz_mm_values = [10, 5, 3, 1]
tilt_mgrad_values = [40, 16, 10, 4]

# Only these combinations are computed (white cells):
# Format per deformation:
# - beule / tz: (depth, value_mm)
# - tilt:       (depth, value_mgrad)
run_combinations = {
    "beule": [
        (12, 10),
        (12, 5),
        (12, 3),
        (12, 1),
        (13, 5),
        (13, 3),
        (13, 1),
        (14, 1),
    ],
    "tz": [
        (12, 10),
        (12, 5),
        (12, 3),
        (12, 1),
        (13, 5),
        (13, 3),
        (13, 1),
        (14, 1),
    ],
    "tilt": [
        (12, 40),
        (12, 16),
        (12, 10),
        (12, 4),
        (13, 16),
        (13, 10),
        (13, 4),
        (14, 4),
    ],
}

# Normal estimation parameters
normal_radius_m = 0.05
normal_max_nn = 30

# Poisson parameters
scale = 1.05
linear_fit = True
density_quantile = 0.02

out_root_name = "PSR_SWEEP_DEFORMATION_MATRIX"
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


def get_scan_positions_from_dataset(ds: Dataset) -> np.ndarray:
    positions = []
    for i in range(len(ds)):
        _points, T = ds[i]
        T = np.asarray(T)
        if T.shape == (4, 4):
            positions.append(T[:3, 3].astype(np.float64))
        elif T.shape in [(3,), (3, 1)]:
            positions.append(T.reshape(3).astype(np.float64))
        else:
            raise RuntimeError(f"Unbekanntes Transformationsformat bei ds[{i}]: {T.shape}")
    if not positions:
        raise RuntimeError("Keine Scannerpositionen gefunden.")
    return np.vstack(positions)


def compute_outward_dir_from_scanners(points: np.ndarray, scanner_positions: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    v = scanner_positions - center[None, :]
    d = v.mean(axis=0)
    n = np.linalg.norm(d)
    if not np.isfinite(n) or n < 1e-12:
        raise RuntimeError("Konnte preferred direction aus Scannerpositionen nicht bestimmen.")
    return d / n


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


def build_reference_psr(depth: int) -> Tuple[o3d.geometry.TriangleMesh, o3d.t.geometry.RaycastingScene, np.ndarray]:
    scan_paths = [str(data_dir / fn) for fn in epoch1_scans]
    ds = Dataset(scan_txt_paths=scan_paths, diagnostics_path=str(data_dir / diag_file))
    scanner_positions = get_scan_positions_from_dataset(ds)
    ref_points = np.vstack([np.asarray(points, dtype=np.float64) for points in ds.points_list]).astype(np.float64)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref_points))
    ensure_normals(pcd)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        scale=scale,
        linear_fit=linear_fit,
    )
    densities = np.asarray(densities)
    threshold = float(np.quantile(densities, density_quantile))
    mesh.remove_vertices_by_mask(densities < threshold)
    mesh.compute_vertex_normals()

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)
    preferred_dir = compute_outward_dir_from_scanners(ref_points, scanner_positions)
    print(f"[INFO] REF preferred_dir: {preferred_dir}")
    return mesh, scene, preferred_dir


def compute_signed_distance_mm(
    scene: o3d.t.geometry.RaycastingScene, points: np.ndarray, preferred_dir: np.ndarray
) -> np.ndarray:
    q = o3d.core.Tensor(points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    closest = scene.compute_closest_points(q)
    closest_pts = closest["points"].numpy().astype(np.float32, copy=False)
    normals = closest["primitive_normals"].numpy().astype(np.float32, copy=False)

    flip = np.sum(normals * preferred_dir[None, :], axis=1) < 0.0
    normals[flip] *= -1.0

    delta = points.astype(np.float32, copy=False) - closest_pts
    sign = np.sign(np.sum(delta * normals, axis=1)).astype(np.float32)
    sign[sign == 0.0] = 1.0
    dist_m = np.linalg.norm(delta, axis=1)
    return (1000.0 * sign * dist_m).astype(np.float32)


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


def run_one(
    deformation_type: str,
    depth: int,
    value: int,
    run_root: Path,
    ref_mesh: o3d.geometry.TriangleMesh,
    ref_scene: o3d.t.geometry.RaycastingScene,
    preferred_dir: np.ndarray,
) -> Dict[str, object]:
    t0 = time.time()
    input_paths = build_epoch2_file_list(deformation_type, value)
    points, las_all, soll_defo_mm, has_any_soll = load_las_points(input_paths)
    ist_defo_mm = compute_signed_distance_mm(ref_scene, points, preferred_dir)

    run_name = f"PSR_{deformation_type}_{value_label(deformation_type, value)}_d{depth}"
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
    n_verts = int(np.asarray(ref_mesh.vertices).shape[0])
    n_tris = int(np.asarray(ref_mesh.triangles).shape[0])

    print(
        f"[OK] {run_name}: verts={n_verts}, tris={n_tris}, "
        f"runtime={runtime_s:.2f}s, LAS={las_path.name}"
    )
    return {
        "run_name": run_name,
        "deformation_type": deformation_type,
        "depth": depth,
        "value": value,
        "value_label": value_label(deformation_type, value),
        "n_points": int(points.shape[0]),
        "n_vertices": n_verts,
        "n_triangles": n_tris,
        "density_threshold": np.nan,
        "mean_ist_defo_mm": float(np.nanmean(ist_defo_mm)),
        "runtime_s": runtime_s,
        "las_path": str(las_path),
    }


def write_summary(rows: List[Dict[str, object]], out_path: Path) -> None:
    keys = [
        "run_name",
        "deformation_type",
        "depth",
        "value",
        "value_label",
        "n_points",
        "n_vertices",
        "n_triangles",
        "density_threshold",
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


def validate_combinations() -> None:
    valid_depth = set(depth_values)
    valid_vals = {
        "beule": set(beule_mm_values),
        "tz": set(tz_mm_values),
        "tilt": set(tilt_mgrad_values),
    }
    for deformation_type, combos in run_combinations.items():
        if deformation_type not in valid_vals:
            raise ValueError(f"Unknown key in run_combinations: {deformation_type}")
        for depth, value in combos:
            if depth not in valid_depth:
                raise ValueError(f"Invalid depth in run_combinations[{deformation_type}]: {depth}")
            if value not in valid_vals[deformation_type]:
                raise ValueError(
                    f"Invalid value in run_combinations[{deformation_type}]: {value}"
                )


def main() -> None:
    run_root = results_root / out_root_name
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "summary.tsv"

    validate_combinations()

    print("[INFO] Start PSR sweep")
    print("[INFO] run_combinations:", run_combinations)
    print("[INFO] output root:", run_root)

    ref_cache: Dict[int, Tuple[o3d.geometry.TriangleMesh, o3d.t.geometry.RaycastingScene, np.ndarray]] = {}
    rows: List[Dict[str, object]] = []
    for deformation_type in ("beule", "tz", "tilt"):
        combos = run_combinations.get(deformation_type, [])
        if not combos:
            continue
        print(f"[INFO] deformation_type={deformation_type}, n_runs={len(combos)}")
        for depth, value in combos:
            print(
                f"[RUN] type={deformation_type}, depth={depth}, "
                f"value={value_label(deformation_type, value)}"
            )
            if depth not in ref_cache:
                ref_cache[depth] = build_reference_psr(depth)
            ref_mesh, ref_scene, preferred_dir = ref_cache[depth]
            row = run_one(
                deformation_type=deformation_type,
                depth=depth,
                value=value,
                run_root=run_root,
                ref_mesh=ref_mesh,
                ref_scene=ref_scene,
                preferred_dir=preferred_dir,
            )
            rows.append(row)

    write_summary(rows, summary_path)
    print("[OK] Summary:", summary_path)
    print("[DONE]")


if __name__ == "__main__":
    main()
