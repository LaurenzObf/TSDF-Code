# -*- coding: utf-8 -*-
"""
Build TSDF from multiple LAS scans and export:
  1) OpenVDB grids (TSDF + weights) to .vdb
  2) Extracted mesh to .ply

No self-checks, no querying, no downsampling.
"""

import os
import time
import numpy as np
import open3d as o3d

from typing import List, Tuple
from vdbfusion.pybind.vdb_volume import VDBVolume
from dataset import Dataset


# ========================= INPUT =========================
results_root = "/home/laurenz/MQPW/results"
data_dir = os.path.join(os.path.dirname(__file__), "data")
diag_file = "Brucher_202405_P50_georef_diagnostics.txt"

# Multiple LAS files
# scan_files = ["s1_f1.las", "s2_f1.las", "s3_f1.las"]
scan_files = ["s2_f1.las"]

voxel_size = 0.05                 # [m]
sdf_trunc = 4.0 * voxel_size      # [m]
space_carving = True

# Mesh extraction
fill_holes = True
min_weight_mesh = 0.5

# Output naming
run_name = "TSDF_build_s1s2s3_f1_vx05"
export_mesh_ply = True
export_vdb = True
# =========================================================


def build_volume(scan_files: List[str],
                 diag_file: str,
                 voxel_size: float,
                 sdf_trunc: float,
                 space_carving: bool) -> Tuple[VDBVolume, Dataset]:
    scan_paths = [os.path.join(data_dir, fn) for fn in scan_files]
    diag_path = os.path.join(data_dir, diag_file)

    ds = Dataset(scan_txt_paths=scan_paths, diagnostics_path=diag_path)
    vol = VDBVolume(voxel_size, sdf_trunc, space_carving=space_carving)

    print("[INFO] start TSDF Integration")
    total_points = 0
    for i in range(len(ds)):
        points, T_WC = ds[i]
        points = np.asarray(points, dtype=np.float64)
        total_points += points.shape[0]
        print(f"[INFO] integrate {i+1}/{len(ds)} ({ds.scan_names[i]}): {points.shape[0]} points")
        if points.shape[0] > 0:
            vol.integrate(points, T_WC)

    print(f"[INFO] integrated total points: {total_points}")
    return vol, ds


def extract_mesh(vol: VDBVolume,
                 fill_holes: bool,
                 min_weight_mesh: float) -> o3d.geometry.TriangleMesh:
    verts, faces = vol.extract_triangle_mesh(fill_holes=fill_holes, min_weight=min_weight_mesh)
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int32)

    if V.size == 0 or F.size == 0:
        raise RuntimeError("Mesh extraction returned empty mesh. Check integration/min_weight_mesh.")

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V),
        triangles=o3d.utility.Vector3iVector(F),
    )
    mesh.compute_vertex_normals()
    return mesh


if __name__ == "__main__":
    t0 = time.time()

    run_dir = os.path.join(results_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    vdb_path = os.path.join(run_dir, f"{run_name}.vdb")
    mesh_path = os.path.join(run_dir, f"{run_name}.ply")

    # 1) Build TSDF
    vol, _ = build_volume(
        scan_files=scan_files,
        diag_file=diag_file,
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        space_carving=space_carving,
    )

    # 1b) Punktdichte: wie viele Punkte pro Voxel (Median)
    ijk, counts = vol.export_voxels_ijk_counts(min_weight=0.0, only_active=True)
    counts = np.asarray(counts).astype(np.int64)
    p50 = np.percentile(counts, 50)
    p75 = np.percentile(counts, 75)
    p90 = np.percentile(counts, 90)
    p95 = np.percentile(counts, 95)
    p99 = np.percentile(counts, 99)
    print(f"[INFO] quantiles: p50={p50:.2f}, p75={p75:.2f}, p90={p90:.2f}, p95={p95:.2f}, p99={p99:.2f}")

    if counts.size == 0:
        print("[WARN] Keine Voxels exportiert (counts leer). Prüfe only_active/min_weight.")
    else:
        med = float(np.median(counts))
        mean = float(np.mean(counts))
        p90 = float(np.percentile(counts, 90))
        mx = int(np.max(counts))

        print(f"[INFO] voxels exported: {counts.size}")
        print(f"[INFO] points per voxel: median={med:.2f}, mean={mean:.2f}, p90={p90:.2f}, max={mx}")

    # 2) Export OpenVDB grids (TSDF + weights)
    if export_vdb:
        if not hasattr(vol, "extract_vdb_grids"):
            raise AttributeError(
                "vol has no extract_vdb_grids(). "
                "Your pybind should expose _extract_vdb_grids as extract_vdb_grids in python wrapper."
            )
        vol.extract_vdb_grids(vdb_path)
        print("[OK] TSDF grids saved:", vdb_path)

    # 3) Extract and export mesh
    if export_mesh_ply:
        mesh = extract_mesh(vol, fill_holes=fill_holes, min_weight_mesh=min_weight_mesh)
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print("[OK] mesh saved:", mesh_path)

    import numpy as np
    import matplotlib.pyplot as plt

    c = counts[counts > 0]
    bins = np.unique(np.logspace(np.log10(1), np.log10(c.max()), 50).astype(int))
    plt.hist(c, bins=bins)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("points per voxel")
    plt.ylabel("voxel count")
    plt.title("Counts distribution (log-log)")
    plt.show()

    print("[OK] runtime: {:.2f} s".format(time.time() - t0))