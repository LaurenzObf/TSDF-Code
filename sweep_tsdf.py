# -*- coding: utf-8 -*-
"""
SWEEP TSDF PARAMETERS (voxel_size x trunc_factor)

Build TSDF from multiple LAS scans and export per run:
  1) OpenVDB grids (TSDF + weights) to .vdb
  2) Extracted mesh to .ply

No self-checks, no querying, no downsampling.
"""

import os
import time
import itertools
import numpy as np
import open3d as o3d
from datetime import datetime
from typing import List, Tuple

from vdbfusion.pybind.vdb_volume import VDBVolume
from dataset import Dataset


# ========================= SETTINGS =========================
results_root = "/home/laurenz/MQPW/results"
data_dir = os.path.join(os.path.dirname(__file__), "data")
diag_file = "Brucher_202405_P50_georef_diagnostics.txt"

# Multiple LAS files
scan_files = ["s1_f1.las", "s2_f1.las", "s3_f1.las"]

# Sweep grid
VOXELS = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]  # [m]
TRUNC_FACTORS = [4.0]

# TSDF integration
space_carving = True

# Mesh extraction
fill_holes = True
min_weight_mesh = 0.5

# Outputs
export_vdb = True
export_mesh_ply = True

RUN_PREFIX = "TSDFSweep_s1s2s3_f1"
# ===========================================================


def make_sweep_root() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(results_root, f"{RUN_PREFIX}_{ts}")
    os.makedirs(root, exist_ok=False)
    return root


def build_volume(scan_files: List[str],
                 diag_file: str,
                 voxel_size: float,
                 sdf_trunc: float,
                 space_carving: bool) -> Tuple[VDBVolume, Dataset, int]:
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
    return vol, ds, total_points


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


def run_one(sweep_root: str, voxel_size: float, trunc_factor: float):
    sdf_trunc = float(voxel_size) * float(trunc_factor)

    tag = f"vx{voxel_size:.3f}_tr{trunc_factor:.1f}"
    run_dir = os.path.join(sweep_root, tag)
    os.makedirs(run_dir, exist_ok=True)

    vdb_path = os.path.join(run_dir, f"{RUN_PREFIX}_{tag}.vdb")
    mesh_path = os.path.join(run_dir, f"{RUN_PREFIX}_{tag}.ply")
    log_path = os.path.join(run_dir, "log.txt")

    print(f"\n[RUN] {tag}  voxel={voxel_size:.3f}  trunc={sdf_trunc:.3f} (factor={trunc_factor:.1f})")
    t0 = time.time()

    # 1) TSDF
    vol, _, total_points = build_volume(
        scan_files=scan_files,
        diag_file=diag_file,
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        space_carving=space_carving,
    )

    # 2) Export VDB
    if export_vdb:
        if not hasattr(vol, "extract_vdb_grids"):
            raise AttributeError(
                "vol has no extract_vdb_grids(). "
                "Your pybind should expose _extract_vdb_grids as extract_vdb_grids in python wrapper."
            )
        vol.extract_vdb_grids(vdb_path)
        print("[OK] VDB saved:", vdb_path)

    # 3) Export mesh
    n_verts = n_tris = 0
    if export_mesh_ply:
        mesh = extract_mesh(vol, fill_holes=fill_holes, min_weight_mesh=min_weight_mesh)
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        n_verts = np.asarray(mesh.vertices).shape[0]
        n_tris = np.asarray(mesh.triangles).shape[0]
        print("[OK] mesh saved:", mesh_path)
        print(f"[INFO] mesh verts={n_verts}  tris={n_tris}")

    dt = time.time() - t0

    # log
    with open(log_path, "w") as f:
        f.write("### TSDF Sweep Run ###\n")
        f.write(f"scan_files: {scan_files}\n")
        f.write(f"diag_file: {diag_file}\n")
        f.write(f"space_carving: {space_carving}\n")
        f.write(f"voxel_size: {voxel_size}\n")
        f.write(f"trunc_factor: {trunc_factor}\n")
        f.write(f"sdf_trunc: {sdf_trunc}\n")
        f.write(f"fill_holes: {fill_holes}\n")
        f.write(f"min_weight_mesh: {min_weight_mesh}\n")
        f.write(f"integrated_points: {total_points}\n")
        f.write(f"export_vdb: {export_vdb}\n")
        f.write(f"vdb_path: {vdb_path if export_vdb else 'N/A'}\n")
        f.write(f"export_mesh_ply: {export_mesh_ply}\n")
        f.write(f"mesh_path: {mesh_path if export_mesh_ply else 'N/A'}\n")
        f.write(f"mesh_verts: {n_verts}\n")
        f.write(f"mesh_tris: {n_tris}\n")
        f.write(f"runtime_s: {dt:.2f}\n")

    print(f"[OK] done -> {run_dir}  time={dt/60.0:.2f} min")


def main():
    sweep_root = make_sweep_root()
    print("[INFO] sweep_root:", sweep_root)
    print("[INFO] scans:", scan_files)
    print("[INFO] VOXELS:", VOXELS)
    print("[INFO] TRUNC_FACTORS:", TRUNC_FACTORS)

    for vx, tf in itertools.product(VOXELS, TRUNC_FACTORS):
        run_one(sweep_root, voxel_size=float(vx), trunc_factor=float(tf))


if __name__ == "__main__":
    main()