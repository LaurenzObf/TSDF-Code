# -*- coding: utf-8 -*-
"""
Build TSDF from LAS scans and export:
  1) OpenVDB grids (TSDF + weights) to .vdb
  2) Extracted mesh to .ply
  3) Colored point cloud (voxel centers colored by counts) to .ply
  4) Legend (colorbar) as separate .png

Counts are clipped: counts > CLIP_MAX -> CLIP_MAX.
"""

import os
import time
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from typing import List, Tuple
from vdbfusion.pybind.vdb_volume import VDBVolume
from dataset import Dataset


# ========================= INPUT =========================
results_root = "/home/laurenz/MQPW/results"
data_dir = os.path.join(os.path.dirname(__file__), "data")
diag_file = "Brucher_202405_P50_georef_diagnostics.txt"

# scan_files = ["s2_f1.las"]  # or ["s1_f1.las","s2_f1.las","s3_f1.las"]
scan_files = ["s1_f1.las", "s2_f1.las", "s3_f1.las"]
# scan_files = ["s3_f1_ausg3.las"]

voxel_size = 0.03                 # [m]
sdf_trunc = 4.0 * voxel_size      # [m]
space_carving = True

# Mesh extraction
fill_holes = True
min_weight_mesh = 0.5

# Counts export / visualization
ONLY_ACTIVE = True
MIN_WEIGHT_COUNTS = 0.0
CLIP_MAX = 30
COLORMAP = "viridis"  # you can change (e.g. "plasma", "turbo")

# Output naming
run_name = "TSDF_build_s1s2s3_f1_vx03"
export_mesh_ply = True
export_vdb = True
export_colored_counts_cloud = True
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


def export_colored_counts_pointcloud_and_legend(
    vol: VDBVolume,
    out_ply: str,
    out_legend_png: str,
    min_weight: float = 0.0,
    only_active: bool = True,
    clip_max: int = 30,
    cmap_name: str = "viridis",
):
    """
    Creates a colored point cloud from voxel centers:
      - points = voxel centers (world)
      - color  = counts (clipped)
    Saves:
      - out_ply: colored point cloud for CloudCompare
      - out_legend_png: separate legend image (HORIZONTAL)
    """

    # Export centers and counts (same filter settings!)
    ijk_c, centers = vol.export_voxels_ijk_centers(min_weight=min_weight, only_active=only_active)
    ijk_k, counts = vol.export_voxels_ijk_counts(min_weight=min_weight, only_active=only_active)

    centers = np.asarray(centers, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)

    if centers.shape[0] == 0 or counts.shape[0] == 0:
        raise RuntimeError("No centers/counts exported. Check only_active/min_weight.")

    if centers.shape[0] != counts.shape[0]:
        raise RuntimeError("centers and counts size mismatch. Check export logic/order.")

    # Clip counts for visualization
    counts_clip = np.clip(counts, 0.0, float(clip_max))

    # Map counts -> RGB with matplotlib
    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=0.0, vmax=float(clip_max))
    colors_rgba = cmap(norm(counts_clip))          # Nx4
    colors_rgb = colors_rgba[:, :3].astype(np.float64)

    # Build Open3D point cloud (with RGB)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

    # Save PLY (CloudCompare reads RGB)
    o3d.io.write_point_cloud(out_ply, pcd, write_ascii=False, compressed=False)
    print(f"[OK] colored counts point cloud saved: {out_ply}")
    print(f"[INFO] (hint) CloudCompare point size is viewer-side; try 2-4 px.")

    # ---------- HORIZONTAL colorbar ----------
    fig = plt.figure(figsize=(6.0, 1.6), dpi=200)              # wide + low
    ax = fig.add_axes([0.08, 0.45, 0.84, 0.18])  # [left,bottom,width,height]
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax, orientation="horizontal")  # <-- horizontal
    cbar.set_label(f"points/voxel (clipped at {clip_max})")    # font size stays default

    ticks = [0, 6, 12, 18, 24, 30] if clip_max == 30 else np.linspace(0, clip_max, 6)
    cbar.set_ticks(ticks)

    fig.savefig(out_legend_png, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"[OK] legend saved: {out_legend_png}")


if __name__ == "__main__":
    t0 = time.time()

    run_dir = os.path.join(results_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    vdb_path = os.path.join(run_dir, f"{run_name}.vdb")
    mesh_path = os.path.join(run_dir, f"{run_name}.ply")
    counts_cloud_path = os.path.join(run_dir, f"{run_name}_voxelcenters_countsRGB.ply")
    legend_path = os.path.join(run_dir, f"{run_name}_counts_legend.png")

    # 1) Build TSDF
    vol, _ = build_volume(
        scan_files=scan_files,
        diag_file=diag_file,
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        space_carving=space_carving,
    )

    # 1b) quick stats for points/voxel (MORE OUTPUT)
    ijk, counts = vol.export_voxels_ijk_counts(min_weight=MIN_WEIGHT_COUNTS, only_active=ONLY_ACTIVE)
    counts = np.asarray(counts).astype(np.int64).reshape(-1)

    if counts.size > 0:
        p05 = float(np.percentile(counts, 5))
        p50 = float(np.percentile(counts, 50))
        p95 = float(np.percentile(counts, 95))
        p99 = float(np.percentile(counts, 99))
        mean = float(np.mean(counts))
        median = float(np.median(counts))
        mx = int(np.max(counts))

        print(f"[INFO] voxels exported: {counts.size}")
        print("[INFO] points/voxel stats:")
        print(f"       mean={mean:.2f}, median={median:.2f}, p05={p05:.2f}, p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}, max={mx:d}")
    else:
        print("[WARN] No voxels exported for counts. Check ONLY_ACTIVE/MIN_WEIGHT_COUNTS.")

    # 2) Export OpenVDB grids
    if export_vdb:
        vol.extract_vdb_grids(vdb_path)
        print("[OK] TSDF grids saved:", vdb_path)

    # 3) Extract and export mesh
    if export_mesh_ply:
        mesh = extract_mesh(vol, fill_holes=fill_holes, min_weight_mesh=min_weight_mesh)
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print("[OK] mesh saved:", mesh_path)

    # 4) Export colored voxel-center point cloud + legend
    if export_colored_counts_cloud:
        export_colored_counts_pointcloud_and_legend(
            vol=vol,
            out_ply=counts_cloud_path,
            out_legend_png=legend_path,
            min_weight=MIN_WEIGHT_COUNTS,
            only_active=ONLY_ACTIVE,
            clip_max=CLIP_MAX,
            cmap_name=COLORMAP,
        )

    print("[OK] runtime: {:.2f} s".format(time.time() - t0))



# # -*- coding: utf-8 -*-
# """
# Build TSDF from LAS scans and export:
#   1) OpenVDB grids (TSDF + weights) to .vdb
#   2) Extracted mesh to .ply
#   3) Colored point cloud (voxel centers colored by counts) to .ply
#   4) Legend (colorbar) as separate .png

# Counts are clipped: counts > CLIP_MAX -> CLIP_MAX.
# """

# import os
# import time
# import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.colors import Normalize

# from typing import List, Tuple
# from vdbfusion.pybind.vdb_volume import VDBVolume
# from dataset import Dataset


# # ========================= INPUT =========================
# results_root = "/home/laurenz/MQPW/results"
# data_dir = os.path.join(os.path.dirname(__file__), "data")
# diag_file = "Brucher_202405_P50_georef_diagnostics.txt"

# # scan_files = ["s2_f1.las"]  # or ["s1_f1.las","s2_f1.las","s3_f1.las"]
# scan_files = ["s1_f1.las","s2_f1.las","s3_f1.las"]
 
# voxel_size = 0.03                 # [m]
# sdf_trunc = 4.0 * voxel_size      # [m]
# space_carving = True

# # Mesh extraction
# fill_holes = True
# min_weight_mesh = 0.5

# # Counts export / visualization
# ONLY_ACTIVE = True
# MIN_WEIGHT_COUNTS = 0.0
# CLIP_MAX = 30
# COLORMAP = "viridis"  # you can change (e.g. "plasma", "turbo")

# # Output naming
# run_name = "TSDF_build_s1s2s3_f1_vx03"
# export_mesh_ply = True
# export_vdb = True
# export_colored_counts_cloud = True
# # =========================================================


# def build_volume(scan_files: List[str],
#                  diag_file: str,
#                  voxel_size: float,
#                  sdf_trunc: float,
#                  space_carving: bool) -> Tuple[VDBVolume, Dataset]:
#     scan_paths = [os.path.join(data_dir, fn) for fn in scan_files]
#     diag_path = os.path.join(data_dir, diag_file)

#     ds = Dataset(scan_txt_paths=scan_paths, diagnostics_path=diag_path)
#     vol = VDBVolume(voxel_size, sdf_trunc, space_carving=space_carving)

#     print("[INFO] start TSDF Integration")
#     total_points = 0
#     for i in range(len(ds)):
#         points, T_WC = ds[i]
#         points = np.asarray(points, dtype=np.float64)
#         total_points += points.shape[0]
#         print(f"[INFO] integrate {i+1}/{len(ds)} ({ds.scan_names[i]}): {points.shape[0]} points")
#         if points.shape[0] > 0:
#             vol.integrate(points, T_WC)

#     print(f"[INFO] integrated total points: {total_points}")
#     return vol, ds


# def extract_mesh(vol: VDBVolume,
#                  fill_holes: bool,
#                  min_weight_mesh: float) -> o3d.geometry.TriangleMesh:
#     verts, faces = vol.extract_triangle_mesh(fill_holes=fill_holes, min_weight=min_weight_mesh)
#     V = np.asarray(verts, dtype=np.float64)
#     F = np.asarray(faces, dtype=np.int32)

#     if V.size == 0 or F.size == 0:
#         raise RuntimeError("Mesh extraction returned empty mesh. Check integration/min_weight_mesh.")

#     mesh = o3d.geometry.TriangleMesh(
#         vertices=o3d.utility.Vector3dVector(V),
#         triangles=o3d.utility.Vector3iVector(F),
#     )
#     mesh.compute_vertex_normals()
#     return mesh


# def export_colored_counts_pointcloud_and_legend(
#     vol: VDBVolume,
#     out_ply: str,
#     out_legend_png: str,
#     min_weight: float = 0.0,
#     only_active: bool = True,
#     clip_max: int = 30,
#     cmap_name: str = "viridis",
#     point_size_hint: float = 1.0,  # just info for you; PLY stores points only
# ):
#     """
#     Creates a colored point cloud from voxel centers:
#       - points = voxel centers (world)
#       - color  = counts (clipped)
#     Saves:
#       - out_ply: colored point cloud for CloudCompare
#       - out_legend_png: separate legend image
#     """

#     # Export centers and counts (same filter settings!)
#     ijk_c, centers = vol.export_voxels_ijk_centers(min_weight=min_weight, only_active=only_active)
#     ijk_k, counts = vol.export_voxels_ijk_counts(min_weight=min_weight, only_active=only_active)

#     centers = np.asarray(centers, dtype=np.float64)
#     counts = np.asarray(counts, dtype=np.float64)

#     if centers.shape[0] == 0 or counts.shape[0] == 0:
#         raise RuntimeError("No centers/counts exported. Check only_active/min_weight.")

#     if centers.shape[0] != counts.shape[0]:
#         # If something ever mismatches, you can align via ijk keys. For now, assume consistent.
#         raise RuntimeError("centers and counts size mismatch. Check export logic/order.")

#     # Clip counts for visualization
#     counts_clip = np.clip(counts, 0.0, float(clip_max))

#     # Map counts -> RGB with matplotlib
#     cmap = cm.get_cmap(cmap_name)
#     norm = Normalize(vmin=0.0, vmax=float(clip_max))
#     colors_rgba = cmap(norm(counts_clip))          # Nx4
#     colors_rgb = colors_rgba[:, :3].astype(np.float64)

#     # Build Open3D point cloud (with RGB)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(centers)
#     pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

#     # Save PLY (CloudCompare reads RGB)
#     o3d.io.write_point_cloud(out_ply, pcd, write_ascii=False, compressed=False)
#     print(f"[OK] colored counts point cloud saved: {out_ply}")
#     print(f"[INFO] (hint) CloudCompare point size is viewer-side; try 2-4 px.")

#     # Save legend as separate PNG
#     fig = plt.figure(figsize=(2.2, 6.0), dpi=200)
#     ax = fig.add_axes([0.35, 0.05, 0.25, 0.9])  # [left,bottom,width,height]
#     sm = cm.ScalarMappable(norm=norm, cmap=cmap)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, cax=ax)
#     cbar.set_label(f"points/voxel (clipped at {clip_max})", rotation=90)
#     # nicer ticks
#     ticks = [0, 6, 12, 18, 24, 30] if clip_max == 30 else np.linspace(0, clip_max, 6)
#     cbar.set_ticks(ticks)
#     fig.savefig(out_legend_png, bbox_inches="tight", transparent=True)
#     plt.close(fig)
#     print(f"[OK] legend saved: {out_legend_png}")


# if __name__ == "__main__":
#     t0 = time.time()

#     run_dir = os.path.join(results_root, run_name)
#     os.makedirs(run_dir, exist_ok=True)

#     vdb_path = os.path.join(run_dir, f"{run_name}.vdb")
#     mesh_path = os.path.join(run_dir, f"{run_name}.ply")
#     counts_cloud_path = os.path.join(run_dir, f"{run_name}_voxelcenters_countsRGB.ply")
#     legend_path = os.path.join(run_dir, f"{run_name}_counts_legend.png")

#     # 1) Build TSDF
#     vol, _ = build_volume(
#         scan_files=scan_files,
#         diag_file=diag_file,
#         voxel_size=voxel_size,
#         sdf_trunc=sdf_trunc,
#         space_carving=space_carving,
#     )

#     # quick stats
#     ijk, counts = vol.export_voxels_ijk_counts(min_weight=MIN_WEIGHT_COUNTS, only_active=ONLY_ACTIVE)
#     counts = np.asarray(counts).astype(np.int64)
#     if counts.size > 0:
#         print(f"[INFO] voxels exported: {counts.size}")
#         print(f"[INFO] quantiles: p50={np.percentile(counts,50):.2f}, p75={np.percentile(counts,75):.2f}, "
#               f"p90={np.percentile(counts,90):.2f}, p95={np.percentile(counts,95):.2f}, p99={np.percentile(counts,99):.2f}")
#         print(f"[INFO] points per voxel: median={np.median(counts):.2f}, mean={np.mean(counts):.2f}, max={counts.max():d}")

#     # 2) Export OpenVDB grids
#     if export_vdb:
#         vol.extract_vdb_grids(vdb_path)
#         print("[OK] TSDF grids saved:", vdb_path)

#     # 3) Extract and export mesh
#     if export_mesh_ply:
#         mesh = extract_mesh(vol, fill_holes=fill_holes, min_weight_mesh=min_weight_mesh)
#         o3d.io.write_triangle_mesh(mesh_path, mesh)
#         print("[OK] mesh saved:", mesh_path)

#     # 4) Export colored voxel-center point cloud + legend
#     if export_colored_counts_cloud:
#         export_colored_counts_pointcloud_and_legend(
#             vol=vol,
#             out_ply=counts_cloud_path,
#             out_legend_png=legend_path,
#             min_weight=MIN_WEIGHT_COUNTS,
#             only_active=ONLY_ACTIVE,
#             clip_max=CLIP_MAX,
#             cmap_name=COLORMAP,
#         )

#     print("[OK] runtime: {:.2f} s".format(time.time() - t0))