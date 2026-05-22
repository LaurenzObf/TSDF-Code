# -*- coding: utf-8 -*-
import os
import time
import numpy as np

from typing import Tuple

from vdbfusion.pybind.vdb_volume import VDBVolume
from dataset import Dataset

import laspy
import open3d as o3d


# ========================= INPUT =========================
project_dir = "/home/laurenz/MQPW"
data_dir = os.path.join(project_dir, "data")
results_dir = os.path.join(project_dir, "results")

# TSDF wird aus EINEM Scan gebaut
scan_files = ["s2_f1_part1.las"]
diag_file = "Brucher_202405_P50_georef_diagnostics.txt"

# TLS Punktwolke zum Vergleich (kann gleich sein wie scan_files[0])
tls_las_file = os.path.join(data_dir, "s2_f1_part1.las")

# TSDF Parameter
voxel_size = 0.05
sdf_trunc = 4.0 * voxel_size
space_carving = True

# Query / Mesh
min_weight_query = 0.5
min_weight_mesh = 0.5
fill_holes = True

# Performance
chunk = 250_000  # nur Batch-Größe, KEIN Downsampling!

basename = "TLS_vs_TSDF_scan_s2_vx05_full"
save_npz = True
# =========================================================


def build_volume(scan_files, diag_file, voxel_size, sdf_trunc):
    scan_paths = [os.path.join(data_dir, fn) for fn in scan_files]
    diag_path = os.path.join(data_dir, diag_file)

    ds = Dataset(scan_txt_paths=scan_paths, diagnostics_path=diag_path)
    vol = VDBVolume(voxel_size, sdf_trunc, space_carving=space_carving)

    print("[INFO] start TSDF Integration")
    total = 0
    for i in range(len(ds)):
        points, T_WC = ds[i]
        total += len(points)
        print("[INFO] integrate Scan {} ({}): {} Punkte".format(i + 1, ds.scan_names[i], len(points)))
        vol.integrate(points, T_WC)
    print("[INFO] Gesamtpunkte integriert: {}".format(total))

    return vol, ds


def query_sdf_chunked(vol, P, min_weight, chunk):
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P muss Shape (N,3) haben")

    out = np.empty((P.shape[0],), dtype=np.float32)
    for s in range(0, P.shape[0], chunk):
        e = min(s + chunk, P.shape[0])
        out[s:e] = np.asarray(vol.query_sdf(P[s:e], min_weight=min_weight), dtype=np.float32).reshape(-1)
    return out


def load_las_points(path):
    las = laspy.read(path)
    return np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)


def point_to_mesh_distance(mesh, P):
    """
    Exakte point->mesh distance via Open3D RaycastingScene.
    """
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)

    P_t = o3d.core.Tensor(P.astype(np.float32))
    d = scene.compute_distance(P_t).numpy().astype(np.float32)
    return d


def summarize(name, x):
    x = np.asarray(x)
    finite = np.isfinite(x)
    xf = x[finite]
    if xf.size == 0:
        return {
            "name": name,
            "n": int(x.size),
            "finite_frac": float(finite.mean()),
            "min": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "name": name,
        "n": int(x.size),
        "finite_frac": float(finite.mean()),
        "min": float(np.min(xf)),
        "mean": float(np.mean(xf)),
        "median": float(np.median(xf)),
        "p95": float(np.percentile(xf, 95)),
        "max": float(np.max(xf)),
    }


if __name__ == "__main__":
    t0 = time.time()

    run_dir = os.path.join(results_dir, "Eval_{}".format(basename))
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "log.txt")
    npz_path = os.path.join(run_dir, "eval_{}.npz".format(basename))
    mesh_path = os.path.join(run_dir, "mesh_{}.ply".format(basename))

    # 1) TSDF bauen
    vol, _ = build_volume(scan_files, diag_file, voxel_size, sdf_trunc)
    print("has query_sdf:", hasattr(vol, "query_sdf"))

    # 2) Mesh aus TSDF (nur als Referenz-Oberfläche für Distanz)
    print("[INFO] extract mesh from TSDF (for point->mesh distance)")
    verts, faces = vol.extract_triangle_mesh(fill_holes=fill_holes, min_weight=min_weight_mesh)
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int32)

    if V.shape[0] == 0 or F.shape[0] == 0:
        raise RuntimeError("Mesh ist leer. Check Integration / min_weight_mesh.")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print("[OK] Mesh saved:", mesh_path)

    # 3) TLS Punkte laden (FULL, kein Downsample)
    print("[INFO] load LAS:", tls_las_file)
    P = load_las_points(tls_las_file)
    print("[INFO] points loaded:", P.shape[0])

    # 4) TSDF an TLS Punkten (Chunking)
    print("[INFO] query TSDF on TLS points (chunked)")
    tsdf_vals = query_sdf_chunked(vol, P, min_weight=min_weight_query, chunk=chunk)

    # 5) Point->Mesh Distanz
    print("[INFO] compute point->mesh distances via RaycastingScene (exact)")
    p2m = point_to_mesh_distance(mesh, P)

    # 6) Stats
    st_abs_tsdf = summarize("|tsdf(P)|", np.abs(tsdf_vals))
    st_p2m = summarize("point_to_mesh(P)", p2m)

    print("\n[STATS] |TSDF| (m)")
    for k, v in st_abs_tsdf.items():
        if k != "name":
            print("  {}: {}".format(k, v))

    print("\n[STATS] point->mesh distance (m)")
    for k, v in st_p2m.items():
        if k != "name":
            print("  {}: {}".format(k, v))

    with open(log_path, "w") as f:
        f.write("### TLS vs TSDF Evaluation (FULL, no downsample) ###\n")
        f.write("scan_files (tsdf built from): {}\n".format(scan_files))
        f.write("tls_las_file: {}\n".format(tls_las_file))
        f.write("voxel_size: {}\n".format(voxel_size))
        f.write("sdf_trunc: {}\n".format(sdf_trunc))
        f.write("min_weight_query: {}\n".format(min_weight_query))
        f.write("min_weight_mesh: {}\n".format(min_weight_mesh))
        f.write("chunk: {}\n".format(chunk))
        f.write("\n[ABS_TSDF]\n")
        f.write(str(st_abs_tsdf) + "\n")
        f.write("\n[POINT2MESH]\n")
        f.write(str(st_p2m) + "\n")

    print("[OK] log saved:", log_path)

    if save_npz:
        np.savez_compressed(
            npz_path,
            P=P.astype(np.float64),
            tsdf=tsdf_vals.astype(np.float32),
            p2m=p2m.astype(np.float32),
            voxel_size=np.float64(voxel_size),
            sdf_trunc=np.float64(sdf_trunc),
            min_weight_query=np.float32(min_weight_query),
            min_weight_mesh=np.float32(min_weight_mesh),
        )
        print("[OK] npz saved:", npz_path)

    print("[OK] runtime: {:.2f} s".format(time.time() - t0))
