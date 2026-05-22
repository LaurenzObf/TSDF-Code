
# -*- coding: utf-8 -*-
import numpy as np
import open3d as o3d


mesh_path = "/home/laurenz/MQPW/results/Mesh_Brucher_multiScan/tsdf_Brucher_multiScan.ply"

# 3 Pick-Punkte aus der Punktwolke (CloudCompare)
P = np.array([
    [704.526123, 419.977814, 358.819397],
    [705.158752, 369.716919, 359.960205],
    [704.224121, 406.123810, 356.709381],
], dtype=np.float32)


def mesh_to_tensor_mesh(mesh_legacy: o3d.geometry.TriangleMesh) -> o3d.t.geometry.TriangleMesh:
    V = np.asarray(mesh_legacy.vertices, dtype=np.float32)
    T = np.asarray(mesh_legacy.triangles, dtype=np.int32)
    return o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V), o3d.core.Tensor(T))


if __name__ == "__main__":
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise RuntimeError("Mesh ist leer / konnte nicht geladen werden.")

    # kleines Cleanup (wie in evaluation.py)
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()

    # Raycasting Szene
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_to_tensor_mesh(mesh))

    pts = o3d.core.Tensor(P, dtype=o3d.core.Dtype.Float32)

    # Unsigned Distanz Punkt -> Mesh
    d = scene.compute_distance(pts).numpy().reshape(-1)

    # Optional: nächster Punkt auf dem Mesh (hilfreich zum Debuggen)
    cp = scene.compute_closest_points(pts)
    closest = cp["points"].numpy()
    tri_id = cp["primitive_ids"].numpy().reshape(-1)

    print("Mesh:", mesh_path)
    for i in range(P.shape[0]):
        print(f"\nP{i+1}: {P[i,0]:.6f} {P[i,1]:.6f} {P[i,2]:.6f}")
        print(f"  dist (m): {float(d[i]):.6f}")
        print(f"  closest : {closest[i,0]:.6f} {closest[i,1]:.6f} {closest[i,2]:.6f}")
        print(f"  tri_id  : {int(tri_id[i])}")







### Mesh vs. TLS-Punktwolke ###
# # -*- coding: utf-8 -*-
# """
# evaluation.py

# Vergleich Mesh vs. TLS-Punktwolke:
# - berechnet für jeden Punkt den 3D-Abstand zur Mesh-Oberfläche (closest point distance)
# - speichert Kennzahlen + optional Distanz-Array (npz) + optional farbiges PLY

# Benutzung (Beispiele):
#   /bin/python3 evaluation.py --mesh results/.../mesh_xxx.ply --las data/s2_f1_part1.las --out results/Eval_run1
#   /bin/python3 evaluation.py --mesh mesh.ply --las scan.las --max-points 2000000 --chunk 200000
# """

# import os
# import argparse
# import time
# import numpy as np

# import open3d as o3d
# import laspy


# def read_las_xyz(las_path: str) -> np.ndarray:
#     las = laspy.read(las_path)
#     P = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
#     return P


# def downsample_points(P: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
#     if max_points <= 0 or P.shape[0] <= max_points:
#         return P
#     rng = np.random.default_rng(seed)
#     idx = rng.choice(P.shape[0], size=max_points, replace=False)
#     return P[idx]


# def voxel_downsample_numpy(P: np.ndarray, voxel_size: float) -> np.ndarray:
#     if voxel_size <= 0:
#         return P
#     pc = o3d.geometry.PointCloud()
#     pc.points = o3d.utility.Vector3dVector(P)
#     pc_ds = pc.voxel_down_sample(voxel_size=voxel_size)
#     return np.asarray(pc_ds.points, dtype=np.float64)


# def mesh_to_tensor_mesh(mesh_legacy: o3d.geometry.TriangleMesh) -> "o3d.t.geometry.TriangleMesh":
#     # Open3D "t" API benötigt Float32
#     V = np.asarray(mesh_legacy.vertices, dtype=np.float32)
#     T = np.asarray(mesh_legacy.triangles, dtype=np.int32)
#     mesh_t = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V), o3d.core.Tensor(T))
#     return mesh_t


# def compute_distances_raycast(mesh_legacy: o3d.geometry.TriangleMesh, P: np.ndarray, chunk: int = 250_000) -> np.ndarray:
#     """
#     Exakte unsigned distance point->mesh via RaycastingScene (Open3D t.geometry).
#     Gibt pro Punkt den Abstand zur nächsten Dreiecksfläche zurück.
#     """
#     mesh_t = mesh_to_tensor_mesh(mesh_legacy)
#     scene = o3d.t.geometry.RaycastingScene()
#     _ = scene.add_triangles(mesh_t)

#     out = np.empty((P.shape[0],), dtype=np.float32)
#     for s in range(0, P.shape[0], chunk):
#         e = min(s + chunk, P.shape[0])
#         pts = o3d.core.Tensor(P[s:e].astype(np.float32))
#         d = scene.compute_distance(pts).numpy().astype(np.float32).reshape(-1)
#         out[s:e] = d
#     return out


# def compute_distances_approx(mesh_legacy: o3d.geometry.TriangleMesh, P: np.ndarray, n_samples: int = 2_000_000, chunk: int = 250_000) -> np.ndarray:
#     """
#     Fallback, wenn RaycastingScene nicht verfügbar ist:
#     - mesht zu einer dichten Punktwolke sampeln
#     - Distanz point->(mesh-samples) via KDTree (approximativ)
#     """
#     mesh_samples = mesh_legacy.sample_points_uniformly(number_of_points=int(n_samples))
#     Q = np.asarray(mesh_samples.points, dtype=np.float64)

#     pcd_Q = o3d.geometry.PointCloud()
#     pcd_Q.points = o3d.utility.Vector3dVector(Q)
#     kdtree = o3d.geometry.KDTreeFlann(pcd_Q)

#     out = np.empty((P.shape[0],), dtype=np.float32)
#     for s in range(0, P.shape[0], chunk):
#         e = min(s + chunk, P.shape[0])
#         block = P[s:e]
#         for i in range(block.shape[0]):
#             _, idx, dist2 = kdtree.search_knn_vector_3d(block[i], 1)
#             out[s + i] = np.sqrt(dist2[0]).astype(np.float32)
#     return out


# def summarize(name: str, d: np.ndarray) -> dict:
#     d = np.asarray(d, dtype=np.float64)
#     finite = np.isfinite(d)
#     x = d[finite]
#     return {
#         "name": name,
#         "n": int(d.size),
#         "finite_frac": float(finite.mean()),
#         "mean": float(np.mean(x)) if x.size else float("nan"),
#         "median": float(np.median(x)) if x.size else float("nan"),
#         "p95": float(np.percentile(x, 95)) if x.size else float("nan"),
#         "max": float(np.max(x)) if x.size else float("nan"),
#         "min": float(np.min(x)) if x.size else float("nan"),
#     }


# def export_colored_ply(points: np.ndarray, distances: np.ndarray, out_ply: str, clip_p95: bool = True) -> None:
#     """
#     Speichert Punktwolke als PLY mit Farben nach Distanz (simple blue->red).
#     """
#     d = distances.astype(np.float64)
#     finite = np.isfinite(d)
#     if not np.any(finite):
#         raise RuntimeError("Keine finiten Distanzwerte zum Export.")

#     d_valid = d[finite]
#     dmax = np.percentile(d_valid, 95) if clip_p95 else np.max(d_valid)
#     dmax = max(dmax, 1e-12)

#     t = np.zeros_like(d, dtype=np.float64)
#     t[finite] = np.clip(d[finite] / dmax, 0.0, 1.0)

#     # blau->rot Gradient
#     colors = np.zeros((points.shape[0], 3), dtype=np.float64)
#     colors[:, 0] = t
#     colors[:, 2] = 1.0 - t
#     colors[~finite] = np.array([0.2, 0.2, 0.2], dtype=np.float64)

#     pc = o3d.geometry.PointCloud()
#     pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
#     pc.colors = o3d.utility.Vector3dVector(colors)
#     o3d.io.write_point_cloud(out_ply, pc)


# def main() -> None:
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--max-points", type=int, default=2_000_000, help="Random downsample auf max N Punkte (0=aus)")
#     ap.add_argument("--voxel-ds", type=float, default=0.0, help="Optional voxel downsample (m), 0=aus")
#     ap.add_argument("--chunk", type=int, default=250_000, help="Chunkgröße für Distanzabfrage")
#     ap.add_argument("--save-npz", action="store_true", help="Speichert points+distances als npz")
#     ap.add_argument("--export-colored-ply", action="store_true", help="Exportiert farbige Punktwolke als PLY")
#     ap.add_argument("--colored-ply-name", default="cloud_colored_by_dist.ply")
#     ap.add_argument("--approx-fallback", action="store_true", help="Wenn Raycasting nicht geht: approx via mesh sampling")
#     ap.add_argument("--mesh-samples", type=int, default=2_000_000, help="Nur für approx-fallback: # Mesh-Samples")

#     ap.add_argument("--mesh", default="/home/laurenz/MQPW/results/Mesh_Brucher_multiScan/tsdf_Brucher_multiScan.ply")
#     ap.add_argument("--las", default="/home/laurenz/MQPW/data/s2_f1_part1.las")
#     ap.add_argument("--out", default="/home/laurenz/MQPW/results/Eval_test")

#     args = ap.parse_args()

#     os.makedirs(args.out, exist_ok=True)
#     log_path = os.path.join(args.out, "log.txt")
#     t0 = time.time()

#     print("[INFO] load mesh:", args.mesh)
#     mesh = o3d.io.read_triangle_mesh(args.mesh)
#     if mesh.is_empty():
#         raise RuntimeError("Mesh ist leer / konnte nicht geladen werden.")

#     mesh.remove_duplicated_vertices()
#     mesh.remove_unreferenced_vertices()
#     mesh.remove_degenerate_triangles()

#     print("[INFO] load LAS:", args.las)
#     P = read_las_xyz(args.las)
#     print("[INFO] points loaded:", P.shape[0])

#     if args.voxel_ds > 0:
#         P = voxel_downsample_numpy(P, voxel_size=args.voxel_ds)
#         print("[INFO] after voxel downsample:", P.shape[0])

#     if args.max_points > 0:
#         P = downsample_points(P, max_points=args.max_points, seed=0)
#         print("[INFO] after random downsample:", P.shape[0])

#     # Distanzberechnung (point -> mesh)
#     use_raycast = True
#     distances = None

#     try:
#         print("[INFO] compute point->mesh distances via RaycastingScene (exact)")
#         distances = compute_distances_raycast(mesh, P, chunk=args.chunk)
#     except Exception as e:
#         use_raycast = False
#         print("[WARN] RaycastingScene Distanz fehlgeschlagen:", repr(e))
#         if not args.approx_fallback:
#             raise
#         print("[INFO] fallback: approx distance via mesh sampling + KDTree")
#         distances = compute_distances_approx(mesh, P, n_samples=args.mesh_samples, chunk=min(args.chunk, 50_000))

#     stats = summarize("point_to_mesh_distance", distances)

#     print("\n[STATS] point->mesh distance (m)")
#     for k in ["n", "finite_frac", "min", "mean", "median", "p95", "max"]:
#         print(f"  {k}: {stats[k]}")

#     # Save outputs
#     with open(log_path, "w") as f:
#         f.write("### Mesh vs PointCloud Evaluation ###\n")
#         f.write(f"mesh: {args.mesh}\n")
#         f.write(f"las: {args.las}\n")
#         f.write(f"out: {args.out}\n")
#         f.write(f"method: {'raycasting_exact' if use_raycast else 'approx_kdtree'}\n")
#         f.write(f"max_points: {args.max_points}\n")
#         f.write(f"voxel_ds: {args.voxel_ds}\n")
#         f.write(f"chunk: {args.chunk}\n")
#         f.write("\n[STATS] point_to_mesh_distance (m)\n")
#         for k in ["n", "finite_frac", "min", "mean", "median", "p95", "max"]:
#             f.write(f"{k}: {stats[k]}\n")
#         f.write(f"\n[RUNTIME]\nseconds: {time.time() - t0:.2f}\n")

#     print("[OK] log saved:", log_path)

#     if args.save_npz:
#         out_npz = os.path.join(args.out, "point_to_mesh_distances.npz")
#         np.savez_compressed(out_npz, points=P.astype(np.float64), distances=distances.astype(np.float32))
#         print("[OK] npz saved:", out_npz)

#     if args.export_colored_ply:
#         out_ply = os.path.join(args.out, args.colored_ply_name)
#         export_colored_ply(P, distances, out_ply, clip_p95=True)
#         print("[OK] colored ply saved:", out_ply)

#     print("[OK] runtime: {:.2f} s".format(time.time() - t0))


# if __name__ == "__main__":
#     main()