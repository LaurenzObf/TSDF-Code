import os
import time
from pathlib import Path
from typing import List

import laspy
import numpy as np
import open3d as o3d  # type: ignore


def load_las_points(paths: List[Path]) -> np.ndarray:
    chunks = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Eingabedatei nicht gefunden: {p}")
        las = laspy.read(str(p))
        pts = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)
        chunks.append(pts)
        print(f"[INFO] geladen: {p.name} ({pts.shape[0]} Punkte)")
    return np.vstack(chunks).astype(np.float64)


def ensure_normals(pcd: o3d.geometry.PointCloud, radius_m: float, max_nn: int) -> None:
    if not pcd.has_normals():
        print("[INFO] Berechne Normalen")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_m, max_nn=max_nn)
        )


if __name__ == "__main__":
    start_time = time.time()

    base_dir = Path("/home/laurenz/MQPW")
    epoch2_dir = base_dir / "deformed_scans"
    output_dir = base_dir / "results" / "Ball pivoting"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Eingabedateien (LAS) im MQPW-Format
    input_files = ["s1_f1_tilt+0.04deg.las", "s2_f1_tilt+0.04deg.las", "s3_f1_tilt+0.04deg.las",]

    normal_radius_m = 0.05
    normal_max_nn = 30

    input_paths = [epoch2_dir / fn for fn in input_files]
    points = load_las_points(input_paths)
    print(f"[INFO] Gesamtpunkte: {points.shape[0]}")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    ensure_normals(pcd, radius_m=normal_radius_m, max_nn=normal_max_nn)

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = float(np.mean(distances))
    print(f"[INFO] Mittlerer Punktabstand: {avg_dist:.6f} m")

    radii = [avg_dist * f for f in (4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0)]
    print(f"[INFO] Ball-Radien: {[round(r, 6) for r in radii]}")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    mesh.compute_vertex_normals()

    run_tag = "tilt04mgrad"
    output_path = output_dir / f"BPA_{run_tag}.ply"
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    print(f"[OK] BPA-Mesh gespeichert unter: {output_path}")

    elapsed_min = (time.time() - start_time) / 60.0
    print(f"[DONE] Laufzeit: {elapsed_min:.2f} min")
