# -*- coding: utf-8 -*-
import os
from typing import Tuple, List

import numpy as np
import laspy


def upsert_extra_dim_float32(las: laspy.LasData, name: str, values: np.ndarray) -> None:
    """
    Schreibt/überschreibt ein LAS-Extra-Feld als float32.
    Falls das Feld noch nicht existiert, wird es angelegt.
    """
    values = np.asarray(values, dtype=np.float32)
    existing_dims = set(las.point_format.dimension_names)

    if name not in existing_dims:
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))

    setattr(las, name, values)


def pca_basis(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centroid = P.mean(axis=0)
    Q = P - centroid
    C = (Q.T @ Q) / max(Q.shape[0] - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    R = eigvecs[:, idx]

    # ensure right-handed basis
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0

    return centroid, R


def tilt_global_pca_y(P: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Global tilt of ALL points:
    - PCA basis from P
    - shift PCA-x/y so they start at 0 (keeps your "positive xy" convention)
    - rotate around PCA-Y by angle_deg
    - back to world

    Rückgabe:
    - P_out: verkippte Punkte in Weltkoordinaten
    - signed_defo_mm: signierte Vor-/Rückwärtsverschiebung in mm
    """
    centroid, R = pca_basis(P)
    Q = P - centroid
    Q_pc = Q @ R

    # shift so min x/y is 0 (optional, but you had it)
    shift = np.array([np.min(Q_pc[:, 0]), np.min(Q_pc[:, 1]), 0.0], dtype=float)
    Q_pc_shifted = Q_pc - shift

    th = np.deg2rad(angle_deg)
    Ry = np.array([[np.cos(th), 0.0, -np.sin(th)],
                   [0.0,        1.0,  0.0],
                   [np.sin(th), 0.0,  np.cos(th)]], dtype=float)

    Q_pc_rot = Q_pc_shifted @ Ry.T
    Q_pc_rot = Q_pc_rot + shift

    P_out = Q_pc_rot @ R.T + centroid
    delta_world = P_out - P

    # PCA-X ist die Vor-/Rückwärts-Richtung der Rotationswirkung.
    forward_dir_world = R[:, 0]
    signed_defo_mm = 1000.0 * (delta_world @ forward_dir_world)

    return P_out, signed_defo_mm


def tilt_las_file(in_path: str, out_path: str, angle_deg: float) -> None:
    las = laspy.read(in_path)
    P = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    print(f"[INFO] {os.path.basename(in_path)}: points={P.shape[0]}")

    if P.shape[0] < 50:
        raise RuntimeError(f"Zu wenige Punkte für PCA: {in_path}")

    P_new, soll_defo_mm = tilt_global_pca_y(P, angle_deg=angle_deg)

    # only overwrite coordinates; all extra dims remain unchanged
    las.x = P_new[:, 0]
    las.y = P_new[:, 1]
    las.z = P_new[:, 2]
    upsert_extra_dim_float32(las, "soll_defo_mm", soll_defo_mm)

    las.write(out_path)
    print(
        f"[INFO] Feld 'soll_defo_mm' geschrieben "
        f"(min={float(np.min(soll_defo_mm)):.3f} mm, max={float(np.max(soll_defo_mm)):.3f} mm)"
    )
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "deformed_scans")
    os.makedirs(out_dir, exist_ok=True)

    # Deine aktuellen Punktwolken:
    # scan_files: List[str] = ["s1_f1_ausg3.las", "s2_f1_ausg3.las", "s3_f1_ausg3.las"]
    scan_files = ["s1_f1.las", "s2_f1.las", "s3_f1.las"]  

    angle_deg = 0.04

    for fn in scan_files:
        in_path = os.path.join(data_dir, fn)
        if not os.path.isfile(in_path):
            raise FileNotFoundError(f"Missing input: {in_path}")

        stem = os.path.splitext(fn)[0]
        out_name = f"{stem}_tilt{angle_deg:+.1f}deg.las"
        out_path = os.path.join(out_dir, out_name)

        tilt_las_file(in_path, out_path, angle_deg=angle_deg)
