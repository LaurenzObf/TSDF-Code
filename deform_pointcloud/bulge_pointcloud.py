# deform_scans_las.py
# Python 3.8 kompatibel
#
# Deformiert eine "Beule" (Halbkugel/Dome) auf der Vorderseite der Staumauer,
# wobei die "Vorderseite" automatisch aus den Scan-Standpunkten (Diagnostics) abgeleitet wird.
#
# - KEINE Normalen-Berechnung (nx,ny,nz wird nicht angefasst)
# - KEINE Bounding Box / keine ROI: wird auf die GESAMTE Punktwolke angewendet
# - Deformation wirkt nur in einem Tangential-Radius um 'center' (radius_m)
# - Richtung kommt aus Scanner-Standpunkten: Mittelwert der (center - scanner_pos)

import os
import sys
import numpy as np
import laspy
from typing import List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TSDF_DIR = os.path.join(PROJECT_ROOT, "TSDF")
if TSDF_DIR not in sys.path:
    sys.path.insert(0, TSDF_DIR)

from dataset import Dataset  # nutzt diag_file zur Pose/Standpunkt-Zuordnung


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


def get_scan_positions_from_dataset(ds: Dataset) -> np.ndarray:
    """
    Extrahiere Scanner-Standpunkte aus deinem Dataset.
    Wir versuchen robust mehrere naheliegende Fälle:
      - ds[i] liefert (points, T_WC) und T_WC ist 4x4 (World-from-Cam), dann Position = T_WC[:3,3]
      - oder T_WC ist 3er-Vektor (origin), dann direkt.
    """
    positions = []
    for i in range(len(ds)):
        _points, T = ds[i]

        T = np.asarray(T)
        if T.shape == (4, 4):
            positions.append(T[:3, 3].astype(float))
        elif T.shape in [(3,), (3, 1)]:
            positions.append(T.reshape(3).astype(float))
        else:
            raise RuntimeError(f"Unbekanntes Extrinsic/origin-Format bei ds[{i}]: shape={T.shape}")

    if len(positions) == 0:
        raise RuntimeError("Keine Scan-Positionen gefunden (Dataset leer?).")

    return np.vstack(positions)  # (M,3)


def estimate_local_surface_normal(
    points_list: List[np.ndarray],
    center: np.ndarray,
    radius_m: float,
    preferred_dir: np.ndarray,
) -> np.ndarray:
    """
    Schätzt eine lokale Flächennormale per PCA aus allen Referenzscans gemeinsam.
    Die Normale wird so orientiert, dass sie in preferred_dir zeigt.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    preferred_dir = np.asarray(preferred_dir, dtype=np.float64).reshape(3)
    preferred_dir /= (np.linalg.norm(preferred_dir) + 1e-12)

    local_chunks = []
    radius2 = float(radius_m) * float(radius_m)

    for points in points_list:
        P = np.asarray(points, dtype=np.float64)
        d2 = np.sum((P - center[None, :]) ** 2, axis=1)
        local = P[d2 <= radius2]
        if local.size > 0:
            local_chunks.append(local)

    if not local_chunks:
        raise RuntimeError(
            f"Keine Punkte für Normalenschätzung gefunden (center={center.tolist()}, radius_m={radius_m})."
        )

    P_local = np.vstack(local_chunks)
    if P_local.shape[0] < 3:
        raise RuntimeError(
            f"Zu wenige Punkte für Normalenschätzung: {P_local.shape[0]} innerhalb radius_m={radius_m}."
        )

    centroid = np.mean(P_local, axis=0)
    Q = P_local - centroid[None, :]
    C = (Q.T @ Q) / max(P_local.shape[0] - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(C)
    normal = eigvecs[:, np.argmin(eigvals)]
    normal /= (np.linalg.norm(normal) + 1e-12)

    if np.dot(normal, preferred_dir) < 0.0:
        normal = -normal

    return normal


def compute_outward_dir_from_scanners(center: np.ndarray, scanner_positions: np.ndarray) -> np.ndarray:
    """
    Bestimme eine robuste "Front"-Richtung:
    - Scanner stehen vor der Wand.
    - Vektor vom Scanner zum center zeigt grob zur Wand.
    - Wir nutzen mean(center - scanner_pos) als outward Richtung.
    """
    v = scanner_positions - center[None, :]   # Richtung zum Scanner (Front)
    d = v.mean(axis=0)
    n = np.linalg.norm(d)

    # Mittelwert und Normierung
    d = v.mean(axis=0)
    n = np.linalg.norm(d)
    if not np.isfinite(n) or n < 1e-12:
        raise RuntimeError("Konnte outward_dir nicht bestimmen (degenerierte Scanner-Positionen?).")
    return d / n


def deform_front_hemisphere(
    P: np.ndarray,
    center: np.ndarray,
    dir_used: np.ndarray,
    radius_m: float,
    dome_height_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dome/Beule:
      - projiziere Punkte auf Ebene senkrecht zu dir_used (Tangentialebene)
      - innerhalb radius_m: h = dome_height_m * sqrt(1 - (r/radius)^2)
      - displacement = h * dir_used

    Rückgabe:
      - disp (N,3)
      - h (N,)  (Skalarhöhe)
    """
    P = np.asarray(P, float)
    center = np.asarray(center, float).reshape(3)
    n = np.asarray(dir_used, float).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)

    v_rel = P - center[None, :]
    dist_n = v_rel @ n                      # (N,)
    v_tan = v_rel - dist_n[:, None] * n[None, :]
    r_tan = np.linalg.norm(v_tan, axis=1)   # (N,)

    inside = r_tan <= radius_m
    h = np.zeros_like(r_tan)

    if np.any(inside):
        ratio = r_tan[inside] / radius_m
        h[inside] = dome_height_m * np.sqrt(np.maximum(0.0, 1.0 - ratio**2))

    disp = h[:, None] * n[None, :]
    return disp, h


def deform_las_files(
    data_dir: str,
    out_dir: str,
    scan_files: List[str],
    diag_file: str,
    center: np.ndarray,
    radius_m: float,
    dome_height_mm: float,
    normal_estimation_radius_m: float,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1) Dataset laden -> Standpunkte -> Richtung
    diag_path = os.path.join(data_dir, diag_file)
    scan_paths_for_ds = [os.path.join(data_dir, fn) for fn in scan_files]
    ds = Dataset(scan_txt_paths=scan_paths_for_ds, diagnostics_path=diag_path)

    scanner_positions = get_scan_positions_from_dataset(ds)
    outward_dir = compute_outward_dir_from_scanners(center=center, scanner_positions=scanner_positions)
    dir_used = estimate_local_surface_normal(
        points_list=ds.points_list,
        center=center,
        radius_m=normal_estimation_radius_m,
        preferred_dir=outward_dir,
    )

    dome_height_m = dome_height_mm / 1000.0  # mm -> m (Vorzeichen bleibt wie du es setzt)
    print("[INFO] Scanner-derived outward direction:", outward_dir)
    print("[INFO] Local surface normal used for dome:", dir_used)
    print("[INFO] dome_height_m:", dome_height_m, "radius_m:", radius_m)
    print("[INFO] normal_estimation_radius_m:", normal_estimation_radius_m)

    # 2) LAS deformieren
    for fn in scan_files:
        in_path = os.path.join(data_dir, fn)
        out_name = os.path.splitext(fn)[0] + f"_deformed_{dome_height_mm:+.0f}mm.las"
        out_path = os.path.join(out_dir, out_name)

        print(f"\n[INFO] Lade: {in_path}")
        las = laspy.read(in_path)

        P = np.column_stack([las.x, las.y, las.z]).astype(np.float64, copy=False)
        print(f"[INFO] Punkte: {P.shape[0]}")

        disp, h = deform_front_hemisphere(
            P=P,
            center=center,
            dir_used=dir_used,
            radius_m=radius_m,
            dome_height_m=dome_height_m,
        )
        P_def = P + disp
        soll_defo_mm = (1000.0 * h).astype(np.float32)

        las.x = P_def[:, 0]
        las.y = P_def[:, 1]
        las.z = P_def[:, 2]
        upsert_extra_dim_float32(las, "soll_defo_mm", soll_defo_mm)

        n_def = int(np.count_nonzero(np.abs(h) > 1e-12))
        print(f"[INFO] Deformierte Punkte: {n_def} / {P.shape[0]}  (max|h|={float(np.max(np.abs(h))):.6f} m)")
        print(
            f"[INFO] Feld 'soll_defo_mm' geschrieben "
            f"(min={float(np.min(soll_defo_mm)):.3f} mm, max={float(np.max(soll_defo_mm)):.3f} mm)"
        )

        las.write(out_path)
        print(f"[OK] Gespeichert: {out_path}")


if __name__ == "__main__":
    base_dir = PROJECT_ROOT
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "deformed_scans")
    diag_file = "Brucher_202405_P50_georef_diagnostics.txt"

    # Deine aktuellen Punktwolken:
    scan_files = ["s1_f1.las","s2_f1.las","s3_f1.las"]

    # Dome-Parameter
    center = np.array([703.383728, 375.882935, 361.387207], dtype=float)
    radius_m = 4.0  # m
    normal_estimation_radius_m = 1.5  # m

    # Vorzeichen: dome_height_mm > 0 --> Beule kommt zur Scanner-Seite raus (Front)
    dome_height_mm = 1  # mm 10, 5, 3, 1

    deform_las_files(
        data_dir=data_dir,
        out_dir=out_dir,
        scan_files=scan_files,
        diag_file=diag_file,
        center=center,
        radius_m=radius_m,
        dome_height_mm=dome_height_mm,
        normal_estimation_radius_m=normal_estimation_radius_m,
    )
