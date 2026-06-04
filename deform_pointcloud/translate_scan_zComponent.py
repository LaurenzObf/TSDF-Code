# -*- coding: utf-8 -*-
"""
translate_scans_z.py  (Python 3.8 kompatibel)

Verschiebt LAS-Punktwolken um eine konstante Translation in Z (Höhe).
- keine Normalen / Zusatzfelder werden verändert
- wahlweise: global (alle Punkte) ODER nur innerhalb eines Radius um ein Center (ROI)

Outputs:
- pro Eingabedatei eine neue LAS im out_dir mit Suffix: _tz+Xmm.las
"""

import os
import numpy as np
import laspy
from typing import List, Optional


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


def load_las_xyz(path: str) -> np.ndarray:
    las = laspy.read(path)
    P = np.column_stack([las.x, las.y, las.z]).astype(np.float64, copy=False)
    return las, P


def apply_z_translation(
    P: np.ndarray,
    dz_m: float,
    center: Optional[np.ndarray] = None,
    radius_m: Optional[float] = None,
) -> np.ndarray:
    """
    P: (N,3)
    dz_m: translation in meters (added to z)
    center + radius_m optional:
      - if both are provided -> shift only points with ||P- center|| <= radius_m
      - else -> shift all points
    """
    P_out = P.copy()

    if center is not None and radius_m is not None:
        c = np.asarray(center, dtype=np.float64).reshape(3)
        r = float(radius_m)

        d = P_out - c[None, :]
        mask = (d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2) <= (r * r)
        P_out[mask, 2] += dz_m
        return P_out

    # global shift
    P_out[:, 2] += dz_m
    return P_out


def translate_las_files_z(
    data_dir: str,
    out_dir: str,
    scan_files: List[str],
    dz_mm: float,
    center: Optional[np.ndarray] = None,
    radius_m: Optional[float] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    dz_m = float(dz_mm) / 1000.0
    mode = "ROI" if (center is not None and radius_m is not None) else "GLOBAL"

    print(f"[INFO] Z-Translation: dz_mm={dz_mm:+.3f} mm (dz_m={dz_m:+.6f} m), mode={mode}")
    if mode == "ROI":
        print(f"[INFO] ROI center={np.asarray(center).reshape(3)}  radius_m={float(radius_m):.3f}")

    for fn in scan_files:
        in_path = os.path.join(data_dir, fn)
        if not os.path.isfile(in_path):
            raise FileNotFoundError(f"Missing input LAS: {in_path}")

        base = os.path.splitext(fn)[0]
        out_name = f"{base}_tz{dz_mm:+.0f}mm.las"
        out_path = os.path.join(out_dir, out_name)

        print(f"\n[INFO] Lade: {in_path}")
        las, P = load_las_xyz(in_path)
        print(f"[INFO] Punkte: {P.shape[0]}")

        P_def = apply_z_translation(P, dz_m=dz_m, center=center, radius_m=radius_m)

        # stats
        dz_applied = P_def[:, 2] - P[:, 2]
        n_shifted = int(np.count_nonzero(np.abs(dz_applied) > 0.5 * abs(dz_m) if dz_m != 0 else 0))
        print(f"[INFO] Verschobene Punkte: {n_shifted} / {P.shape[0]}")
        if dz_m != 0:
            print(f"[INFO] max(|Δz|)={float(np.max(np.abs(dz_applied))):.6f} m")

        las.x = P_def[:, 0]
        las.y = P_def[:, 1]
        las.z = P_def[:, 2]
        upsert_extra_dim_float32(las, "soll_defo_mm", 1000.0 * dz_applied)
        las.write(out_path)

        print(
            f"[INFO] Feld 'soll_defo_mm' geschrieben "
            f"(min={float(np.min(1000.0 * dz_applied)):.3f} mm, max={float(np.max(1000.0 * dz_applied)):.3f} mm)"
        )
        print(f"[OK] Gespeichert: {out_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "deformed_scans")

    # Eingabedateien (anpassen)
    scan_files = ["s1_f1.las", "s2_f1.las", "s3_f1.las"]

    # Translation (mm): + nach oben, - nach unten
    dz_mm = 1.0 # 10 mm, 5 mm, 3 mm, 1 mm

    # --- OPTIONAL: nur im ROI verschieben (sonst beide auf None lassen) --- 
    USE_ROI = False
    center = np.array([703.383728, 375.882935, 361.387207], dtype=float) if USE_ROI else None
    radius_m = 4.0 if USE_ROI else None
    # ---------------------------------------------------------------------

    translate_las_files_z(
        data_dir=data_dir,
        out_dir=out_dir,
        scan_files=scan_files,
        dz_mm=dz_mm,
        center=center,
        radius_m=radius_m,
    )
