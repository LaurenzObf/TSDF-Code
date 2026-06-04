# dataset.py
import os
import re
import numpy as np
import laspy
from typing import List, Tuple, Dict, Optional


def axis_angle_to_R(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    axis = axis / n

    angle = np.deg2rad(angle_deg)
    x, y, z = axis
    K = np.array([[0.0, -z,  y],
                  [z,  0.0, -x],
                  [-y, x,  0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def parse_scanworld_transformations(diag_path: str) -> Dict[str, np.ndarray]:
    if not os.path.isfile(diag_path):
        raise FileNotFoundError(f"Diagnostics-Datei nicht gefunden: {diag_path}")

    with open(diag_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # --- nur den Abschnitt "ScanWorld Transformations" betrachten ---
    msec = re.search(r"ScanWorld Transformations\s*(.*)$", text, flags=re.DOTALL)
    if not msec:
        raise ValueError("Abschnitt 'ScanWorld Transformations' nicht gefunden.")

    sec = msec.group(1)

    # Block-Parser: Name, translation, rotation (mit optionalen Leerzeilen dazwischen)
    pat = re.compile(
        r"^\s*(?P<name>[^\r\n]+)\s*\n"
        r"\s*translation:\s*\(\s*(?P<tx>[-+0-9.eE]+)\s*,\s*(?P<ty>[-+0-9.eE]+)\s*,\s*(?P<tz>[-+0-9.eE]+)\s*\)\s*m\s*\n"
        r"\s*rotation:\s*\(\s*(?P<ax>[-+0-9.eE]+)\s*,\s*(?P<ay>[-+0-9.eE]+)\s*,\s*(?P<az>[-+0-9.eE]+)\s*\)\s*:\s*(?P<ang>[-+0-9.eE]+)\s*deg\s*$",
        flags=re.MULTILINE
    )

    out: Dict[str, np.ndarray] = {}
    for m in pat.finditer(sec):
        raw_name = m.group("name").strip()
        name = raw_name.lower()

        t = np.array([float(m.group("tx")), float(m.group("ty")), float(m.group("tz"))], dtype=np.float64)
        axis = np.array([float(m.group("ax")), float(m.group("ay")), float(m.group("az"))], dtype=np.float64)
        ang = float(m.group("ang"))

        R = axis_angle_to_R(axis, ang)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

        out[name] = T

    if not out:
        raise ValueError("Keine ScanWorld-Transformationen gefunden. Prüfe die Diagnostics-Datei/Format.")
    return out


def read_las_points(las_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Liest LAS/LAZ und gibt (points Nx3 float64, extra optional) zurück.
    Extra ist hier erstmal None (kannst du später z.B. intensity/classification zurückgeben).
    """
    if not os.path.isfile(las_path):
        raise FileNotFoundError(f"Datei nicht gefunden: {las_path}")

    las = laspy.read(las_path)

    # las.x/las.y/las.z sind bereits "skalierte" Koordinaten in float
    points = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)
    points = np.ascontiguousarray(points)

    extra = None
    return points, extra


def scanfile_to_sname(scan_path: str) -> str:
    """
    Für Dateinamen wie:
      s1_f1_part1.las, s2_f1_part1.las, s3_f1_part1.las
    extrahiert -> s1_f1
    """
    base = os.path.splitext(os.path.basename(scan_path))[0]
    m = re.search(r"(s\d+_f\d+)", base, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Kann Scan-Name nicht aus Dateiname lesen: {scan_path} (erwarte z.B. 's1_f1_...')")
    return m.group(1).lower()


class Dataset:
    def __init__(self, scan_txt_paths: Optional[List[str]] = None,
                 scan_paths: Optional[List[str]] = None,
                 diagnostics_path: str = ""):

        paths = scan_paths if scan_paths is not None else scan_txt_paths
        if not paths:
            raise ValueError("scan_paths/scan_txt_paths ist leer.")

        self.T_by_name = parse_scanworld_transformations(diagnostics_path)

        self.points_list = []
        self.extras_list = []
        self.T_list = []
        self.scan_names = []

        for path in paths:
            points, extra = read_las_points(path)   # <-- LAS Reader
            sname = scanfile_to_sname(path)         # <-- zieht 's1_f1' aus Dateiname

            if sname not in self.T_by_name:
                available = ", ".join(sorted(self.T_by_name.keys()))
                raise KeyError(
                    f"Für '{sname}' keine Transformation in Diagnostics gefunden (aus Datei: {path}).\n"
                    f"Vorhanden in Diagnostics: {available}"
                )

            self.points_list.append(points)
            self.extras_list.append(extra)
            self.T_list.append(self.T_by_name[sname])
            self.scan_names.append(sname)

        self.n_scans = len(self.points_list)

    def __len__(self) -> int:
        return self.n_scans

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.points_list[idx], self.T_list[idx]
    




### für Jubach ###
#
# # dataset.py
# import os
# import numpy as np
# from typing import Tuple

# class Dataset:
#     """
#     Erwartet pro Zeile: x y z (weitere Spalten werden ignoriert).
#     Gibt (points Nx3 float64, origin 3,) zurück.
#     """
#     def __init__(self, txt_path: str):
#         if not os.path.isfile(txt_path):
#             raise FileNotFoundError(f"Datei nicht gefunden: {txt_path}")

#         arr = np.loadtxt(txt_path, comments="#", ndmin=2, dtype=float)
#         if arr.shape[1] < 3:
#             raise ValueError("Erwarte mindestens 3 Spalten (x y z).")

#         self.points = np.ascontiguousarray(arr[:, :3].astype(np.float64))
#         self.origin = np.zeros(3, dtype=np.float64)  # kein Ursprung in Datei
#         self.n_scans = 1

#         # Optional: restliche Spalten aufbewahren (z.B. Label/Normalen)
#         self.extra = arr[:, 3:] if arr.shape[1] > 3 else None

#     def __len__(self) -> int:
#         return self.n_scans

#     def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
#         return self.points, self.origin