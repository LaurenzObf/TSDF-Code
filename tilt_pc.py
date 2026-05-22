import os
import numpy as np
import laspy


def upsert_extra_dim_float32(las: laspy.LasData, name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float32)
    existing_dims = set(las.point_format.dimension_names)
    if name not in existing_dims:
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))
    setattr(las, name, values)


def transform_dam_wall(P: np.ndarray, angle_deg: float) -> np.ndarray:
    mean_x = np.mean(P[:, 0])
    mean_y = np.mean(P[:, 1])
    min_z = np.min(P[:, 2]) + 3.0
    shift = np.array([mean_x, mean_y, min_z], dtype=np.float64)
    P1 = P - shift

    th = np.deg2rad(angle_deg)
    Ry = np.array(
        [
            [np.cos(th), 0.0, -np.sin(th)],
            [0.0, 1.0, 0.0],
            [np.sin(th), 0.0, np.cos(th)],
        ],
        dtype=np.float64,
    )

    P_rot = P1 @ Ry.T
    P_out = P_rot + shift
    return P_out


def run_one_file(in_path: str, out_path: str, angle_deg: float) -> None:
    las = laspy.read(in_path)
    P = np.column_stack((las.x, las.y, las.z)).astype(np.float64)

    P_new = transform_dam_wall(P, angle_deg=angle_deg)

    delta = P_new - P
    signed_defo_mm = (1000.0 * delta[:, 0]).astype(np.float32)
    upsert_extra_dim_float32(las, "soll_defo_mm", signed_defo_mm)

    las.x = P_new[:, 0]
    las.y = P_new[:, 1]
    las.z = P_new[:, 2]
    las.write(out_path)

    print(
        f"[OK] {os.path.basename(in_path)} -> {os.path.basename(out_path)}  "
        f"soll_defo_mm: min={float(np.min(signed_defo_mm)):.3f} mm, "
        f"max={float(np.max(signed_defo_mm)):.3f} mm"
    )


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    out_dir = os.path.join(script_dir, "deformed_scans")
    os.makedirs(out_dir, exist_ok=True)

    scan_files = ["s1_f1.las", "s2_f1.las", "s3_f1.las"]

    # 4 mgrad -> 0.004 deg
    angle_deg = 0.004
    mgrad = int(round(angle_deg * 1000.0))

    for fn in scan_files:
        in_path = os.path.join(data_dir, fn)
        if not os.path.isfile(in_path):
            raise FileNotFoundError(f"Input-LAS nicht gefunden: {in_path}")

        stem = os.path.splitext(fn)[0]
        out_name = f"{stem}_tilt+{angle_deg:g}deg.las"
        out_path = os.path.join(out_dir, out_name)
        run_one_file(in_path, out_path, angle_deg=angle_deg)

    print(f"[DONE] Tilt fuer {len(scan_files)} Dateien geschrieben nach: {out_dir}")
