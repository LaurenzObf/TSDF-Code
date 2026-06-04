# TSDF-Code

Code für TSDF-, PSR- und Ball-Pivoting-basierte Deformationsanalyse auf TLS-Punktwolken.

## Enthaltene Dateien/Ordner
- `TSDF/`: TSDF-Code inkl. `dataset.py`, TSDF-Sweepern und Vergleichsskripten.
- `deform_pointcloud/`: Skripte fuer Beule, Verkippung und z-Translation mit `soll_defo_mm`.
- `PSR/`: Poisson-Surface-Reconstruction-Code inkl. Sweepern fuer dieselben Deformationsfaelle.
- `BPA/`: Ball-Pivoting-Code inkl. Sweepern fuer dieselben Deformationsfaelle.
- `README.md`: Kurzueberblick ueber Struktur und Zweck der wichtigsten Dateien.

## Daten (nicht im Repo)
Lokale Datenordner wie `data/`, `deformed_scans/` und `results/` werden nicht mit GitHub synchronisiert.

## Abhaengigkeiten
- Python 3.8+
- `numpy`, `laspy`, `open3d`
- angepasstes `vdbfusion`: https://github.com/LaurenzObf/vdbfusion-custom
