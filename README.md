# TSDF-Code

Kompakter Code fuer TSDF-, PSR- und Ball-Pivoting-basierte Deformationsanalyse auf TLS-Punktwolken.

## Enthaltene Dateien/Ordner
- `dataset.py`: Laedt LAS-Scans und liefert die zugehoerigen Scanner-Transformationen aus der Diagnostics-Datei.
- `tilt_pointcloud.py`: Erzeugt verkippte Scans und schreibt `soll_defo_mm` in die Ausgabe-LAS.
- `bulge_pointcloud.py` (bzw. `deform_pointcloud.py`): Erzeugt lokale Beulen-Deformationen und schreibt `soll_defo_mm`.
- `translate_scan_zComponent.py`: Erzeugt z-Translationen und schreibt `soll_defo_mm`.
- `Sweeper_tsdf_deformation.py`: Einheitlicher TSDF-Sweeper fuer `beule`, `tz` und `tilt` mit LAS-Export (`tsdf_value_mm`).
- `Poisson SR/`: PSR-Code inkl. Sweepern fuer dieselben Deformationsfaelle.
- `Ball pivoting/`: BPA-Code inkl. Sweepern fuer dieselben Deformationsfaelle.
- `README.md`: Kurzueberblick ueber Struktur und Zweck der wichtigsten Dateien.

## Daten (nicht im Repo)
Lokale Datenordner wie `data/`, `deformed_scans/` und `results/` werden nicht mit GitHub synchronisiert.

## Abhaengigkeiten
- Python 3.8+
- `numpy`, `laspy`, `open3d`
- angepasstes `vdbfusion`: https://github.com/LaurenzObf/vdbfusion-custom
