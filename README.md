# TSDF-Code (MQPW)

Kurzes Projekt zur TSDF-basierten Deformationsanalyse auf TLS-Punktwolken.

## Ziel
- Referenz-TSDF aus `Epoch1`-Scans aufbauen.
- Deformierte Punktwolken (`Epoch2`) gegen das TSDF abfragen.
- Soll-Ist-Deformation pro Punkt vergleichen:
  - `soll_defo_mm` (künstlich eingebrachte Deformation)
  - `tsdf_value_mm` (gequeryter TSDF-Wert)

## Wichtige Skripte
- `tilt_pc.py`: erzeugt verkippte `s1/s2/s3`-Scans und schreibt `soll_defo_mm`.
- `deform_pointcloud.py`: lokale Beulen-Deformation mit `soll_defo_mm`.
- `translate_scan_zComponent.py`: z-Translation mit `soll_defo_mm`.
- `tsdf_deformation_Heatmap.py`: baut Referenz-TSDF und exportiert LAS mit `tsdf_value_mm`.

## Datenstruktur (lokal)
- Eingabe: `data/`
- Deformierte Daten: `deformed_scans/`
- Ausgaben: `results/`


## Schnellstart
1. Deformation erzeugen (Beispiel Tilt):
```bash
python3 tilt_pc.py
```

2. Einzel-TSDF-Export:
```bash
python3 tsdf_deformation_Heatmap.py
```

3. Matrix-Sweep:
```bash
python3 Sweeper_tsdf_tilt_matrix.py
```

## Abhaengigkeiten
- Python 3.8+
- `numpy`, `laspy`
- angepasstes `vdbfusion`-Repo:
  - https://github.com/LaurenzObf/vdbfusion-custom

Das Custom-`vdbfusion` muss auf dem Zielrechner gebaut/installiert sein, damit `query_sdf` und die Exportfunktionen verfuegbar sind.

