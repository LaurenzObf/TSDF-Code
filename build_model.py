'''
function to bild model. based on TSDF, here small postprecessing
'''

# import os
import numpy as np
import open3d as o3d
# import pymeshlab as ml
# from vdbfusion.pybind.vdb_volume import VDBVolume
# from dataset_jubach import Dataset


def make_manifold_with_open3d(mesh: o3d.geometry.TriangleMesh, min_cluster_tris=50):
    """Bereinigt das Mesh direkt in-memory (kein Speichern)."""
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()

    # kleine Fragmente entfernen
    if min_cluster_tris > 0:
        tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        cluster_n_tris = np.asarray(cluster_n_tris)
        small = np.where(cluster_n_tris < min_cluster_tris)[0]
        if len(small) > 0:
            mask = np.isin(tri_clusters, small)
            mesh.remove_triangles_by_mask(mask)
            mesh.remove_unreferenced_vertices()
    return mesh


def close_holes_safely(ms,
                       passes=(500),
                       pass_stop_pct=0.03,
                       cum_stop_pct=0.05):
    """Füllt kleine/technische Löcher; stoppt automatisch bei zu vielen neuen Faces.
       Gibt pro Pass Statistiken zurück."""
    def face_count():
        return ms.current_mesh().face_number()

    f0 = face_count()
    cum_added = 0
    stats = []  # Liste für Logging nach außen

    for i, mh in enumerate(passes, 1):
        try:
            f_before = face_count()
            ms.apply_filter('meshing_close_holes',
                            maxholesize=int(mh),
                            selfintersection=False,
                            newfaceselected=False)
            f_after = face_count()
        except Exception as e:
            print(f"[WARN] Close Holes (mh={mh}) fehlgeschlagen: {e}")
            break

        added = max(0, f_after - f_before)
        cum_added += added
        pass_pct = added / max(1, f_before)
        cum_pct  = cum_added / max(1, f0)

        print(f"[INFO] Pass {i}: maxholesize={mh} → +{added} Faces "
              f"(pass {pass_pct:.2%}, cumul {cum_pct:.2%})")

        stats.append({
            "pass": i,
            "maxholesize": int(mh),
            "added_faces": int(added),
            "pass_pct": float(pass_pct),
            "cum_pct": float(cum_pct),
        })

        # Stopkriterien
        if pass_pct > pass_stop_pct:
            print(f"[STOP] Abbruch: Pass über Schwelle ({pass_pct:.2%} > {pass_stop_pct:.2%}).")
            break
        if cum_pct > cum_stop_pct:
            print(f"[STOP] Abbruch: kumulativ über Schwelle ({cum_pct:.2%} > {cum_stop_pct:.2%}).")
            break

    return stats
