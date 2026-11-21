#!/usr/bin/env python3
"""Step 2. Compute Distance_River (exact Euclidean distance, metres, EPSG:32646) from each lattice cell centre to the nearest
river line segment, and write the two sidecar files in the format read by reassemble_restricted_fields.py.

Inputs : --rivers  .npz from step 1, OR a GeoPackage line layer in EPSG:32646 (UPLAND_SKM >= 25 reaches)
         --master  public master CSV (.csv/.csv.gz) -> Raster_Row, Raster_Column, X, Y (cell centres, EPSG:32646)
         --shap-csv public SHAP sample CSV (X, Y per row; row order = Sample_Row_Index)
Output : REGEN_Distance_River_Lattice.csv.gz and REGEN_Distance_River_SHAP_Sample.csv.gz in --out-dir (private; derived from
         HydroRIVERS, so redistribution is subject to the HydroSHEDS licence).
Distance is computed per point to each straight segment (clamped projection), no rasterisation. Values stored as float32.
Dependencies: numpy only.
"""
import argparse, csv, gzip, io, os, numpy as np
from riverdist import load_segments, point_seg_dist
ap = argparse.ArgumentParser()
ap.add_argument("--rivers", required=True); ap.add_argument("--master", required=True)
ap.add_argument("--shap-csv", required=True); ap.add_argument("--out-dir", required=True)
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
S = load_segments(a.rivers); print("segments:", len(S))
def opn(p): return gzip.open(p, "rt", newline="") if p.endswith(".gz") else open(p, newline="")
with opn(a.master) as f:
    rd = csv.DictReader(f); rows = [(r["Raster_Row"], r["Raster_Column"], float(r["X"]), float(r["Y"])) for r in rd]
d = point_seg_dist(np.array([r[2] for r in rows]), np.array([r[3] for r in rows]), S).astype(np.float32)
def write(path, header, recs):
    with gzip.open(path, "wt", newline="") as g:
        w = csv.writer(g, lineterminator="\r\n"); w.writerow(header); w.writerows(recs)
write(os.path.join(a.out_dir, "REGEN_Distance_River_Lattice.csv.gz"), ["Raster_Row", "Raster_Column", "Distance_River"],
      [(r[0], r[1], repr(float(v))) for r, v in zip(rows, d)])
with opn(a.shap_csv) as f:
    sr = list(csv.DictReader(f))
ds = point_seg_dist(np.array([float(r["X"]) for r in sr]), np.array([float(r["Y"]) for r in sr]), S).astype(np.float32)
write(os.path.join(a.out_dir, "REGEN_Distance_River_SHAP_Sample.csv.gz"), ["Sample_Row_Index", "X", "Y", "Distance_River_Value"],
      [(i, r["X"], r["Y"], repr(float(v))) for i, (r, v) in enumerate(zip(sr, ds))])
print("lattice n=%d min=%.4f max=%.2f mean=%.3f" % (len(d), d.min(), d.max(), d.mean()))
