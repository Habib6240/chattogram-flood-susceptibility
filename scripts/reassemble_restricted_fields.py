#!/usr/bin/env python3
"""Rebuild the original ML master dataset (CSV.gz, NPZ) and SHAP explanation sample.

The original tables combine columns under different terms, so the record ships them in parts:
  * public tables WITHOUT Distance_River and Distance_Road (05_ml_outputs/master_dataset, 07_shap_outputs);
  * the OSM-derived Distance_Road sidecars (ODbL 1.0, 05_ml_outputs/osm_derived_ODbL);
  * the separate RESTRICTED component record (chattogram-flood-susceptibility-restricted-distance-river-v1.0.0): the model-input tables WITH the
    HydroRIVERS-derived Distance_River (and without Distance_Road). Available only to recipients who signed the End User
    Licence Agreement (see 10_metadata/Access_Request_Procedure.md in the public dataset record).

Usage:
  python reassemble_restricted_fields.py --dataset <extracted dataset record> --out <dir> [--distance-river-dir <dir>]
Mode A: --distance-river-dir (default: a sibling folder named chattogram-flood-susceptibility-restricted-distance-river-v1.0.0 next to <dataset>, if present)
  contains the three "..._with_Distance_River_HydroSHEDS_EULA" tables. Distance_Road is inserted from the ODbL sidecars.
  Result: byte-identical (CSV, SHAP CSV) / array-identical (NPZ) to the archived originals; hashes in
  10_metadata/Original_Model_Input_Table_Hashes.csv.
Mode B: --distance-river-dir contains regenerated sidecars REGEN_Distance_River_Lattice.csv.gz and
  REGEN_Distance_River_SHAP_Sample.csv.gz (see 10_metadata/Distance_River_Regeneration.md). Values agree with the originals to
  about one 30 m cell; the result is NOT identical to the archived tables.
Rebuilt files combine ODbL-licensed and HydroSHEDS-licensed columns; each column keeps its own terms (DATA_LICENSE.md).
Do not redistribute the rebuilt files.
"""
import argparse, csv, gzip, os, sys
import numpy as np

def read_gz(p): return list(csv.reader(gzip.open(p, "rt", newline="")))
def write_gz(p, rows):
    with gzip.open(p, "wt", newline="", encoding="utf-8") as f: csv.writer(f, lineterminator="\r\n").writerows(rows)

ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--distance-river-dir", default=None)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
S = os.path.join(a.dataset, "05_ml_outputs", "osm_derived_ODbL"); M = os.path.join(a.dataset, "05_ml_outputs", "master_dataset")
_sib = os.path.join(os.path.dirname(os.path.abspath(a.dataset)), "chattogram-flood-susceptibility-restricted-distance-river-v1.0.0")
dr = a.distance_river_dir or (_sib if os.path.isdir(_sib) else None)
if dr is None:
    sys.exit("No Distance_River folder given and no sibling folder chattogram-flood-susceptibility-restricted-distance-river-v1.0.0 found. Use --distance-river-dir with the restricted component record (needs the signed EULA) or with regenerated sidecars (see Distance_River_Regeneration.md).")
SUF = "_with_Distance_River_HydroSHEDS_EULA"
A_files = [os.path.join(dr, "BUET_B04U_ML_Master_150m" + SUF + ".csv.gz"), os.path.join(dr, "BUET_B04U_ML_Master_150m" + SUF + ".npz"),
           os.path.join(dr, "BUET_B08U_XGB_SHAP_Explanation_Sample" + SUF + ".csv")]
B_files = [os.path.join(dr, "REGEN_Distance_River_Lattice.csv.gz"), os.path.join(dr, "REGEN_Distance_River_SHAP_Sample.csv.gz")]
mode = "A" if all(os.path.exists(p) for p in A_files) else ("B" if all(os.path.exists(p) for p in B_files) else None)
if mode is None:
    sys.exit("Distance_River tables not found in " + dr + "\nNeed the restricted component record (signed EULA) or regenerated sidecars: see Distance_River_Regeneration.md")
print("Mode", mode, "- using", dr)
road = {(r[0], r[1]): r[2] for r in read_gz(os.path.join(S, "OSM_Derived_Distance_Road_Lattice_ODbL.csv.gz"))[1:]}
if mode == "A":
    base = read_gz(A_files[0]); h = base[0]; pos = h.index("Distance_River") + 1
else:
    river = {(r[0], r[1]): r[2] for r in read_gz(B_files[0])[1:]}
    base = read_gz(os.path.join(M, "BUET_B04U_ML_Master_150m_without_restricted_fields.csv.gz")); h = base[0]
    pos = h.index("NDVI") + 2; assert h[pos - 1] == "HSG" and h[pos] == "Flood_2019"
ir, ic = h.index("Raster_Row"), h.index("Raster_Column")
assert len(road) == len(base) - 1
out = [h[:pos] + (["Distance_River"] if mode == "B" else []) + ["Distance_Road"] + h[pos:]]
for r in base[1:]:
    k = (r[ir], r[ic]); out.append(r[:pos] + ([river[k]] if mode == "B" else []) + [road[k]] + r[pos:])
write_gz(os.path.join(a.out, "BUET_B04U_ML_Master_150m.csv.gz"), out)
z = np.load(A_files[1] if mode == "A" else os.path.join(M, "BUET_B04U_ML_Master_150m_without_restricted_fields.npz")); d = {k: z[k] for k in z.files}
rr, cc = d["Raster_Row"], d["Raster_Column"]
arr = lambda m: np.array([np.float32(m[(str(int(x)), str(int(y)))]) for x, y in zip(rr, cc)], dtype=np.float32)
new = {}
for k in d:
    new[k] = d[k]
    if mode == "A" and k == "Distance_River": new["Distance_Road"] = arr(road)
    if mode == "B" and k == "HSG": new["Distance_River"] = arr(river); new["Distance_Road"] = arr(road)
np.savez(os.path.join(a.out, "BUET_B04U_ML_Master_150m.npz"), **new)
sd = {int(r[0]): r[3] for r in read_gz(os.path.join(S, "OSM_Derived_Distance_Road_SHAP_Sample_ODbL.csv.gz"))[1:]}
if mode == "A":
    with open(A_files[2], newline="", encoding="utf-8") as f: R = list(csv.reader(f))
    H = R[0]; q = H.index("Distance_Road_SHAP")
    RR = [H[:q] + ["Distance_Road_Value"] + H[q:]] + [r[:q] + [sd[i]] + r[q:] for i, r in enumerate(R[1:])]
else:
    sr = {int(r[0]): r[3] for r in read_gz(B_files[1])[1:]}
    with open(os.path.join(a.dataset, "07_shap_outputs", "BUET_B08U_XGB_SHAP_Explanation_Sample_without_restricted_values.csv"), newline="", encoding="utf-8") as f: R = list(csv.reader(f))
    H = R[0]; p = H.index("Distance_River_SHAP"); q = H.index("Distance_Road_SHAP")
    RR = [H[:p] + ["Distance_River_Value"] + H[p:q] + ["Distance_Road_Value"] + H[q:]]
    for i, r in enumerate(R[1:]): RR.append(r[:p] + [sr[i]] + r[p:q] + [sd[i]] + r[q:])
with open(os.path.join(a.out, "BUET_B08U_XGB_SHAP_Explanation_Sample.csv"), "w", newline="", encoding="utf-8") as f: csv.writer(f, lineterminator="\r\n").writerows(RR)
print("Rebuilt files written to", a.out)
