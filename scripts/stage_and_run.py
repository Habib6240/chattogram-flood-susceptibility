#!/usr/bin/env python3
"""Stage the archived inputs into the directory layout expected by the original scripts,
and write patched copies of the scripts that point to the staged BASE.

The original scripts (scripts/ml, scripts/shap, scripts/comparison) are NOT modified: they contain a
hard-coded Windows BASE path from the authors' workstation. This helper never alters them.

Usage
  python scripts/stage_and_run.py --repo . --zenodo /path/to/extracted_zenodo_deposit --base /path/to/work
  python /path/to/work/_scripts/B08V_BUET_UrbanAware_SHAP_Dependence.py      # example run

Requires the Zenodo deposit (large inputs) in addition to this repository (small inputs).
"""
import argparse, os, re, shutil, glob, gzip, sys

LAYOUT = {  # filename -> relative directory under BASE (as expected by the original scripts)
 "BUET_B04U_ML_Master_150m.npz": "13_BUET_ML/02B_ML_Dataset_UrbanAware",
 "BUET_B06_5km_Block_Spatial_Folds.csv": "13_BUET_ML/04_Spatial_CV",
 "BUET_B07U_LOEO_OutOfFold_Predictions.npz": "13_BUET_ML/05B_Strict_LOEO_UrbanAware",
 "BUET_B08U_XGB_SHAP_Explanation_Sample.csv": "13_BUET_ML/06B_Explainability_UrbanAware",
 "BUET_B09A_RF_SusceptibilityScore_30m.tif": "13_BUET_ML/07_Final_30m_Maps_UrbanAware",
 "BUET_B09A_XGB_SusceptibilityScore_30m.tif": "13_BUET_ML/07_Final_30m_Maps_UrbanAware",
 "BUET_B09A_RF_XGB_ConsensusScore_30m.tif": "13_BUET_ML/07_Final_30m_Maps_UrbanAware",
 "08_Final_Common_Analysis_Mask_30m.tif": "11_FINAL_ANALYSIS_INPUTS",
 "DU10_AHP_Susceptibility_Index_01.tif": "12_MCDA",
 "DU11_FAHP_Susceptibility_Index_01.tif": "12_MCDA",
 "DU12_ANP_Susceptibility_Index_01.tif": "12_MCDA",
}
SCRIPTS = ["scripts/ml/B06U_BUET_5Fold_Spatial_CV_UrbanAware.py",
           "scripts/ml/B07U_BUET_Strict_SpatioTemporal_LOEO_UrbanAware.py",
           "scripts/shap/B08U_BUET_XGBoost_TreeSHAP_UrbanAware.py",
           "scripts/shap/B08V_BUET_UrbanAware_SHAP_Dependence.py",
           "scripts/comparison/B09B_BUET_ML_vs_MCDA_Strict_Comparison.py"]
BASE_RX = re.compile(r'BASE\s*=\s*\(\s*r"[^"]*"\s*r"[^"]*"\s*\)', re.S)

def find(name, roots):
    for r in roots:
        for p in glob.glob(os.path.join(r, "**", name), recursive=True) + glob.glob(os.path.join(r, "**", name + ".gz"), recursive=True):
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="."); ap.add_argument("--zenodo", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--distance-river-dir", default=None, help="folder with the Distance_River tables: the unpacked restricted component record (signed EULA) or regenerated sidecars (default: sibling folder chattogram-flood-susceptibility-restricted-distance-river-v1.0.0 next to the dataset record, if present)")
    a = ap.parse_args(); roots = [a.repo, a.zenodo]; missing = []
    side = os.path.join(a.zenodo, "05_ml_outputs", "osm_derived_ODbL", "OSM_Derived_Distance_Road_Lattice_ODbL.csv.gz")
    if os.path.exists(side):   # public master/SHAP files lack Distance_River (separate restricted record) and Distance_Road (ODbL sidecar)
        _sib = os.path.join(os.path.dirname(os.path.abspath(a.zenodo)), "chattogram-flood-susceptibility-restricted-distance-river-v1.0.0")
        if not a.distance_river_dir and os.path.isdir(_sib):
            a.distance_river_dir = _sib
        if a.distance_river_dir:
            import subprocess; re_dir = os.path.join(a.base, "_reassembled_restricted")
            subprocess.check_call([sys.executable, os.path.join(a.repo, "scripts", "reassemble_restricted_fields.py"), "--dataset", a.zenodo, "--out", re_dir, "--distance-river-dir", a.distance_river_dir])
            roots = [re_dir] + roots
        else:
            print("WARNING: no Distance_River sidecars found (the restricted component record is not next to the dataset record and --distance-river-dir was not given), so the master dataset and SHAP sample cannot be rebuilt;"
                  " ML scripts B06U/B07U/B08U/B08V cannot be run. Other inputs are staged.")
    for name, rel in LAYOUT.items():
        src = find(name, roots)
        if not src: missing.append(name); continue
        dst_dir = os.path.join(a.base, *rel.split("/")); os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, name)
        if src.endswith(".gz") and not name.endswith(".gz"):
            with gzip.open(src, "rb") as f, open(dst, "wb") as g: shutil.copyfileobj(f, g)
        elif not os.path.exists(dst): shutil.copy2(src, dst)
    out = os.path.join(a.base, "_scripts"); os.makedirs(out, exist_ok=True)
    for s in SCRIPTS:
        txt = open(os.path.join(a.repo, s), encoding="utf-8").read()
        new, n = BASE_RX.subn('BASE = r"%s"' % a.base.replace("\\", "\\\\") if os.name == "nt" else 'BASE = %r' % a.base, txt)
        if n != 1: sys.exit("BASE literal not found exactly once in " + s)
        open(os.path.join(out, os.path.basename(s)), "w", encoding="utf-8").write(new)
    print("Staged under", a.base); print("Patched scripts in", out)
    if missing: print("MISSING inputs (not found in repo or Zenodo folder):", *missing, sep="\n  ")

if __name__ == "__main__": main()
