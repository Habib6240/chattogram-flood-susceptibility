# Materials not in the archive (upload request)

These are the only files that would close genuine reproducibility gaps. Nothing here changes any published number; it concerns provenance and the ability to rebuild upstream steps. Batches contain at most 20 files.

## Batch 1: scripts, logs and parameter records (priority order)

| # | Needed file | Why it matters |
|---|---|---|
| 1 | Google Earth Engine script(s) that build the 2019/2021/2023/2026 Sentinel-1 flood-reference labels (B03U; open-water and urban-aware pathways; Equations 2-5) | Only upstream step for the labels; currently unrebuildable |
| 2 | Earlier B03 (open-water, non-urban) GEE script, if separate from #1 | Produces the original 2021 open-water mask used in S5-S8 |
| 3 | GEE export script(s) for CHIRPS, WorldCover, NDVI (Sentinel-2 + Cloud Score+), HYSOGs250m and GHSL | Predictor provenance |
| 4 | GRASS/QGIS processing record: commands, graphical-model files or QGIS processing history (r.slope.aspect, r.watershed, r.stream.extract, line density, distance rasters) | Terrain, TWI, drainage-density provenance; software versions |
| 5 | Script that assembles the ML master dataset (B04U; plus B04 if B04U depends on it) | Produces `BUET_B04U_ML_Master_150m.npz/csv` |
| 6 | B05U baseline spatial-holdout script | SI Table S12 Panel A |
| 7 | B09A full-data scoring/rasterisation script (RF, XGBoost, consensus at 30 m) | Produces the three ML susceptibility rasters |
| 8 | MCDA scripts: DU10 (AHP), DU11 (FAHP), DU12 (ANP), DU13 (class map), DU14 (2021 validation), DU15 (sensitivity) | Weights and surfaces are archived; generating code is not |
| 9 | AHP pairwise-comparison matrix file | SI Table S2 note: matrix file not in archive |
| 10 | Final-figure scripts: Figure 1 (Q1_FINAL), 2 (v4), 3 (v2), 4 (Q1_FINAL), 5 (v4), 7 (Q1_FINAL) | The archived scripts belong to earlier versions of these figures |
| 11 | `pip freeze` / `conda list` from the training machine, with the scikit-learn version used for Random Forest | The only recorded gap in the software tier table |
| 12 | GRASS and QGIS version numbers | Same |
| 13 | Sentinel-1 scene list and pair-coverage screening log | Manuscript Table 1 coverage values (disclosed as recorded, not re-derived) |
| 14 | Domain P05/P95 table used for Equation 1 (if saved) | Would upgrade Eq. 1/11 from partial to full reproduction |

## Batch 2: data (only if you wish to enable full upstream rebuild; large, for Zenodo)

| # | Needed file | Note |
|---|---|---|
| 15 | The 13 aligned 30 m predictor rasters (or the GEE exports) | Needed to rebuild the master dataset and Equation 1 |
| 16 | Event pre/observation Sentinel-1 composites (VV, VH) for the four events | Needed to rebuild Equations 2-5; can be re-exported from GEE once script #1 exists |

Everything in the ML, SHAP, comparison and figure-data layers is already present, so the repository can be released without Batch 2 if the upstream gap is stated in the README (it is).


## Distance_River (separate restricted record)
`Distance_River` (HydroRIVERS-derived) is not in this public repository; the original values are in the separate restricted record https://doi.org/10.5281/zenodo.24398548 (signed EULA). See README and the public dataset record's `10_metadata/HydroSHEDS_Licence_Analysis.md` and `Distance_River_Regeneration.md`.
