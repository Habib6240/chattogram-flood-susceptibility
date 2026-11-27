# Software environment

Only versions that are evidenced are stated. Nothing has been assumed.

| Component | Version | Evidence tier |
|---|---|---|
| XGBoost | 3.4.1 | Confirmed: stamped inside the archived full-data model (`BUET_B08U_XGB_FullComposite_Model.joblib`) |
| scikit-learn (preprocessor / one-hot encoder) | 1.9.1 | Confirmed: stamped inside the archived preprocessor joblib |
| scikit-learn (Random Forest runs) | 1.9.1 | Confirmed: stamped inside the Random Forest model files (B05, B05U baselines and the B09A full-data model); read without unpickling |
| Python | 3.12.13 | QGIS Python console (27 November 2026); also observed in the independent verification run |
| NumPy | 2.4.6 | QGIS Python console (27 November 2026) and the independent verification run |
| SHAP | not used | TreeSHAP was computed with XGBoost's native `pred_contribs` |
| GDAL (`osgeo`) | 3.13.1 | QGIS Python console (27 November 2026) |
| SciPy 1.17.1, pandas 3.0.3 | Installed in the QGIS Python console (27 November 2026); matplotlib and joblib versions not recorded | Required by scripts; any recent release is expected to work, but none is verified for training-time identity |
| GRASS GIS / QGIS / GDAL | GRASS 8.5.0; QGIS 3.44.12; GDAL 3.13.1 | Installed in the QGIS Python environment on 27 November 2026 (`environment/QGIS_environment_report_2026-11-27.txt`); the QGIS project was last saved by 3.44.12 on 18 September 2026; command parameters not recorded |
| Google Earth Engine | Python/JavaScript API; datasets listed in `Source_Data_Manifest.csv` | Versions of the client are not applicable; dataset IDs are recorded |

Key hyper-parameters are in the scripts: Random Forest 500 trees, `class_weight="balanced_subsample"`, `random_state=42`; XGBoost 600 trees, learning rate 0.05, depth 5, subsample 0.80, column sample 0.80, `tree_method="hist"`, `random_state=42`. Explanation sample: `numpy.random.default_rng(42)`, 4,000 rows per fold.

Re-running training on a different library version can change model outputs slightly. The archived outputs, not a re-run, are the reference for the published numbers.

## Environment of the Distance_River sensitivity re-runs (`scripts/sensitivity`)
Linux x86-64; Python 3.13.18; scikit-learn 1.9.1; NumPy 2.5.3; XGBoost 3.4.1 built from the official source tag v3.4.1 (commit 6fe8c547bdd2). These are the versions in which the sensitivity re-runs were executed; they are not the (partly unrecorded) training-time versions. Random Forest re-fits on the original data reproduced the archived predictions exactly; XGBoost re-fits did not (small differences; platform explanation not proven). Seed 42 for all regenerated-data runs; seeds 43 and 44 on original data only.
