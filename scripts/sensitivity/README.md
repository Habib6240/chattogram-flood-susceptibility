# Distance_River predictor-reconstruction sensitivity scripts

These scripts produced the sensitivity results reported in manuscript Section 3.8 / Table 7 and Supplementary Information Section S12-S13 (Tables S20-S29). They are **analysis-record scripts, kept with the paths they were run with**; they are not a one-command pipeline.

**Design.** Only the `Distance_River` column was replaced; rows, the other 14 columns, labels, preprocessing, folds, hyperparameters and evaluation code are those of the archived B06U/B07U scripts. Random Forest and XGBoost were re-fitted under the archived designs (B06U spatial 5-fold CV and B07U strict held-event and held-geography evaluation), and the five MCDA surfaces were recomputed with the archived weights. The locked results are not replaced by any of these runs. No inferential test and no equivalence margin were used; changes are described against a seed-variation band (XGBoost `random_state` 43 and 44 on the original data) that was fixed in `docs/sensitivity_reports/PREREGISTERED_CRITERIA_XGB.txt` before any XGBoost result was computed.

| Script | Purpose |
|---|---|
| `ml_sens.py` | Random Forest (and a gradient-boosting surrogate, not used for any manuscript claim) re-fit, B06U and B07U designs, for a chosen `Distance_River` vector |
| `ml_metrics.py`, `ml_report.py` | Metrics and tables for the Random Forest runs |
| `xgb_run.py` | Original-implementation XGBoost re-fit, B06U and B07U (`SEED`, `TAG` environment variables for the seed-variation runs) |
| `xgb_report.py` | XGBoost, consensus (archived RF + XGBoost), class-stability and spatial-agreement tables |
| `mcda_sens.py`, `mcda_run.py` | MCDA surfaces recomputed with the archived weights and each `Distance_River` variant |

**Inputs not distributed here.** The scripts read `dist_orig.npy`, `dist_vector.npy` and `dist_raster30.npy` (per-location `Distance_River` vectors in master-table row order). These are HydroRIVERS-derived values and are not in this public repository. `orig` is the original column from the separate restricted record (signed EULA); `vector` is produced by `scripts/regenerate_distance_river/` (exact point-to-segment distance); `raster30` is a secondary variant (river rasterised at 30 m on the UTM-anchored grid, distance to the nearest river-pixel centre) whose construction code was run interactively and is **not** included as a standalone script. The scripts also read the master table `BUET_B04U_ML_Master_150m.npz` (rebuilt with `scripts/stage_and_run.py`) and archived outputs from the dataset record. Edit the absolute paths near the top of each script (`/tmp/orig/...`, `/home/claude/release/...`, `/home/claude/final/sens/...`) to your layout.

**Environment of these runs.** Linux x86-64, Python 3.13.18, scikit-learn 1.9.1, NumPy 2.5.3, XGBoost 3.4.1 built from the official source tag v3.4.1 (commit 6fe8c547bdd2). Random Forest re-fits on the original data reproduced the archived predictions exactly; XGBoost re-fits on the original data did **not** reproduce the archived predictions exactly on this platform (small differences, comparable to seed variation; the cause was not established), so the same-platform XGBoost re-run on the original data is the paired baseline for the XGBoost comparison. The consensus uses the archived Random Forest predictions, not a regenerated Random Forest. Regenerated-data runs used seed 42 only; seeds 43 and 44 were run on the original data only.

**Reports** (`docs/sensitivity_reports/`): `01` Random Forest and MCDA (historical statements about XGBoost are superseded by `03`), `03` XGBoost and consensus, `03a` its tables, and the pre-registered criteria file.
