# Distance_River regeneration: targeted sensitivity validation (20 Nov 2026)
> **Status note (added at packaging).** This is the first of four sensitivity reports and was written before XGBoost could be built in the sandbox. Statements below that XGBoost retraining was 'impossible' and the HistGradientBoosting 'surrogate' results are historical: XGBoost 3.4.1 was subsequently built and run (report 03), and the manuscript and Supplementary Information rely on report 03 for XGBoost and the consensus. The surrogate is not the original model and is not used for any manuscript claim. Scripts cited here as '/home/claude/final/sens/ (not packaged)' are now in `scripts/sensitivity/` (as-run paths; see its README).

Nothing in the manuscript, other predictors, locked figures, archived outputs or the delivered archives was changed. Nothing published. All numbers below were computed in this session; scripts and result files are in `/home/claude/final/sens/` (not packaged).

## 0. What could not be done
| Item | Status | Blocker |
|---|---|---|
| Step 1 with the official HydroRIVERS v1.0 download | **Not executed** | hydrosheds.org is refused by the session's egress policy (checked again; not circumvented). River input was the author's pre-processed `05_River_HydroRIVERS_UPLAND25_UTM46N.gpkg` (3,378 segments, UPLAND_SKM 25.0-13,956.6). |
| XGBoost retraining (and therefore RF-XGBoost consensus) | **Impossible here** | `xgboost` is not installed and PyPI is blocked; the archived XGBoost `.joblib` cannot be loaded without it. |
| Replacement for XGBoost | Surrogate only | scikit-learn `HistGradientBoostingClassifier` (600 iterations, lr 0.05, depth 5, L2 1.0, 80% feature sampling, class-balancing weights, seed 42). **It is not the original model**; it is a sensitivity proxy for gradient boosting. |

## 1. Origin of the 32.38 m maximum difference
| Candidate source | Test | Result |
|---|---|---|
| UPLAND_SKM filter | Thresholds 30-100 km2 applied to the GeoPackage | Much worse (>=30: mean error 708 m). 25 is consistent. Lower thresholds untestable (file already filtered). |
| Coordinate transformation | Best rigid shift of the river vs the lattice (+/-40 m) | Gain only 8.0 to 7.4 m mean error: no translation/datum offset. Raw lon/lat reprojection itself untestable (Step 1). |
| Euclidean calculation / scale | Error vs distance, 0 m to 15 km | Flat (mean abs error 7.6-8.1 m in every distance bin): not a scale or projection effect. |
| **Rasterisation / cell-centre convention** | Original values repeat heavily (73,038 unique of 188,895; most common 7.099, 22.901, 37.099, ... and 55.325, 94.675 ...). These equal 30 m multiples plus the offsets of the lattice from a 30 m grid anchored at multiples of 30 m in UTM (lattice origin 335302.901 / 2544715.325). | Rasterise the same river at 30 m on that UTM-anchored grid (integer Bresenham), measure from lattice cell centres to nearest river-pixel centre: **73.9% of cells within 1 m of the original (median abs diff 0.046 m)**, mean abs diff 4.7 m vs 8.0 m for exact vector distance. The remaining 26% choose a different river pixel (differences about 20-44 m; max 43.7 m). |
| Original software | Tool/version not recorded | The line-rasterisation rule that would reproduce the other 26% was not identified. |
**Conclusion.** The discrepancy is mainly an implementation difference: the original appears to be a raster-based nearest-river-pixel distance on a 30 m river grid not aligned with the 150 m lattice, whereas the published workflow uses exact vector distance. About one 30 m pixel of difference is expected; exact replication needs the original tool. Preprocessing/UPLAND/reprojection beyond the above could not be separated from this. Not bit-identical, no claim of equivalence beyond the metrics below.

## 2. Model retraining (all other inputs, folds, labels, hyperparameters and seed 42 unchanged)
Environment check: rerunning RF on the **original** data reproduces the archived out-of-fold predictions exactly (Pearson 1.000000, max difference 0) in both designs, including the locked macro values (AUC 0.875839, AP 0.437801). So the comparison isolates the predictor.

**Random Forest (original model class), regenerated = exact vector distance**
| Design | Metric | Original | Regenerated | Change |
|---|---|---|---|---|
| 5-fold spatial CV (B06U) | pooled AUC / AP | 0.904597 / 0.738241 | 0.904696 / 0.738273 | +0.000098 / +0.000032 |
| Strict event+geography (B07U), macro of 4 events | pooled AUC / AP | 0.875839 / 0.437801 | 0.875624 / 0.436738 | -0.000215 / -0.001063 |
| B07U per event AUC (2019/21/23/26) | | 0.8883/0.8880/0.8390/0.8880 | 0.8882/0.8880/0.8386/0.8877 | -0.0001/0.0000/-0.0005/-0.0003 |
| B07U per event AP change | | | | -0.0010/-0.0014/-0.0004/-0.0014 |
Prediction level: Pearson 0.9972-0.9976, Spearman 0.992-0.994, mean abs change 0.013, max 0.116. Single scenarios (event x fold) moved by AUC -0.0045 to +0.0048.
Raster-style regeneration (B06U only): AUC -0.000017, AP +0.000259.
**Noise reference (B06U, RF seed 43 vs 42 on the original data):** AUC -0.00026, AP -0.00078, Pearson 0.99757, mean abs change 0.0134. The regenerated-predictor change (Pearson 0.99766, mean abs 0.0132) is the same size as changing the random seed. A seed control for B07U was not run.

**Gradient-boosting surrogate (not XGBoost)**
| Design | Original | Regenerated | Change |
|---|---|---|---|
| B06U pooled AUC / AP | 0.903659 / 0.725305 | 0.904142 / 0.729130 | +0.000483 / +0.003825 |
| B07U macro AUC / AP | 0.871430 / 0.430806 | 0.871020 / 0.431585 | -0.000410 / +0.000778 |
| B07U per event AUC change | | | +0.0000/+0.0009/-0.0051/+0.0025 |
| B07U per event AP change | | | +0.0001/-0.0012/-0.0057/+0.0099 |
The surrogate is more sensitive than RF at event level (up to 0.005 AUC and 0.010 AP). Its macro AUC (0.8714) is close to the archived XGBoost value (0.872429), but this does not show XGBoost behaves like it.

## 3. MCDA sensitivity
Recomputed from the master data with the archived weights and the 5th-95th-percentile standardisation (my reproduction of the archived AHP/FAHP/ANP rasters: r = 1.000000, mean abs error 1.5-2.4e-4; macro AUC reproduces to 1.5e-5 of the locked values). Distance_River weights: AHP 8.47%, FAHP 8.66%, ANP 8.68%. Standardisation bounds moved 168.43 to 168.66 m and 7,144.26 to 7,144.81 m.
| | AHP | FAHP | ANP |
|---|---|---|---|
| Score Pearson (orig vs regen) | 0.9999997 | 0.9999997 | 0.9999998 |
| Mean / max abs score change | 9e-5 / 3.7e-4 | 9e-5 / 3.8e-4 | 9e-5 / 3.8e-4 |
| Top-20% Jaccard | 0.9987 | 0.9988 | 0.9991 |
| Cells changing 5-class (archived FAHP breaks) | 0.07% | 0.08% | 0.07% |
| Macro AUC change | +0.000016 | +0.000016 | +0.000011 |
| Macro AP change | +0.000013 | +0.000015 | +0.000016 |
Locked MCDA values stay as archived; none is altered by this.

## 4. What this does and does not show
- Shown: regenerated values agree within about one 30 m pixel; MCDA is practically unchanged; RF metrics change by at most 0.0002 AUC and 0.001 AP (macro), comparable to a seed change in the one design where I measured that.
- Not shown: any XGBoost or consensus result (the locked 0.872429 and 0.876142 are archived, untested), official-download Step 1, B07U seed noise, or equivalence. No equivalence margin was pre-specified, so I make no equivalence claim.
- Note: an RF rerun on regenerated values would change the third decimal of the macro AP (0.438 to 0.437).

## 5. Recommendation on reproducibility claims
Revision of wording is needed; the results themselves need none.
1. Do not state that results are fully reproducible from the public archives. The archived outputs reproduce all reported numbers, and the pipeline is deterministic (original-data rerun is exact), but the master dataset omits Distance_River.
2. Suggested sentence: "The river-distance predictor is not redistributed; it can be regenerated from the official HydroRIVERS download. Regenerated values agree with the originals to about one 30 m grid cell, and re-running the Random Forest pipeline with them changed macro ROC-AUC and AP by at most 0.0002 and 0.001; archived outputs are the reference for all reported values."
3. State that XGBoost and consensus sensitivity were not re-tested unless you run it first.
4. Before submission, if you want a stronger claim: (a) run Step 1 on the official file and then the verification script; (b) run the archived B06U/B07U scripts unchanged on the master rebuilt from the regenerated sidecars on a machine with xgboost, and compare with the archived outputs.
