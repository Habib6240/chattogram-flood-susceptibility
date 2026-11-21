# XGBoost and consensus sensitivity to the regenerated Distance_River predictor

Date: 20 Nov 2026. Scope: the remaining validation after the Random Forest and MCDA analyses. Locked results, figures and conclusions are not modified; all numbers below are new reruns, shown next to the archived values.

## 0. Conclusion in brief
- The original XGBoost model was run with the original implementation (XGBoost **3.4.1**, the version recorded in the archive, built from the official source tag because PyPI is not reachable here). Its archived predictions and metrics were **not reproduced exactly** on this platform; the discrepancy is small and has the same size as seed-to-seed variation (Section 1).
- Replacing only Distance_River with the regenerated values changes macro ROC-AUC by +0.000272 and macro AP by +0.000576 in the strict event- and geography-withheld evaluation (same-platform paired rerun), and the consensus by +0.000173 AUC and +0.000085 AP. No quintile class moves by more than one class; the mean top-20% overlap (Jaccard) is 0.961.
- At macro level and for the consensus the changes lie inside the descriptive seed-noise reference. For five of the eight per-event metrics in the strict design (vector) they exceed it slightly (largest change 0.0021 AUC, 0.0017 AP). This is numerical stability, **not** statistical equivalence: no equivalence margin was defined or tested.
- The ordering and interpretation of the original findings (consensus slightly above RF above XGBoost; ML above MCDA; 2023 weakest) are unchanged. The archived values remain the reference.

## 1. Original reproduction (Phases 1-2)
**Recovered implementation.** Scripts `B06U_BUET_5Fold_Spatial_CV_UrbanAware.py` and `B07U_BUET_Strict_SpatioTemporal_LOEO_UrbanAware.py` (repository `scripts/ml`), `B08U...TreeSHAP_UrbanAware.py` (full-data model). Features: the 15 raw columns (13 factors, with aspect as sin/cos/flat, plus Distance_Road); ColumnTransformer: numeric pass-through plus one-hot for LULC and HSG, fitted on training rows only. XGBClassifier: n_estimators 600, learning_rate 0.05, max_depth 5, min_child_weight 2, subsample 0.8, colsample_bytree 0.8, reg_lambda 1, reg_alpha 0, binary:logistic, logloss, scale_pos_weight = neg/pos of the training set, tree_method hist, n_jobs -1, random_state 42. Folds: the archived 5 km block folds; B07U trains on any-flood of the other three events (all three labels valid) and tests on the held event in the held fold. Archived predictions: `06_oof_predictions`. Not recorded in the archive: hardware, operating system, compiler, thread count; the original paths indicate Windows. Archived model artifact reports XGBoost 3.4.1, CPU, seed 42, 600 rounds.

**Run environment.** XGBoost 3.4.1 compiled from tag v3.4.1 (commit 6fe8c547bdd2), Linux x86_64, CPU, Python 3.13.18, scikit-learn 1.9.1, NumPy 2.5.3. Environment check: Random Forest predictions are reproduced exactly with this stack (earlier task), so data, folds and labels match the archive.

**Result: not reproduced exactly.** Criterion fixed before running (`PREREGISTERED_CRITERIA_XGB.txt`): equal to six decimals and max prediction difference <= 1e-6.
- B07U macro AUC 0.872718 vs archived 0.872429 (+0.000289); macro AP 0.432848 vs 0.432363 (+0.000485); per-event values in Table B (differences up to 0.0015 AUC and 0.0028 AP).
- B06U pooled AUC 0.903324 vs 0.901732; max prediction difference 0.35, Pearson 0.995.
- Archived consensus: macro AUC 0.876142, AP 0.439899 reproduced as 0.876251 and 0.439977 (the consensus is built from the rerun XGBoost).

**Investigation of the discrepancy.** (a) The same-code refit of the archived full-data model (archived preprocessor, identical parameters and seed) differs from the archived model by up to 0.24 in probability (mean 0.014); in-sample AUC 0.93955 vs 0.93929. (b) The difference is present from the first tree (max margin difference 0.155 after one tree, 2.14 after 600), so it is not a late numerical drift. (c) It does not depend on thread count: n_jobs 1, 2 and -1 give identical predictions here. (d) The data are not the cause: RF reproduces exactly and labels, folds and coverage equal the archive. The most likely cause is platform-dependent random sampling (row and column subsampling use the C++ standard library generator, which differs between the original Windows build and this Linux build). This is a likely explanation, not a proven one; it could not be tested without the original platform. Consequence: the paired comparison below uses a same-platform rerun on the original Distance_River as the baseline, and archived values are shown for reference.

## 2. Regenerated XGBoost results (Phases 3-4)
Only the Distance_River column was replaced (188,894 of 188,895 values differ from the original; max difference 32.4 m, median 7.0 m for the vector regeneration). Same rows, features, labels, preprocessing, folds, hyperparameters, seed 42 and evaluation. "vector" is the documented regeneration workflow; "raster30" is a 30 m raster nearest-pixel variant (median difference 0.05 m, max 43.7 m) run as a second regenerated version. Seeds 43 and 44 were run on the original data only.

### A. B06U 5 km spatial block CV (XGBoost; valid composite rows n=169900)
| run | pooled AUC | pooled AP | fold-mean AUC | fold-mean AP | max abs dp vs archived | Pearson vs archived |
|---|---|---|---|---|---|---|
| archived | 0.901732 | 0.722925 | 0.901543 | 0.733702 | - | - |
| orig | 0.903324 | 0.727953 | 0.903614 | 0.739319 | 0.348 | 0.99517 |
| vector | 0.903239 | 0.726938 | 0.903286 | 0.737134 | 0.338 | 0.99521 |
| raster30 | 0.903810 | 0.729671 | 0.904139 | 0.741064 | 0.352 | 0.99472 |
| orig_s43 | 0.902770 | 0.725924 | 0.902475 | 0.735679 | 0.381 | 0.99525 |
| orig_s44 | 0.902290 | 0.723223 | 0.902153 | 0.733819 | 0.359 | 0.99548 |


### B. B07U strict event- and geography-withheld (XGBoost, macro over 4 events)
| run | macro AUC | macro AP | macro fold-mean AUC | macro fold-mean AP | AUC 2019/2021/2023/2026 | AP 2019/2021/2023/2026 |
|---|---|---|---|---|---|---|
| archived | 0.872429 | 0.432363 | 0.867956 | 0.430717 | 0.8867 / 0.8781 / 0.8326 / 0.8923 | 0.5582 / 0.4706 / 0.1844 / 0.5162 |
| orig | 0.872718 | 0.432848 | 0.868721 | 0.430352 | 0.8877 / 0.8796 / 0.8318 / 0.8918 | 0.5606 / 0.4732 / 0.1842 / 0.5134 |
| vector | 0.872990 | 0.433424 | 0.868889 | 0.430547 | 0.8877 / 0.8786 / 0.8339 / 0.8917 | 0.5589 / 0.4741 / 0.1856 / 0.5151 |
| raster30 | 0.872911 | 0.432958 | 0.868793 | 0.430604 | 0.8874 / 0.8796 / 0.8329 / 0.8917 | 0.5582 / 0.4736 / 0.1843 / 0.5158 |
| orig_s43 | 0.872398 | 0.432706 | 0.868407 | 0.431012 | 0.8872 / 0.8797 / 0.8310 / 0.8917 | 0.5577 / 0.4731 / 0.1852 / 0.5149 |
| orig_s44 | 0.872539 | 0.431756 | 0.868870 | 0.430268 | 0.8870 / 0.8806 / 0.8312 / 0.8913 | 0.5576 / 0.4732 / 0.1841 / 0.5122 |


### C. Absolute changes (regenerated minus baseline)
| design | comparison | dMacro AUC | dMacro AP | per-event dAUC (2019/21/23/26) | per-event dAP | pred diff mean / p99 / max | Pearson | Spearman |
|---|---|---|---|---|---|---|---|---|
| B06U | vector - rerun-orig | -0.000085 (pooled AUC) | -0.001015 (pooled AP) | - | - | 0.0140 / 0.0872 / 0.2484 | 0.99760 | 0.99782 |
| B06U | raster30 - rerun-orig | +0.000486 (pooled AUC) | +0.001718 (pooled AP) | - | - | 0.0139 / 0.0895 / 0.2512 | 0.99756 | 0.99783 |
| B07U | vector - rerun-orig | +0.000272 | +0.000576 | -0.0000 / -0.0010 / +0.0021 / -0.0000 | -0.0017 / +0.0009 / +0.0014 / +0.0017 | 0.0147 / 0.0965 / 0.3399 | 0.99706 | 0.99747 |
| B07U | raster30 - rerun-orig | +0.000193 | +0.000110 | -0.0003 / -0.0000 / +0.0011 / -0.0001 | -0.0024 / +0.0004 / +0.0000 / +0.0024 | 0.0145 / 0.0949 / 0.3528 | 0.99713 | 0.99752 |


### D. Seed-noise reference (XGBoost random_state 42 vs 43 vs 44 on ORIGINAL data, and on regenerated vector)
| design | metric | orig seeds 42/43/44 | max abs seed change (orig) | regenerated(vector)-orig (seed 42) | within band? |
|---|---|---|---|---|---|
| B06U | auc | 0.903324 / 0.902770 / 0.902290 | 0.001034 | -0.000085 | yes |
| B06U | ap | 0.727953 / 0.725924 / 0.723223 | 0.004730 | -0.001015 | yes |
| B07U | macro_auc | 0.872718 / 0.872398 / 0.872539 | 0.000320 | +0.000272 | yes |
| B07U | macro_ap | 0.432848 / 0.432706 / 0.431756 | 0.001092 | +0.000576 | yes |
| B07U | 2019 auc | 0.887694 / 0.887225 / 0.887002 | 0.000691 | -0.000010 | yes |
| B07U | 2019 ap | 0.560559 / 0.557671 / 0.557562 | 0.002997 | -0.001667 | yes |
| B07U | 2021 auc | 0.879626 / 0.879700 / 0.880610 | 0.000984 | -0.000985 | no |
| B07U | 2021 ap | 0.473219 / 0.473109 / 0.473198 | 0.000111 | +0.000901 | no |
| B07U | 2023 auc | 0.831783 / 0.830958 / 0.831226 | 0.000825 | +0.002123 | no |
| B07U | 2023 ap | 0.184234 / 0.185155 / 0.184105 | 0.000921 | +0.001391 | no |
| B07U | 2026 auc | 0.891769 / 0.891708 / 0.891318 | 0.000451 | -0.000041 | yes |
| B07U | 2026 ap | 0.513381 / 0.514890 / 0.512160 | 0.001510 | +0.001678 | no |


## 3. Consensus sensitivity (Phase 5)
The consensus rule is unchanged: (RF + XGBoost)/2 of out-of-fold probabilities, using the archived original RF predictions.
### E. RF-XGBoost consensus (B07U, archived RF + XGBoost; consensus=(RF+XGB)/2)
| consensus built from | macro AUC | macro AP | AUC 2019/21/23/26 | AP 2019/21/23/26 |
|---|---|---|---|---|
| archived RF + archived XGB | 0.876142 | 0.439899 | 0.8894 / 0.8852 / 0.8377 / 0.8922 | 0.5651 / 0.4864 / 0.1881 / 0.5199 |
| archived RF + XGB rerun (orig) | 0.876251 | 0.439977 | 0.8899 / 0.8859 / 0.8373 / 0.8919 | 0.5663 / 0.4873 / 0.1878 / 0.5186 |
| archived RF + XGB rerun (vector) | 0.876424 | 0.440062 | 0.8900 / 0.8855 / 0.8384 / 0.8919 | 0.5661 / 0.4872 / 0.1884 / 0.5186 |
| archived RF + XGB rerun (raster30) | 0.876365 | 0.440124 | 0.8899 / 0.8860 / 0.8378 / 0.8918 | 0.5659 / 0.4873 / 0.1880 / 0.5193 |
| archived RF + XGB rerun (orig_s43) | 0.876063 | 0.439779 | 0.8897 / 0.8859 / 0.8367 / 0.8919 | 0.5652 / 0.4872 / 0.1877 / 0.5190 |
| archived RF + XGB rerun (orig_s44) | 0.876139 | 0.439499 | 0.8896 / 0.8864 / 0.8369 / 0.8917 | 0.5651 / 0.4873 / 0.1879 / 0.5178 |

| consensus comparison | dMacro AUC | dMacro AP | mean / p99 / max abs prediction change | Pearson | Spearman | quintile class changed (mean over events) | moved >1 class | top-20% Jaccard (mean of events) |
|---|---|---|---|---|---|---|---|---|
| regen(vector) - rerun-orig consensus | +0.000173 | +0.000085 | 0.0073 / 0.0484 / 0.1700 | 0.99904 | 0.99907 | 0.0406 | 0.0000 | 0.9608 |
| regen(raster30) - rerun-orig consensus | +0.000114 | +0.000147 | 0.0073 / 0.0474 / 0.1764 | 0.99907 | 0.99908 | 0.0399 | 0.0000 | 0.9613 |
| rerun-orig - archived consensus | +0.000109 | +0.000078 | 0.0097 / 0.0622 / 0.1882 | 0.99837 | 0.99840 | 0.0541 | 0.0000 | 0.9479 |
| seed 43 - seed 42 (orig data) | -0.000188 | -0.000198 | 0.0096 / 0.0602 / 0.2198 | 0.99843 | 0.99841 | 0.0538 | 0.0000 | 0.9494 |
| seed 44 - seed 42 (orig data) | -0.000112 | -0.000478 | 0.0097 / 0.0615 / 0.2312 | 0.99839 | 0.99840 | 0.0539 | 0.0000 | 0.9487 |
| regen(vector) - archived consensus | +0.000282 | +0.000163 | 0.0097 / 0.0614 / 0.2024 | 0.99839 | 0.99841 | 0.0539 | 0.0000 | 0.9486 |

per-event consensus top-20% Jaccard and quintile change, regen(vector) vs rerun-orig:
- 2019: class changed 0.0410, moved >1 class 0.0000, top-20% Jaccard 0.9593
- 2021: class changed 0.0421, moved >1 class 0.0000, top-20% Jaccard 0.9650
- 2023: class changed 0.0393, moved >1 class 0.0000, top-20% Jaccard 0.9581
- 2026: class changed 0.0399, moved >1 class 0.0000, top-20% Jaccard 0.9608
Classes are five equal-count percentile classes of the consensus within each evaluated event; Jaccard is for the top 20% (class 5). Both were defined before the results.

## 4. Robustness checks (Phase 6)
- **Leakage and strictness.** In every one of the 20 B07U scenarios per run the code asserted that training and test rows are disjoint and that training rows are never in the test fold; the held-event label is never used in the training target (built only from the other three events); preprocessing is fitted on training rows only. Predictions cover exactly the archived held-event rows with no missing values. Only Distance_River differs between original and regenerated inputs; labels and folds equal the archived ones. Event and geography withholding therefore stayed strict.
- **Seed variation.** Two extra seeds on the original data. Macro AUC range 0.000320, macro AP range 0.001092 (B07U); B06U pooled AUC range 0.001034. These ranges rest on three runs and are a descriptive reference only.
- **Within or outside the reference band.** Macro AUC, macro AP and the B06U pooled metrics change by less than the seed ranges (Table D). Per-event: outside the reference band for 2021 AUC and AP, 2023 AUC and AP, and 2026 AP; inside for 2019 and 2026 AUC and 2019 AP. The largest per-event changes are +0.0021 AUC (2023) and +0.0017 AP (2019, 2026 with opposite signs) for the vector regeneration, and +-0.0024 AP (2019, 2026) for raster30; all are much smaller than the differences between events (2023 AUC is about 0.06 lower than the other events).
- **Numerical stability versus equivalence.** Predictions change by a mean of 0.015 (probability scale) with rank correlation 0.997; for the consensus 0.007 and 0.999; 4% of cells change quintile, none by more than one class. Changes between seeds on the original data are as large or larger (consensus mean 0.0097, class change 5.4%) than regenerated-minus-original. These support numerical stability. No equivalence margin was specified, so statistical equivalence is neither claimed nor tested.

## 5. Remaining limitations
1. XGBoost results are not bit-for-bit reproducible across platforms; the archive reference could not be matched exactly, and the cause is inferred.
2. Only the regenerated vector and raster30 variants were tested; the regenerated runs were done at seed 42 only (no seed repeats on regenerated data).
3. The consensus uses the archived RF predictions as the protocol requires; RF re-fits with regenerated data were reported in the earlier sensitivity analysis.
4. Full-data models and the 30 m full-domain RF/XGBoost/consensus maps, and the spatial-agreement figures with FAHP (Pearson 0.655320, Jaccard 0.381276), were not recomputed: the 30 m conditioning rasters are not available. Spatial agreement here is original versus regenerated consensus on the evaluated lattice cells.
5. Same-cohort comparisons remain descriptive; no inferential test was performed.
6. XGBoost was compiled from the official GitHub tag; the PyPI wheel used originally could not be installed.

## 6. Conclusion
Using the original XGBoost implementation, replacing the Distance_River values by regenerated ones changes macro ROC-AUC by at most 0.0003 and macro AP by at most 0.0006 (XGBoost) and by at most 0.0002 and 0.0002 (consensus) against a same-platform baseline, with a seed-noise reference of 0.0003 and 0.0011. Event-wise changes are at most 0.0024 (AP) and 0.0021 (AUC). The rankings and the locked findings remain intact, and the archived results are unchanged and remain the reference. The result supports numerical stability of the findings to this predictor, not formal equivalence, and the XGBoost archive could not be reproduced exactly on this platform.

Files: `xgb_run.py`, `xgb_report.py`, `PREREGISTERED_CRITERIA_XGB.txt` (kept in the working folder).
