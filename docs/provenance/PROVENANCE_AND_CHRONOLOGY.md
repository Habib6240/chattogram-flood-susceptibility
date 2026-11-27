# Provenance, chronology and verification notes (added in version 1.1.0)

All statements below come from the author's scripts, output files and statements, or from checks run on those files on 27 November 2026. Nothing was reconstructed. Items that remain unrecorded are listed in section 9.

## 1. Script map (where each pipeline stage lives)
| Stage | Files | Status |
|---|---|---|
| Sentinel-1 availability and pair screening (Google Earth Engine, JavaScript) | `scripts/gee/B01_*`, `B01C_*`, `B02A_*`, `B02C_*`; `scripts/ml/B02B_Final_Event_Selection_Audit.py` | Final. B02B is an offline Python step on the exported tables |
| Original open-water labels | `scripts/gee/B03_MultiEvent_Sentinel1_Flood_Mask_Master_ORIGINAL_OPEN_WATER.js` | Original label version (produced first) |
| Urban-aware labels | `scripts/gee/B03U_UrbanAware_...js`; QC: `scripts/ml/B03U_QC_Urban_Label_Revision.py` | Final label version used for all reported ML and MCDA results except the stated 2021 original-label MCDA validation |
| Predictor exports (GEE) | `scripts/gee/01_CHIRPSv3...`, `02_ESA_WorldCover...`, `03_Sentinel2_NDVI...` | Final. GHSL enters only inside B03U |
| `04_SoilGrids_Clay_...CANDIDATE_NOT_USED.js` | `scripts/gee/` | Candidate layer, not a model predictor (the hydrologic soil group is HYSOGs250m; the QGIS project also holds unused BARC soil and TRI candidate layers) |
| Factor audits, aspect fix, common mask (QGIS Python console) | `scripts/qgis_python/common_qc/` | Final |
| MCDA (standardisation, AHP, FAHP, ANP, comparison, 2021 validation, sensitivity) | `scripts/qgis_python/mcda/` (DU08 to DU15, DU09A) | Final; run in the QGIS Python console |
| ML dataset, baseline, final maps, figures | `scripts/ml/` (B04U, B05U, B06U, B07U, B09A, B10A), `scripts/shap/`, `scripts/comparison/` | Final |
| Original-label ML chain | `scripts/original_label_pipeline/` (B04 to B08C) | Superseded by the urban-aware chain; retained because it documents the label chronology and supports the label-version sensitivity |

The GEE scripts read private Earth Engine assets (`AOI_Boundary`, `Master_Area_Extraction`) that are not distributed here; the QGIS and ML scripts contain the author's original absolute Windows paths. They document the method; they cannot be re-run from this record alone.

## 2. Chronology of the label versions
The author confirmed on 27 November 2026 that the original B03 labels were produced first. The original-label ML chain was then run, an explainability dependence check (B08B) showed almost no flooded built-up cells, a label audit (B08C: "systematic in the full Sentinel-1 label dataset, or only a SHAP-sample effect") was run, and the labels were revised once to the urban-aware B03U version, after which the urban-aware chain was run. File modification times in the author's output archive are consistent with this order: B03 masks 2026-09-10 17:54; original-label models 2026-09-11 10:48 to 12:24; B03U masks 2026-09-11 15:23; B04U 15:41; B05U 15:55; B07U 16:37; B08U 16:46; B09A 17:10 (file times are supporting evidence only).
The revision can only add positives (open pathway identical to B03; the QC script requires zero old-flood-to-new-non-flood cells). No model hyper-parameter was retuned.
Built-up flood prevalence in the composite target rose from 0.13% (8 of 6,170 built-up cells) to 64.2% (3,962 of 6,170) (`BUET_B03U_QC_Composite_Comparison.csv`).

## 3. Sensitivity of the results to the label version
Computed on 27 November 2026 from the archived out-of-fold predictions and rasters by `scripts/verification/label_version_and_insample_check.py`; outputs in `data/verification/Label_version_and_insample_check_2026-11-27_*.csv`.
| Quantity | Original B03 labels | Urban-aware B03U labels |
|---|---|---|
| Strict macro ROC-AUC, RF / XGBoost / consensus | 0.9097 / 0.9073 / 0.9101 | 0.8758 / 0.8724 / 0.8761 |
| Strict macro AP, RF / XGBoost / consensus | 0.4603 / 0.4562 / 0.4626 | 0.4378 / 0.4324 / 0.4399 |
| MCDA macro ROC-AUC on all valid cells, AHP / FAHP / ANP | 0.7957 / 0.8012 / 0.8019 | 0.7707 / 0.7755 / 0.7652 |
The urban-aware revision lowered both families' discrimination; it did not raise the learned scores. The original-label strict ML run is the archived B07 file; the original-label MCDA values use all valid mask cells, not the 150 m cohort, and are descriptive.

## 4. In-sample optimism of the final full-data maps
The final 30 m ML maps (B09A) are fitted on all four events. Scored on those same events (all valid mask cells, urban-aware labels), macro ROC-AUC/AP are RF 0.905/0.515, XGBoost 0.902/0.515 and consensus 0.905/0.520, against 0.876/0.440 (consensus) in the strict held-event and held-geography evaluation. The full-domain surfaces therefore support spatial-agreement analysis, not accuracy claims.

## 5. Factor scoring for the MCDA (not five equal classes)
`scripts/qgis_python/mcda/09_DU_Common_Criterion_Standardization.py` and `data/mcda/DU09_Standardization_Rules.csv`: the ten continuous factors are scaled to 0-1 by linear membership between the factor's own 5th and 95th percentile over the common domain (direction increasing or decreasing as listed); LULC, HSG and aspect use prescribed class scores (script lines listing `LULC_SCORES`, `HSG_SCORES`, `ASPECT_SCORES`). The five-class maps (DU13, B09A) are equal-count classes of the final scores and are cartographic only; they were not used for validation or factor scoring. The class scores have no cited source.

## 6. MCDA weights
The AHP pairwise matrix (`10_DU_AHP...py`, `data/mcda/DU10_AHP_Pairwise_Matrix.csv`) is analyst-specified; no survey or expert elicitation was carried out. The script comment "literature-informed" is not backed by a citation and should not be read as one. FAHP reuses the AHP matrix; ANP reuses it with an analyst-defined dependency network (`DU12_ANP_Network_Links.csv`). Weight sensitivity: `DU15_*` files.

## 7. MCDA surface rebuild check (27 November 2026)
`scripts/verification/rebuild_mcda_surfaces.py` recomputed the AHP, FAHP and ANP surfaces as the weighted sum of the 13 `DU09_*_STD01` rasters with the archived weights. Over the 4,722,216 cells of the common mask the largest absolute difference from the archived rasters is 2.98e-08 for all three surfaces (float32 precision; no missing cells). The Elevation standardized raster recomputed from the DEM with the archived P05 and P95 differs by at most 7.24e-08. Output: `data/verification/MCDA_surface_rebuild_check_2026-11-27.csv`. The standardized rasters and the DEM are not redistributed.

## 8. Software evidence (Python/QGIS environment)
`environment/QGIS_environment_report_2026-11-27.txt`, produced by `scripts/qgis_python/qgis_environment_report.py` in the QGIS Python console used for the work: QGIS 3.44.12, GRASS 8.5.0, GDAL 3.13.1, Python 3.12.13, NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.3, scikit-learn 1.9.1, XGBoost 3.4.1, Qt 5.15.13, Windows 11. These are the versions installed on 27 November 2026; the QGIS project file (not distributed) was last saved by QGIS 3.44.12 on 18 September 2026. The Random Forest model files record scikit-learn 1.9.1 inside the pickles (`BUET_B05_RF_Baseline.joblib`, `BUET_B05U_RF_Baseline.joblib`, `BUET_B09A_RF_FullComposite_Model.joblib`; SHA-256 beginning c846186b, 1c67e4e7 and 0088ab7c; read without unpickling). The Random Forest pickles (225-284 MB) are not redistributed, as unpickling executes code.

## 9. Still unrecorded
- GRASS and QGIS command parameters (the QGIS processing-history database was empty); the SI states z-scale 1, TWI convergence factor 5, stream threshold 2,000 cells and a 1,000 m line-density radius, which are not independently recorded.
- Alignment of HYSOGs250m to the 30 m grid (only layer names in the QGIS project record it).
- Google Earth Engine collections are not date-pinned; the scripts record collection identifiers and parameters, not retrieval dates.
- Whether the XGBoost non-reproduction on a different platform arose from the Windows/Python 3.12 versus Linux/Python 3.13 difference is not established.
- Final main-figure scripts other than `B10A` are not included.

## 10. QGIS working-folder inventory (from the author's file listing, 27 November 2026)
The author's QGIS working folder (file names and modification dates as shown by Windows Explorer; file contents not re-examined) holds, in order of first appearance:
- 31 Aug 2026: DEM clip (EPSG:4326 and UTM 46N), slope, aspect, TRI (Riley).
- 1 Sep 2026: hydrology DEM (30 m), MFD flow direction and accumulation, TWI (MFD), curvature layers, streams at 1,000- and 2,000-cell thresholds, drainage density (2,000 version used for the final stack).
- 3 Sep 2026: master area extraction (AOI); CHIRPS, WorldCover and NDVI exports aligned to the 30 m grid; SoilGrids clay and BARC permeability candidates.
- 4-7 Sep 2026: HYSOGs250m alignment chain (`Aligned30m`, `NoData255`, `Chattogram_30m`, `STACK`); HydroRIVERS clipped, reprojected, rasterized and distance-to-river computed (5-6 Sep); OSM 2021 functional roads and distance-to-road (6-7 Sep).
- 8-9 Sep 2026: `*_STACK` versions of the factors and the candidate-factors common valid mask.
- 10 Sep 2026: aspect flat-area fix.
- 18 Sep 2026: project `Chattogong Project_A-Z.qgz` and a water display raster.
Layers present in the working folder but not among the 13 model predictors: SoilGrids clay, BARC permeability, TRI, tangential and profile curvature, the 1,000-cell stream layer, and a temporary water-class raster. The reason each was not used is not recorded and is not asserted. The ordering supports the statement that the 13-factor stack was assembled on 8-9 September, before the Sentinel-1 label scripts of 10-11 September. These files are intermediates and are not redistributed.
