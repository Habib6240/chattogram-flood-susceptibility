# Chattogram flood susceptibility: prescribed (MCDA) versus learned (ML) susceptibility

Code, derived data and documentation for the manuscript *Prescribed versus Learned Flood Susceptibility: MCDA and Machine Learning Under Event- and Geography-Withheld Evaluation in Chattogram, Bangladesh* (author: Md. Ahsan Habib, Department of Civil Engineering, Bangladesh University of Engineering and Technology (BUET), Dhaka-1000, Bangladesh; ORCID 0009-0009-6932-0567; contact 1904062@ce.buet.ac.bd).

**Version 1.1.0** adds the upstream scripts (Google Earth Engine, QGIS Python console, dataset assembly, final mapping, MCDA), a provenance and label-chronology note, a label-version sensitivity check, an in-sample-optimism check, an MCDA rebuild check and environment evidence. No published number changed. See `CHANGELOG.md`.

**Links.** GitHub repository: https://github.com/Habib6240/chattogram-flood-susceptibility. Software record (this repository, archived on release): https://doi.org/10.5281/zenodo.24397165. Public dataset record: https://doi.org/10.5281/zenodo.24397676. Restricted `Distance_River` record (signed EULA): https://doi.org/10.5281/zenodo.24398548. DOIs were reserved in Zenodo before publication; a DOI link resolves only once its record is published.

## What the study does
One 13-factor conditioning system is used to build (i) prescribed MCDA susceptibility (AHP, Buckley FAHP, ANP) and (ii) learned susceptibility (Random Forest, XGBoost and their consensus). Both are evaluated against four Sentinel-1-derived flood-reference observations (2019, 2021, 2023, 2026). Predictive discrimination, spatial agreement and factor attribution are reported separately and are never interchanged.

Interpretive limits: the references are remotely sensed observations, not field-verified flood truth; the strict evaluation withholds the target event and the target geography but is not forward forecasting; MCDA judgments are analyst-specified; ML scores are susceptibility scores, not calibrated probabilities; the same-cohort MCDA-ML comparison is descriptive, with no inferential superiority test; SHAP is model-specific and associational.

## Repository layout
| Path | Contents |
|---|---|
| `scripts/gee` | Google Earth Engine scripts: Sentinel-1 availability, pair screening and flood-label generation (original B03 and urban-aware B03U); CHIRPS, WorldCover and NDVI exports |
| `scripts/qgis_python` | QGIS Python-console scripts: factor audits and common mask (`common_qc`), MCDA standardisation, AHP, FAHP, ANP, comparison, validation and sensitivity (`mcda`), and an environment-report script |
| `scripts/ml` | B02B event selection, B03U label QC, B04U dataset assembly, B05U baseline, B06U spatial 5-fold CV, B07U strict held-event and held-geography evaluation, B09A final 30 m maps, B10A figures/tables |
| `scripts/original_label_pipeline` | Original (pre-revision) label ML chain B04 to B08C, retained for the label chronology |
| `docs/provenance` | Provenance, label chronology, scoring rules, MCDA rebuild check, software evidence |
| `scripts/shap` | B08U full-data XGBoost TreeSHAP; B08V dependence summaries |
| `scripts/comparison` | B09B ML-versus-MCDA comparison and spatial agreement |
| `scripts/stage_and_run.py` | Builds the directory layout the original scripts expect and writes patched copies pointing to it (originals are untouched) |
| `scripts/sensitivity` | Distance_River predictor-reconstruction sensitivity analysis (analysis-record scripts; see its README) |
| `scripts/verification` | Independent recomputation scripts from the pre-submission audit (archival) |
| `data/` | Compact CSV outputs: MCDA weights and sensitivity, folds and performance, SHAP summaries, comparison tables, verification outputs, flood-mask summaries (no boundary geometry) |
| `figures/source` | Figure scripts and plotted-data tables (see `FIGURE_SOURCE_STATUS.md`) |
| `supplement/` | Supplementary tables workbook, provenance and callout crosswalk |
| `environment/` | Evidence-tiered software record, `requirements.txt`, `environment.yml` |
| `docs/sensitivity_reports` | Sensitivity reports and the pre-registered criteria file |
| `docs/MISSING_MATERIALS.md` | Status of upstream materials (resolved and still missing) |
| `Source_Data_Manifest.csv` | Third-party inputs, identifiers, access routes, licence notes |
| `File_Inventory.csv`, `SHA256SUMS.txt` | Inventory and checksums of every packaged file |

Large products (rasters, master dataset, out-of-fold predictions, model files, SHAP sample, final figures) are in the companion Zenodo dataset record (numbered folders 01-11, same checksum scheme).

## Reproducibility levels
Five levels are kept separate: (1) scientific robustness, (2) numerical reconstruction, (3) computational reproducibility, (4) data accessibility, (5) independent end-to-end reproducibility.
| Layer | Status |
|---|---|
| Numerical reconstruction: published metrics from archived out-of-fold predictions, SHAP outputs and rasters | Reproducible from archived files; independently recomputed in the audit (`data/verification`) |
| Computational reproducibility: ML training (B06U, B07U), SHAP (B08U), dependence (B08V), comparison (B09B) from the master dataset | Scripts included and runnable after staging. B08V reproduced the archived summary exactly (maximum difference 0.0). On the original data the Random Forest B06U/B07U re-fit reproduced the archived out-of-fold predictions exactly, whereas the XGBoost re-fit did not (small differences, comparable to seed variation; cause not established); the archived outputs remain the reference for published numbers |
| Scientific robustness to the rebuilt `Distance_River` column | Single-regeneration sensitivity analysis (`scripts/sensitivity`, `docs/sensitivity_reports`); summary below. Descriptive only: no inferential test and no equivalence claim |
| MCDA weights and surfaces | DU08-DU15 scripts included. The three surfaces were recomputed from the standardized rasters and archived weights (maximum difference 3e-08, `docs/provenance`); the standardized rasters themselves are not redistributed. Weights are analyst-specified |
| Data accessibility | Public dataset record; `Distance_River` in a separate access-restricted record (signed EULA); `Distance_Road` as an ODbL sidecar |
| Independent end-to-end reproducibility (Sentinel-1 labels, Equations 2-5; predictor stack GRASS/QGIS/GEE; master-dataset assembly) | **Not demonstrated.** The generating scripts are now included (`scripts/gee`, `scripts/qgis_python`, `scripts/ml`), but they have not been re-run from raw data, they refer to private Earth Engine assets and original local paths, and GRASS/QGIS command parameters and HYSOGs250m alignment are not independently recorded (`docs/provenance/PROVENANCE_AND_CHRONOLOGY.md`, section 9) |

## Execution
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r environment/requirements.txt     # GDAL bindings (osgeo) via conda-forge/system GDAL for B09B
# 1. Download and extract the Zenodo dataset record (folders 01-11) to ../zenodo_data
python scripts/stage_and_run.py --repo . --zenodo ../zenodo_data --base /path/to/work [--distance-river-dir /path/to/distance_river_sidecars]   # rebuilds the original master/SHAP files when the restricted component record (signed EULA) is unpacked next to the dataset record or given with --distance-river-dir, or a regenerated-Distance_River variant (see below)
# 2. Run in order (patched copies; each reads the previous step's outputs)
cd /path/to/work/_scripts
python B06U_BUET_5Fold_Spatial_CV_UrbanAware.py
python B07U_BUET_Strict_SpatioTemporal_LOEO_UrbanAware.py
python B08U_BUET_XGBoost_TreeSHAP_UrbanAware.py
python B08V_BUET_UrbanAware_SHAP_Dependence.py
python B09B_BUET_ML_vs_MCDA_Strict_Comparison.py
```
Retraining may differ marginally with other library versions; the archived outputs are the reference for the published numbers. Compare regenerated files with the archived CSVs, and check data files with `sha256sum -c SHA256SUMS.txt`.

## Licences
Scoped by component (`LICENSING.md`, `DATA_LICENSE.md`, `THIRD_PARTY_NOTICES.md`): **MIT applies to source code only**; author-generated tables and documentation are under CC BY 4.0 (author-approved; third-party material excluded); third-party data keep their own terms (see `Source_Data_Manifest.csv`); OpenStreetMap-derived `Distance_Road` values are only in a separate ODbL 1.0 sidecar in the dataset record; HydroRIVERS-derived `Distance_River` values are not in this repository; they are in the separate restricted Zenodo record https://doi.org/10.5281/zenodo.24398548 (signed End User Licence Agreement required; `docs/EULA_HydroSHEDS_Distance_River_DRAFT_v2.docx` is an author-drafted draft, not reviewed by a lawyer and not issued or approved by WWF). No blanket CC BY 4.0 applies to the mixed dataset. Required attribution for Copernicus GLO-30-derived values: Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

## Boundary data
No administrative-boundary file is included. The study-area outline carried GADM attribute fields and was removed. To obtain boundaries independently, download GADM (BGD_adm2, version 2 schema; the study used the Chittagong district feature, ID_2 = 10) from gadm.org under the GADM licence. The exact AOI geometry used in the study is not published. The figure-generation scripts under `figures/source/` expect an `AOI_Boundary` shapefile as an external input that is not distributed here; they cannot be re-run without supplying one.

## Citation
See `CITATION.cff`. Cite the article and the Zenodo software record (https://doi.org/10.5281/zenodo.24397165); cite the dataset record (https://doi.org/10.5281/zenodo.24397676) when using the data.

## Script classification
| Script group | Role | Status |
|---|---|---|
| `scripts/ml/B06U_*`, `B07U_*` | Spatial CV and strict event-and-geography evaluation | Original archived scripts, essential; runnable after staging |
| `scripts/shap/B08U_*`, `B08V_*` | TreeSHAP attribution and dependence | Original archived scripts, essential |
| `scripts/comparison/B09B_*` | Same-cohort ML-MCDA comparison, spatial agreement | Original archived script, essential |
| `scripts/sensitivity/*` | Distance_River sensitivity analysis | Analysis-record scripts with as-run paths; require restricted or regenerated `Distance_River` vectors not distributed here |
| `scripts/stage_and_run.py`, `scripts/reassemble_restricted_fields.py` | Staging helper; rejoins the separately licensed OSM-derived column | New helpers written for this package (not original analysis scripts) |
| `scripts/verification/*` | Audit recomputation | Archival; paths must be edited |
| Sentinel-1 label generation (GEE), factor preprocessing (QGIS Python), master-dataset assembly (B04U), MCDA DU08-DU15 scripts | Upstream | Included as the author's original scripts (version 1.1.0); not re-run from raw data; see `docs/provenance` |
| Third-party tools (QGIS, GRASS, Google Earth Engine) | Processing | Non-redistributable; versions as in `environment/Software_Environment.md` |

## Known limitations
- Sentinel-1 references are remotely sensed observations, not field-verified flood truth; 2023 is post-peak persistent inundation.
- Strict evaluation withholds event and geography but is not forward forecasting; no buffer between adjacent 5 km blocks.
- Predictor stack is static; rainfall is a 1991-2020 climatology; GHSL is used only in the label-support pathway, not as a predictor.
- ML outputs are susceptibility scores, not calibrated probabilities. The ML-MCDA comparison is descriptive (no inferential test). SHAP is associational.
- Archived outputs are the reference for published numbers. Library versions at training time are only partly recorded; the sensitivity re-runs used the environment stated in `scripts/sensitivity/README.md`, and the XGBoost re-fit on the original data did not reproduce the archived XGBoost predictions exactly.

## Data-source acknowledgements
ESA/Copernicus (Sentinel-1, Sentinel-2, WorldCover, GLO-30), UCSB Climate Hazards Center (CHIRPS), European Commission JRC (GHSL, Global Surface Water), NASA/ORNL DAAC (HYSOGs250m, SRTM), HydroSHEDS (HydroRIVERS), OpenStreetMap contributors (ODbL), GADM and Google Earth Engine. Full identifiers: `Source_Data_Manifest.csv`.

## Quality-assurance checks
```bash
sha256sum -c SHA256SUMS.txt          # every packaged file; expected: all OK
python -m py_compile scripts/*/*.py scripts/*.py   # syntax check of all scripts
```
After the staged run, compare regenerated CSVs with the archived files in `data/` (numerical differences beyond floating-point noise should be investigated). Full end-to-end reproduction from raw satellite data is **not** supported by this package (see Reproducibility levels).

## Reproducibility and the Distance_River predictor
`Distance_River` (HydroRIVERS-derived) is not in this public repository. The complete original values are in the separate access-restricted Zenodo record https://doi.org/10.5281/zenodo.24398548, accessible after a signed EULA; with that record unpacked next to the dataset record (or `--distance-river-dir`) `scripts/stage_and_run.py` and `scripts/reassemble_restricted_fields.py` rebuild the original model-input tables exactly (SHA-256 of the originals: dataset `10_metadata/Original_Model_Input_Table_Hashes.csv`; the archive rebuild was re-tested: master CSV and SHAP CSV hashes identical and 28 of 28 NPZ arrays identical). Without it, regenerate the column from the official HydroRIVERS download with `scripts/regenerate_distance_river/`. Regenerated values are **not** identical to the originals (vector regeneration: median difference 6.98 m, mean 8.03 m, maximum 32.38 m; Pearson 0.99999; 188,894 of 188,895 values differ).

**Sensitivity of the results to the rebuilt column** (single regeneration; only `Distance_River` changed; details in `scripts/sensitivity/README.md` and `docs/sensitivity_reports/`; locked results are unchanged):
- Random Forest, strict event-and-geography evaluation: macro ROC-AUC 0.875839 to 0.875624 (-0.000215), macro AP 0.437801 to 0.436738 (-0.001063).
- XGBoost, against a same-platform re-run on the original data: macro ROC-AUC +0.000272, macro AP +0.000576 (archived 0.872429 / 0.432363; same-platform re-run 0.872718 / 0.432848; regenerated 0.872990 / 0.433424). Five of eight per-event metrics fell outside the seed-variation band.
- Consensus (archived Random Forest plus XGBoost): macro ROC-AUC 0.876142 (archived) / 0.876251 (re-run) / 0.876424 (regenerated); Pearson 0.99904 between re-run and regenerated consensus scores; 4.06% of cells change susceptibility class; top-20% Jaccard 0.9608. Changing only the XGBoost seed changed about 5.4% of classes (Jaccard about 0.949).
- MCDA surfaces recomputed with the archived weights: mean score change 9e-5 (maximum 3.8e-4), class change 0.07-0.08%, top-20% Jaccard 0.9987-0.9991.

These are descriptions of numerical stability, not equivalence tests (no margin was defined). The XGBoost platform explanation for the non-reproduction is not proven. The consensus in the sensitivity analysis uses the archived Random Forest predictions. Archived outputs remain the reference for all published numbers. Step 1 of the regeneration (official shapefile to UTM) was not executed; steps 2-5 were run on the author's pre-processed GeoPackage.
