# Changelog

## 1.1.0 (staging; release date set at publication)
Added
- `scripts/gee` (10 Google Earth Engine scripts), `scripts/qgis_python` (factor QC and MCDA DU08-DU15 and an environment-report script), `scripts/ml` additions (B02B, B03U QC, B04U, B05U, B09A, B10A) and `scripts/original_label_pipeline` (B04-B08C).
- `docs/provenance/PROVENANCE_AND_CHRONOLOGY.md`: script map, label-version chronology, scoring rules, weights statement, MCDA rebuild check, software evidence, remaining gaps.
- `scripts/verification/rebuild_mcda_surfaces.py` and `label_version_and_insample_check.py` with their outputs in `data/verification/`.
- 16 MCDA construction CSVs in `data/mcda/` (standardization rules, AHP matrix, FAHP fuzzy matrix, ANP network and supermatrices, comparison and class tables).
- `environment/QGIS_environment_report_2026-11-27.txt`.
Changed
- README, `docs/MISSING_MATERIALS.md`, `environment/Software_Environment.md`, `CITATION.cff`, `.zenodo.json` updated to the above (scikit-learn 1.9.1 for the Random Forest, QGIS/GRASS/GDAL versions, remaining gaps).
No published number changed. No model, label or figure was regenerated.

## 1.0.0
First release.
