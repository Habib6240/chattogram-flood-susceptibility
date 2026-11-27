# Status of upstream materials (version 1.1.0)

Version 1.0.0 listed 16 items needed to close reproducibility gaps. Status after the author supplied scripts and records on 27 November 2026:

| # | Item | Status in 1.1.0 |
|---|---|---|
| 1 | GEE scripts for the urban-aware Sentinel-1 labels (B03U) | Included (`scripts/gee`) |
| 2 | Original B03 open-water script | Included |
| 3 | GEE exports for CHIRPS, WorldCover, NDVI; GHSL | Included for CHIRPS, WorldCover, NDVI. GHSL is used only inside B03U. HYSOGs250m was downloaded and aligned in QGIS; its alignment step is not recorded |
| 4 | GRASS/QGIS processing record | Partly: versions recorded (QGIS 3.44.12, GRASS 8.5.0, GDAL 3.13.1); command parameters not recorded (QGIS processing history empty) |
| 5 | Master-dataset assembly (B04U; B04) | Included |
| 6 | B05U baseline spatial-holdout script | Included |
| 7 | B09A full-data scoring and rasterisation | Included |
| 8 | MCDA scripts DU08-DU15 (and DU09A) | Included (`scripts/qgis_python/mcda`) |
| 9 | AHP pairwise-comparison matrix file | Included (`data/mcda/DU10_AHP_Pairwise_Matrix.csv`, and in the DU10 script) |
| 10 | Final-figure scripts | Partly: `B10A` included; scripts for each final main figure are not |
| 11 | `pip freeze` / `conda list`; Random Forest scikit-learn version | Partly: core library versions from the QGIS Python environment report; scikit-learn 1.9.1 read from the Random Forest model files. No full package list |
| 12 | GRASS and QGIS versions | Included (installed versions, 27 November 2026) |
| 13 | Sentinel-1 scene list and pair-coverage screening | Included (GEE audit scripts B01, B01C, B02A, B02C; B02B; their CSV outputs are in the dataset record or author outputs) |
| 14 | Domain P05/P95 table | Included (`data/mcda/DU09_Standardization_Rules.csv`) |
| 15 | 13 aligned 30 m predictor rasters | Not provided: several inputs cannot be redistributed |
| 16 | Event Sentinel-1 composites | Not provided; re-exportable by running the B03/B03U scripts |

Remaining gaps are listed in `docs/provenance/PROVENANCE_AND_CHRONOLOGY.md`, section 9. Independent end-to-end reproduction from raw data has not been demonstrated.

## Distance_River (separate restricted record)
`Distance_River` (HydroRIVERS-derived) is not in this public repository; the original values are in the separate restricted record https://doi.org/10.5281/zenodo.24398548 (signed EULA). See README and the public dataset record's `10_metadata/HydroSHEDS_Licence_Analysis.md` and `Distance_River_Regeneration.md`. The standardized Distance_River raster (DU09) is derived from those values and is likewise not deposited.
