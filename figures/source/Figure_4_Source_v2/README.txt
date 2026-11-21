Figure 4 source package (v2) — strict evaluation design
Changes from v1 (presentation only): bottom summary and note wording; "illustrative example only" added to the example subtitle; panel (a) map enlarged by about 7% and internal block-grid line weight reduced (0.5 to 0.35 pt). Fold colours, block geometry, AOI and schematic logic are unchanged (Figure_4_Block_Geometry_v2.csv is identical to the v1 table).
Files
- make_figure4.py : generates the PNG/SVG and Figure_4_Block_Geometry_v2.csv.
- Figure_4_Block_Geometry_v2.csv : one row per 5 km block (233): Block_5km, Spatial_Fold, Centroid_X/Y, Grid_bx/Grid_by, Left/Right/Bottom/Top (EPSG:32646).
- Figure_4_Strict_Evaluation_Design_v2.png (300 dpi) / .svg (editable text).
Inputs (not included; edit the UP path in the script): BUET_B06U_5km_Block_Spatial_Folds.csv, BUET_B04U_ML_Master_150m.npz, AOI_Boundary.zip, B07U_BUET_Strict_SpatioTemporal_LOEO_UrbanAware.py.
Build summary: 5 km squares recovered from the 30 m grid indices (floor(col*30/5000), floor(row*30/5000)); AOI shapefile (EPSG:4326) projected to EPSG:32646 and used as a vector clip path; panel (b) text taken from the B07U script (4 events, 5 folds, training = other 3 events and outside the held fold, test = held event and inside the held fold, block-overlap check).
