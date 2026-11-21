Supplementary Figures (final v1)
================================

Figures
- Figure_S1_Conditioning_Factors_v1.png / .svg       the 13 final conditioning factors on the common 30 m analysis domain
- Figure_S2_MCDA_sensitivity.png / .svg              sensitivity of MCDA ROC-AUC (three panels)
- Figure_S3_fold_level_performance.png / .svg        fold-level performance of the machine-learning models (two panels)
- figure_captions.txt                                final captions as used in the Supplement

Scripts and data
- build_figure_s1.py            read-only checks and drawing of Figure S1 (needs the factor rasters; see below)
- make_supplementary_figures.py generates Figures S2 and S3 from plotted_data/ (or from the audited source CSVs with --prepare DIR)
- plotted_data/                 exact values plotted in Figures S2 and S3 (columns of the audited source files)
- helper_tables/                Figure S1 checks: panel-file map, grid alignment, valid-cell reconciliation, mask and area-of-interest checks,
                                display statistics, categorical class counts

Running
  python make_supplementary_figures.py                  # Figures S2 and S3 from plotted_data/
  python make_supplementary_figures.py --prepare DIR    # rebuild plotted_data/ from the audited source CSVs in DIR first
  python build_figure_s1.py --data DATA_DIR --out OUTPUT_DIR
Figure S1 inputs (not included because of size): the 13 final selected-factor GeoTIFFs, 08_Final_Common_Analysis_Mask_30m.tif,
07A_Selected13_Factor_Audit.csv, 07B_Final_Mask_Summary.csv and AOI_Boundary.zip; the exact file names are listed in
helper_tables/panel_file_map.csv. Python 3 with numpy, pandas, Pillow and matplotlib is required; software versions are not recorded here.

Display rules (Figure S1)
- Common valid domain: 4,722,216 cells; cells outside are white; the area-of-interest outline is the same in every panel.
- Continuous factors: 2nd-98th percentile display stretch on the valid domain (not the P05-P95 bounds used for MCDA standardization).
- Drainage density and curvature: relative Low-High scale because units are not recorded; aspect 0-360 degrees with flat terrain in grey.
- LULC and HSG: categorical legends. Rainfall (about 0.05 degrees) and HSG (about 250 m) have coarser native support than the common grid.

Notes
- Values are plotted without estimation or smoothing; horizontal offsets of markers in Figure S3 change display position, not data values.
- Changes in ROC-AUC are descriptive; no inferential test was applied. The figures document inputs and evaluation results; they are not forecasts.
- Palette: Okabe-Ito colour-blind-safe colours for Figures S2 and S3; the standard WorldCover palette is used for LULC in Figure S1.
