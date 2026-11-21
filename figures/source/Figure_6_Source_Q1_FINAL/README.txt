Figure 6 (Q1 FINAL) - ML consensus vs FAHP cross-family spatial comparison, rebuilt from the raw rasters
=======================================================================================================
INPUTS (user-supplied, not redistributed; never printed in the figure)
  08_Final_Common_Analysis_Mask_30m.tif            common valid domain (mask == 1; 4,722,216 cells)
  BUET_B09A_RF_XGB_ConsensusScore_30m.tif          continuous ML consensus surface (float32, nodata -9999)
  DU11_FAHP_Susceptibility_Index_01.tif            continuous FAHP surface (float32, nodata -9999)
  BUET_B09A_Consensus_Common5Class_30m.tif, DU13_FAHP_CommonClass_1to5.tif   supplied common-score class rasters: QA comparison only (47.2245% exact agreement)
  AOI_Boundary.zip                                 study-area boundary (EPSG:4326; projected to EPSG:32646 with the Kruger-series code, round-trip error < 1e-11 deg)
GRID: all rasters 4233 x 2825 cells, 30 m, EPSG:32646, upper-left (335302.901, 2544715.325) m; origins agree within 4.0e-5 m (script asserts < 0.01 m). Common mask: mask == 1; both continuous surfaces are valid at every mask cell (asserted).
CLASS DEFINITION (panel c): the LOCKED analysis uses method-specific percentile (quintile) classes derived from the continuous surfaces: class = 1 + number of the surface's own 20th/40th/60th/80th percentile breaks
  (numpy linear interpolation, computed on the 4,722,216 common valid cells) that are <= the value. Source: Supplementary Table S15 (exact agreement 47.3576% with percentile classes, 47.2245% with common-score classes), Table 6 and the B09B pairwise outputs
  (47.3576%). The supplied class rasters are common-score classes and are not displayed. Use --classes supplied only to inspect that alternative.
DIFFERENCE: D = C(ML consensus) - C(FAHP), integers -4..+4 (derived/Figure_6_ClassDifference_D_30m.tif, int8, nodata -99). D = 0 same class; D > 0 higher ML class; D < 0 higher FAHP class. No raw-score subtraction.
GATES: the script stops unless Pearson r 0.655320, exact agreement 47.3576%, within +/-1 class 88.1979% and top-20% Jaccard 0.381276 are reproduced (locked values are checks, never plotting inputs).
COLOUR: (a) ColorBrewer YlOrRd, (b) YlGnBu (different ramps on purpose), linear stretch between the 2nd and 98th percentiles of each surface on the common valid cells (display only; values beyond are clipped to the end colours); (c) 9-class discrete RdBu diverging, centred on 0 (grey #EDEDED), boundaries at +/-0.5.
DISPLAY RESAMPLING (rendering only): continuous panels are 2 x 2 block means of the 30 m cells (a display cell is drawn if >= 2 of its 4 cells are valid); the class-difference panel takes the nearest source cell; 1412 x 2116 display cells, drawn with nearest-neighbour; no smoothing. All statistics use the full 30 m data.
LAYOUT: 7.48 x 5.09 in; three maps 2.30 x 3.446 in on the same extent (84.75 x 126.99 km), AOI outline 0.5 pt; one scale bar (panel a) and one north arrow (panel c; grid north).
FONTS: Liberation Sans; labels 7.0 pt; panel titles 8.5 pt bold; panel letters 10 pt bold; footnote 6.7 pt (minimum 6.7 pt).
EXPORT: PNG 600 dpi (4488 x 3051 px); PDF with Liberation Sans embedded and maps at 600 ppi (1380 x 2068 px); SVG with live text (svg.fonttype = none) and the maps embedded at 600 ppi.
FILES: make_figure6_q1.py (figure + derived rasters + statistics), qc_figure6.py (independent QC -> derived/Figure_6_QC_results.txt), cvd_check_fig6.py, geoutil.py, run_figure6.sh;
  derived/ (D raster, per-method percentile-class rasters, class breaks, D distribution incl. the supplied-class alternative, summary statistics JSON, QC results, colour-vision results); qa_renders/ (CVD/greyscale simulations, independent Chromium SVG and poppler PDF renders); output/ (PNG, SVG, PDF).
DATA ATTRIBUTION: not drawn in the figure; belongs in the manuscript caption / data statement.
CHANGE FROM Q1_v1: none to the figure (the FINAL PNG is pixel-identical to Q1_v1); file names carry _FINAL; the QC script gained three guard checks (statistics never use stretched/display arrays; percentile classes are the default; one scale bar and one north arrow).
CLASS-SYSTEM GUARDRAIL: percentile classes reproduce 47.3576% exact agreement; the supplied common-score class rasters reproduce 47.2245% (QA only, never displayed). The figure uses the former because it matches Table 6 / Table S15 and the locked Figure 6 logic.
