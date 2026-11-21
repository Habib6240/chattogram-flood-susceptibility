Figure 2 source package (v2) — four Sentinel-1-derived urban-aware flood-reference masks
v2 change: legend wording only ("Sentinel-1-derived flood-reference observation"); the PNG above the legend strip is pixel-identical to v1 (0 differing pixels). Internal identifiers (B03U, file names) remain only in scripts, CSV tables and the QA note.
Files
- make_figure2.py (data QA + figure), geoutil.py (shapefile reader, UTM 46N projection).
- Figure_2_Event_Raster_Statistics_v2.csv, Figure_2_CSV_CrossCheck_v2.csv, Figure_2_Lattice_CrossCheck_v2.csv, Figure_2_Event_Metadata_v2.csv, Figure_2_Area_Convention_Check_v2.csv (valid and flood areas by cell count x 900 m2, by AOI-fraction weighting, and from the summary CSV).
- Figure_2_FloodReference_Events_v2.png (300 dpi) / .svg.
Inputs (not included; edit UP in the script): the four event mask rasters, BUET_B03U_Flood_Mask_Summary.csv, BUET_B04U_ML_Master_150m.npz (cross-check only), AOI_Boundary.zip (unzipped to ./AOI_Boundary).
Coding verified: 1 = flood-reference, 0 = valid non-flood reference, 255 = NoData. No percentages or areas are displayed in the figure.
