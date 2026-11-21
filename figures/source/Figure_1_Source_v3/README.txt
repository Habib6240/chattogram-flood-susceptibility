Figure 1 source package (v3)
Changes from v2 (text only): internal stage codes removed from the workflow and replaced by descriptive labels; MCDA header note now reads "prescribed criteria priorities; no flood-label training" (set at 6.2 pt to fit its box); legend entry now reads "Coastal water / sea (DEM-derived)" (two lines). Map, river hierarchy, colours, layout, DEM backdrop, inset and all scientific content are unchanged.
Files
- make_figure1.py, geoutil.py : figure generation. Inputs (not included; edit UP/DATA in the script): AOI_Boundary.zip, BGD_adm0_National_Boundary.zip, BGD_adm2_District_Boundary.zip, rasters_COP30_tar.gz (output_hh.tif, Copernicus GLO-30), 05_River_HydroRIVERS_UPLAND25_UTM46N.gpkg. Upazila boundaries and roads are not drawn.
- analyze_rivers.py, Figure_1_River_Source_Comparison_v1.csv, Figure_1_River_Diagnostic_*.png : evaluation of BGD_water_lines_dcw (DIVA) against HydroRIVERS. Decision: HydroRIVERS only; DIVA not drawn.
- Figure_1_AOI_District_Overlap_v1.csv : the AOI lies 100% inside Chattogram (GADM "Chittagong") district.
- Figure_1_StudyArea_Workflow_v3.png (300 dpi) / .svg (editable text; terrain backdrop embedded as raster).
Rivers: HydroRIVERS principal = stream order >= 5 (dark blue), secondary = order 1-4 (light blue).
