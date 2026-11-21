# Regenerate Distance_River from the official HydroRIVERS download (optional independent route)
The original values are in the separate restricted record (signed EULA); regenerated values are not identical to them.
MIT (code only). Inputs/outputs derived from HydroRIVERS are subject to the HydroSHEDS v1 licence and must not be redistributed by you without checking it.
Run in order: `1_prepare_hydrorivers_utm46n.py` (geopandas) -> `2_compute_distance_river.py` (numpy) -> `3_verify_distance_river.py` -> `../reassemble_restricted_fields.py`.
Full instructions, validation results, tolerances and limits: dataset record `10_metadata/Distance_River_Regeneration.md`.
Tested: steps 2, 3 and the rebuild (author's pre-processed GeoPackage). Not tested: step 1 (official shapefile unavailable in the authoring session).
`riverdist.py` is a helper (GeoPackage line reader + exact point-to-segment distance).
