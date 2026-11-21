# Zenodo form text: software record

Plain text for pasting into the Zenodo form; generated from `.zenodo.json` (which holds the same content, JSON-escaped: paste from here, not from the JSON, so quotation marks carry no backslashes).

**Resource type:** Software  
**Title:** Code for: Prescribed versus Learned Flood Susceptibility: MCDA and Machine Learning Under Event- and Geography-Withheld Evaluation in Chattogram, Bangladesh  
**Creator:** Habib, Md. Ahsan; ORCID 0009-0009-6932-0567; Department of Civil Engineering, Bangladesh University of Engineering and Technology (BUET), Dhaka-1000, Bangladesh  
**Version:** 1.0.0  
**Access:** open  
**Keywords:** flood susceptibility; Sentinel-1; AHP; fuzzy AHP; analytic network process; random forest; XGBoost; SHAP; Chattogram; Bangladesh  

**Description**

Scripts, derived summary data and documentation supporting the cited article, including the Distance_River predictor-reconstruction sensitivity scripts and reports (scripts/sensitivity, docs/sensitivity_reports). Upstream Sentinel-1 label-generation, GRASS/QGIS preprocessing and MCDA scripts are not included (see docs/MISSING_MATERIALS.md); independent end-to-end reproduction from raw data is not supported. Large data products are in the companion dataset record. Licence scope: MIT applies to source code only; supporting tables and documentation are under CC BY 4.0 (author-owned contributions only; third-party material excluded); see LICENSING.md. Not in this record: HydroRIVERS-derived Distance_River values (original values are in the separate access-restricted Zenodo record https://doi.org/10.5281/zenodo.24398548; independent regeneration scripts in scripts/regenerate_distance_river/) and OSM-derived Distance_Road values (ODbL sidecar in the dataset record); the ML scripts cannot be re-run from this record alone (see README). No blanket CC BY 4.0 applies to the mixed dataset. Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

**Related identifiers**

| Relation | Identifier | Scheme |
|---|---|---|
| isSupplementTo | https://github.com/Habib6240/chattogram-flood-susceptibility | url |
| isSupplementTo | 10.5281/zenodo.24397676 | doi |
| isSupplementTo | 10.5281/zenodo.24398548 | doi |

**Licences:** MIT (source code only); Creative Commons Attribution 4.0 International for the small author-owned supporting tables and documentation (third-party material excluded). No licence is claimed for third-party data; see `LICENSING.md`, `DATA_LICENSE.md` and `THIRD_PARTY_NOTICES.md`.
