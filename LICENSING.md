# Licensing scope (single authoritative statement)

| Component | Terms | Status |
|---|---|---|
| Source code: `scripts/`, `environment/` helper files | MIT (`LICENSE`) | Applies to software only |
| Author-generated documentation and compact data tables in this repository (`README.md`, `docs/`, `data/`, `supplement/`, `figures/source/`, summary CSVs) | CC BY 4.0, for the author-owned eligible contributions | Author-approved; subject to third-party exclusions |
| Third-party inputs (Sentinel-1/2, CHIRPS, WorldCover, GHSL, GLO-30, SRTM, HydroRIVERS, JRC GSW, HYSOGs250m, OpenStreetMap, GADM) | Each provider's own terms (`Source_Data_Manifest.csv`) | Not relicensed here; GADM, GLO-30 tiles, HydroRIVERS and the OSM extract are not redistributed |
| OpenStreetMap-derived `Distance_Road` values | ODbL 1.0 | Not in this repository; separate sidecar in the Zenodo dataset record |
| HydroRIVERS-derived `Distance_River` values | HydroSHEDS-compatible EULA (author-drafted, not WWF-issued; needs legal review) | **Not in this public repository.** Provided only in the separate restricted Zenodo record https://doi.org/10.5281/zenodo.24398548, access after a signed EULA |
| Values derived from Copernicus GLO-30 and Sentinel data | Copernicus licences, mandatory notices | `THIRD_PARTY_NOTICES.md` |

No blanket CC BY 4.0 applies to the mixed dataset: MIT for code, CC BY 4.0 for author-owned eligible contributions, ODbL 1.0 for the OSM component, the EULA for the HydroSHEDS-derived component, provider terms for third-party material.

The MIT licence does not apply to data files. The CC BY 4.0 proposal does not apply to code or to third-party material.

**Rights confirmation (author).** The author has confirmed (author's own statement, not independently verified) ownership of the eligible original contributions and approved CC BY 4.0 for them.

**Boundary data.** No administrative-boundary file is distributed. Anyone needing it must download GADM (BGD_adm2, version 2 schema, 64 districts) from gadm.org under the GADM licence.

**Not legal advice.** The ODbL treatment of the OpenStreetMap-derived values is a precautionary arrangement, not a legal determination; see `DATA_LICENSE.md`.

**Zenodo software record.** `.zenodo.json` sets MIT (the single-licence field). The record also contains small supporting tables (CC BY 4.0, author-approved): add CC-BY-4.0 as a second licence in the Zenodo form.
