# Data terms (repository)

See `LICENSING.md` for the scope table and `THIRD_PARTY_NOTICES.md` for required attributions.

1. **Author-generated tables and documentation:** CC BY 4.0 for the author-owned eligible contributions (author-approved), subject to the third-party exclusions below (see `LICENSING.md`).
2. **Third-party data:** each provider's terms apply; `Source_Data_Manifest.csv` records, per source, what is redistributed, the evidence for the terms, required attribution and unresolved items. Values derived from Copernicus GLO-30 (in the companion dataset record) carry the mandatory notice in `THIRD_PARTY_NOTICES.md`.
3. **OpenStreetMap-derived values** are not in this repository; in the dataset record they are in a separate ODbL 1.0 sidecar. `scripts/reassemble_restricted_fields.py` rejoins them.
4. **HydroRIVERS-derived `Distance_River` values** are not in this repository. The complete original values are in the separate restricted Zenodo record https://doi.org/10.5281/zenodo.24398548, accessible after a signed End User Licence Agreement (draft: `docs/EULA_HydroSHEDS_Distance_River_DRAFT_v2.docx`; author-drafted, not reviewed by a lawyer, not issued or approved by WWF). Whether independently computed distances are Derivative Works or contain Licensed Materials is not settled by the HydroSHEDS v1 licence text (Appendix A ss.1.1, 2.1.1-2.1.3), so access is gated as a precaution; no legal conclusion is drawn. An independent regeneration workflow is in `scripts/regenerate_distance_river/`; regenerated values are not identical to the originals.
5. **GADM:** no GADM-derived geometry is distributed.
6. **Software:** MIT, code only.

**Rights confirmation (author).** The author has confirmed (author's own statement, not independently verified) ownership of the eligible original contributions and approved CC BY 4.0 for them. CC BY 4.0 applies to author-owned eligible contributions only, subject to third-party exclusions; it does not apply to code (MIT) or to third-party material.

**Not legal advice.** Licence readings here are good-faith summaries of the quoted provider texts. No legal determination was made.
