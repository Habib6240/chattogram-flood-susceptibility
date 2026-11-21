# Independent recomputation scripts (archival)

These scripts were written during the 2026 pre-submission audit to recompute AHP/FAHP/ANP weights, Equations 1 and 11 (lattice reproduction), the six-method spatial agreement, SI/manuscript table values and the SHAP aggregation from the archived outputs. They were run in the review environment, where input files carried upload-hash prefixes and were located by `glob` patterns under `/mnt/user-data/`.

They are provided as an audit trail, not as turnkey tools: the input-path constants at the top of each script must be edited to the local data layout before use. Results are in `data/verification/independent_recomputation/` and are summarised in the Supplementary Information (Table S18).
