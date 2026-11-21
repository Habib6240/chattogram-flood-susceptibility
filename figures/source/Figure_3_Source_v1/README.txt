Figure 3 source package (v1) — prescribed MCDA weights and rank structure
Files
- make_figure3.py : reads the four weight CSVs (source of truth), runs the numerical QA and draws the figure.
- Figure_3_Weight_Rank_Table_v1.csv : cleaned table (full-precision weights and ranks for AHP, FAHP, ANP; rank changes).
- Figure_3_QA_Checks_v1.csv : weight sums, AHP consistency statistics, individual-versus-comparison-file difference, Spearman correlations.
- Figure_3_QA_LockedValue_Comparison_v1.csv : the 39 supplied expected values versus the CSV values rounded to the same precision (39 of 39 match).
- Figure_3_MCDA_Weight_Rank_Structure_v1.png (300 dpi) / .svg.
Inputs (not included; edit UP in the script): DU10_AHP_Weights_and_Consistency.csv, DU11_FAHP_Weights.csv, DU12_ANP_Final_Weights.csv, DU12_AHP_FAHP_ANP_Weight_Comparison.csv.
Display: bar labels in panel (a) are rounded to 3 decimals; all computations use full precision. Ranks: 1 = highest weight; factors in panel (a) are ordered by AHP weight for all three methods.
