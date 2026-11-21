Figure 8 (Q1 final) source package: prescribed MCDA weights versus grouped XGBoost SHAP attribution
=====================================================================================================
Reproduce:  python3 make_figure8_q1.py data output     (needs Python 3, numpy, pandas, matplotlib, Liberation Sans font)
QC:         python3 qc_figure8.py data output ; python3 qc_render_cvd8.py data output (needs Pillow, scikit-image, playwright+Chromium, poppler-utils)
            python3 crosscheck8.py data output <Supplementary_Tables_Final_v1.xlsx> <manuscript.docx>  (optional; needs the locked supplement and manuscript files, which are not included)
Every plotted number is read from data/*.csv; nothing is typed into the script.

data/
  DU10_AHP_Weights_and_Consistency.csv, DU11_FAHP_Weights.csv, DU12_ANP_Final_Weights.csv   primary MCDA weights (13 each)
  BUET_B08U_XGB_SHAP_Global_13Factor_Importance.csv    primary grouped SHAP output (13 factors)
  BUET_B08U_XGB_SHAP_Encoded_Feature_Importance.csv    primary encoded (25-column) SHAP output
  BUET_B08U_SHAP_QC.csv                                primary SHAP quality-control record
  BUET_B08U_XGB_SHAP_Explanation_Sample.csv            primary 20,000-row explanation sample (row-level grouped SHAP); used to recompute the shares. NOTE (licensing correction): the Distance_River_Value (HydroSHEDS-derived, restricted record) and Distance_Road_Value (OSM-derived, ODbL sidecar) columns were removed from this copy; the figure code reads only the *_SHAP, Predicted_Probability and SHAP_Bias columns, which are unchanged
  B08U_BUET_XGBoost_TreeSHAP_UrbanAware.py             script that produced the SHAP outputs (documentation only; not run here)
output/
  Figure_8_Weights_SHAP_Q1_FINAL.png (600 dpi) / .svg (live text) / .pdf (vector, embedded fonts)
  Figure_8_Numerical_Audit.csv                  all plotted values, ranks and the SHAP recomputation difference
  Figure_8_Source_to_Value_Traceability.csv     every plotted value -> source file and column
  Figure_8_Aggregation_Sensitivity.csv          row-wise grouping (plotted) versus sum of encoded columns (not plotted)
  Figure_8_CrossCheck_Supplement_Manuscript.csv comparison with Tables S3/S16 and manuscript v3 Results 3.6
  Figure_8_QC_Report.txt, qc_results.json, qc_render_cvd.json
Figure_8_Caption_Recommended.txt

Provenance status
- MCDA weights: primary CSVs, 39 of 39 values and ranks verified; each method sums to 1.
- SHAP shares: primary B08U grouped CSV supplied; the 13 shares were independently recomputed from the 20,000-row explanation sample (max difference 1.6e-6 percentage points). Sample additivity holds.
- NOT performed: re-execution of the XGBoost fit and TreeSHAP itself (xgboost is not installable in the audit environment; the fitted model and the B04U input are not part of this package). The sample CSV and global CSV are therefore taken as the outputs of the B08U script.
Interpretation limits
- MCDA weights are analyst-specified priorities. SHAP shares come from a full-data explanatory XGBoost model (composite any-flood target), not from the strict event- and geography-withheld evaluation, and are model-specific and associational, not causal.
- Shares are estimates from a 20,000-row sample (4,000 per spatial fold); no uncertainty interval was computed.
- Weights and SHAP shares are different quantities; panel (c) compares ordering only. No inferential comparison is made.
