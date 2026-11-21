#!/usr/bin/env python3
"""Compare a RERUN of B08U with the outputs behind Figure 8, and verify full 20,000-row SHAP additivity.
Usage: python3 compare_b08u_rerun.py <ORIGINAL_dir> <RERUN_dir> <out_dir> [Figure_8_Numerical_Audit.csv]
Each dir holds the four B08U CSVs (Global_13Factor_Importance, Encoded_Feature_Importance, Explanation_Sample, SHAP_QC), file names as in B08U.
Pass ORIGINAL_dir == RERUN_dir to run the additivity/consistency self-check on the stored outputs only.
Tolerances below are PROPOSED and must be approved by the author BEFORE the rerun is opened."""
import sys,os,json,numpy as np,pandas as pd
O,R,OUT=sys.argv[1:4]; AUD=sys.argv[4] if len(sys.argv)>4 else None; os.makedirs(OUT,exist_ok=True)
G='BUET_B08U_XGB_SHAP_Global_13Factor_Importance.csv'; E='BUET_B08U_XGB_SHAP_Encoded_Feature_Importance.csv'; S='BUET_B08U_XGB_SHAP_Explanation_Sample.csv'; Q='BUET_B08U_SHAP_QC.csv'
TA=dict(row_shap=1e-4,mean_abs_rel=1e-4,share_pp=1e-3); TB=dict(share_pp=0.05)   # Tier A (same stack) / Tier B (different xgboost version or hardware)
SCRIPT_LIMIT=dict(margin=1e-4,prob=1e-5)
res={}; rows=[]; fails=[]
def rec(name,val,limit,ok): rows.append((name,val,limit,'PASS' if ok else 'FAIL')); (None if ok else fails.append(name))
o=pd.read_csv(f'{O}/{G}').set_index('Factor'); r=pd.read_csv(f'{R}/{G}').set_index('Factor'); assert set(o.index)==set(r.index)==set(o.index) and len(o)==13
d_share=(o.Relative_Importance_Percent-r.Relative_Importance_Percent.reindex(o.index)).abs()
d_rel=((o.Mean_Absolute_Grouped_SHAP-r.Mean_Absolute_Grouped_SHAP.reindex(o.index)).abs()/o.Mean_Absolute_Grouped_SHAP)
rk_o=o.Relative_Importance_Percent.rank(ascending=False).astype(int); rk_r=r.Relative_Importance_Percent.reindex(o.index).rank(ascending=False).astype(int)
lab_eq=(o.Relative_Importance_Percent.round(2)==r.Relative_Importance_Percent.reindex(o.index).round(2)).all()
ranks_eq=(rk_o==rk_r).all(); top9_hsg_eq=((rk_o<=9)|(rk_o==13)).pipe(lambda m:(rk_o[m]==rk_r[m]).all())
res.update(max_share_diff_pp=float(d_share.max()),max_meanabs_rel_diff=float(d_rel.max()),ranks_identical=bool(ranks_eq),two_decimal_labels_identical=bool(lab_eq))
pd.DataFrame({'orig_share':o.Relative_Importance_Percent,'rerun_share':r.Relative_Importance_Percent.reindex(o.index),'diff_pp':d_share,'rel_diff_meanabs':d_rel,'rank_orig':rk_o,'rank_rerun':rk_r}).to_csv(f'{OUT}/grouped_comparison.csv',float_format='%.9g')
eo=pd.read_csv(f'{O}/{E}').set_index('Encoded_Feature'); er=pd.read_csv(f'{R}/{E}').set_index('Encoded_Feature')
res['encoded_max_abs_diff']=float((eo.Mean_Absolute_SHAP-er.Mean_Absolute_SHAP.reindex(eo.index)).abs().max()); assert len(eo)==len(er)==25
so=pd.read_csv(f'{O}/{S}'); sr=pd.read_csv(f'{R}/{S}'); assert len(sr)==20000
same_rows=np.array_equal(so[['X','Y']].values,sr[['X','Y']].values); res['same_rows_same_order']=bool(same_rows)
fac=[c[:-5] for c in sr.columns if c.endswith('_SHAP') and c!='SHAP_Bias']; assert len(fac)==13
res['max_row_shap_diff']=float(max((so[f+'_SHAP']-sr[f+'_SHAP']).abs().max() for f in fac)) if same_rows else None
res['max_row_bias_diff']=float((so.SHAP_Bias-sr.SHAP_Bias).abs().max()) if same_rows else None
# full 20,000-row additivity on the RERUN sample (float64, from stored columns)
tot=sr[[f+'_SHAP' for f in fac]].sum(axis=1)+sr.SHAP_Bias; p=sr.Predicted_Probability.clip(1e-12,1-1e-12)
res['additivity_max_margin_err']=float((tot-np.log(p/(1-p))).abs().max()); res['additivity_max_prob_err']=float((1/(1+np.exp(-tot))-sr.Predicted_Probability).abs().max()); res['additivity_rows']=len(sr)
qr=dict(pd.read_csv(f'{R}/{Q}').values); res['qc_reported_margin_err']=float(qr['Maximum_Margin_Reconstruction_Error']); res['qc_reported_prob_err']=float(qr['Maximum_Probability_Reconstruction_Error']); res['qc_status']=qr['Additivity_Status']
res['sample_per_fold']=sr.Spatial_Fold.value_counts().sort_index().to_dict()
if AUD:
    a=pd.read_csv(AUD).set_index('Factor_key'); res['figure_labels_equal_rerun_2dp']=bool((a.SHAP_share_pct_grouped_csv.round(2)==r.Relative_Importance_Percent.reindex(a.index).round(2)).all())
# checks
rec('A1 additivity margin (20,000 rows)',res['additivity_max_margin_err'],SCRIPT_LIMIT['margin'],res['additivity_max_margin_err']<=SCRIPT_LIMIT['margin'])
rec('A2 additivity probability (20,000 rows)',res['additivity_max_prob_err'],SCRIPT_LIMIT['prob'],res['additivity_max_prob_err']<=SCRIPT_LIMIT['prob'])
rec('A3 QC file additivity status PASS',res['qc_status'],'PASS',res['qc_status']=='PASS')
rec('A4 4,000 rows per fold, 5 folds',str(res['sample_per_fold']),'4000 x 5',sorted(res['sample_per_fold'].values())==[4000]*5)
rec('A5 same 20,000 rows in same order',res['same_rows_same_order'],True,same_rows)
tierA=all(x for x in [same_rows, res['max_row_shap_diff'] is not None and res['max_row_shap_diff']<=TA['row_shap'], res['max_meanabs_rel_diff']<=TA['mean_abs_rel'], res['max_share_diff_pp']<=TA['share_pp'], ranks_eq, lab_eq])
tierB=(res['max_share_diff_pp']<=TB['share_pp']) and top9_hsg_eq
rec('T-A row-level max |dSHAP|',res['max_row_shap_diff'],TA['row_shap'],same_rows and res['max_row_shap_diff'] is not None and res['max_row_shap_diff']<=TA['row_shap'])
rec('T-A grouped mean|SHAP| max relative diff',res['max_meanabs_rel_diff'],TA['mean_abs_rel'],res['max_meanabs_rel_diff']<=TA['mean_abs_rel'])
rec('T-A share diff (pp)',res['max_share_diff_pp'],TA['share_pp'],res['max_share_diff_pp']<=TA['share_pp'])
rec('T-A all 13 ranks identical',res['ranks_identical'],True,ranks_eq); rec('T-A all 13 two-decimal labels identical',lab_eq,True,lab_eq)
add_ok=not any(f.startswith('A') for f in fails)
verdict='FAIL (additivity / sampling / structure)' if not add_ok else ('TIER A PASS: figure unchanged' if tierA else ('TIER B REVIEW: within 0.05 pp and top-9/HSG ranks stable, but rerun differs from figure labels or ranks 10-12; author decision, figure would need re-render' if tierB else 'FAIL: outside Tier B tolerance'))
res['verdict']=verdict; pd.DataFrame(rows,columns=['check','value','limit','result']).to_csv(f'{OUT}/rerun_checks.csv',index=False)
json.dump(res,open(f'{OUT}/rerun_summary.json','w'),indent=1,default=str); print(json.dumps(res,indent=1,default=str)); print('VERDICT:',verdict)
