import numpy as np,pandas as pd,glob,json,openpyxl
from sklearn.metrics import roc_auc_score,average_precision_score
A=glob.glob('/mnt/user-data/attachments/*/')[0]
g=lambda p:glob.glob(A+p)[0]
wb=openpyxl.load_workbook(g('*1359f757-Supplementary_Tables_Final_v1.xlsx'),data_only=True)
def rows(s): return [r for r in wb[s].iter_rows(values_only=True) if any(c is not None for c in r)]
res=[]
def rec(tab,item,si,src,tol=5e-7):
    d=abs(float(si)-float(src)); res.append(dict(table=tab,item=item,SI=si,source=src,absdiff=d,ok=d<=tol))
# S9 vs B03U
b=pd.read_csv(g('*a7af73af-BUET_B03U_Flood_Mask_Summary.csv'))
for r in rows('S9'):
    if r[0] in ('2019','2021','2023','2026') and isinstance(r[1],float):
        s=b[b.event_id.str.contains(r[0])].iloc[0]
        for i,c in enumerate(['open_otsu_threshold_db','open_applied_threshold_db','urban_otsu_vv_increase_db','urban_applied_vv_increase_db','valid_label_area_km2','original_open_flood_area_km2','urban_newly_added_area_km2','combined_flood_area_km2','combined_flood_fraction_pct'],1):
            rec('S9 B',f'{r[0]} {c}',r[i],s[c],5e-6)
# S12 B1 vs B06U
f=pd.read_csv(g('*2695d56a-BUET_B06U_Fold_Performance.csv'))
for r in rows('S12'):
    if isinstance(r[0],int) and r[1] in('RF','XGBoost') and len(r)>8 and isinstance(r[2],int):
        m='Random_Forest' if r[1]=='RF' else 'XGBoost'
        s=f[(f.Spatial_Fold==r[0])&(f.Model==m)].iloc[0]
        for i,c in [(2,'Train_N'),(3,'Test_N'),(4,'Test_Flood_N'),(5,'Train_Block_N'),(6,'Test_Block_N'),(7,'ROC_AUC'),(8,'Average_Precision')]:
            rec('S12 B1',f'f{r[0]} {r[1]} {c}',r[i],s[c],5e-6)
cv=pd.read_csv(g('*1e74700e-BUET_B06U_Spatial_CV_Summary.csv'))
for r in rows('S12'):
    if r[0] in('RF','XGBoost') and isinstance(r[1],float):
        s=cv[cv.Model==('Random_Forest' if r[0]=='RF' else 'XGBoost')].iloc[0]
        for i,c in [(1,'Mean_Fold_AUC'),(2,'SD_Fold_AUC'),(3,'Min_Fold_AUC'),(4,'Max_Fold_AUC'),(5,'Mean_Fold_AP'),(6,'SD_Fold_AP'),(7,'Pooled_OOF_AUC'),(8,'Pooled_OOF_AP')]:
            rec('S12 B2',f'{r[0]} {c}',r[i],s[c],5e-6)
# S11 C vs B06U
for r in rows('S11'):
    if isinstance(r[0],int) and len(r)>6 and isinstance(r[1],int) and r[0]<=5:
        s=f[(f.Spatial_Fold==r[0])&(f.Model=='XGBoost')].iloc[0]
        for i,c in [(1,'Train_N'),(2,'Test_N'),(3,'Test_Flood_N'),(4,'Test_Flood_Percent'),(5,'Train_Block_N'),(6,'Test_Block_N')]:
            rec('S11 C',f'f{r[0]} {c}',r[i],s[c],5e-6)
# S13 B from OOF
z=np.load(g('*c2832f3a-BUET_B07U_LOEO_OutOfFold_Predictions.npz'))
fold=z['Spatial_Fold']; blk=z['Block_5km']
for r in rows('S13'):
    if r[0] in(2019,2021,2023,2026) and r[2] in('RF','XGBoost') and len(r)>9 and isinstance(r[1],int):
        y=z[f'Observed_{r[0]}']; ok=(y>=0)&(fold==r[1])
        p=z[f'{"RF" if r[2]=="RF" else "XGB"}_{r[0]}'][ok]
        rec('S13 B',f'{r[0]} f{r[1]} {r[2]} testrows',r[5],ok.sum(),0)
        rec('S13 B',f'{r[0]} f{r[1]} {r[2]} flooded',r[6],int((y[ok]==1).sum()),0)
        rec('S13 B',f'{r[0]} f{r[1]} {r[2]} AUC',r[7],roc_auc_score(y[ok],p),5e-6)
        rec('S13 B',f'{r[0]} f{r[1]} {r[2]} AP',r[8],average_precision_score(y[ok],p),5e-6)
        trb=len(np.unique(blk[(y>=0)&(fold!=r[1])])); teb=len(np.unique(blk[ok]))
        rec('S13 B',f'{r[0]} f{r[1]} {r[2]} testblocks',r[4],teb,0)
# S14 vs B09B
pc=pd.read_csv(g('*d2eacf43-BUET_B09B_FullDomain_Pairwise_Correlation.csv')).set_index(['Method_1','Method_2'])
pe=pd.read_csv(g('*8a68f166-BUET_B09B_Percentile5Class_Agreement.csv')).set_index(['Method_1','Method_2'])
tj=pd.read_csv(g('*e43f1cb4-BUET_B09B_Top20_Susceptibility_Overlap.csv')).set_index(['Method_1','Method_2'])
nm={'RF':'RF','XGBoost':'XGBoost','ML consensus':'ML_Consensus','AHP':'AHP','FAHP':'FAHP','ANP':'ANP'}
for r in rows('S14'):
    if isinstance(r[0],str) and ' - ' in r[0] and isinstance(r[2],float):
        a,b2=[nm[x.strip()] for x in r[0].split(' - ')]; k=(a,b2)
        rec('S14',r[0]+' Pearson',r[2],pc.loc[k,'Pearson_r'],5e-7)
        rec('S14',r[0]+' Spearman',r[3],pc.loc[k,'Spearman_rho'],5e-7)
        rec('S14',r[0]+' exact%',r[4],pe.loc[k,'Exact_5Class_Agreement']*100,5e-5)
        rec('S14',r[0]+' within1%',r[5],pe.loc[k,'Within_One_Class_Agreement']*100,5e-5)
        rec('S14',r[0]+' Jaccard',r[6],tj.loc[k,'Jaccard'],5e-7)
        rec('S14',r[0]+' inter',r[7],tj.loc[k,'Top20_Intersection_Pixels'],0)
        rec('S14',r[0]+' union',r[8],tj.loc[k,'Top20_Union_Pixels'],0)
df=pd.DataFrame(res); df.to_csv('/home/claude/m1/out/table_verification_detail.csv',index=False)
print(df.groupby('table').ok.agg(['count','sum']))
print(df[~df.ok].to_string())
