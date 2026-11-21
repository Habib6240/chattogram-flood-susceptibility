import glob,numpy as np,pandas as pd,openpyxl,json,hashlib
from PIL import Image
from sklearn.metrics import roc_auc_score
Image.MAX_IMAGE_PIXELS=None
A=glob.glob('/mnt/user-data/attachments/*/')[0];g=lambda p:glob.glob(A+p)[0]
res=[]
def rec(t,i,a,b,tol=5e-5):
    res.append(dict(table=t,item=i,SI_or_claim=a,source=b,absdiff=abs(float(a)-float(b)),ok=abs(float(a)-float(b))<=tol))
# six-method vs S14
six=pd.read_csv('m1/out/six_method_recomputed.csv')
wb=openpyxl.load_workbook(g('*1359f757*'),data_only=True)
nm={'RF':'RF','XGBoost':'XGBoost','ML consensus':'ML_Consensus','AHP':'AHP','FAHP':'FAHP','ANP':'ANP'}
for r in wb['S14'].iter_rows(values_only=True):
    if isinstance(r[0],str) and ' - ' in r[0] and isinstance(r[2],float):
        a,b=[nm[x.strip()] for x in r[0].split(' - ')]
        s=six[((six.m1==a)&(six.m2==b))|((six.m1==b)&(six.m2==a))].iloc[0]
        for i,c,tol in [(2,'pearson',5e-7),(3,'spearman',5e-7),(4,'exact',5e-6),(5,'within1',5e-6),(6,'jaccard',5e-7),(7,'inter',0),(8,'union',0)]:
            rec('S14 vs rasters',r[0]+' '+c,r[i],s[c],tol)
# DU14 sample
s=pd.read_csv(g('*fb2ce133*'));y=s.Observed_2021_Flood_Label
base={}
for m in('AHP','FAHP','ANP'):
    base[m]=roc_auc_score(y,s[m+'_Score']);
d15=pd.read_csv(g('*089ee274*')).set_index('Metric').Value
for m in base: rec('DU15 baseline AUC',m,float(d15[m+'_AUC_300m_Representative']),base[m],1e-9)
rec('DU15 sample','N',len(s),float(d15['Sample_N']),0);rec('DU15 sample','Flood',y.sum(),float(d15['Flood_N']),0)
# scores vs rasters and label vs B03U 2021 mask
rd=lambda p:np.asarray(Image.open(g(p))).astype('float64')
ras={'AHP':rd('*8456c989*'),'FAHP':rd('*a676a929*'),'ANP':rd('*5a5b1b64*')}
r,c=s.Raster_Row.values,s.Raster_Column.values
for m in ras: rec('DU14 sample vs raster',m+' max score diff',np.abs(ras[m][r,c]-s[m+'_Score']).max(),0,1e-6)
fm=np.asarray(Image.open(g('*357bffb5*')))
lab=fm[r,c]
res.append(dict(table='DU14 sample vs B03U 2021',item='flood labels not flooded in B03U (must be 0)',SI_or_claim=int(((y==1)&(lab!=1)).sum()),source=0,absdiff=int(((y==1)&(lab!=1)).sum()),ok=int(((y==1)&(lab!=1)).sum())==0))
res.append(dict(table='DU14 sample vs B03U 2021',item='sample cells with invalid B03U label',SI_or_claim=int((lab==255).sum()),source=0,absdiff=int((lab==255).sum()),ok=int((lab==255).sum())==0))
# DU14 offsets
o=pd.read_csv(g('*e67cac5c*'));sm=pd.read_csv(g('*3686b499*')).dropna(how='all').set_index('Method')
for m in('AHP','FAHP','ANP'):
    x=o[m+'_AUC']
    for k,v in [('Mean_AUC',x.mean()),('SD_AUC',x.std(ddof=1)),('Minimum_AUC',x.min()),('Median_AUC',x.median()),('Maximum_AUC',x.max())]: rec('DU14 offsets recompute',m+' '+k,sm.loc[m,k],v,1e-12)
    rec('DU14 offsets',m+' N offsets',100,len(x),0)
rec('DU14 offsets','offset(5,5) AUC vs DU15 baseline AHP',o[(o.Row_Offset==5)&(o.Column_Offset==5)].AHP_AUC.iloc[0],d15['AHP_AUC_300m_Representative'],1e-12)
pp=pd.read_csv(g('*b61588d6*'))
for _,q in pp.iterrows():
    a,b=q.Method_1,q.Method_2;dd=o[a+'_AUC']-o[b+'_AUC']
    rec('DU15 paired recompute',f'{a}-{b} mean',q.Mean_Paired_AUC_Difference,dd.mean(),1e-12)
    rec('DU15 paired recompute',f'{a}-{b} P2.5',q.P02_5_Paired_Difference,np.percentile(dd,2.5),1e-12)
    rec('DU15 paired recompute',f'{a}-{b} P97.5',q.P97_5_Paired_Difference,np.percentile(dd,97.5),1e-12)
    rec('DU15 paired recompute',f'{a}-{b} win%',q.Method1_Win_Percent,(dd>0).mean()*100,1e-9)
# DU14 validation summary vs S5 & sample counts & score stats
v=pd.read_csv(g('*6c7d220a*')).set_index('Method')
for r_ in wb['S5'].iter_rows(values_only=True):
    if r_[0] in('AHP','FAHP','ANP') and isinstance(r_[1],float):
        q=v.loc[r_[0]]
        for i,cn in [(1,'ROC_AUC'),(2,'Youden_Optimal_Threshold'),(3,'Sensitivity'),(4,'Specificity'),(5,'Precision'),(6,'F1'),(7,'Balanced_Accuracy'),(8,'TP'),(9,'TN'),(10,'FP'),(11,'FN')]: rec('S5 vs DU14 summary',r_[0]+' '+cn,r_[i],q[cn],5e-6 if i<8 else 0)
for m in v.index:
    q=v.loc[m];res.append(dict(table='DU14 internal',item=m+' TP+TN+FP+FN',SI_or_claim=int(q.TP+q.TN+q.FP+q.FN),source=4251529,absdiff=abs(int(q.TP+q.TN+q.FP+q.FN)-4251529),ok=int(q.TP+q.TN+q.FP+q.FN)==4251529))
    rec('DU14 internal',m+' sens=TP/(TP+FN)',q.Sensitivity,q.TP/(q.TP+q.FN),1e-12)
# S6 vs csv
for r_ in wb['S6'].iter_rows(values_only=True):
    if r_[0] in('AHP','FAHP','ANP') and r_[1]==100:
        q=sm.loc[r_[0]]
        for i,cn in [(2,'Mean_AUC'),(3,'SD_AUC'),(4,'Minimum_AUC'),(5,'Median_AUC'),(6,'Maximum_AUC')]: rec('S6 vs DU14 summary',r_[0]+' '+cn,r_[i],q[cn],5e-6)
    if isinstance(r_[0],str) and ' - ' in r_[0] and isinstance(r_[1],float):
        a,b=r_[0].split(' - ');q=pp[(pp.Method_1==a)&(pp.Method_2==b)].iloc[0];rec('S6 B vs DU15',r_[0]+' mean',r_[1],q.Mean_Paired_AUC_Difference,5e-6)
# S7/S8 vs DU15
wp=pd.read_csv(g('*c5411c13*'));lo=pd.read_csv(g('*bec36e04*'));lr=pd.read_csv(g('*7cbd4b92*'))
fmap=lambda x:x.replace(' ','_').replace('Distance_to_river','Distance_River').replace('Distance_to_road','Distance_Road').replace('Drainage_density','Drainage_Density').replace('Profile_curvature','Profile_Curvature').replace('Tangential_curvature','Tangential_Curvature')
def f(x):
    d={'Drainage density':'Drainage_Density','Distance to river':'Distance_River','Distance to road':'Distance_Road','Profile curvature':'Profile_Curvature','Tangential curvature':'Tangential_Curvature'};return d.get(x,x)
sec=None
for r_ in wb['S7'].iter_rows(values_only=True):
    if r_[0] in('AHP','FAHP','ANP') and r_[2] in('-20%','+20%'):
        q=wp[(wp.Method==r_[0])&(wp.Perturbed_Criterion==f(r_[1]))&(wp.Perturbation==r_[2])].iloc[0]
        for i,cn in [(3,'Original_Weight'),(4,'Perturbed_Normalized_Weight'),(5,'Baseline_AUC'),(6,'Scenario_AUC'),(7,'Delta_AUC')]: rec('S7 A vs DU15',f'{r_[0]} {r_[1]} {r_[2]} {cn}',r_[i],q[cn],5e-6)
    elif r_[0] in('AHP','FAHP','ANP') and isinstance(r_[2],float) and r_[1] not in('-20%','+20%') and len(r_)>5 and r_[5] is not None and r_[3] is not None and isinstance(r_[3],float) and r_[5] is not None and (r_[4] is not None):
        q=lo[(lo.Method==r_[0])&(lo.Removed_Criterion==f(r_[1]))]
        if len(q):
            q=q.iloc[0]
            for i,cn in [(2,'Removed_Original_Weight'),(3,'Baseline_AUC'),(4,'LeaveOneOut_AUC'),(5,'Delta_AUC')]: rec('S7 B vs DU15',f'{r_[0]} {r_[1]} {cn}',r_[i],q[cn],5e-6)
for r_ in wb['S8'].iter_rows(values_only=True):
    if isinstance(r_[2],float) and isinstance(r_[5],int) and r_[6]=='PASS':
        q=lr[(lr.Source_Criterion==f(r_[0]))&(lr.Removed_Link_Target==f(r_[1]))].iloc[0]
        for i,cn,t in [(2,'Scenario_AUC',5e-6),(3,'Delta_AUC',5e-6),(4,'L1_Weight_Vector_Shift',5e-6),(5,'Limit_Iterations',0)]: rec('S8 vs DU15',f'{r_[0]}->{r_[1]} {cn}',r_[i],q[cn],t)
rec('S8 count','links',57,len(lr),0);rec('S4 convergence','max final diff 9.956e-13',9.956e-13,lr.Final_Difference.max(),5e-16);rec('S4','max col spread 7.763e-13',7.763e-13,lr.Limit_Column_Spread.max(),5e-16);rec('S4','min iters',32,lr.Limit_Iterations.min(),0);rec('S4','max iters',63,lr.Limit_Iterations.max(),0)
df=pd.DataFrame(res);df.to_csv('m1/out/du_six_verification_detail.csv',index=False)
print(df.groupby('table').ok.agg(['count','sum']).to_string());print(df[~df.ok].to_string())
# duplicates md5
import collections
h=collections.defaultdict(list)
for fpath in glob.glob(A+'*'):
    import os
    if os.path.isfile(fpath) and os.path.getsize(fpath)>1000: h[hashlib.md5(open(fpath,'rb').read()).hexdigest()].append(fpath.split('/')[-1])
dup=[v for v in h.values() if len(v)>1];json.dump(dup,open('m1/out/dup_groups.json','w'),indent=0)
print(len(dup),'duplicate groups');[print(d) for d in dup if any(('DU1' in x or 'B09' in x or 'B05' in x or 'B03U' in x or 'Mask' in x or 'B07U' in x or 'B08V' in x or 'B06U' in x) for x in d)]
