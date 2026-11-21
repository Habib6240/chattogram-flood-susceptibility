"""Figure 3 — prescribed MCDA weights and rank structure (AHP, FAHP, ANP), v1.
Reads the four weight CSVs (source of truth), cross-checks them, then draws (a) grouped weights and (b) a rank slopegraph."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
UP='/mnt/user-data/uploads/'
ahp_raw=pd.read_csv(UP+'DU10_AHP_Weights_and_Consistency.csv',nrows=13).set_index('Criterion')
cons=pd.read_csv(UP+'DU10_AHP_Weights_and_Consistency.csv',skiprows=15); cons=dict(zip(cons.iloc[:,0],cons.iloc[:,1]))
fahp_raw=pd.read_csv(UP+'DU11_FAHP_Weights.csv').set_index('Criterion')
anp_raw=pd.read_csv(UP+'DU12_ANP_Final_Weights.csv',nrows=13).set_index('Criterion'); anp_meta=pd.read_csv(UP+'DU12_ANP_Final_Weights.csv',skiprows=15)
cmp_=pd.read_csv(UP+'DU12_AHP_FAHP_ANP_Weight_Comparison.csv').set_index('Criterion')
NAME={'Rainfall':'Rainfall','Elevation':'Elevation','Slope':'Slope','Drainage_Density':'Drainage density','Distance_River':'Distance to river','TWI':'TWI','LULC':'LULC','HSG':'HSG','NDVI':'NDVI','Distance_Road':'Distance to road','Profile_Curvature':'Profile curvature','Tangential_Curvature':'Tangential curvature','Aspect':'Aspect'}
W=pd.DataFrame({'AHP':ahp_raw.Weight,'FAHP':fahp_raw.Final_Normalized_Weight,'ANP':anp_raw.ANP_Weight}).loc[list(ahp_raw.index)]
assert len(W)==13 and set(W.index)==set(NAME)
# ---- numerical QA
chk={'sum_AHP':W.AHP.sum(),'sum_FAHP':W.FAHP.sum(),'sum_ANP':W.ANP.sum(),'CR':float(cons['CR']),'lambda_max':float(cons['Lambda_Max']),'CI':float(cons['CI']),'RI':float(cons['RI']),'status':cons['Consistency_Status']}
for k in ('AHP','FAHP','ANP'): assert abs(W[k].sum()-1)<1e-9,k
assert abs(chk['CR']-0.066610)<5e-7 and chk['status']=='PASS'
mx=max(np.abs(W.AHP-cmp_.AHP_Weight).max(),np.abs(W.FAHP-cmp_.FAHP_Weight).max(),np.abs(W.ANP-cmp_.ANP_Weight).max()); assert mx<1e-12,mx
R=W.rank(ascending=False,method='min').astype(int); assert all(R[k].is_unique for k in R)
exp_ahp=['Rainfall','Elevation','Slope','Drainage_Density','Distance_River','TWI','LULC','HSG','NDVI','Distance_Road','Profile_Curvature','Tangential_Curvature','Aspect']
exp_anp=['Drainage_Density','Elevation','Slope','Rainfall','TWI','Distance_River','LULC','NDVI','HSG','Profile_Curvature','Tangential_Curvature','Distance_Road','Aspect']
assert list(R.AHP.sort_values().index)==exp_ahp and list(R.FAHP.sort_values().index)==exp_ahp and list(R.ANP.sort_values().index)==exp_anp
rho={'AHP_vs_FAHP':spearmanr(W.AHP,W.FAHP)[0],'AHP_vs_ANP':spearmanr(W.AHP,W.ANP)[0],'FAHP_vs_ANP':spearmanr(W.FAHP,W.ANP)[0]}
print('sums',chk['sum_AHP'],chk['sum_FAHP'],chk['sum_ANP'],'| CR',chk['CR'],'| max |indiv - comparison| =',mx,'| rho',{k:round(v,6) for k,v in rho.items()})
# locked-value expectations (given precision)
EXP={'Rainfall':(0.231109,0.221284,0.134516),'Elevation':(0.1819,0.1811,0.1764),'Slope':(0.1415,0.1431,0.1390),'Drainage_Density':(0.1095,0.1113,0.2620),'Distance_River':(0.0847,0.0866,0.0868),'TWI':(0.0653,0.0668,0.1180),'LULC':(0.0503,0.0516,0.0276),'HSG':(0.0388,0.0398,0.0139),'NDVI':(0.0299,0.0306,0.0195),'Distance_Road':(0.0232,0.0237,0.0016),'Profile_Curvature':(0.0179,0.0183,0.0110),'Tangential_Curvature':(0.0142,0.0143,0.0083),'Aspect':(0.0116,0.0115,0.0013)}
rows=[]
for f,e in EXP.items():
    for m,v in zip(('AHP','FAHP','ANP'),e):
        dec=len(str(v).split('.')[1]); rows.append((NAME[f],m,v,W.loc[f,m],round(W.loc[f,m],dec),abs(round(W.loc[f,m],dec)-v)<1e-12))
qa=pd.DataFrame(rows,columns=['Factor','Method','Locked_value_as_given','CSV_value','CSV_rounded_to_given_precision','Match']); qa.to_csv('Figure_3_QA_LockedValue_Comparison_v1.csv',index=False); print('locked-value matches:',int(qa.Match.sum()),'of',len(qa))
assert qa.Match.all()
tbl=pd.DataFrame({'Factor':[NAME[i] for i in W.index],'AHP_weight':W.AHP.values,'FAHP_weight':W.FAHP.values,'ANP_weight':W.ANP.values,'AHP_rank':R.AHP.values,'FAHP_rank':R.FAHP.values,'ANP_rank':R.ANP.values,'ANP_minus_AHP_weight':(W.ANP-W.AHP).values,'ANP_minus_AHP_rank_change':(R.AHP-R.ANP).values}); tbl.to_csv('Figure_3_Weight_Rank_Table_v1.csv',index=False,float_format='%.12f')
pd.DataFrame([('sum_AHP',chk['sum_AHP']),('sum_FAHP',chk['sum_FAHP']),('sum_ANP',chk['sum_ANP']),('AHP_lambda_max',chk['lambda_max']),('AHP_CI',chk['CI']),('AHP_RI',chk['RI']),('AHP_CR',chk['CR']),('AHP_status',chk['status']),('max_abs_diff_individual_vs_comparison_CSV',mx),('Spearman_AHP_vs_FAHP',rho['AHP_vs_FAHP']),('Spearman_AHP_vs_ANP',rho['AHP_vs_ANP']),('Spearman_FAHP_vs_ANP',rho['FAHP_vs_ANP']),('ANP_limit_iterations',float(anp_meta.iloc[0,1]))],columns=['Check','Value']).to_csv('Figure_3_QA_Checks_v1.csv',index=False,float_format='%.12g')
# ---------------- figure
C={'AHP':'#1F3A5F','FAHP':'#6C9BCB','ANP':'#D08A2E'}
fig=plt.figure(figsize=(9.2,6.7),facecolor='white')
axA=fig.add_axes([0.165,0.115,0.325,0.795]); order=list(W.index)   # AHP-descending (CSV order is already AHP-descending)
y=np.arange(13); h=0.26
for k,(m,off) in enumerate((('AHP',-h),('FAHP',0),('ANP',h))):
    axA.barh(y+off,W.loc[order,m],height=h,color=C[m],label=m,zorder=3)
    for yi,v in zip(y+off,W.loc[order,m]): axA.text(v+0.003,yi,f'{v:.3f}',va='center',ha='left',fontsize=5.6,color='#333333')
axA.set_yticks(y); axA.set_yticklabels([NAME[i] for i in order],fontsize=8.2); axA.invert_yaxis(); axA.set_xlim(0,0.30); axA.set_xlabel('Normalized criterion weight',fontsize=8.5)
axA.xaxis.grid(True,color='#e3e3e3',lw=0.6,zorder=0); axA.set_axisbelow(True); axA.tick_params(axis='x',labelsize=7.8); axA.tick_params(axis='y',length=0)
for s in ('top','right','left'): axA.spines[s].set_visible(False)
axA.legend(loc='lower right',frameon=False,fontsize=8.2,handlelength=1.2,labelspacing=0.5)
fig.text(0.012,0.945,'(a)',fontsize=12,fontweight='bold',va='bottom'); fig.text(0.045,0.945,'Final prescribed weights',fontsize=11,fontweight='bold',va='bottom')
# panel b
axB=fig.add_axes([0.545,0.115,0.445,0.795]); axB.set_xlim(-2.3,4.3); axB.set_ylim(13.6,-0.7); axB.axis('off')
xs={'AHP':0.0,'FAHP':1.0,'ANP':2.0}
for m,x in xs.items(): axB.text(x,-0.25,m,ha='center',va='bottom',fontsize=9,fontweight='bold',color=C[m])
UPC,DNC,GR='#2A9D8F','#8E5A9B','#BDBDBD'
for f in order:
    ra,rf,rn=R.loc[f,'AHP'],R.loc[f,'FAHP'],R.loc[f,'ANP']; d=ra-rn
    col,lw,z=(UPC,1.9,4) if d>=2 else (DNC,1.9,4) if d<=-2 else (GR,1.0,2)
    axB.plot([0,1,2],[ra,rf,rn],color=col,lw=lw,zorder=z,solid_capstyle='round'); axB.scatter([0,1,2],[ra,rf,rn],s=14 if z==4 else 8,color=col,zorder=z+1,edgecolor='white',linewidth=0.4)
    fw='bold' if z==4 else 'normal'; tc='#111111' if z==4 else '#555555'
    axB.text(-0.09,ra,f'{NAME[f]}  {ra}',ha='right',va='center',fontsize=7.2,color=tc,fontweight=fw)
    axB.text(2.09,rn,f'{rn}  {NAME[f]}',ha='left',va='center',fontsize=7.2,color=tc,fontweight=fw)
fig.text(0.545,0.945,'(b)',fontsize=12,fontweight='bold',va='bottom'); fig.text(0.58,0.945,'Rank structure across MCDA methods',fontsize=11,fontweight='bold',va='bottom')
fig.text(0.545,0.074,'1 = highest weight. Teal / purple lines: rank rises / falls',fontsize=6.9,color='#555555',va='center',ha='left')
fig.text(0.545,0.058,'by at least two positions under ANP.',fontsize=6.9,color='#555555',va='center',ha='left')
fig.text(0.545,0.036,'AHP and FAHP retain nearly identical rankings;',fontsize=6.9,color='#555555',va='center',ha='left',style='italic')
fig.text(0.545,0.016,'ANP redistributes priority toward drainage density.',fontsize=6.9,color='#555555',va='center',ha='left',style='italic')
fig.savefig('Figure_3_MCDA_Weight_Rank_Structure_v1.png',dpi=300,facecolor='white'); fig.savefig('Figure_3_MCDA_Weight_Rank_Structure_v1.svg',facecolor='white'); plt.close(fig); print('saved')
