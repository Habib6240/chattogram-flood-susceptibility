#!/usr/bin/env python3
"""Figure 8 (Q1 final): prescribed MCDA weights (AHP, FAHP, ANP) versus grouped XGBoost SHAP attribution.
All plotted numbers are read from the primary CSVs in ./data (no typed values). Usage: python3 make_figure8_q1.py [data_dir] [out_dir]"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE=os.path.dirname(os.path.abspath(__file__))
D=sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE,'data')
O=sys.argv[2] if len(sys.argv)>2 else os.path.join(HERE,'output')
os.makedirs(O,exist_ok=True)
plt.rcParams.update({'font.family':'Liberation Sans','svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'font.size':7})
KEYS=['Rainfall','Elevation','Slope','Drainage_Density','Distance_River','TWI','LULC','HSG','NDVI','Distance_Road','Profile_Curvature','Tangential_Curvature','Aspect']
NICE={'Distance_Road':'Distance to road','Distance_River':'Distance to river','Tangential_Curvature':'Tangential curvature','Drainage_Density':'Drainage density','Profile_Curvature':'Profile curvature'}
lab=lambda k:NICE.get(k,k)
# ---------- read primary CSVs
ahp=pd.read_csv(f'{D}/DU10_AHP_Weights_and_Consistency.csv',nrows=13).set_index('Criterion').Weight
fahp=pd.read_csv(f'{D}/DU11_FAHP_Weights.csv').set_index('Criterion').Final_Normalized_Weight
anp=pd.read_csv(f'{D}/DU12_ANP_Final_Weights.csv',nrows=13).set_index('Criterion').ANP_Weight
g=pd.read_csv(f'{D}/BUET_B08U_XGB_SHAP_Global_13Factor_Importance.csv').set_index('Factor')
shap=g.Relative_Importance_Percent
for s in (ahp,fahp,anp,shap): assert set(s.index)==set(KEYS) and len(s)==13
# ---------- independent recomputation of SHAP shares from the 20,000-row explanation sample
smp=pd.read_csv(f'{D}/BUET_B08U_XGB_SHAP_Explanation_Sample.csv')
rm=smp[[k+'_SHAP' for k in KEYS]].abs().mean(); rm.index=KEYS
shap_re=100*rm/rm.sum()
# ---------- ranks (1 = largest), computed here
rk=pd.DataFrame({'AHP':ahp.rank(ascending=False,method='min'),'FAHP':fahp.rank(ascending=False,method='min'),'ANP':anp.rank(ascending=False,method='min'),'SHAP':shap.rank(ascending=False,method='min')}).astype(int)
order=list(rk.sort_values(['SHAP']).index)   # factors ordered by SHAP rank
# ---------- audit tables
aud=pd.DataFrame({'Factor':[lab(k) for k in order],'Factor_key':order,
 'AHP_weight':[ahp[k] for k in order],'FAHP_weight':[fahp[k] for k in order],'ANP_weight':[anp[k] for k in order],
 'SHAP_share_pct_grouped_csv':[shap[k] for k in order],'SHAP_share_pct_recomputed_from_sample':[shap_re[k] for k in order],
 'SHAP_mean_abs_grouped_csv':[g.Mean_Absolute_Grouped_SHAP[k] for k in order],'SHAP_mean_abs_recomputed':[rm[k] for k in order]})
aud['SHAP_share_abs_diff']=(aud.SHAP_share_pct_grouped_csv-aud.SHAP_share_pct_recomputed_from_sample).abs()
for m in ['AHP','FAHP','ANP','SHAP']: aud[m+'_rank']=[rk.loc[k,m] for k in order]
aud.to_csv(f'{O}/Figure_8_Numerical_Audit.csv',index=False,float_format='%.15g')
src={'AHP_weight':'DU10_AHP_Weights_and_Consistency.csv : Weight','FAHP_weight':'DU11_FAHP_Weights.csv : Final_Normalized_Weight','ANP_weight':'DU12_ANP_Final_Weights.csv : ANP_Weight','SHAP_share_pct':'BUET_B08U_XGB_SHAP_Global_13Factor_Importance.csv : Relative_Importance_Percent (recomputed from BUET_B08U_XGB_SHAP_Explanation_Sample.csv : <Factor>_SHAP)'}
tr=[]
for k in order:
    for col,(v,s) in {'AHP_weight':(ahp[k],src['AHP_weight']),'FAHP_weight':(fahp[k],src['FAHP_weight']),'ANP_weight':(anp[k],src['ANP_weight']),'SHAP_share_pct':(shap[k],src['SHAP_share_pct'])}.items():
        tr.append({'Panel':'a' if 'weight' in col else 'b','Factor':lab(k),'Quantity':col,'Value':v,'Source_file_and_column':s,'Source_row_key':k})
    for m in ['AHP','FAHP','ANP','SHAP']:
        tr.append({'Panel':'c','Factor':lab(k),'Quantity':m+'_rank','Value':rk.loc[k,m],'Source_file_and_column':'rank(descending, ties=min) of the panel (a)/(b) value','Source_row_key':k})
pd.DataFrame(tr).to_csv(f'{O}/Figure_8_Source_to_Value_Traceability.csv',index=False,float_format='%.15g')
# ---------- figure
C={'AHP':'#0072B2','FAHP':'#56B4E9','ANP':'#009E73','SHAP':'#D55E00'}
M={'AHP':'o','FAHP':'s','ANP':'^','SHAP':'D'}
fig=plt.figure(figsize=(7.6,4.9))
gs=fig.add_gridspec(1,3,width_ratios=[1.45,1.1,1.5],left=0.155,right=0.985,top=0.905,bottom=0.215,wspace=0.14)
axa=fig.add_subplot(gs[0]); axb=fig.add_subplot(gs[1],sharey=axa); axc=fig.add_subplot(gs[2],sharey=axa)
y=np.arange(13)
for ax in (axa,axb,axc):
    ax.set_axisbelow(True); ax.tick_params(labelsize=7,length=2.5,pad=2)
# (a)
axa.grid(axis='x',color='#e3e3e3',lw=0.5)
for i in range(13): axa.axhline(i,color='#f0f0f0',lw=0.5,zorder=0)
offa={'AHP':-0.2,'FAHP':0.0,'ANP':0.2}
for m,s in (('AHP',ahp),('FAHP',fahp),('ANP',anp)):
    axa.scatter([s[k] for k in order],y+offa[m],s=15 if m!='ANP' else 17,marker=M[m],color=C[m],edgecolor='white' if m!='FAHP' else '#1b5b82',linewidth=0.35,zorder=3)
axa.set_xlim(-0.005,0.285); axa.set_xticks([0,0.05,0.10,0.15,0.20,0.25]); axa.set_xticklabels(['0','0.05','0.10','0.15','0.20','0.25'])
axa.set_xlabel('Normalised MCDA weight',fontsize=7,labelpad=3)
axa.set_yticks(y); axa.set_yticklabels([lab(k) for k in order],fontsize=7); axa.set_ylim(12.6,-0.6)
# (b)
axb.grid(axis='x',color='#e3e3e3',lw=0.5)
vals=[shap[k] for k in order]
axb.barh(y,vals,height=0.58,color=C['SHAP'],zorder=3)
for i,v in enumerate(vals): axb.text(v+0.5,i,f'{v:.2f}',va='center',ha='left',fontsize=6.5,color='#222222')
axb.set_xlim(0,33); axb.set_xticks([0,10,20,30])
axb.set_xlabel('Grouped mean |SHAP| share (%)',fontsize=7,labelpad=3)
plt.setp(axb.get_yticklabels(),visible=False); axb.tick_params(axis='y',length=0)
# (c)
axc.grid(axis='x',color='#e3e3e3',lw=0.5)
off={'AHP':-0.27,'FAHP':-0.09,'ANP':0.09,'SHAP':0.27}
for i,k in enumerate(order):
    r=[rk.loc[k,m] for m in ('AHP','FAHP','ANP','SHAP')]
    axc.plot([min(r),max(r)],[i,i],color='#bdbdbd',lw=0.9,zorder=1,solid_capstyle='round')
    for m in ('AHP','FAHP','ANP','SHAP'):
        axc.scatter(rk.loc[k,m],i+off[m],s=13,marker=M[m],color=C[m],edgecolor='white' if m!='FAHP' else '#1b5b82',linewidth=0.3,zorder=3)
axc.set_xlim(0.2,13.8); axc.set_xticks(range(1,14))
axc.set_xlabel('Rank (1 = largest weight or SHAP share)',fontsize=7,labelpad=3)
plt.setp(axc.get_yticklabels(),visible=False); axc.tick_params(axis='y',length=0)
for ax,t,tt in ((axa,'(a)','Prescribed MCDA weights'),(axb,'(b)','XGBoost SHAP attribution'),(axc,'(c)','Rank by factor')):
    tl=ax.text(0,1.035,t,transform=ax.transAxes,fontsize=10,fontweight='bold',ha='left',va='bottom')
    fig.canvas.draw(); bb=tl.get_window_extent(); w=bb.width/ax.get_window_extent().width
    ax.text(w+0.025,1.045,tt,transform=ax.transAxes,fontsize=8.5,fontweight='bold',ha='left',va='bottom')
h=[Line2D([],[],marker=M[m],color='none',markerfacecolor=C[m],markeredgecolor='white' if m!='FAHP' else '#1b5b82',markersize=5.5,label=l) for m,l in (('AHP','AHP'),('FAHP','FAHP'),('ANP','ANP'),('SHAP','XGBoost SHAP'))]
fig.legend(handles=h,loc='lower center',ncol=4,frameon=False,fontsize=7,bbox_to_anchor=(0.55,0.085),handletextpad=0.3,columnspacing=1.6)
fig.text(0.5,0.028,'MCDA weights (prescribed priorities) and SHAP shares (model attribution) are different quantities on separate axes; panel (c) compares ordering only.\nSHAP values are model-specific and associational, not causal.',ha='center',va='bottom',fontsize=6.5,color='#333333',linespacing=1.35)
for ext,kw in (('png',dict(dpi=600)),('svg',dict(dpi=600)),('pdf',dict(dpi=600))):
    fig.savefig(f'{O}/Figure_8_Weights_SHAP_Q1_FINAL.{ext}',**kw)
plt.close(fig); print('ok',O)
