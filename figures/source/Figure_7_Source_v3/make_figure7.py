"""Figure 7 (v3: footnote wording only changed from v2): complete six-method pairwise spatial-agreement structure (15 unordered pairs).
(a) Pearson r, (b) exact five-class agreement (B09B method-specific percentile classes), (c) top-20% Jaccard.
Data: fig7_data_v3.csv (written by this script). Seven pairs are the B09B values supplied for Table 6 at source precision;
the other eight pairs are taken from the integrated reference book table (3-decimal precision; cross-checked against the seven
supplied pairs) and are flagged in the Source column until the B09B pairwise CSV is available at full precision."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
M=['RF','XGBoost','ML consensus','AHP','FAHP','ANP']
S='B09B pairwise values supplied for Table 6 (source precision)'; R='Integrated reference book B09B table (3 decimals)'
rows=[('RF','XGBoost',0.963245,78.66,0.861023,S),('RF','ML consensus',0.989,87.4,0.933,R),('RF','AHP',0.617,45.8,0.364,R),('RF','FAHP',0.623,46.5,0.376,R),('RF','ANP',0.603,45.8,0.374,R),
('XGBoost','ML consensus',0.992,90.9,0.923,R),('XGBoost','AHP',0.666,46.9,0.372,R),('XGBoost','FAHP',0.672,47.6,0.384,R),('XGBoost','ANP',0.651,46.5,0.381,R),
('ML consensus','AHP',0.649443,46.66,0.3687,S),('ML consensus','FAHP',0.655320,47.36,0.381276,S),('ML consensus','ANP',0.634384,46.54,0.3788,S),
('AHP','FAHP',0.999784,97.71,0.962272,S),('AHP','ANP',0.9495,65.60,0.5476,S),('FAHP','ANP',0.9541,66.79,0.5661,S)]
df=pd.DataFrame(rows,columns=['Method_A','Method_B','Pearson_r','Exact_5class_agreement_pct','Top20_Jaccard','Source']); df.to_csv('fig7_data_v3.csv',index=False)
assert len(df)==15
idx={m:i for i,m in enumerate(M)}
def mat(col,scale=1.0):
    A=np.full((6,6),np.nan)
    for _,r in df.iterrows():
        i,j=idx[r.Method_A],idx[r.Method_B]; A[i,j]=A[j,i]=r[col]*scale
    return A
panels=[('(a) Pearson correlation',mat('Pearson_r'),'p'),('(b) Exact five-class agreement',mat('Exact_5class_agreement_pct',0.01),'c'),('(c) Top-20% Jaccard overlap',mat('Top20_Jaccard'),'j')]
def fmt(v,kind):
    if kind=='p': return f"{v:.4f}" if v>0.9995 else f"{v:.3f}"
    if kind=='c': return f"{v*100:.1f}%"
    return f"{v:.3f}"
fig,axs=plt.subplots(1,3,figsize=(13.2,6.0))
cmap=plt.get_cmap('YlGnBu').copy()
for ax,(title,A,kind) in zip(axs,panels):
    Z=np.ma.masked_invalid(A)
    im=ax.imshow(Z,cmap=cmap,vmin=0.3,vmax=1.0)
    for i in range(6):
        for j in range(6):
            if i==j:
                ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,facecolor='#e6e6e6',edgecolor='none')); ax.text(j,i,'—',ha='center',va='center',fontsize=10,color='#777777'); continue
            v=A[i,j]; ax.text(j,i,fmt(v,kind),ha='center',va='center',fontsize=9.5,color='white' if v>0.72 else '#1a1a1a')
    ax.set_xticks(range(6)); ax.set_xticklabels(M,rotation=40,ha='left',fontsize=10); ax.xaxis.tick_top()
    ax.set_yticks(range(6)); ax.set_yticklabels(M if ax is axs[0] else [],fontsize=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    for k in (2.5,):
        ax.axhline(k,color='white',lw=4); ax.axvline(k,color='white',lw=4)
    ax.set_xlim(-0.5,5.5); ax.set_ylim(5.5,-0.5)
fig.subplots_adjust(left=0.085,right=0.995,top=0.76,bottom=0.24,wspace=0.06)
for ax,(title,A,kind) in zip(axs,panels):
    p=ax.get_position(); fig.text(p.x0,0.955,title,fontsize=12,fontweight='bold',ha='left',va='top')
cax=fig.add_axes([0.35,0.175,0.30,0.022]); cb=fig.colorbar(im,cax=cax,orientation='horizontal'); cb.set_label('Value (class agreement shown as proportion)',fontsize=9); cb.ax.tick_params(labelsize=9)
fig.text(0.01,0.01,'Full 4,722,216-cell domain; all 15 unordered surface pairs. Panel (b) uses method-specific percentile (quintile) classes computed separately for each surface, not the common-score classes used for the ML maps.\nTop-20% Jaccard is the overlap of the highest-susceptibility sets and is not accuracy or coincidence with observed flooding. White lines separate ML surfaces (RF, XGBoost, consensus) from MCDA surfaces.',fontsize=8,color='#444444',va='bottom')
fig.savefig('Figure_7_Spatial_Agreement_v3.png',dpi=300); fig.savefig('Figure_7_Spatial_Agreement_v3.svg'); plt.close(fig)
print('ok')
