"""Figure 5 (v2): six-method event-level ROC-AUC and AP. Plotted values are read directly from BUET_B09B_Strict_Event_Performance_All_Methods.csv (macro mean = unweighted mean of the four events). v2 changes the footnote wording only."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'font.size':9})
U='/mnt/user-data/uploads/'
# ---------- Figure 5
ev=pd.read_csv(U+'BUET_B09B_Strict_Event_Performance_All_Methods.csv')
P=ev.pivot(index='Held_Event',columns='Method',values='ROC_AUC'); Q=ev.pivot(index='Held_Event',columns='Method',values='Average_Precision')
meth=[('Random_Forest','RF','#d95f02','o'),('XGBoost','XGBoost','#e6ab02','s'),('ML_Consensus','ML consensus','#a6001a','D'),('AHP','AHP','#08519c','^'),('FAHP','FAHP','#3182bd','v'),('ANP','ANP','#6baed6','P')]
rows=[]
for e in [2019,2021,2023,2026]:
    for m,l,_,_ in meth: rows.append([e,l,P.loc[e,m],Q.loc[e,m]])
for m,l,_,_ in meth: rows.append(['Macro mean',l,P[m].mean(),Q[m].mean()])
pd.DataFrame(rows,columns=['Held_event','Method','ROC_AUC','Average_Precision']).round(6).to_csv('fig5_data_v2.csv',index=False)
cats=['2019','2021','2023','2026','Macro\nmean']
fig,axs=plt.subplots(1,2,figsize=(10.5,4.2))
for ax,(D,yl,lim) in zip(axs,[(P,'ROC-AUC',(0.70,0.92)),(Q,'Average precision (AP)',(0.0,0.62))]):
    for k,(m,l,c,mk) in enumerate(meth):
        vals=[D.loc[e,m] for e in [2019,2021,2023,2026]]+[D[m].mean()]
        xs=np.arange(5)+(k-2.5)*0.12
        ax.scatter(xs,vals,color=c,marker=mk,s=34,label=l,zorder=3,edgecolor='white',linewidth=0.4)
    ax.axvline(3.5,color='#bbbbbb',lw=0.8,ls='--')
    ax.set_xticks(range(5)); ax.set_xticklabels(cats); ax.set_ylabel(yl); ax.set_ylim(*lim); ax.grid(axis='y',color='#e5e5e5',lw=0.6,zorder=0)
axs[0].set_title('(a) ROC-AUC',loc='left',fontsize=10,fontweight='bold'); axs[1].set_title('(b) Average precision',loc='left',fontsize=10,fontweight='bold')
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,ncol=6,frameon=False,loc='lower center',bbox_to_anchor=(0.5,0.045),fontsize=8.5)
fig.text(0.5,0.005,'Identical held-event cohorts for all six methods. Descriptive comparison; macro mean = unweighted mean of the four events.',ha='center',fontsize=7.5,color='#444444')
fig.tight_layout(rect=(0,0.11,1,1))
fig.savefig('Figure_5_Strict_Performance_v2.png',dpi=300); fig.savefig('Figure_5_Strict_Performance_v2.svg'); plt.close(fig)
