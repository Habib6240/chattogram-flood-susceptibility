import numpy as np, csv, glob, os, json
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
from ml_metrics import loeo_metrics, EV, L, FOLD
D=np.load('/tmp/orig/BUET_B04U_ML_Master_150m.npz');y=D['Composite_AnyFlood_All4Valid'].astype(int);valid=y>=0
# archived B06U OOF aligned to master rows
rows=list(csv.DictReader(open('/home/claude/release/chattogram-flood-susceptibility-data-v1.0.0/06_oof_predictions/BUET_B06U_OutOfFold_Predictions.csv')))
key={(int(a),int(b)):i for i,(a,b) in enumerate(zip(D['Raster_Row'],D['Raster_Column']))}
arch={'RF':np.full(len(y),np.nan,np.float32),'XGB':np.full(len(y),np.nan,np.float32)}
for q in rows:
    i=key[(int(q['Raster_Row']),int(q['Raster_Column']))];arch['RF'][i]=float(q['RF_OOF_Probability']);arch['XGB'][i]=float(q['XGB_OOF_Probability'])
def b06(p):
    out={'pooled':(AUC(y[valid],p[valid]),AP(y[valid],p[valid]))}
    fa=[];fp=[]
    for f in range(1,6):
        k=valid&(FOLD==f);fa.append(AUC(y[k],p[k]));fp.append(AP(y[k],p[k]))
    out['fold_auc']=fa;out['fold_ap']=fp;out['foldmean']=(np.mean(fa),np.mean(fp));return out
def cmp(a,b,mask):
    a,b=a[mask].astype(float),b[mask].astype(float);d=b-a
    return dict(pearson=np.corrcoef(a,b)[0,1],spearman=spearmanr(a,b)[0],mean_abs=np.abs(d).mean(),p99_abs=np.percentile(np.abs(d),99),max_abs=np.abs(d).max())
R={}
print('=== B06U (5-fold geographic spatial CV) ===')
for k in ['RF','HGB']:
    base=None
    for v in ['orig','vector','raster30']:
        f=f'res_B06U_{v}_RF-HGB.npz'
        if not os.path.exists(f): continue
        p=np.load(f)[f'{k}_oof'];m=b06(p);R[('B06U',k,v)]=m
        if v=='orig': base=p
        line=f"{k:3s} {v:9s} pooled AUC {m['pooled'][0]:.6f} AP {m['pooled'][1]:.6f} | fold-mean AUC {m['foldmean'][0]:.6f} AP {m['foldmean'][1]:.6f}"
        if v!='orig': c=cmp(base,p,valid);line+=f" | vs orig-rerun: dAUC {m['pooled'][0]-R[('B06U',k,'orig')]['pooled'][0]:+.6f} dAP {m['pooled'][1]-R[('B06U',k,'orig')]['pooled'][1]:+.6f} Pearson {c['pearson']:.6f} Spearman {c['spearman']:.6f} mean|dp| {c['mean_abs']:.5f} p99 {c['p99_abs']:.4f} max {c['max_abs']:.4f}"
        print(line)
for k in ['RF']:
    m=b06(arch[k]);print(f"{k} ARCHIVED (original run) pooled AUC {m['pooled'][0]:.6f} AP {m['pooled'][1]:.6f} | fold-mean AUC {m['foldmean'][0]:.6f}")
    if os.path.exists('res_B06U_orig_RF-HGB.npz'):
        p=np.load('res_B06U_orig_RF-HGB.npz')['RF_oof'];c=cmp(arch[k],p,valid);print(f"  environment check (archived vs rerun on ORIGINAL data): Pearson {c['pearson']:.6f} mean|dp| {c['mean_abs']:.5f} max {c['max_abs']:.4f}; dAUC {b06(p)['pooled'][0]-m['pooled'][0]:+.6f}")
print('\n=== B07U (strict event- and geography-withheld LOEO, 20 scenarios) ===')
base={}
for k in ['RF','HGB']:
    for v in ['orig','vector','raster30']:
        f=f'res_B07U_{v}_RF-HGB.npz'
        if not os.path.exists(f): continue
        Z=np.load(f);pred={e:Z[f'{k}_{e}'] for e in EV};m=loeo_metrics(pred);R[('B07U',k,v)]=m
        if v=='orig': base[k]=pred
        line=f"{k:3s} {v:9s} macro pooled AUC {m['macro_pooled_auc']:.6f} AP {m['macro_pooled_ap']:.6f} | macro fold-mean AUC {m['macro_foldmean_auc']:.6f} AP {m['macro_foldmean_ap']:.6f}"
        line+=' | per-event AUC '+' '.join(f"{m[e]['pooled_auc']:.4f}" for e in EV)
        print(line)
        if v!='orig':
            o=R[('B07U',k,'orig')];print(f"      delta vs orig-rerun: macro AUC {m['macro_pooled_auc']-o['macro_pooled_auc']:+.6f} AP {m['macro_pooled_ap']-o['macro_pooled_ap']:+.6f}; per-event dAUC "+' '.join(f"{m[e]['pooled_auc']-o[e]['pooled_auc']:+.5f}" for e in EV)+'; per-event dAP '+' '.join(f"{m[e]['pooled_ap']-o[e]['pooled_ap']:+.5f}" for e in EV))
            for e in EV:
                obs=L['Observed_'+e].astype(int);vv=obs>=0;c=cmp(base[k][e],pred[e],vv)
                print(f"      {e}: Pearson {c['pearson']:.6f} Spearman {c['spearman']:.6f} mean|dp| {c['mean_abs']:.5f} max {c['max_abs']:.4f}; fold AUC range of change {min(np.array(m[e]['fold_auc'])-np.array(o[e]['fold_auc'])):+.5f}..{max(np.array(m[e]['fold_auc'])-np.array(o[e]['fold_auc'])):+.5f}")
if os.path.exists('res_B07U_orig_RF-HGB.npz'):
    m=loeo_metrics({e:L['RF_'+e] for e in EV});o=R[('B07U','RF','orig')]
    print(f"RF ARCHIVED macro pooled AUC {m['macro_pooled_auc']:.6f} AP {m['macro_pooled_ap']:.6f}; orig-rerun - archived: dAUC {o['macro_pooled_auc']-m['macro_pooled_auc']:+.6f} dAP {o['macro_pooled_ap']-m['macro_pooled_ap']:+.6f}")
    for e in EV:
        obs=L['Observed_'+e].astype(int);vv=obs>=0;c=cmp(L['RF_'+e],base['RF'][e],vv);print(f"   env check {e}: Pearson {c['pearson']:.6f} mean|dp| {c['mean_abs']:.5f} max {c['max_abs']:.4f}")
