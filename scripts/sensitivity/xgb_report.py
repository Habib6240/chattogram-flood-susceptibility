import numpy as np, csv, os, json
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
EV=['2019','2021','2023','2026']
L=np.load('/home/claude/release/chattogram-flood-susceptibility-data-v1.0.0/06_oof_predictions/BUET_B07U_LOEO_OutOfFold_Predictions.npz'); FOLD=L['Spatial_Fold'].astype(int)
D=np.load('/tmp/orig/BUET_B04U_ML_Master_150m.npz'); y=D['Composite_AnyFlood_All4Valid'].astype(int); valid=y>=0
rows=list(csv.DictReader(open('/home/claude/release/chattogram-flood-susceptibility-data-v1.0.0/06_oof_predictions/BUET_B06U_OutOfFold_Predictions.csv')))
key={(int(a),int(b)):i for i,(a,b) in enumerate(zip(D['Raster_Row'],D['Raster_Column']))}
arch06={k:np.full(len(y),np.nan,np.float32) for k in('RF','XGB')}
for q in rows:
    i=key[(int(q['Raster_Row']),int(q['Raster_Column']))];arch06['RF'][i]=float(q['RF_OOF_Probability']);arch06['XGB'][i]=float(q['XGB_OOF_Probability'])
def b06(p):
    fa=[];fp=[]
    for f in range(1,6):
        k=valid&(FOLD==f);fa.append(AUC(y[k],p[k]));fp.append(AP(y[k],p[k]))
    return dict(auc=AUC(y[valid],p[valid]),ap=AP(y[valid],p[valid]),fauc=np.mean(fa),fap=np.mean(fp))
def b07(pred):
    o={}
    for e in EV:
        obs=L['Observed_'+e].astype(int);p=pred[e];v=obs>=0
        fa=[AUC(obs[v&(FOLD==f)],p[v&(FOLD==f)]) for f in range(1,6)];fp=[AP(obs[v&(FOLD==f)],p[v&(FOLD==f)]) for f in range(1,6)]
        o[e]=dict(auc=AUC(obs[v],p[v]),ap=AP(obs[v],p[v]),fauc=np.mean(fa),fap=np.mean(fp))
    for k in('auc','ap','fauc','fap'): o['macro_'+k]=float(np.mean([o[e][k] for e in EV]))
    return o
def ld(design,variant,tag=''):
    f=f'xres_{design}_{variant}{tag}.npz'
    if not os.path.exists(f): return None
    Z=np.load(f); return Z['XGB_oof'] if design=='B06U' else {e:Z[f'XGB_{e}'] for e in EV}
def cons(r,x): return {e:((r[e].astype(np.float64)+x[e].astype(np.float64))/2).astype(np.float32) for e in EV}
RF={e:L['RF_'+e] for e in EV}; XA={e:L['XGB_'+e] for e in EV}
def dif(a,b,m):
    a,b=a[m].astype(float),b[m].astype(float);d=b-a
    return dict(pearson=np.corrcoef(a,b)[0,1],spearman=spearmanr(a,b)[0],mean_abs=np.abs(d).mean(),p99=np.percentile(np.abs(d),99),max_abs=np.abs(d).max())
def classes(p,m):
    q=np.zeros(len(p),int);v=np.where(m)[0];r=np.argsort(np.argsort(p[v],kind='stable'),kind='stable');q[v]=np.minimum(5,1+(r*5)//len(v));return q
def ccmp(a,b,m):
    ca,cb=classes(a,m),classes(b,m);ta=(ca==5)&m;tb=(cb==5)&m
    return dict(changed=float((ca[m]!=cb[m]).mean()),gt1=float((np.abs(ca[m]-cb[m])>1).mean()),jac=float((ta&tb).sum()/(ta|tb).sum()))
R={}
def S(name,design,variant,tag=''):
    p=ld(design,variant,tag)
    if p is None: return None
    R[name]=(design,p); return p
# collect
for d in('B06U','B07U'):
    S(f'{d}_orig',d,'orig'); S(f'{d}_vector',d,'vector'); S(f'{d}_raster30',d,'raster30')
    for s in(43,44): S(f'{d}_orig_s{s}',d,'orig',f'_seed{s}')
    for s in(43,44): S(f'{d}_vector_s{s}',d,'vector',f'_seed{s}')
f=lambda x:f'{x:.6f}';fs=lambda x:f'{x:+.6f}'
L_=[]
def P(s=''): print(s); L_.append(s)
M06={k:b06(v[1]) for k,v in R.items() if k.startswith('B06U')}
M07={k:b07(v[1]) for k,v in R.items() if k.startswith('B07U')}
A06=b06(arch06['XGB']);A07=b07(XA);C07=b07(cons(RF,XA))
P('### A. B06U 5 km spatial block CV (XGBoost; valid composite rows n=%d)'%valid.sum())
P('| run | pooled AUC | pooled AP | fold-mean AUC | fold-mean AP | max abs dp vs archived | Pearson vs archived |');P('|---|---|---|---|---|---|---|')
P(f"| archived | {f(A06['auc'])} | {f(A06['ap'])} | {f(A06['fauc'])} | {f(A06['fap'])} | - | - |")
for k,m in M06.items():
    d=dif(arch06['XGB'],R[k][1],valid);P(f"| {k[5:]} | {f(m['auc'])} | {f(m['ap'])} | {f(m['fauc'])} | {f(m['fap'])} | {d['max_abs']:.3f} | {d['pearson']:.5f} |")
P('\n### B. B07U strict event- and geography-withheld (XGBoost, macro over 4 events)')
P('| run | macro AUC | macro AP | macro fold-mean AUC | macro fold-mean AP | AUC 2019/2021/2023/2026 | AP 2019/2021/2023/2026 |');P('|---|---|---|---|---|---|---|')
def row(n,m): return f"| {n} | {f(m['macro_auc'])} | {f(m['macro_ap'])} | {f(m['macro_fauc'])} | {f(m['macro_fap'])} | "+' / '.join(f"{m[e]['auc']:.4f}" for e in EV)+' | '+' / '.join(f"{m[e]['ap']:.4f}" for e in EV)+' |'
P(row('archived',A07))
for k,m in M07.items(): P(row(k[5:],m))
# paired deltas
P('\n### C. Absolute changes (regenerated minus baseline)')
P('| design | comparison | dMacro AUC | dMacro AP | per-event dAUC (2019/21/23/26) | per-event dAP | pred diff mean / p99 / max | Pearson | Spearman |');P('|---|---|---|---|---|---|---|---|---|')
def pdiff(a,b,design):
    if design=='B06U': return dif(a,b,valid)
    ds=[dif(a[e],b[e],L['Observed_'+e]>=0) for e in EV]
    A=np.concatenate([a[e][L['Observed_'+e]>=0] for e in EV]);B=np.concatenate([b[e][L['Observed_'+e]>=0] for e in EV]);return dif(A,B,np.ones(len(A),bool))
for design,M,Rr in(('B06U',M06,None),('B07U',M07,None)):
    for var in('vector','raster30'):
        for base in('orig',):
            kk=f'{design}_{var}';bk=f'{design}_{base}'
            if kk not in M or bk not in M: continue
            m,b=M[kk],M[bk];d=pdiff(R[bk][1],R[kk][1],design)
            if design=='B06U': P(f"| {design} | {var} - rerun-orig | {fs(m['auc']-b['auc'])} (pooled AUC) | {fs(m['ap']-b['ap'])} (pooled AP) | - | - | {d['mean_abs']:.4f} / {d['p99']:.4f} / {d['max_abs']:.4f} | {d['pearson']:.5f} | {d['spearman']:.5f} |")
            else: P(f"| {design} | {var} - rerun-orig | {fs(m['macro_auc']-b['macro_auc'])} | {fs(m['macro_ap']-b['macro_ap'])} | "+' / '.join(f"{m[e]['auc']-b[e]['auc']:+.4f}" for e in EV)+' | '+' / '.join(f"{m[e]['ap']-b[e]['ap']:+.4f}" for e in EV)+f" | {d['mean_abs']:.4f} / {d['p99']:.4f} / {d['max_abs']:.4f} | {d['pearson']:.5f} | {d['spearman']:.5f} |")
# seed noise band
P('\n### D. Seed-noise reference (XGBoost random_state 42 vs 43 vs 44 on ORIGINAL data, and on regenerated vector)')
P('| design | metric | orig seeds 42/43/44 | max abs seed change (orig) | regenerated(vector)-orig (seed 42) | within band? |');P('|---|---|---|---|---|---|')
for design,M in(('B06U',M06),('B07U',M07)):
    ks=[k for k in (f'{design}_orig',f'{design}_orig_s43',f'{design}_orig_s44') if k in M]
    if len(ks)<2 or f'{design}_vector' not in M: continue
    mets=['auc','ap'] if design=='B06U' else ['macro_auc','macro_ap']
    for mt in mets:
        vals=[M[k][mt] for k in ks];band=max(abs(vals[i]-vals[0]) for i in range(1,len(vals)));ch=M[f'{design}_vector'][mt]-vals[0]
        P(f"| {design} | {mt} | {' / '.join(f(v) for v in vals)} | {band:.6f} | {fs(ch)} | {'yes' if abs(ch)<=band else 'no'} |")
    if design=='B07U':
        for e in EV:
            for mt in('auc','ap'):
                vals=[M[k][e][mt] for k in ks];band=max(abs(vals[i]-vals[0]) for i in range(1,len(vals)));ch=M['B07U_vector'][e][mt]-vals[0]
                P(f"| B07U | {e} {mt} | {' / '.join(f(v) for v in vals)} | {band:.6f} | {fs(ch)} | {'yes' if abs(ch)<=band else 'no'} |")
# Phase 5 consensus
P('\n### E. RF-XGBoost consensus (B07U, archived RF + XGBoost; consensus=(RF+XGB)/2)')
P('| consensus built from | macro AUC | macro AP | AUC 2019/21/23/26 | AP 2019/21/23/26 |');P('|---|---|---|---|---|')
def crow(n,m): return f"| {n} | {f(m['macro_auc'])} | {f(m['macro_ap'])} | "+' / '.join(f"{m[e]['auc']:.4f}" for e in EV)+' | '+' / '.join(f"{m[e]['ap']:.4f}" for e in EV)+' |'
P(crow('archived RF + archived XGB',C07))
CC={}
for v in('orig','vector','raster30','orig_s43','orig_s44'):
    k=f'B07U_{v}'
    if k in R: CC[v]=cons(RF,R[k][1]);P(crow(f'archived RF + XGB rerun ({v})',b07(CC[v])))
base=cons(RF,XA)
P('\n| consensus comparison | dMacro AUC | dMacro AP | mean / p99 / max abs prediction change | Pearson | Spearman | quintile class changed (mean over events) | moved >1 class | top-20% Jaccard (mean of events) |');P('|---|---|---|---|---|---|---|---|---|')
def crow2(n,A,B):
    ds=[dif(A[e],B[e],L['Observed_'+e]>=0) for e in EV];cs=[ccmp(A[e],B[e],L['Observed_'+e]>=0) for e in EV]
    mA,mB=b07(A),b07(B);g=lambda k:np.mean([d[k] for d in ds]);h=lambda k:np.mean([c[k] for c in cs])
    P(f"| {n} | {fs(mB['macro_auc']-mA['macro_auc'])} | {fs(mB['macro_ap']-mA['macro_ap'])} | {g('mean_abs'):.4f} / {g('p99'):.4f} / {max(d['max_abs'] for d in ds):.4f} | {g('pearson'):.5f} | {g('spearman'):.5f} | {h('changed'):.4f} | {h('gt1'):.4f} | {h('jac'):.4f} |")
if 'orig' in CC:
    if 'vector' in CC: crow2('regen(vector) - rerun-orig consensus',CC['orig'],CC['vector'])
    if 'raster30' in CC: crow2('regen(raster30) - rerun-orig consensus',CC['orig'],CC['raster30'])
    crow2('rerun-orig - archived consensus',base,CC['orig'])
    if 'orig_s43' in CC: crow2('seed 43 - seed 42 (orig data)',CC['orig'],CC['orig_s43'])
    if 'orig_s44' in CC: crow2('seed 44 - seed 42 (orig data)',CC['orig'],CC['orig_s44'])
if 'vector' in CC: crow2('regen(vector) - archived consensus',base,CC['vector'])
P('\nper-event consensus top-20% Jaccard and quintile change, regen(vector) vs rerun-orig:')
if 'orig' in CC and 'vector' in CC:
    for e in EV:
        c=ccmp(CC['orig'][e],CC['vector'][e],L['Observed_'+e]>=0);P(f"- {e}: class changed {c['changed']:.4f}, moved >1 class {c['gt1']:.4f}, top-20% Jaccard {c['jac']:.4f}")
open('xgb_report_tables.md','w').write('\n'.join(L_))
