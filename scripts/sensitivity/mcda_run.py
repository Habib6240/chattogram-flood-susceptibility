import numpy as np, json
from mcda_sens import *
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
from PIL import Image
EV=['2019','2021','2023','2026']
V={'orig':D['Distance_River'].astype('float64'),'vector':np.load('dist_vector.npy').astype('float64'),'raster30':np.load('dist_raster30.npy').astype('float64')}
S={k:scores(v)[0] for k,v in V.items()}
print('Distance_River p5,p95 orig/vector/raster:',[tuple(np.round(np.percentile(v,[5,95]),2)) for v in V.values()])
print('weights of Distance_River:',{m:round(float(Wt[m][4]),5) for m in Wt})
cls=np.asarray(Image.open(T+'DU13_FAHP_CommonClass_1to5.tif')).astype(int)[r,c]
# class edges recovered from archived FAHP index
edges=[ (np.nanmax(ARCH['FAHP'][cls==k]),np.nanmin(ARCH['FAHP'][cls==k+1])) for k in range(1,5)];print('class edges (archived, FAHP):',[tuple(np.round(e,4)) for e in edges])
brk=[(a+b)/2 for a,b in edges]
def klass(s): return 1+np.digitize(s,brk)
def met(s,lab):
    v=lab>=0;return roc_auc_score(lab[v],s[v]),average_precision_score(lab[v],s[v])
res={}
for m in ['AHP','FAHP','ANP']:
    for nm,sc_ in [('archived',ARCH[m]),('orig',S['orig'][m]),('vector',S['vector'][m]),('raster30',S['raster30'][m])]:
        a=[met(sc_,D['Flood_'+e].astype(int)) for e in EV]
        res[(m,nm)]=(np.mean([x[0] for x in a]),np.mean([x[1] for x in a]),[round(x[0],6) for x in a])
    print(m,'macro AUC/AP  archived %.6f/%.6f | recomputed(orig dist) %.6f/%.6f | vector %.6f/%.6f | raster30 %.6f/%.6f'%(res[(m,'archived')][0],res[(m,'archived')][1],res[(m,'orig')][0],res[(m,'orig')][1],res[(m,'vector')][0],res[(m,'vector')][1],res[(m,'raster30')][0],res[(m,'raster30')][1]))
    for v in ['vector','raster30']:
        a,b=S['orig'][m],S[v][m];d=b-a
        thr=lambda s:s>=np.quantile(s,0.8);ja,jb=thr(a),thr(b);jac=(ja&jb).sum()/(ja|jb).sum()
        print(f'   {v}: Pearson {np.corrcoef(a,b)[0,1]:.8f} Spearman {spearmanr(a,b)[0]:.8f} MAE {np.abs(d).mean():.5f} max|d| {np.abs(d).max():.5f} top20%% Jaccard {jac:.5f} class changed (5-class) {np.mean(klass(a)!=klass(b))*100:.3f}%  AUC diff(macro) {res[(m,v)][0]-res[(m,"orig")][0]:+.6f} AP diff {res[(m,v)][1]-res[(m,"orig")][1]:+.6f}')
print('per-event AUC orig vs vector (AHP):',res[('AHP','orig')][2],res[('AHP','vector')][2])
