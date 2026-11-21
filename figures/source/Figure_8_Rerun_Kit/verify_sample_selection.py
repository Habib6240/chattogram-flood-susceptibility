#!/usr/bin/env python3
"""Partial B08U provenance check that does NOT need xgboost: rebuilds the explanation-row selection exactly as B08U does
(valid rows, block->fold map, numpy default_rng(42), 4,000 per fold) and compares it with the supplied explanation sample.
Usage: python3 verify_sample_selection.py <B04U.npz> <B06_folds.csv> <B08U_Explanation_Sample.csv>"""
import sys,csv,numpy as np,pandas as pd
npz,folds,samp=sys.argv[1:4]
D=np.load(npz,allow_pickle=False); y=D['Composite_AnyFlood_All4Valid'].astype(np.int8); V=y>=0; oi=np.where(V)[0]
yv=y[V]; blk=D['Block_5km'].astype(np.int64)[V]; N=len(yv); print('valid rows',N,'flood',int((yv==1).sum()),'non-flood',int((yv==0).sum()))
b2f={}
for r in csv.DictReader(open(folds,encoding='utf-8')): b2f[int(r['Block_5km'])]=int(r['Spatial_Fold'])
sf=np.array([b2f[int(b)] for b in blk],dtype=np.uint8)
rng=np.random.default_rng(42); idx=[]
for f in range(1,6):
    c=np.where(sf==f)[0]; idx.extend(rng.choice(c,size=min(4000,len(c)),replace=False).tolist())
idx=np.array(idx,dtype=np.int64); S=pd.read_csv(samp); ok=True
def chk(n,c,d=''):
    global ok; ok&=bool(c); print(('PASS ' if c else 'FAIL ')+n,d)
chk('sample length 20,000',len(S)==len(idx)==20000)
X=D['X'].astype(np.float64)[V][idx]; Y=D['Y'].astype(np.float64)[V][idx]
chk('X,Y of every row equal',np.array_equal(X,S.X.values) and np.array_equal(Y,S.Y.values))
chk('Spatial_Fold equal',np.array_equal(sf[idx],S.Spatial_Fold.values))
chk('observed label equal',np.array_equal(yv[idx],S.Observed_Composite_Label.values))
for k in ['Elevation','Slope','TWI','Rainfall','NDVI','LULC','HSG','Distance_River','Distance_Road','Drainage_Density','Profile_Curvature','Tangential_Curvature']:
    v=D[k][oi][idx].astype(np.float64); chk(f'{k}_Value equal (float32 tolerance)',np.allclose(v,S[k+'_Value'].values,rtol=1e-6,atol=1e-9))
print('ALL PASS' if ok else 'SOME FAIL')
