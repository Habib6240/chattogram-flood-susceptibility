"""Sensitivity re-run of the archived B06U / B07U designs (RF; optional HGB surrogate) with a chosen Distance_River vector.
Everything else identical to the archived scripts: features, preprocessing, folds, labels, RF hyperparameters, seed 42."""
import sys, time, json, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
variant, design, models = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
D = np.load('/tmp/orig/BUET_B04U_ML_Master_150m.npz')
L = np.load('/home/claude/release/chattogram-flood-susceptibility-data-v1.0.0/06_oof_predictions/BUET_B07U_LOEO_OutOfFold_Predictions.npz')
assert np.array_equal(L['Raster_Row'], D['Raster_Row']) and np.array_equal(L['Raster_Column'], D['Raster_Column'])
FOLD = L['Spatial_Fold'].astype(np.int64)
dist = np.load(f'/home/claude/final/sens/dist_{variant}.npy')
names = ["Elevation","Slope","Aspect_sin","Aspect_cos","Aspect_flat","Profile_Curvature","Tangential_Curvature","TWI","Drainage_Density","Rainfall","LULC","NDVI","HSG","Distance_River","Distance_Road"]
X = np.column_stack([dist if n == "Distance_River" else D[n] for n in names]).astype(np.float32)
NUM=[0,1,2,3,4,5,6,7,8,9,11,13,14]; CAT=[10,12]
def pre(): return ColumnTransformer([("numeric","passthrough",NUM),("categorical",OneHotEncoder(handle_unknown="ignore",sparse_output=False,dtype=np.float32),CAT)],remainder="drop",sparse_threshold=0.0)
def fit_predict(kind, Xtr, ytr, Xte):
    if kind == 'RF':
        m = RandomForestClassifier(n_estimators=500,max_features="sqrt",min_samples_leaf=2,class_weight="balanced_subsample",bootstrap=True,n_jobs=-1,random_state=int(__import__('os').environ.get('SEED','42')))
        m.fit(Xtr, ytr)
    else:  # surrogate for XGBoost (xgboost not installable here): NOT the original model
        w = np.where(ytr==1, (ytr==0).sum()/(ytr==1).sum(), 1.0)
        m = HistGradientBoostingClassifier(max_iter=600,learning_rate=0.05,max_depth=5,l2_regularization=1.0,max_features=0.8,random_state=42)
        m.fit(Xtr, ytr, sample_weight=w)
    return m.predict_proba(Xte)[:,1].astype(np.float32)
EV=["2019","2021","2023","2026"]; LAB={e:D[f"Flood_{e}"].astype(np.int8) for e in EV}
out = {}
t0=time.time()
if design == 'B06U':
    y = D["Composite_AnyFlood_All4Valid"].astype(np.int8); valid = y >= 0
    for k in models:
        oof = np.full(len(y), np.nan, np.float32)
        for f in range(1,6):
            tr = np.where(valid & (FOLD!=f))[0]; te = np.where(valid & (FOLD==f))[0]
            p = pre(); Xtr = p.fit_transform(X[tr]).astype(np.float32); Xte = p.transform(X[te]).astype(np.float32)
            oof[te] = fit_predict(k, Xtr, y[tr], Xte); print(variant,design,k,'fold',f,'%.0fs'%(time.time()-t0),flush=True)
        out[f'{k}_oof']=oof
else:
    for k in models:
        for e in EV: out[f'{k}_{e}']=np.full(len(FOLD),np.nan,np.float32)
    import os
    part=f'/home/claude/final/sens/part_{design}_{variant}.npz'
    if os.path.exists(part):
        P=np.load(part)
        for kk in P.files: out[kk]=P[kk]
        done_ev=[e for e in EV if f'DONE_{e}' in P.files]
    else: done_ev=[]
    for held in EV:
        if held in done_ev: continue
        tm=np.column_stack([LAB[e] for e in EV if e!=held]); vt=np.all(tm>=0,axis=1)
        tt=np.full(len(FOLD),-1,np.int8); tt[vt]=np.any(tm[vt]==1,axis=1).astype(np.int8)
        hv = LAB[held]>=0
        for f in range(1,6):
            tr=np.where(vt&(FOLD!=f))[0]; te=np.where(hv&(FOLD==f))[0]
            p=pre(); Xtr=p.fit_transform(X[tr]).astype(np.float32); Xte=p.transform(X[te]).astype(np.float32)
            for k in models:
                out[f'{k}_{held}'][te]=fit_predict(k,Xtr,tt[tr],Xte)
            print(variant,design,held,'fold',f,'%.0fs'%(time.time()-t0),flush=True)
        np.savez_compressed(part, **out, **{f'DONE_{e}':np.array(1) for e in done_ev+[held]}); done_ev.append(held)
np.savez_compressed(f'/home/claude/final/sens/res_{design}_{variant}{__import__("os").environ.get("TAG","")}_{"-".join(models)}.npz', **out)
print('done %.0fs'%(time.time()-t0))
