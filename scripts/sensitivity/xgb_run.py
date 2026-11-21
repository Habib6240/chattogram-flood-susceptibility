"""Original-implementation XGBoost re-run of the archived B06U / B07U designs with a chosen Distance_River vector.
Identical to scripts/ml/B06U*.py and B07U*.py: features, ColumnTransformer (passthrough numeric + OHE LULC/HSG), folds, labels,
XGBClassifier hyperparameters (n_estimators 600, lr 0.05, depth 5, min_child_weight 2, subsample 0.8, colsample_bytree 0.8,
lambda 1, alpha 0, binary:logistic, logloss, scale_pos_weight = neg/pos of the training set, hist, n_jobs -1, random_state 42).
Usage: xgb_run.py VARIANT DESIGN   (env SEED overrides random_state; TAG suffix)"""
import sys, os, time, numpy as np, xgboost
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
variant, design = sys.argv[1], sys.argv[2]
SEED=int(os.environ.get('SEED','42')); TAG=os.environ.get('TAG','')
D = np.load('/tmp/orig/BUET_B04U_ML_Master_150m.npz')
L = np.load('/home/claude/release/chattogram-flood-susceptibility-data-v1.0.0/06_oof_predictions/BUET_B07U_LOEO_OutOfFold_Predictions.npz')
assert np.array_equal(L['Raster_Row'], D['Raster_Row']) and np.array_equal(L['Raster_Column'], D['Raster_Column'])
FOLD = L['Spatial_Fold'].astype(np.int64)
dist = np.load(f'/home/claude/final/sens/dist_{variant}.npy')
names = ["Elevation","Slope","Aspect_sin","Aspect_cos","Aspect_flat","Profile_Curvature","Tangential_Curvature","TWI","Drainage_Density","Rainfall","LULC","NDVI","HSG","Distance_River","Distance_Road"]
X = np.column_stack([dist if n == "Distance_River" else D[n] for n in names]).astype(np.float32)
NUM=[0,1,2,3,4,5,6,7,8,9,11,13,14]; CAT=[10,12]
def pre(): return ColumnTransformer([("numeric","passthrough",NUM),("categorical",OneHotEncoder(handle_unknown="ignore",sparse_output=False,dtype=np.float32),CAT)],remainder="drop",sparse_threshold=0.0)
def fit_predict(Xtr,ytr,Xte):
    spw=(ytr==0).sum()/(ytr==1).sum()
    m=XGBClassifier(n_estimators=600,learning_rate=0.05,max_depth=5,min_child_weight=2,subsample=0.80,colsample_bytree=0.80,reg_lambda=1.0,reg_alpha=0.0,objective="binary:logistic",eval_metric="logloss",scale_pos_weight=spw,tree_method="hist",n_jobs=-1,random_state=SEED)
    m.fit(Xtr,ytr); return m.predict_proba(Xte)[:,1].astype(np.float32)
EV=["2019","2021","2023","2026"]; LAB={e:D[f"Flood_{e}"].astype(np.int8) for e in EV}
t0=time.time(); out={}
print('xgboost',xgboost.__version__,'variant',variant,design,'seed',SEED,flush=True)
if design=='B06U':
    y=D["Composite_AnyFlood_All4Valid"].astype(np.int8); valid=y>=0; oof=np.full(len(y),np.nan,np.float32)
    for f in range(1,6):
        tr=np.where(valid&(FOLD!=f))[0]; te=np.where(valid&(FOLD==f))[0]
        p=pre(); Xtr=p.fit_transform(X[tr]).astype(np.float32); Xte=p.transform(X[te]).astype(np.float32)
        oof[te]=fit_predict(Xtr,y[tr],Xte); print('fold',f,'%.0fs'%(time.time()-t0),flush=True)
    out['XGB_oof']=oof
else:
    for e in EV: out[f'XGB_{e}']=np.full(len(FOLD),np.nan,np.float32)
    for held in EV:
        tm=np.column_stack([LAB[e] for e in EV if e!=held]); vt=np.all(tm>=0,axis=1)
        tt=np.full(len(FOLD),-1,np.int8); tt[vt]=np.any(tm[vt]==1,axis=1).astype(np.int8); hv=LAB[held]>=0
        for f in range(1,6):
            tr=np.where(vt&(FOLD!=f))[0]; te=np.where(hv&(FOLD==f))[0]
            assert not set(tr)&set(te) and (FOLD[tr]!=f).all() and (FOLD[te]==f).all()
            p=pre(); Xtr=p.fit_transform(X[tr]).astype(np.float32); Xte=p.transform(X[te]).astype(np.float32)
            out[f'XGB_{held}'][te]=fit_predict(Xtr,tt[tr],Xte); print(held,'fold',f,'%.0fs'%(time.time()-t0),flush=True)
np.savez_compressed(f'/home/claude/final/sens/xres_{design}_{variant}{TAG}.npz',**out); print('done %.0fs'%(time.time()-t0))
