import glob,numpy as np,pandas as pd,itertools,json,hashlib,tifffile
from PIL import Image
from scipy.stats import rankdata
Image.MAX_IMAGE_PIXELS=None
A=glob.glob('/mnt/user-data/attachments/*/')[0]
F={'mask':'b23da44e-08_Final','AHP':'8456c989-DU10','FAHP':'a676a929-DU11','ANP':'5a5b1b64-DU12','RF':'003c4b46-BUET_B09A_RF_Susc','XGBoost':'f285f228-BUET_B09A_XGB','ML_Consensus':'6865cff3-BUET_B09A_RF_XGB'}
meta=[];R={}
for k,p in F.items():
    f=glob.glob(A+p+'*')[0]
    with tifffile.TiffFile(f) as t:
        pg=t.pages[0];tg=pg.tags;g=lambda n:tg[n].value if n in tg else None
        meta.append(dict(layer=k,file=f.split('/')[-1],shape=str(pg.shape),dtype=str(pg.dtype),pixel=str(g('ModelPixelScaleTag')),tie=str(g('ModelTiepointTag')[3:5]),epsg=str((t.geotiff_metadata or {}).get('ProjectedCSTypeGeoKey')),nodata=str(g('GDAL_NODATA')),md5=hashlib.md5(open(f,'rb').read()).hexdigest()))
    R[k]=np.asarray(Image.open(f)).astype('float64')
m=R['mask']==1
for d in meta:
    a=R[d['layer']];d['n_nodata_-9999']=int((a==-9999).sum()) if d['layer']!='mask' else None
    if d['layer']!='mask':
        v=a[m];d['valid_in_mask']=int((v>-9999).sum());d['min_valid']=float(v.min());d['max_valid']=float(v.max());d['outside_mask_finite_valid']=int(((a>-9999)&~m).sum())
    else: d['mask_valid']=int(m.sum())
json.dump(meta,open('m1/out/six_meta.json','w'),default=str)
for d in meta:print(d)
names=['RF','XGBoost','ML_Consensus','AHP','FAHP','ANP']
V={k:R[k][m] for k in names}
rk={k:rankdata(v) for k,v in V.items()}
br={k:np.percentile(v,[20,40,60,80]) for k,v in V.items()}
cl={k:1+sum((V[k]>=b).astype(np.int8) for b in br[k]) for k in names}
top={k:V[k]>=br[k][3] for k in names}
rows=[]
for a,b in itertools.combinations(names,2):
    pr=np.corrcoef(V[a],V[b])[0,1];sp=np.corrcoef(rk[a],rk[b])[0,1]
    ex=(cl[a]==cl[b]).mean();w1=(abs(cl[a]-cl[b])<=1).mean()
    i=(top[a]&top[b]).sum();u=(top[a]|top[b]).sum()
    rows.append(dict(m1=a,m2=b,pearson=pr,spearman=sp,exact=ex*100,within1=w1*100,jaccard=i/u,inter=int(i),union=int(u)))
df=pd.DataFrame(rows);df.to_csv('m1/out/six_method_recomputed.csv',index=False)
print(df.round(6).to_string())
print({k:int(top[k].sum()) for k in names})
