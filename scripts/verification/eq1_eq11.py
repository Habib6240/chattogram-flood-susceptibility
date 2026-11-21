import numpy as np,pandas as pd,glob
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
A=glob.glob('/mnt/user-data/attachments/*/')[0];g=lambda p:glob.glob(A+p)[0]
d=pd.read_csv('/mnt/user-data/uploads/BUET_B04U_ML_Master_150m.csv')
r,c=d.Raster_Row.values,d.Raster_Column.values
R={k:np.asarray(Image.open(g(p))).astype('float64')[r,c] for k,p in [('AHP','*8456c989*'),('FAHP','*a676a929*'),('ANP','*5a5b1b64*')]}
W={'AHP':[0.23110902422921192,0.181866493121561,0.1415113769013037,0.10949987007658181,0.08470119847872382,0.06534024956970849,0.05034656100520957,0.03883585884298039,0.02989251359339813,0.023197602396559266,0.017949904924371228,0.014165332115014436,0.01158401474537608]}
import csv
names=["Rainfall","Elevation","Slope","Drainage_Density","Distance_River","TWI","LULC","HSG","NDVI","Distance_Road","Profile_Curvature","Tangential_Curvature","Aspect"]
def rw(p,col):
    rows=[x for x in list(csv.reader(open(g(p))))[1:] if len(x)>col and x[0] in names];m={x[0]:float(x[col]) for x in rows};return [m[n] for n in names]
W['AHP']=rw('*880b7a79*',1);W['FAHP']=rw('*39922c1e*',8);W['ANP']=rw('*d322b720*',1)
print({k:sum(v) for k,v in W.items()})
inc={'Rainfall','TWI','Drainage_Density'}
col={'Rainfall':'Rainfall','Elevation':'Elevation','Slope':'Slope','Drainage_Density':'Drainage_Density','Distance_River':'Distance_River','TWI':'TWI','NDVI':'NDVI','Distance_Road':'Distance_Road','Profile_Curvature':'Profile_Curvature','Tangential_Curvature':'Tangential_Curvature'}
Z={}
bounds={}
for n,cn in col.items():
    x=d[cn].values;p5,p95=np.percentile(x,[5,95]);bounds[n]=(p5,p95)
    t=np.clip((x-p5)/(p95-p5),0,1);Z[n]=t if n in inc else 1-t
lu={10:.1,20:.3,30:.4,40:.7,50:1.,60:.6,90:1.,95:.8}
Z['LULC']=d.LULC.map(lu).values
hs={3:.5,4:1.,13:.75,14:1.}
print('HSG codes',sorted(d.HSG.unique()),'LULC',sorted(d.LULC.unique()))
Z['HSG']=d.HSG.map(hs).values
print('nan LULC',np.isnan(Z['LULC']).sum(),'HSG',np.isnan(Z['HSG']).sum())
sc={'N':.2,'NE':.1,'E':.3,'SE':.5,'S':.75,'SW':1.,'W':.8,'NW':.7}
dirs=['N','NE','E','SE','S','SW','W','NW']
best=None
for conv in ['compass','ccw_from_east']:
    th=np.degrees(np.arctan2(d.Aspect_sin.values,d.Aspect_cos.values))%360   # aspect angle as given by sin/cos
    if conv=='compass': az=th
    else: az=(90-th)%360
    idx=((az+22.5)//45).astype(int)%8
    z=np.array([sc[dirs[i]] for i in idx]);z=np.where(d.Aspect_flat.values==1,1.0,z)
    Z['Aspect_'+conv]=z
for conv in ['compass','ccw_from_east']:
    Z['Aspect']=Z['Aspect_'+conv]
    print('--',conv)
    for m in ['AHP','FAHP','ANP']:
        S=sum(w*Z[n] for w,n in zip(W[m],names))
        dif=S-R[m]
        print(m,'r',np.corrcoef(S,R[m])[0,1],'MAE',np.abs(dif).mean(),'max',np.abs(dif).max(),'frac<1e-3',(np.abs(dif)<1e-3).mean(),'frac<1e-6',(np.abs(dif)<1e-6).mean())
print({k:tuple(np.round(v,4)) for k,v in bounds.items()})
