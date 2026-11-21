import numpy as np, csv, json
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
T='/home/claude/release/chattogram-flood-susceptibility-data-v1.0.0/04_mcda_outputs/'
W='/home/claude/repo/chattogram-flood-susceptibility/figures/source/Figure_8_Source_Q1_FINAL/Figure_8_Source_Q1_FINAL/data/'
D=np.load('/tmp/orig/BUET_B04U_ML_Master_150m.npz')
r,c=D['Raster_Row'],D['Raster_Column']
names=["Rainfall","Elevation","Slope","Drainage_Density","Distance_River","TWI","LULC","HSG","NDVI","Distance_Road","Profile_Curvature","Tangential_Curvature","Aspect"]
def rw(p,col):
    m={x[0]:float(x[col]) for x in list(csv.reader(open(W+p)))[1:] if x and x[0] in names};return np.array([m[n] for n in names])
Wt={'AHP':rw('DU10_AHP_Weights_and_Consistency.csv',1),'FAHP':rw('DU11_FAHP_Weights.csv',8),'ANP':rw('DU12_ANP_Final_Weights.csv',1)}
ARCH={k:np.asarray(Image.open(T+f)).astype('float64')[r,c] for k,f in [('AHP','DU10_AHP_Susceptibility_Index_01.tif'),('FAHP','DU11_FAHP_Susceptibility_Index_01.tif'),('ANP','DU12_ANP_Susceptibility_Index_01.tif')]}
inc={'Rainfall','TWI','Drainage_Density'}
lu={10:.1,20:.3,30:.4,40:.7,50:1.,60:.6,90:1.,95:.8}; hs={3:.5,4:1.,13:.75,14:1.}
sc={'N':.2,'NE':.1,'E':.3,'SE':.5,'S':.75,'SW':1.,'W':.8,'NW':.7};dirs=['N','NE','E','SE','S','SW','W','NW']
def scores(dist, fixed_bounds=None):
    Z={}
    for n in names:
        if n in('LULC','HSG','Aspect'):continue
        x=dist if n=='Distance_River' else D[n].astype('float64')
        p5,p95=np.percentile(x,[5,95]) if not(fixed_bounds and n=='Distance_River') else fixed_bounds
        t=np.clip((x-p5)/(p95-p5),0,1);Z[n]=t if n in inc else 1-t
    Z['LULC']=np.array([lu[int(v)] for v in D['LULC']]);Z['HSG']=np.array([hs[int(v)] for v in D['HSG']])
    th=np.degrees(np.arctan2(D['Aspect_sin'],D['Aspect_cos']))%360
    idx=((th+22.5)//45).astype(int)%8;z=np.array([sc[dirs[i]] for i in idx]);Z['Aspect']=np.where(D['Aspect_flat']==1,1.0,z)
    return {m:sum(w*Z[n] for w,n in zip(Wt[m],names)) for m in Wt},Z
if __name__=='__main__':
    S0,Z0=scores(D['Distance_River'].astype('float64'))
    for m in Wt:
        d=S0[m]-ARCH[m];print(m,'reproduction vs archived raster: r=%.6f MAE=%.2e max=%.2e frac<1e-3=%.4f'%(np.corrcoef(S0[m],ARCH[m])[0,1],np.abs(d).mean(),np.abs(d).max(),np.mean(np.abs(d)<1e-3)))
