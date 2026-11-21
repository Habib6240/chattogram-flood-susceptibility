import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import cKDTree
from geoutil import *
UP='/mnt/user-data/uploads/'
# AOI
aoi_rings=read_shp('data/AOI_Boundary/AOI_Boundary.shp')[0]
aoi_utm=[np.column_stack(utm46(r[:,0],r[:,1])) for r in aoi_rings]
verts=[];codes=[]
for v in aoi_utm: verts+=list(v)+[v[0]]; codes+=[Path.MOVETO]+[Path.LINETO]*(len(v)-1)+[Path.CLOSEPOLY]
aoi=Path(np.array(verts),codes); allv=np.vstack(aoi_utm); xmin,ymin=allv.min(0)-1000; xmax,ymax=allv.max(0)+1000
# DIVA lines
diva_recs=read_shp('data/diva/BGD_water_lines_dcw.shp'); diva_att=read_dbf('data/diva/BGD_water_lines_dcw.dbf')
D=[]
for i,(rr,a) in enumerate(zip(diva_recs,diva_att)):
    for r in rr:
        if len(r)<2: continue
        D.append((i,np.column_stack(utm46(r[:,0],r[:,1])),a))
# HydroRIVERS
H=read_gpkg_lines(UP+'05_River_HydroRIVERS_UPLAND25_UTM46N.gpkg')
def densify(l,step=100.0):
    out=[l[0]]
    for a,b in zip(l[:-1],l[1:]):
        n=max(1,int(np.hypot(*(b-a))//step)); out+= [a+(b-a)*k/n for k in range(1,n+1)]
    return np.array(out)
def inwin(l): return (l[:,0].max()>=xmin) & (l[:,0].min()<=xmax) & (l[:,1].max()>=ymin) & (l[:,1].min()<=ymax)
Dw=[(i,l,a) for i,l,a in D if inwin(l)]; Hw=[(lines,o,u) for lines,o,u in H if any(inwin(l) for l in lines)]
print('DIVA parts total',len(D),'in window',len(Dw),'| HydroRIVERS features total',len(H),'in window',len(Hw))
# densified points
hp=[];hmeta=[]
for fi,(lines,o,u) in enumerate(Hw):
    for l in lines:
        d=densify(l); hp.append(d); hmeta+= [(fi,o,u)]*len(d)
hp=np.vstack(hp); hmeta=np.array(hmeta); tH=cKDTree(hp)
dp=[];dmeta=[]
for k,(i,l,a) in enumerate(Dw):
    d=densify(l); dp.append(d); dmeta+= [(k,i)]*len(d)
dp=np.vstack(dp); dmeta=np.array(dmeta); tD=cKDTree(dp)
inwindow_d=(dp[:,0]>=xmin)&(dp[:,0]<=xmax)&(dp[:,1]>=ymin)&(dp[:,1]<=ymax); inaoi_d=aoi.contains_points(dp)
dist_to_H=tH.query(dp)[0]
inwindow_h=(hp[:,0]>=xmin)&(hp[:,0]<=xmax)&(hp[:,1]>=ymin)&(hp[:,1]<=ymax); inaoi_h=aoi.contains_points(hp)
dist_to_D=tD.query(hp)[0]
rows=[]
for tol in (500,1000,2000):
    for dom,md,mh in (('inside AOI',inaoi_d,inaoi_h),('map window (AOI+1 km)',inwindow_d,inwindow_h)):
        Ld=md.sum()*0.1; cov=(dist_to_H[md]<=tol).sum()*0.1; Lh=mh.sum()*0.1; covh=(dist_to_D[mh]<=tol).sum()*0.1
        rows.append((dom,tol,round(Ld,1),round(cov,1),round(100*cov/Ld,1),round(Ld-cov,1),round(Lh,1),round(covh,1),round(100*covh/Lh,1)))
tab=pd.DataFrame(rows,columns=['Domain','Tolerance_m','DIVA_length_km','DIVA_within_tol_of_HydroRIVERS_km','DIVA_covered_pct','DIVA_not_covered_km','HydroRIVERS_length_km','HydroRIVERS_within_tol_of_DIVA_km','HydroRIVERS_covered_pct']); print(tab.to_string(index=False)); tab.to_csv('Figure_1_River_Source_Comparison_v1.csv',index=False)
print('DIVA names in window:',pd.Series([a['NAM'] for i,l,a in Dw]).value_counts().head(10).to_dict())
np.savez('/tmp/riv.npz',dp=dp,dmeta=dmeta,dist=dist_to_H,inwin=inwindow_d,inaoi=inaoi_d,hp=hp,hmeta=hmeta,distD=dist_to_D)
# diagnostic plot
fig,ax=plt.subplots(figsize=(8,11))
for lines,o,u in Hw:
    for l in lines: ax.plot(l[:,0],l[:,1],color='#2c6fb7',lw=0.5+0.25*o,alpha=0.9)
for i,l,a in Dw: ax.plot(l[:,0],l[:,1],color='#d62728',lw=0.7,alpha=0.8)
ax.add_patch(plt.matplotlib.patches.PathPatch(aoi,facecolor='none',edgecolor='k',lw=1.2)); ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_aspect('equal'); ax.axis('off')
fig.savefig('rivers_diag.png',dpi=110,bbox_inches='tight')
