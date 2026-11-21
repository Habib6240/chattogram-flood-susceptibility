"""Figure 1 — study area, data context and analytical workflow (v1). See README for inputs."""
import sys, os, zipfile, numpy as np, pandas as pd, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyBboxPatch, Rectangle, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from geoutil import *
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
UP='/mnt/user-data/uploads/'; DATA='data/'
OUT_TAG='v3'
# ------------------------------------------------ data
def rings_path(rings,proj=True):
    verts=[];codes=[]
    for r in rings:
        if len(r)<3: continue
        v=np.column_stack(utm46(r[:,0],r[:,1])) if proj else r
        verts+=list(v)+[v[0]]; codes+=[Path.MOVETO]+[Path.LINETO]*(len(v)-1)+[Path.CLOSEPOLY]
    return Path(np.array(verts),codes)
aoi_rings=read_shp(DATA+'AOI_Boundary/AOI_Boundary.shp')[0]
aoi_utm=[np.column_stack(utm46(r[:,0],r[:,1])) for r in aoi_rings]
aoi_path=rings_path(aoi_rings)
adm0=read_shp(DATA+'BGD_adm0_National_Boundary/BGD_adm0.shp'); adm2=read_shp(DATA+'BGD_adm2_District_Boundary/BGD_adm2.shp'); dbf2=read_dbf(DATA+'BGD_adm2_District_Boundary/BGD_adm2.dbf')
rivers=read_gpkg_lines(UP+'05_River_HydroRIVERS_UPLAND25_UTM46N.gpkg')
allaoi=np.vstack(aoi_utm); ax0,ay0=allaoi.min(0); ax1,ay1=allaoi.max(0)
M=1000; xmin,xmax,ymin,ymax=ax0-M,ax1+M,ay0-M,ay1+M
# AOI overlap with districts (1 km grid of points inside AOI)
gx,gy=np.meshgrid(np.arange(ax0,ax1,1000.0),np.arange(ay0,ay1,1000.0)); pts=np.column_stack([gx.ravel(),gy.ravel()]); pts=pts[aoi_path.contains_points(pts)]
names=[d['NAME_2'] for d in dbf2]; overlap={}
for nm,rr in zip(names,adm2):
    if not rr: continue
    bb=np.vstack(rr); lo,la=bb.min(0),bb.max(0)
    for p in [rings_path([r]) for r in rr if len(r)>=3]:
        pass
dist_paths=[rings_path(rr) if rr else None for rr in adm2]
for nm,pp in zip(names,dist_paths):
    if pp is None: continue
    k=int(pp.contains_points(pts).sum())
    if k: overlap[nm]=overlap.get(nm,0)+k
tot=len(pts); ov=pd.DataFrame({'District_GADM_name':list(overlap),'Share_of_AOI_1km_points_pct':[round(100*v/tot,2) for v in overlap.values()]}).sort_values('Share_of_AOI_1km_points_pct',ascending=False); ov.to_csv('Figure_1_AOI_District_Overlap_v1.csv',index=False); print(ov.to_string(index=False))
# ------------------------------------------------ DEM: raw Copernicus GLO-30 (geographic) warped to UTM 46N window for hillshade
dem=cv2.imread(DATA+'output_hh.tif',cv2.IMREAD_UNCHANGED).astype(np.float32)
lon0,lat0,sc=91.37277776666669,23.015277788888884,0.0002777777777777778   # pixel-is-point tie of pixel (0,0)
step=60.0; xs=np.arange(xmin,xmax,step); ys=np.arange(ymax,ymin,-step); X,Y=np.meshgrid(xs,ys); lo,la=utm46_inv(X,Y)
mapx=((lo-lon0)/sc).astype(np.float32); mapy=((lat0-la)/sc).astype(np.float32)
z=cv2.remap(dem,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=np.nan)
inside_dem=(mapx>=0)&(mapx<=dem.shape[1]-1)&(mapy>=0)&(mapy<=dem.shape[0]-1); z[~inside_dem]=np.nan
sea=(np.nan_to_num(z,nan=1)==0)
zz=np.where(np.isnan(z),0,z); gy_,gx_=np.gradient(zz*2.0,step)   # 2x vertical exaggeration
slope=np.arctan(np.hypot(gx_,gy_)); aspect=np.arctan2(gy_,-gx_)
az=np.radians(315.0); alt=np.radians(40.0)
hs=np.cos(alt)*np.cos(slope)+np.sin(alt)*np.sin(slope)*np.cos(az-aspect+np.pi/2*0)
hs=np.clip((hs-hs.min())/(np.percentile(hs,99.5)-hs.min()),0,1)
def hexrgb(h): h=h.lstrip('#'); return np.array([int(h[i:i+2],16)/255 for i in (0,2,4)])
lowc,highc=hexrgb('#F6F4EC'),hexrgb('#D8D2BE'); t=np.clip(np.nan_to_num(z)/350.0,0,1)[...,None]
base=lowc*(1-t)+highc*t; rgb=base*(0.62+0.38*hs[...,None])
rgb[sea]=hexrgb('#DCE8F1'); rgb[np.isnan(z)]=1.0
# ------------------------------------------------ figure
FW,FH=11.8,6.7
fig=plt.figure(figsize=(FW,FH),facecolor='white')
axM=fig.add_axes([0.012,0.045,0.335,0.875])
axM.imshow(rgb,extent=(xmin,xmax,ymin,ymax),interpolation='bilinear',origin='upper',zorder=1)
# district boundaries (subtle), clipped to window
win=Path([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax],[xmin,ymin]])
for rr in adm2:
    for r in rr:
        if len(r)<3: continue
        v=np.column_stack(utm46(r[:,0],r[:,1]))
        if v[:,0].max()<xmin or v[:,0].min()>xmax or v[:,1].max()<ymin or v[:,1].min()>ymax: continue
        axM.plot(v[:,0],v[:,1],color='#8C8C8C',lw=0.55,zorder=3,solid_joinstyle='round')
# rivers: HydroRIVERS only, two-tier hierarchy (secondary drawn first, principal on top); DIVA/DCW lines evaluated but not used
SEC='#8FB8E0'; PRI='#1F4E8C'
for tier in ('sec','pri'):
    for lines,o,up in rivers:
        if (o>=5)!=(tier=='pri'): continue
        lw=(1.4 if o==5 else 1.7) if tier=='pri' else (0.45 if o<=2 else 0.7)
        for l in lines:
            if l[:,0].max()<xmin or l[:,0].min()>xmax or l[:,1].max()<ymin or l[:,1].min()>ymax: continue
            axM.plot(l[:,0],l[:,1],color=PRI if tier=='pri' else SEC,lw=lw,alpha=1.0 if tier=='pri' else 0.95,zorder=5 if tier=='pri' else 4,solid_capstyle='round')
# AOI
axM.add_patch(PathPatch(aoi_path,facecolor='none',edgecolor='#111111',linewidth=2.0,zorder=7))
axM.set_xlim(xmin,xmax); axM.set_ylim(ymin,ymax); axM.set_aspect('equal'); axM.axis('off')
# district labels (minimal): visible vertex centroid
lab={'Chittagong':'Chattogram','Khagrachari':'Khagrachhari'}
for nm,rr in zip(names,adm2):
    if nm not in lab or lab[nm] is None or not rr: continue
    v=np.vstack([np.column_stack(utm46(r[:,0],r[:,1])) for r in rr if len(r)>=3]); m=(v[:,0]>xmin)&(v[:,0]<xmax)&(v[:,1]>ymin)&(v[:,1]<ymax)
    if m.sum()<30: continue
    cx,cy=v[m].mean(0)
    if not (xmin+6000<cx<xmax-6000 and ymin+6000<cy<ymax-6000): continue
    axM.text(cx,cy,lab[nm],fontsize=7.5,color='#555555',style='italic',ha='center',va='center',zorder=7)
# Bay of Bengal label (sea cells exist in window)
sy,sx_=np.where(sea); 
if len(sx_)>2000:
    bx=xs[int(np.percentile(sx_,30))]; by=ys[int(np.percentile(sy,75))]; axM.text(bx,by,'Bay of\nBengal',fontsize=8,color='#4F7A99',style='italic',ha='center',va='center',zorder=7,linespacing=1.2)
# scale bar and north arrow
sbx=xmin+3500; sby=ymin+5000; axM.plot([sbx,sbx+20000],[sby,sby],color='#111111',lw=1.8,solid_capstyle='butt',zorder=9)
for xx in (sbx,sbx+10000,sbx+20000): axM.plot([xx,xx],[sby-1000,sby+1000],color='#111111',lw=1.0,zorder=9)
axM.text(sbx+10000,sby+2300,'20 km',ha='center',va='bottom',fontsize=8,zorder=9)
axM.annotate('',xy=(0.90,0.965),xytext=(0.90,0.885),xycoords='axes fraction',arrowprops=dict(arrowstyle='-|>',color='#111111',lw=1.3,mutation_scale=13),zorder=9); axM.text(0.90,0.972,'N',transform=axM.transAxes,ha='center',va='bottom',fontsize=10,fontweight='bold',zorder=9)
fig.text(0.012,0.945,'(a)',fontsize=12,fontweight='bold',va='bottom'); fig.text(0.037,0.945,'Study area and data context',fontsize=11.5,fontweight='bold',va='bottom')
# ---- locator inset: Bangladesh (equirectangular, cos 23.7 deg) with Chittagong district highlighted and AOI outlined
axI=fig.add_axes([0.355,0.50,0.185,0.415]); ck=np.cos(np.radians(23.7))
def sq(r): return np.column_stack([r[:,0]*ck,r[:,1]])
def thin(r,maxn=1800):
    k=max(1,len(r)//maxn); return r[::k]
bd=[thin(r) for rr in adm0 for r in rr if len(r)>=3 and (r[:,0].max()-r[:,0].min())*(r[:,1].max()-r[:,1].min())>0.004]
vs=[];cs=[]
for r in bd:
    v=sq(r); vs+=list(v)+[v[0]]; cs+=[Path.MOVETO]+[Path.LINETO]*(len(v)-1)+[Path.CLOSEPOLY]
axI.add_patch(PathPatch(Path(np.array(vs),cs),facecolor='#E4E4E4',edgecolor='#8a8a8a',linewidth=0.45))
ci=[i for i,n in enumerate(names) if n=='Chittagong'][0]
vs=[];cs=[]
for r in adm2[ci]:
    if len(r)<3: continue
    if (r[:,0].max()-r[:,0].min())*(r[:,1].max()-r[:,1].min())<0.004: continue
    v=sq(thin(r)); vs+=list(v)+[v[0]]; cs+=[Path.MOVETO]+[Path.LINETO]*(len(v)-1)+[Path.CLOSEPOLY]
axI.add_patch(PathPatch(Path(np.array(vs),cs),facecolor='#4A5F7A',edgecolor='#2b3a4d',linewidth=0.4))
for r in aoi_rings: axI.plot(r[:,0]*ck,r[:,1],color='#C0392B',lw=0.9)
axI.set_xlim(87.9*ck,92.8*ck); axI.set_ylim(20.6,26.75); axI.set_aspect('equal'); axI.set_xticks([]); axI.set_yticks([])
for s in axI.spines.values(): s.set_linewidth(0.7); s.set_color('#555555')
axI.text(0.03,0.97,'Bangladesh',transform=axI.transAxes,fontsize=8.5,fontweight='bold',va='top',ha='left')
axI.annotate('Chattogram\ndistrict',xy=(91.85*ck,22.25),xytext=(88.6*ck,21.55),fontsize=7.8,color='#2b3a4d',fontweight='bold',arrowprops=dict(arrowstyle='-',color='#2b3a4d',lw=0.7),ha='left',va='center')
# ---- legend (right of map)
axL=fig.add_axes([0.355,0.045,0.185,0.40]); axL.axis('off'); axL.set_xlim(0,1); axL.set_ylim(0,1)
axL.text(0,0.97,'Map layers',fontsize=9,fontweight='bold',va='top')
items=[('line','#111111',2.0,'Study-area boundary (AOI)'),('line','#8C8C8C',0.8,'District boundaries'),('line','#1F4E8C',2.0,'Principal rivers\n(HydroRIVERS, stream order ≥ 5)'),('line','#8FB8E0',0.9,'Secondary rivers\n(HydroRIVERS, stream order 1–4)'),('box','#D8D2BE',None,'Terrain: Copernicus GLO-30\nhillshade (light backdrop)'),('box','#DCE8F1',None,'Coastal water / sea\n(DEM-derived)'),('inset','#4A5F7A',None,'Chattogram district (inset)')]
y=0.86
for kind,c,lw,txt in items:
    if kind=='line': axL.plot([0,0.16],[y,y],color=c,lw=lw)
    elif kind=='box': axL.add_patch(Rectangle((0,y-0.028),0.16,0.056,facecolor=c,edgecolor='#999999',lw=0.5))
    else: axL.add_patch(Rectangle((0,y-0.028),0.16,0.056,facecolor=c,edgecolor='#2b3a4d',lw=0.5))
    axL.text(0.22,y,txt,fontsize=7.8,va='center',linespacing=1.15); y-=0.123 if '\n' in txt else 0.098
axL.plot([0,0.16],[y,y],color='#C0392B',lw=0.9); axL.text(0.22,y,'AOI outline in inset',fontsize=7.8,va='center')
# ------------------------------------------------ panel (b): workflow
axB=fig.add_axes([0.565,0.02,0.425,0.925]); axB.set_xlim(0,100); axB.set_ylim(0,100); axB.axis('off')
NAVY='#1F3A5F'; STEEL='#CFD8E3'; WARM='#F2E4CF'; WEDGE='#8A5A19'; GREYF='#EEF0F3'; GL='#7a8795'
def box(x,y,w,h,fc,ec,lw=1.0,z=2):
    axB.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.0',facecolor=fc,edgecolor=ec,linewidth=lw,zorder=z))
def T(x,y,t,fs=7.4,bold=False,c='#111111',style='normal',ha='center'):
    axB.text(x,y,t,fontsize=fs,fontweight='bold' if bold else 'normal',color=c,style=style,ha=ha,va='center',zorder=6,linespacing=1.2)
def arrow(x0,y0,x1,y1,c='#444444'):
    axB.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=9,lw=1.0,color=c,zorder=1,shrinkA=0,shrinkB=0))
fig.text(0.565,0.945,'(b)',fontsize=12,fontweight='bold',va='bottom'); fig.text(0.59,0.945,'Overall analytical workflow',fontsize=11.5,fontweight='bold',va='bottom')
# inputs
box(0,83.5,48.5,12,GREYF,GL); T(24.25,92.4,'13 conditioning factors',8.2,True); T(24.25,88.3,'terrain · hydrology · land cover · vegetation\nsoil · rainfall climatology · road access',7.2)
box(51.5,83.5,48.5,12,GREYF,GL); T(75.75,92.4,'Sentinel-1-derived flood-reference\nobservations',8.0,True); T(75.75,87.2,'4 archived events: 2019 · 2021 · 2023 · 2026',7.2)
axB.plot([24.25,24.25],[83.5,80.2],color='#444444',lw=1.0,zorder=1); axB.plot([75.75,75.75],[83.5,80.2],color='#444444',lw=1.0,zorder=1); axB.plot([24.25,75.75],[80.2,80.2],color='#444444',lw=1.0,zorder=1)
arrow(24.25,80.2,24.25,77.4); arrow(75.75,80.2,75.75,77.4)
# branch headers
box(0,66,48.5,11,STEEL,NAVY,lw=1.3); T(24.25,74.6,'Prescribed MCDA branch',8.6,True,NAVY); T(24.25,71.4,'AHP · FAHP · ANP',8.2); T(24.25,68.1,'prescribed criteria priorities; no flood-label training',6.2,c='#44546a',style='italic')
box(51.5,66,48.5,11,WARM,WEDGE,lw=1.3); T(75.75,74.6,'Learned ML branch',8.6,True,WEDGE); T(75.75,71.4,'Random Forest · XGBoost · ML consensus',7.9); T(75.75,68.1,'trained on flood-reference labels',6.8,c='#6b4a1d',style='italic')
# modules
mh=7.6; y1,y2,y3=56.6,47.6,38.6
def mod(x,y,txt,ec): box(x+3,y,45.5-3,mh,'white',ec,lw=0.9); T(x+3+(45.5-3)/2,y+mh/2,txt,7.2)
mod(0,y1,'2021 event-specific\nMCDA assessment',NAVY); mod(0,y2,'MCDA sensitivity\nand robustness',NAVY)
mod(51.5,y1,'5-fold spatial block\ncross-validation',WEDGE); mod(51.5,y2,'strict held-event and\nheld-geography evaluation',WEDGE); mod(51.5,y3,'XGBoost SHAP\nattribution',WEDGE)
axB.plot([1.2,1.2],[y2+mh/2,y1+mh/2],color=NAVY,lw=1.0,zorder=1); 
for yy in (y1,y2): axB.plot([1.2,3],[yy+mh/2,yy+mh/2],color=NAVY,lw=1.0,zorder=1)
axB.plot([52.7,52.7],[y3+mh/2,y1+mh/2],color=WEDGE,lw=1.0,zorder=1)
for yy in (y1,y2,y3): axB.plot([52.7,54.5],[yy+mh/2,yy+mh/2],color=WEDGE,lw=1.0,zorder=1)
axB.plot([1.2,1.2],[y1+mh/2,66.0],color=NAVY,lw=1.0,zorder=1); axB.plot([52.7,52.7],[y1+mh/2,66.0],color=WEDGE,lw=1.0,zorder=1)
# shared comparison
box(0,12.5,100,21,GREYF,GL,lw=1.2); T(50,31.2,'Shared comparison and integration',8.8,True)
cw=47.5; ch=7.4
for (x,y,t,b) in [(2,21.6,'Identical same-cohort event comparison\n(all six methods)',False),(50.5,21.6,'Full-domain spatial\nagreement analysis',False),(2,13.7,'Attribution contrast\n(MCDA priorities vs SHAP)',False),(50.5,13.7,'Integrated interpretation',True)]:
    box(x,y,cw,ch,'white',GL,lw=0.9); T(x+cw/2,y+ch/2,t,7.2 if not b else 7.6,b)
arrow(24.25,y2,24.25,33.6); arrow(75.75,y3,75.75,33.6)
# outputs
arrow(50,12.5,50,10.4)
ow=23.4; xs_=[0,25.5,51,76.6]; labs=['Susceptibility\nmaps','Comparative\nperformance evidence','Spatial\ncorrespondence','Complementary roles\nof MCDA and ML']
for x,t in zip(xs_,labs): box(x,1.2,ow,9,'white',NAVY,lw=1.1); T(x+ow/2,5.7,t,7.0)
fig.savefig(f'Figure_1_StudyArea_Workflow_{OUT_TAG}.png',dpi=300,facecolor='white'); fig.savefig(f'Figure_1_StudyArea_Workflow_{OUT_TAG}.svg',facecolor='white'); plt.close(fig); print('saved')
