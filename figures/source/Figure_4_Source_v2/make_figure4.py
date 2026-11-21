"""Figure 4 — strict evaluation design (v2: wording, subtitle, panel (a) size and grid-line weight refined; science unchanged).
(a) 5-fold spatial block partition reconstructed from BUET_B06U_5km_Block_Spatial_Folds.csv, clipped to AOI_Boundary;
(b) schematic of the B07U strict held-event and held-geography design.
Inputs (edit UP if needed): fold CSV, B04U master npz (block grid verification only), AOI_Boundary.zip, B07U script (design-text check).
Outputs: Figure_4_Strict_Evaluation_Design_v2.png/.svg, Figure_4_Block_Geometry_v2.csv
"""
import io, math, struct, zipfile, re, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle, FancyBboxPatch, Polygon
from matplotlib.collections import PatchCollection
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
UP='/mnt/user-data/uploads/'
FOLD_CSV=UP+'BUET_B06U_5km_Block_Spatial_Folds.csv'; NPZ=UP+'BUET_B04U_ML_Master_150m.npz'; AOI_ZIP=UP+'AOI_Boundary.zip'; B07U=UP+'B07U_BUET_Strict_SpatioTemporal_LOEO_UrbanAware.py'
# ---------------- data
folds=pd.read_csv(FOLD_CSV); assert len(folds)==233 and folds.Block_5km.is_unique and sorted(folds.Spatial_Fold.unique())==[1,2,3,4,5]
D=np.load(NPZ); X,Y,rr,cc,b5=D['X'],D['Y'],D['Raster_Row'].astype(int),D['Raster_Column'].astype(int),D['Block_5km'].astype(int)
ox=np.unique(np.round(X-30*cc,3)); oy=np.unique(np.round(Y+30*rr,3)); assert len(ox)==1 and len(oy)==1
left0=ox[0]-15.0; top0=oy[0]+15.0                      # outer pixel edges of the 30 m grid (cell centres at ox, oy)
bx=(cc*30)//5000; by=(rr*30)//5000
g=pd.DataFrame({'Block_5km':b5,'Grid_bx':bx,'Grid_by':by}).drop_duplicates()
assert len(g)==233 and g.Block_5km.is_unique and set(g.Block_5km)==set(folds.Block_5km)
blk=folds.merge(g,on='Block_5km')
blk['Left']=left0+blk.Grid_bx*5000.0; blk['Right']=blk.Left+5000.0; blk['Top']=top0-blk.Grid_by*5000.0; blk['Bottom']=blk.Top-5000.0
inside=((blk.Centroid_X>=blk.Left)&(blk.Centroid_X<=blk.Right)&(blk.Centroid_Y>=blk.Bottom)&(blk.Centroid_Y<=blk.Top)); assert inside.all()
blk[['Block_5km','Spatial_Fold','Centroid_X','Centroid_Y','Grid_bx','Grid_by','Left','Right','Bottom','Top']].round(3).to_csv('Figure_4_Block_Geometry_v2.csv',index=False)
# ---------------- AOI (EPSG:4326 -> EPSG:32646, Kruger series)
def utm46(lon,lat):
    a=6378137.0; f=1/298.257223563; k0=0.9996; lon0=math.radians(93.0); n=f/(2-f)
    A=a/(1+n)*(1+n**2/4+n**4/64)
    al=[n/2-2*n**2/3+5*n**3/16+41*n**4/180, 13*n**2/48-3*n**3/5+557*n**4/1440, 61*n**3/240-103*n**4/140, 49561*n**4/161280]
    phi=np.radians(lat); lam=np.radians(lon)-lon0; c=2*math.sqrt(n)/(1+n)
    t=np.sinh(np.arctanh(np.sin(phi))-c*np.arctanh(c*np.sin(phi)))
    xi=np.arctan2(t,np.cos(lam)); eta=np.arctanh(np.sin(lam)/np.sqrt(1+t**2))
    xs=xi+sum(al[j]*np.sin(2*(j+1)*xi)*np.cosh(2*(j+1)*eta) for j in range(4)); es=eta+sum(al[j]*np.cos(2*(j+1)*xi)*np.sinh(2*(j+1)*eta) for j in range(4))
    return 500000+k0*A*es, k0*A*xs
z=zipfile.ZipFile(AOI_ZIP); b=z.read('AOI_Boundary.shp'); assert struct.unpack('<i',b[32:36])[0]==5
pos=100; rings=[]
while pos<len(b):
    clen=struct.unpack('>i',b[pos+4:pos+8])[0]*2; c0=pos+8; pos=c0+clen
    st=struct.unpack('<i',b[c0:c0+4])[0]
    if st!=5: continue
    npart,npts=struct.unpack('<2i',b[c0+36:c0+44]); parts=list(struct.unpack('<%di'%npart,b[c0+44:c0+44+4*npart])); p0=c0+44+4*npart
    pts=np.frombuffer(b[p0:p0+16*npts],dtype='<f8').reshape(-1,2); parts.append(npts)
    for i in range(npart): rings.append(pts[parts[i]:parts[i+1]])
verts=[];codes=[];proj=[]
for r in rings:
    ex,ny=utm46(r[:,0],r[:,1]); proj.append(np.column_stack([ex,ny]))
    v=np.column_stack([ex,ny]); verts+=list(v)+[v[0]]; codes+=[Path.MOVETO]+[Path.LINETO]*(len(v)-1)+[Path.CLOSEPOLY]
aoi=Path(np.array(verts),codes)
allp=np.vstack(proj); print('AOI rings',len(rings),'UTM bbox',allp.min(0).round(0),allp.max(0).round(0))
frac=aoi.contains_points(np.column_stack([X,Y])).mean(); print('sample points inside AOI: %.4f'%frac)
area=sum(0.5*abs(np.dot(p[:,0],np.roll(p[:,1],-1))-np.dot(p[:,1],np.roll(p[:,0],-1))) for p in proj)/1e6; print('AOI polygon area km2 (shoelace, outer+holes unsigned): %.1f'%area)
# ---------------- B07U design-text check
src=open(B07U,encoding='utf-8').read()
assert 'EVENTS = ["2019", "2021", "2023", "2026"]' in src and 'range(1, 6)' in src and '4 held-out events x 5 reused B06 spatial folds = 20 scenarios' in src
assert 'train_mask = valid_train_label & (SPATIAL_FOLD != fold)' in src and 'test_mask = held_valid & (SPATIAL_FOLD == fold)' in src
# ---------------- figure
FC={1:'#0072B2',2:'#E69F00',3:'#009E73',4:'#D55E00',5:'#CC79A7'}   # Okabe-Ito
NAVY='#1F3A5F'; STEEL='#CFD8E3'; GREY='#7a7a7a'
fig=plt.figure(figsize=(11.2,5.9),facecolor='white')
axA=fig.add_axes([0.012,0.115,0.395,0.78]); axB=fig.add_axes([0.40,0.02,0.59,0.93])
# ---- panel (a)
rects=[Rectangle((r.Left,r.Bottom),5000,5000) for r in blk.itertuples()]
pc=PatchCollection(rects,facecolor=[FC[f] for f in blk.Spatial_Fold],edgecolor='white',linewidth=0.35); axA.add_collection(pc)
clip=PathPatch(aoi,facecolor='none',edgecolor='none',transform=axA.transData); axA.add_patch(clip); pc.set_clip_path(clip)
axA.add_patch(PathPatch(aoi,facecolor='none',edgecolor='#1a1a1a',linewidth=0.9))
x0,y0=allp.min(0); x1,y1=allp.max(0); pad=1500
axA.set_xlim(x0-pad,x1+pad+6000); axA.set_ylim(y0-pad-6500,y1+pad); axA.set_aspect('equal'); axA.axis('off')
# scale bar (20 km)
sx=x0+2000; sy=y0-pad-2000
axA.plot([sx,sx+20000],[sy,sy],color='#1a1a1a',lw=1.6,solid_capstyle='butt'); 
for xx in (sx,sx+10000,sx+20000): axA.plot([xx,xx],[sy-900,sy+900],color='#1a1a1a',lw=1.0)
axA.text(sx+10000,sy-2600,'20 km',ha='center',va='top',fontsize=8)
# north arrow
axA.annotate('',xy=(0.86,0.93),xytext=(0.86,0.83),xycoords='axes fraction',arrowprops=dict(arrowstyle='-|>',color='#1a1a1a',lw=1.3,mutation_scale=13)); axA.text(0.86,0.945,'N',transform=axA.transAxes,ha='center',va='bottom',fontsize=10,fontweight='bold')
handles=[Rectangle((0,0),1,1,facecolor=FC[k],edgecolor='none') for k in range(1,6)]
axA.legend(handles,[f'Fold {k}' for k in range(1,6)],ncol=5,loc='upper center',bbox_to_anchor=(0.5,0.0),frameon=False,fontsize=8.5,handlelength=1.0,handleheight=1.0,columnspacing=0.9,handletextpad=0.4)
fig.text(0.02,0.935,'(a)',fontsize=12,fontweight='bold',va='bottom'); fig.text(0.052,0.935,'Spatial 5-fold block partition',fontsize=11,fontweight='bold',va='bottom')
fig.text(0.052,0.905,'5 km blocks clipped to the study-area boundary',fontsize=8.5,color='#444444',va='bottom')
# ---- panel (b)
axB.set_xlim(0,100); axB.set_ylim(0,100); axB.axis('off')
def box(x,y,w,h,fc,ec='#555555',lw=0.9,hatch=None,txt=None,tc='#111111',fs=9,bold=False,z=2):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.9',facecolor=fc,edgecolor=ec,linewidth=lw,hatch=hatch,zorder=z); axB.add_patch(p)
    if txt: axB.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color=tc,fontweight='bold' if bold else 'normal',zorder=z+1,linespacing=1.25)
axB.text(0,99,'(b)',fontsize=12,fontweight='bold',va='top'); axB.text(5.2,99,'Strict held-event and held-geography evaluation (B07U)',fontsize=11,fontweight='bold',va='top')
axB.text(5.2,94.4,'Example split: held event 2021, held geography Fold 3 (illustrative example only)',fontsize=8.5,color='#444444',va='top')
# events strip
axB.text(0,86.5,'Event dimension',fontsize=9,fontweight='bold',va='center',color='#222222')
for i,e in enumerate(['2019','2021','2023','2026']):
    held=(e=='2021'); box(1+i*24.5,76,22.5,8,NAVY if held else STEEL,ec=NAVY,txt=e,tc='white' if held else '#111111',fs=10,bold=held)
axB.text(1+1*24.5+11.25,74.0,'held event (test)',ha='center',va='top',fontsize=8,color=NAVY,fontweight='bold')
axB.text(1+3*24.5+22.5,74.0,'other 3 events (training)',ha='right',va='top',fontsize=8,color='#444444')
# folds strip
axB.text(0,66.0,'Geography dimension (5 km blocks grouped into 5 spatial folds)',fontsize=9,fontweight='bold',va='center',color='#222222')
for k in range(1,6):
    held=(k==3); box(1+(k-1)*19.6,55.5,18,7.5,FC[k],ec=NAVY if held else 'white',lw=2.6 if held else 0.9,txt=f'Fold {k}',tc='white' if k in (1,3,4) else '#111111',fs=9.5,bold=held,z=3 if held else 2)
axB.text(1+2*19.6+9,53.4,'held fold (test geography)',ha='center',va='top',fontsize=8,color=NAVY,fontweight='bold')
axB.text(99,53.4,'other folds (training)',ha='right',va='top',fontsize=8,color='#444444')
# matrix
mx,my,cw,ch=26,20,33,11.5
axB.text(mx+cw*0.5,my+2*ch+1.6,'Blocks outside\nheld fold',ha='center',va='bottom',fontsize=8.5,color='#222222',linespacing=1.15)
axB.text(mx+cw*1.5+1,my+2*ch+1.6,'Blocks inside\nheld fold',ha='center',va='bottom',fontsize=8.5,color='#222222',linespacing=1.15)
axB.text(mx-1.5,my+1.5*ch,'Other\n3 events',ha='right',va='center',fontsize=8.5,color='#222222',linespacing=1.15)
axB.text(mx-1.5,my+0.5*ch,'Held\nevent',ha='right',va='center',fontsize=8.5,color='#222222',linespacing=1.15)
box(mx,my+ch+0.6,cw,ch-1.2,STEEL,ec=NAVY,lw=1.6,txt='TRAINING\nlabels: other 3 events\nblocks: outside held fold',fs=8.5)
box(mx+cw+1,my+0.6,cw,ch-1.2,NAVY,ec=NAVY,lw=1.6,txt='TEST\nlabel: held event\nblocks: inside held fold',tc='white',fs=8.5)
box(mx+cw+1,my+ch+0.6,cw,ch-1.2,'white',ec='#bbbbbb',hatch='///',txt='not used',tc='#555555',fs=8.5)
box(mx,my+0.6,cw,ch-1.2,'white',ec='#bbbbbb',hatch='///',txt='not used',tc='#555555',fs=8.5)
axB.text(mx+cw+0.5,my-1.2,'Within each scenario, no 5 km block is shared between training and test,\nand the held-event label is excluded from training.',ha='center',va='top',fontsize=7.8,color='#444444',linespacing=1.3)
# summary
box(1,0.2,98,9.6,'#F3F5F8',ec='#9aa7b8',lw=0.8)
axB.text(50,6.7,'4 held-out events × 5 spatial folds = 20 event–fold splits;',ha='center',va='center',fontsize=9.3,fontweight='bold',color=NAVY)
axB.text(50,3.3,'evaluated with RF and XGBoost = 40 model-specific evaluations.',ha='center',va='center',fontsize=9.3,fontweight='bold',color=NAVY)
fig.savefig('Figure_4_Strict_Evaluation_Design_v2.png',dpi=300,facecolor='white'); fig.savefig('Figure_4_Strict_Evaluation_Design_v2.svg',facecolor='white'); plt.close(fig)
print('QA: blocks',len(blk),'folds',sorted(blk.Spatial_Fold.unique()),'fold block counts',blk.Spatial_Fold.value_counts().sort_index().to_dict(),'| centroids inside reconstructed blocks:',bool(inside.all()))
