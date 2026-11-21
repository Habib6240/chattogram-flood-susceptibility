"""Figure 2 — four Sentinel-1-derived (B03U urban-aware) flood-reference masks, v1.
Data QA first (coding, counts, CSV and lattice cross-checks), then a 2 x 2 map composite on one common extent/scale/symbology."""
import sys, os, numpy as np, pandas as pd, cv2, tifffile, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from geoutil import read_shp, utm46
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
UP='/mnt/user-data/uploads/'
EV=[('2019','BUET_B03U_E03_2019_Jul_FloodMask30m.tif','E03_2019_Jul'),('2021','BUET_B03U_E04_2021_Jul_FloodMask30m.tif','E04_2021_Jul'),('2023','BUET_B03U_E05_2023_Aug_FloodMask30m.tif','E05_2023_Aug'),('2026','BUET_B03U_E07_2026_Jul_FloodMask30m.tif','E07_2026_Jul')]
csv=pd.read_csv(UP+'BUET_B03U_Flood_Mask_Summary.csv').set_index('event_id')
npz=np.load(UP+'BUET_B04U_ML_Master_150m.npz'); lr,lc=npz['Raster_Row'].astype(int),npz['Raster_Column'].astype(int)
# ---------------- data QA
arrs={}; geo={}; stats=[]; xchk=[]; latt=[]
for y,f,eid in EV:
    with tifffile.TiffFile(UP+f) as t:
        p=t.pages[0]; tg={tag.code:tag.value for tag in p.tags.values()}; gk=list(tg[34735]); epsg=[gk[i+3] for i in range(4,len(gk),4) if gk[i]==3072][0]
        geo[y]=dict(shape=p.shape,scale=tuple(tg[33550][:2]),tie=tuple(round(v,3) for v in tg[33922][3:5]),epsg=epsg,nodata=tg.get(42113),dtype=str(p.dtype))
    a=cv2.imread(UP+f,cv2.IMREAD_UNCHANGED); arrs[y]=a
    u,c=np.unique(a,return_counts=True); cnt=dict(zip(u.tolist(),c.tolist())); assert set(cnt)<= {0,1,255} and geo[y]['nodata']=='255'
    n1,n0,nn=cnt.get(1,0),cnt.get(0,0),cnt.get(255,0); nv=n1+n0
    stats.append(dict(Event=y,File=f,CRS=f'EPSG:{epsg}',Rows=a.shape[0],Cols=a.shape[1],Pixel_m=geo[y]['scale'][0],Origin_x=geo[y]['tie'][0],Origin_y=geo[y]['tie'][1],Extent_xmin=geo[y]['tie'][0],Extent_xmax=geo[y]['tie'][0]+a.shape[1]*30,Extent_ymin=geo[y]['tie'][1]-a.shape[0]*30,Extent_ymax=geo[y]['tie'][1],NoData=geo[y]['nodata'],Value_1_flood_cells=n1,Value_0_nonflood_cells=n0,Value_255_nodata_cells=nn,Valid_reference_cells=nv,Flood_fraction_of_valid_pct=round(100*n1/nv,4)))
    r=csv.loc[eid]; va=nv*900/1e6; fa=n1*900/1e6
    xchk.append(dict(Event=y,Label_type_CSV=r.label_type,Valid_area_km2_raster=round(va,3),Valid_area_km2_CSV=round(r.valid_label_area_km2,3),Valid_diff_km2=round(va-r.valid_label_area_km2,3),Valid_diff_pct=round(100*(va-r.valid_label_area_km2)/r.valid_label_area_km2,3),
        Flood_area_km2_raster=round(fa,3),Flood_area_km2_CSV_combined_B03U=round(r.combined_flood_area_km2,3),Flood_diff_km2=round(fa-r.combined_flood_area_km2,3),Flood_diff_pct=round(100*(fa-r.combined_flood_area_km2)/r.combined_flood_area_km2,3),
        Flood_area_km2_CSV_original_open_B03=round(r.original_open_flood_area_km2,3),Flood_fraction_pct_raster=round(100*n1/nv,3),Flood_fraction_pct_CSV=round(r.combined_flood_fraction_pct,3),Fraction_diff_pp=round(100*n1/nv-r.combined_flood_fraction_pct,3)))
    lab=npz['Flood_'+y]; v=(lab==0)|(lab==1); ra=a[lr,lc]
    latt.append(dict(Event=y,Lattice_valid_B04U=int(v.sum()),Lattice_valid_raster=int((ra!=255).sum()),Lattice_flood_B04U=int((lab==1).sum()),Lattice_flood_raster=int((ra==1).sum()),Label_mismatches=int((lab[v]!=ra[v]).sum())))
ref=geo['2019']
for y,g in geo.items(): assert g['shape']==ref['shape'] and g['scale']==ref['scale'] and g['tie']==ref['tie'] and g['epsg']==32646,(y,g)
pd.DataFrame(stats).to_csv('Figure_2_Event_Raster_Statistics_v2.csv',index=False); pd.DataFrame(xchk).to_csv('Figure_2_CSV_CrossCheck_v2.csv',index=False); pd.DataFrame(latt).to_csv('Figure_2_Lattice_CrossCheck_v2.csv',index=False)
meta=pd.DataFrame([('2019','E03_2019_Jul','2019-07-02','2019-07-14','Sentinel-1A','Ascending',41,12,'DURING_EVENT'),('2021','E04_2021_Jul','2021-07-17','2021-07-29','Sentinel-1A','Descending',77,12,'DURING_EVENT'),('2023','E05_2023_Aug','2023-07-31','2023-08-12','Sentinel-1A','Descending',77,12,'POST_PEAK_PERSISTENT_INUNDATION'),('2026','E07_2026_Jul','2026-06-28','2026-07-10','Sentinel-1D','Descending',77,12,'DURING_EVENT')],columns=['Event','Event_ID','Pre_event_date','Post_event_observation_date','Platform','Pass','Relative_orbit','Interval_days','Observation_status'])
for _,m in meta.iterrows():
    r=csv.loc[m.Event_ID]; assert r.pre_date==m.Pre_event_date and r.observation_date==m.Post_event_observation_date and r.platform==m.Platform[-1] and r.orbit_pass.title()==m.Pass and int(r.relative_orbit)==m.Relative_orbit and int(r.gap_days)==12 and r.label_type==m.Observation_status
meta.to_csv('Figure_2_Event_Metadata_v2.csv',index=False)
print(pd.DataFrame(stats)[['Event','Valid_reference_cells','Value_1_flood_cells','Flood_fraction_of_valid_pct']].to_string(index=False)); print(pd.DataFrame(xchk)[['Event','Valid_diff_km2','Valid_diff_pct','Flood_diff_km2','Flood_diff_pct','Fraction_diff_pp']].to_string(index=False)); print(pd.DataFrame(latt).to_string(index=False))
# ---------------- figure
H,W=ref['shape']; x0,y0=ref['tie']; extent=(x0,x0+W*30,y0-H*30,y0); ew=extent[1]-extent[0]; eh=extent[3]-extent[2]
aoi=read_shp('AOI_Boundary/AOI_Boundary.shp')[0]; verts=[];codes=[]
for r in aoi:
    v=np.column_stack(utm46(r[:,0],r[:,1])); verts+=list(v)+[v[0]]; codes+=[Path.MOVETO]+[Path.LINETO]*(len(v)-1)+[Path.CLOSEPOLY]
aoi_path=Path(np.array(verts),codes)
BLUE=np.array([0x1F,0x78,0xB4])/255; GREY=np.array([0xE6,0xE6,0xE6])/255
def rgb(a):
    img=np.ones(a.shape+(3,),np.float32); img[a==0]=GREY; img[a==1]=BLUE
    return cv2.resize(img,(W//2,H//2),interpolation=cv2.INTER_AREA)
FW,FH=7.0,10.6; wf=0.40; hf=wf*FW/(ew/eh)/FH
cols=[0.055,0.545]; blockh=hf+0.056; rows_top=[0.992-0.056,0.992-0.056-blockh]   # map bottom y of each row
fig=plt.figure(figsize=(FW,FH),facecolor='white')
titles={'2019':('(a) 2019 flood-reference observation','Post-event Sentinel-1: 14 Jul 2019',''),'2021':('(b) 2021 flood-reference observation','Post-event Sentinel-1: 29 Jul 2021',''),'2023':('(c) 2023 post-peak flood-reference observation','Post-event Sentinel-1: 12 Aug 2023','post-peak persistent inundation'),'2026':('(d) 2026 flood-reference observation','Post-event Sentinel-1: 10 Jul 2026','')}
pos={'2019':(0,0),'2021':(1,0),'2023':(0,1),'2026':(1,1)}
for y,_,_ in EV:
    ci,ri=pos[y]; ybot=rows_top[ri]-hf; ax=fig.add_axes([cols[ci],ybot,wf,hf])
    ax.imshow(rgb(arrs[y]),extent=extent,interpolation='antialiased',origin='upper'); ax.add_patch(PathPatch(aoi_path,facecolor='none',edgecolor='#2b2b2b',linewidth=0.9))
    ax.set_xlim(extent[0],extent[1]); ax.set_ylim(extent[2],extent[3]); ax.set_aspect('equal'); ax.axis('off')
    t,s1,s2=titles[y]; ytop=ybot+hf
    fig.text(cols[ci],ytop+0.044,t,fontsize=8.5,fontweight='bold',va='bottom',ha='left'); fig.text(cols[ci],ytop+0.027,s1,fontsize=7.8,color='#444444',va='bottom',ha='left')
    if s2: fig.text(cols[ci],ytop+0.012,s2,fontsize=7.8,color='#444444',va='bottom',ha='left',style='italic')
# shared legend + scale bar + north arrow (bottom strip)
yl=0.046
items=[(BLUE,'Sentinel-1-derived\nflood-reference observation','none'),(GREY,'Valid non-flood reference','none'),((1,1,1),'Outside event-reference\ndomain / NoData','#888888')]
xs=[0.055,0.385,0.66]
for (c,t,ec),x in zip(items,xs):
    fig.patches.append(Rectangle((x,yl-0.007),0.026,0.016,transform=fig.transFigure,facecolor=c,edgecolor=ec if ec!='none' else c,linewidth=0.7)); fig.text(x+0.034,yl+0.001,t,fontsize=7.8,va='center',ha='left',linespacing=1.15)
# scale bar (20 km, to map scale) and north arrow
sbl=wf*20000/ew; sx=0.055; sy=0.0155
fig.add_artist(plt.Line2D([sx,sx+sbl],[sy,sy],color='#111111',lw=1.8,solid_capstyle='butt',transform=fig.transFigure))
for xx in (sx,sx+sbl/2,sx+sbl): fig.add_artist(plt.Line2D([xx,xx],[sy-0.003,sy+0.003],color='#111111',lw=1.0,transform=fig.transFigure))
fig.text(sx+sbl+0.012,sy,'20 km (all panels)',fontsize=7.8,va='center',ha='left')
fig.text(0.37,sy,'White areas (including those inside the AOI outline) lie outside the event-specific\nreference domain and do not indicate absence of flooding.',fontsize=6.9,color='#555555',va='center',ha='left',style='italic',linespacing=1.25)
fig.add_artist(matplotlib.patches.FancyArrowPatch((0.965,0.006),(0.965,0.030),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=11,lw=1.2,color='#111111')); fig.text(0.965,0.033,'N',fontsize=9,fontweight='bold',ha='center',va='bottom')
fig.savefig('Figure_2_FloodReference_Events_v2.png',dpi=300,facecolor='white'); fig.savefig('Figure_2_FloodReference_Events_v2.svg',facecolor='white'); plt.close(fig); print('saved')
