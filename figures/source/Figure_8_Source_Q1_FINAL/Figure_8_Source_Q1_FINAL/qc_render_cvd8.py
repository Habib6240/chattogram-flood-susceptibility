#!/usr/bin/env python3
"""Render/accessibility QC for Figure 8: independent poppler + Chromium renders vs PNG, CVD (Machado 2009) and greyscale distinguishability, aggregation sensitivity.
Usage: python3 qc_render_cvd8.py data output"""
import sys,os,subprocess,numpy as np,pandas as pd,json
from PIL import Image
from skimage.color import rgb2lab,deltaE_cie76
D,O=sys.argv[1],sys.argv[2]; W=os.path.join(O,'_work'); os.makedirs(W,exist_ok=True); R={}
pdf=f'{O}/Figure_8_Weights_SHAP_Q1_FINAL.pdf'; svg=f'{O}/Figure_8_Weights_SHAP_Q1_FINAL.svg'; png=f'{O}/Figure_8_Weights_SHAP_Q1_FINAL.png'
P=Image.open(png).convert('RGB'); sz=(1520,980)
subprocess.run(['pdftoppm','-r','200','-png','-singlefile',pdf,f'{W}/pdf'],check=True)
from playwright.sync_api import sync_playwright
s=open(svg).read()
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={'width':1520,'height':980})
    open(f'{W}/svg.html','w').write(f'<html><body style="margin:0;background:white"><img src="file://{svg}" style="width:1520px;display:block"></body></html>')
    pg.goto(f'file://{W}/svg.html'); pg.wait_for_timeout(800)
    pg.screenshot(path=f'{W}/svg.png',full_page=True,timeout=120000); b.close()
from PIL import ImageFilter
def gray(path):
    im=Image.open(path).convert('L').resize(sz).filter(ImageFilter.GaussianBlur(3)); return np.asarray(im,float)
ref=gray(png)
for nm in ('pdf','svg'):
    a=gray(f'{W}/{nm}.png')
    c=np.corrcoef(ref.ravel(),a.ravel())[0,1]; R[f'{nm}_vs_png_pixel_corr']=round(float(c),4); print(nm,'size',Image.open(f'{W}/{nm}.png').size,'corr',round(float(c),4))
# CVD
C={'AHP':'#0072B2','FAHP':'#56B4E9','ANP':'#009E73','SHAP':'#D55E00'}
rgb=lambda h:np.array([int(h[i:i+2],16)/255 for i in (1,3,5)])
M={'protan':[[0.152286,1.052583,-0.204868],[0.114503,0.786281,0.099216],[-0.003882,-0.048116,1.051998]],
   'deutan':[[0.367322,0.860646,-0.227968],[0.280085,0.672501,0.047413],[-0.011820,0.042940,0.968881]],
   'tritan':[[1.255528,-0.076749,-0.178779],[-0.078411,0.930809,0.147602],[0.004733,0.691367,0.303900]]}
lin=lambda c:np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4); gam=lambda c:np.where(c<=0.0031308,12.92*c,1.055*np.clip(c,0,None)**(1/2.4)-0.055)
def sim(c,m): return np.clip(gam(np.clip(np.array(M[m])@lin(c),0,1)),0,1)
res={}
for cond in ['normal','protan','deutan','tritan']:
    cols={k:(rgb(v) if cond=='normal' else sim(rgb(v),cond)) for k,v in C.items()}
    lab={k:rgb2lab(c.reshape(1,1,3))[0,0] for k,c in cols.items()}
    ks=list(C); dmin=min(np.linalg.norm(lab[a]-lab[b]) for i,a in enumerate(ks) for b in ks[i+1:])
    pairs={f'{a}-{b}':round(float(np.linalg.norm(lab[a]-lab[b])),1) for i,a in enumerate(ks) for b in ks[i+1:]}
    res[cond]={'min_dE76':round(float(dmin),1),'pairs':pairs}
    print(cond,'min dE76',round(float(dmin),1),pairs)
R['cvd']=res
# greyscale luminance of the colours
L={k:float(rgb2lab(rgb(v).reshape(1,1,3))[0,0,0]) for k,v in C.items()}; R['luminance_L']=L; print('L*',L)
R['note']='Series are also distinguished by marker shape (circle, square, triangle, diamond) and by row offset, so colour is not the sole carrier.'
# aggregation sensitivity
enc=pd.read_csv(f'{D}/BUET_B08U_XGB_SHAP_Encoded_Feature_Importance.csv').set_index('Encoded_Feature').Mean_Absolute_SHAP
g=pd.read_csv(f'{D}/BUET_B08U_XGB_SHAP_Global_13Factor_Importance.csv').set_index('Factor')
alt={}
for f in g.index:
    if f=='LULC': alt[f]=enc[[i for i in enc.index if i.startswith('LULC_')]].sum()
    elif f=='HSG': alt[f]=enc[[i for i in enc.index if i.startswith('HSG_')]].sum()
    elif f=='Aspect': alt[f]=enc[[i for i in enc.index if i.startswith('Aspect_')]].sum()
    else: alt[f]=enc[f]
alt=pd.Series(alt); sa=100*alt/alt.sum()
t=pd.DataFrame({'share_rowwise_grouped_pct(plotted)':g.Relative_Importance_Percent,'rank_rowwise':g.Relative_Importance_Percent.rank(ascending=False).astype(int),'share_sum_of_encoded_mean_abs_pct':sa,'rank_sum_encoded':sa.rank(ascending=False).astype(int)})
t['rank_change']=t.rank_sum_encoded-t.rank_rowwise; t=t.sort_values('rank_rowwise'); t.to_csv(f'{O}/Figure_8_Aggregation_Sensitivity.csv',float_format='%.6f'); print(t.round(3))
R['aggregation_sensitivity_rank_changes']=int((t.rank_change!=0).sum())
json.dump(R,open(f'{O}/qc_render_cvd.json','w'),indent=1)
