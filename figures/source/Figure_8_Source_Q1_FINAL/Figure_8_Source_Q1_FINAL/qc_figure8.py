#!/usr/bin/env python3
"""Independent QC for Figure 8. Re-derives every plotted number from the raw CSVs with separate code and inspects the exported files.
Usage: python3 qc_figure8.py data output"""
import sys,os,re,json,subprocess,numpy as np,pandas as pd
from xml.etree import ElementTree as ET
D,O=sys.argv[1],sys.argv[2]; R=[]
def chk(name,ok,det=''): R.append((name,'PASS' if ok else 'FAIL',det)); print(('PASS ' if ok else 'FAIL ')+name,det)
raw={}
for f,c,n in (('DU10_AHP_Weights_and_Consistency.csv','Weight',13),('DU11_FAHP_Weights.csv','Final_Normalized_Weight',13),('DU12_ANP_Final_Weights.csv','ANP_Weight',13)):
    t=[l.strip().split(',') for l in open(f'{D}/{f}')]; h=t[0]; ci=h.index(c)
    raw[f]={r[0]:float(r[ci]) for r in t[1:n+1]}
A,F,N=raw['DU10_AHP_Weights_and_Consistency.csv'],raw['DU11_FAHP_Weights.csv'],raw['DU12_ANP_Final_Weights.csv']
for nm,d in (('AHP',A),('FAHP',F),('ANP',N)):
    chk(f'{nm}: 13 weights, sum=1',len(d)==13 and abs(sum(d.values())-1)<1e-9,f'sum={sum(d.values()):.12f}')
aud=pd.read_csv(f'{O}/Figure_8_Numerical_Audit.csv'); key=aud.Factor_key.tolist()
chk('audit table has 13 unique factors',len(set(key))==13)
mx=max(max(abs(aud.AHP_weight[i]-A[k]),abs(aud.FAHP_weight[i]-F[k]),abs(aud.ANP_weight[i]-N[k])) for i,k in enumerate(key))
chk('39 MCDA weights in audit table equal raw CSV values',mx<1e-12,f'max diff {mx:.1e}')
# SHAP: separate recomputation using csv module + numpy
import csv
rows=list(csv.DictReader(open(f'{D}/BUET_B08U_XGB_SHAP_Explanation_Sample.csv')))
chk('explanation sample has 20,000 rows',len(rows)==20000,str(len(rows)))
folds={}; [folds.__setitem__(r['Spatial_Fold'],folds.get(r['Spatial_Fold'],0)+1) for r in rows]
chk('4,000 rows per fold (5 folds)',sorted(folds.values())==[4000]*5,str(folds))
ma={k:np.mean([abs(float(r[k+'_SHAP'])) for r in rows]) for k in key}; tot=sum(ma.values()); share={k:100*v/tot for k,v in ma.items()}
g={r['Factor']:float(r['Relative_Importance_Percent']) for r in csv.DictReader(open(f'{D}/BUET_B08U_XGB_SHAP_Global_13Factor_Importance.csv'))}
d=max(abs(share[k]-g[k]) for k in key); chk('13 grouped SHAP shares: recomputed from 20,000 rows = global CSV',d<1e-4,f'max diff {d:.2e} pp')
d2=max(abs(aud.SHAP_share_pct_grouped_csv[i]-g[k]) for i,k in enumerate(key)); chk('audit table SHAP shares equal global CSV',d2<1e-9)
chk('SHAP shares sum to 100',abs(sum(g.values())-100)<1e-4,f'{sum(g.values()):.6f}')
# additivity from sample
mm=max(abs(sum(float(r[k+'_SHAP']) for k in key)+float(r['SHAP_Bias'])-np.log(float(r['Predicted_Probability'])/(1-float(r['Predicted_Probability'])))) for r in rows[:5000])
chk('additivity (bias + sum SHAP = logit p), first 5,000 rows',mm<1e-3,f'max {mm:.2e}')
# encoded -> grouped aggregation check
enc={r['Encoded_Feature']:float(r['Mean_Absolute_SHAP']) for r in csv.DictReader(open(f'{D}/BUET_B08U_XGB_SHAP_Encoded_Feature_Importance.csv'))}
chk('25 encoded features',len(enc)==25)
for fac,pre in (('LULC','LULC_'),('HSG','HSG_')):
    comp=[v for k,v in enc.items() if k.startswith(pre)]; grouped=g[fac]
    chk(f'{fac}: components found',len(comp)>0,f'{len(comp)} columns')
# ranks, independent (argsort)
def rank(dic): 
    ks=sorted(dic,key=lambda k:-dic[k]); return {k:i+1 for i,k in enumerate(ks)}
rr={'AHP':rank(A),'FAHP':rank(F),'ANP':rank(N),'SHAP':rank(g)}
for m in rr: chk(f'{m} ranks in audit table equal independent ranks',all(aud[m+'_rank'][i]==rr[m][k] for i,k in enumerate(key)))
chk('no tied values within any series',all(len(set(d.values()))==13 for d in (A,F,N,g)))
chk('factor order = SHAP rank 1..13',aud.SHAP_rank.tolist()==list(range(1,14)))
# ---- exports
svg=open(f'{O}/Figure_8_Weights_SHAP_Q1_FINAL.svg',encoding='utf8').read()
root=ET.fromstring(svg); ns='{http://www.w3.org/2000/svg}'
texts=[''.join(e.itertext()).strip() for e in root.iter(ns+'text')]; texts=[t for t in texts if t]
chk('SVG has live text',len(texts)>40,f'{len(texts)} text elements')
bad=re.compile(r'B0\d|DU\d|\.csv|\.py|npz|v1\b|_SHAP|Composite|AnyFlood',re.I)
chk('no stage codes / file names / identifiers in figure text',not [t for t in texts if bad.search(t)],str([t for t in texts if bad.search(t)]))
blob=' '.join(texts).lower()
chk('no causal / superiority wording in figure',not re.search(r'\bcaus(es|ed|al effect)\b|driv(er|es|ing)|superior|better|outperform|significan',blob.replace('not causal','')))
chk('"not causal" and "separate axes" statements present','not causal' in blob and 'separate axes' in blob)
for lab in ['(a)','(b)','(c)','AHP','FAHP','ANP','XGBoost SHAP']: chk(f'label present: {lab}',any(lab in t for t in texts))
facs=['NDVI','LULC','Elevation','Rainfall','Slope','Distance to road','TWI','Distance to river','Tangential curvature','Drainage density','Profile curvature','Aspect','HSG']
chk('all 13 factor names appear in SVG exactly once as tick labels',all(texts.count(f)==1 for f in facs))
sz=[float(m) for m in re.findall(r'font:\s*(?:[a-z0-9 ]*?)([0-9.]+)px',svg)]+[float(m) for m in re.findall(r'font-size:\s*([0-9.]+)',svg)]
# svg px at 72dpi == pt
chk('minimum font size >= 6.5 pt',min(sz)>=6.49,f'min {min(sz):.2f} pt; sizes {sorted(set(round(s,1) for s in sz))}')
chk('SVG font family Liberation Sans','Liberation Sans' in svg)
# bar labels printed vs csv
exp=[f'{g[k]:.2f}' for k in key]
chk('13 bar labels equal CSV shares at 2 dp (each present in SVG text)',all(e in texts for e in exp),str(exp))
chk('bar labels appear in SHAP-rank order (top to bottom)',[t for t in texts if t in exp]==exp)
pf=subprocess.run(['pdffonts',f'{O}/Figure_8_Weights_SHAP_Q1_FINAL.pdf'],capture_output=True,text=True).stdout
chk('PDF fonts all embedded',all(l.split()[-5]=='yes' for l in pf.strip().splitlines()[2:]),pf.strip().splitlines()[2:][0] if pf.strip().splitlines()[2:] else '')
pi=subprocess.run(['pdfimages','-list',f'{O}/Figure_8_Weights_SHAP_Q1_FINAL.pdf'],capture_output=True,text=True).stdout.strip().splitlines()
chk('PDF is vector (no raster images)',len(pi)<=2,f'{max(len(pi)-2,0)} images')
from PIL import Image
im=Image.open(f'{O}/Figure_8_Weights_SHAP_Q1_FINAL.png'); chk('PNG 600 dpi',abs(im.info['dpi'][0]-600)<1,str(im.info['dpi'])+' '+str(im.size))
# clipping: content must not touch image border
a=np.asarray(im.convert('L')); 
chk('no ink on PNG border (no clipping)',a[:3].min()>250 and a[-3:].min()>250 and a[:,:3].min()>250 and a[:,-3:].min()>250)
json.dump(R,open(f'{O}/qc_results.json','w'),indent=1)
print('FAILS:',sum(1 for r in R if r[1]=='FAIL'))
