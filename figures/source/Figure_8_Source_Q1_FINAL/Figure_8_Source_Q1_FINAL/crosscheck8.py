#!/usr/bin/env python3
"""Cross-check Figure 8 values against Supplementary Tables S3/S16 and manuscript v3 text. Usage: crosscheck8.py data output supp.xlsx manuscript.docx"""
import sys,re,openpyxl,docx,pandas as pd
D,O,X,M=sys.argv[1:5]
a=pd.read_csv(f'{O}/Figure_8_Numerical_Audit.csv').set_index('Factor')
wb=openpyxl.load_workbook(X,data_only=True); out=[]
def rows(n): return [r for r in wb[n].iter_rows(values_only=True) if any(c is not None for c in r)]
for r in rows('S3')[2:15]:
    f=r[0]; 
    for j,(c,rc) in enumerate((('AHP_weight','AHP_rank'),('FAHP_weight','FAHP_rank'),('ANP_weight','ANP_rank')),1):
        out.append(('S3',f,c,r[j],a.loc[f,c],abs(r[j]-a.loc[f,c])<5e-7)); out.append(('S3',f,rc,r[j+3],a.loc[f,rc],r[j+3]==a.loc[f,rc]))
for r in rows('S16')[3:16]:
    f=r[1]; out.append(('S16',f,'mean_abs',r[2],a.loc[f,'SHAP_mean_abs_grouped_csv'],abs(r[2]-a.loc[f,'SHAP_mean_abs_grouped_csv'])<5e-7))
    out.append(('S16',f,'share_pct',r[3],a.loc[f,'SHAP_share_pct_grouped_csv'],abs(r[3]-a.loc[f,'SHAP_share_pct_grouped_csv'])<5e-7)); out.append(('S16',f,'rank',r[0],a.loc[f,'SHAP_rank'],r[0]==a.loc[f,'SHAP_rank']))
txt=' '.join(p.text for p in docx.Document(M).paragraphs)
sec=re.search(r'Grouped mean absolute TreeSHAP contribution.*?owing to rounding\.',txt,re.S).group(0)
nice={'Distance to road':'distance to road','Distance to river':'distance to river','Tangential curvature':'tangential curvature','Drainage density':'drainage density','Profile curvature':'profile curvature','Rainfall':'rainfall','Slope':'slope','Elevation':'elevation','Aspect':'aspect'}
for f in a.index:
    v=f'{a.loc[f,"SHAP_share_pct_grouped_csv"]:.2f}'; w=nice.get(f,f)
    out.append(('Manuscript v3 Results 3.6',f,'share_pct_text',v,v,re.search(re.escape(w)+r'[^()]{0,14}\('+re.escape(v)+r'%\)',sec,re.I) is not None))
df=pd.DataFrame(out,columns=['Source','Factor','Quantity','Source_value','Figure_value','Match']); df.to_csv(f'{O}/Figure_8_CrossCheck_Supplement_Manuscript.csv',index=False,float_format='%.10g')
print(df.Match.value_counts().to_dict()); print(df[~df.Match])
