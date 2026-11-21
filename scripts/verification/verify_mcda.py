import numpy as np, csv
A_=\
"/mnt/user-data/attachments/9a6167a7-15aa-5aef-9946-c3abdb14a781/"
names=["Rainfall","Elevation","Slope","Drainage_Density","Distance_River","TWI","LULC","HSG","NDVI","Distance_Road","Profile_Curvature","Tangential_Curvature","Aspect"]
lab={"Rainfall":"Rainfall","Elevation":"Elevation","Slope":"Slope","Drainage density":"Drainage_Density","Distance to river":"Distance_River","TWI":"TWI","LULC":"LULC","HSG":"HSG","NDVI":"NDVI","Distance to road":"Distance_Road","Profile curvature":"Profile_Curvature","Tangential curvature":"Tangential_Curvature","Aspect":"Aspect"}
n=13; g=[2,3,4,4,5,5,6,6,7,7,8,9]
A=np.ones((n,n))
for i in range(n):
    for j in range(i+1,n): A[i,j]=g[j-i-1]; A[j,i]=1/A[i,j]
def pe(M):
    w,v=np.linalg.eig(M); k=np.argmax(w.real); x=np.abs(v[:,k].real); return w[k].real,x/x.sum()
lam,w=pe(A); CI=(lam-n)/(n-1); CR=CI/1.56
rd=lambda f:{r[0]:[float(x) for x in r[1:]] for r in list(csv.reader(open(A_+f)))[1:]}
ahp=rd("880b7a79-DU10_AHP_Weights_and_Consistency.csv") if False else None
rows=list(csv.reader(open(A_+"880b7a79-DU10_AHP_Weights_and_Consistency.csv")))
ahp={r[0]:float(r[1]) for r in rows[1:14]}
meta={r[0]:float(r[1]) for r in rows[16:] if r and r[0] in("Lambda_Max","CI","CR")}
print("lambda_max",lam,meta["Lambda_Max"],"CI",CI,meta["CI"],"CR",CR,meta["CR"])
print("AHP max|diff|",max(abs(w[i]-ahp[names[i]]) for i in range(n)))
# FAHP
def tfn(a):
    if a==1: return (1,1,1)
    if a>1:
        k=round(a); return (k-1,k,k+1) if k<9 else (8,9,9)
    k=round(1/a); l,m,u=(k-1,k,k+1) if k<9 else (8,9,9); return (1/u,1/m,1/l)
T=np.array([[tfn(A[i,j]) for j in range(n)] for i in range(n)])  # n,n,3 (l,m,u)
gm=np.exp(np.log(T).mean(axis=1))  # component-wise geometric mean (n,3)
S=gm.sum(axis=0)  # sum l,m,u
fw=np.stack([gm[:,0]/S[2],gm[:,1]/S[1],gm[:,2]/S[0]],axis=1)
dz=fw.mean(axis=1); fahp=dz/dz.sum()
rows=list(csv.reader(open(A_+"39922c1e-DU11_FAHP_Weights.csv")))[1:]
f={r[0]:r for r in rows}
print("FAHP max|diff| final",max(abs(fahp[i]-float(f[names[i]][8])) for i in range(n)),
      "GM",max(abs(gm[i,k]-float(f[names[i]][1+k])) for i in range(n) for k in range(3)),
      "FW",max(abs(fw[i,k]-float(f[names[i]][4+k])) for i in range(n) for k in range(3)),
      "defuzz",max(abs(dz[i]-float(f[names[i]][7])) for i in range(n)))
# ANP
import re
links={}
for line in open('txt/SI.txt'):
    pass
txt=open('txt/SI.txt').read().split("\n")
st=[i for i,l in enumerate(txt) if l.startswith("Source factor | Number of targets")][0]
for l in txt[st+1:st+14]:
    s,k,t=[x.strip() for x in l.split("|")]
    links[lab[s]]=[lab[x.strip()] for x in t.split(";")]
assert sum(len(v) for v in links.values())==57
idx={nm:i for i,nm in enumerate(names)}
W=np.zeros((n,n))
for s,tg in links.items():
    ti=[idx[t] for t in tg]
    if len(ti)==1: W[ti[0],idx[s]]=1.0
    else:
        _,v=pe(A[np.ix_(ti,ti)]); W[ti,idx[s]]=v
print("column sums",np.round(W.sum(axis=0),12))
Lm=W.copy(); it=0
while True:
    Wn=Lm@W; d=np.abs(Wn-Lm).max(); Lm=Wn; it+=1
    if d<1e-12 or it>10000: break
anp=Lm.mean(axis=1)
r=list(csv.reader(open(A_+"d322b720-DU12_ANP_Final_Weights.csv")))
anp_ref={x[0]:float(x[1]) for x in r[1:14]}
print("ANP max|diff|",max(abs(anp[i]-anp_ref[names[i]]) for i in range(n)),"iters",it,"col spread",np.abs(Lm-Lm[:,[0]]).max(), r[15:])
# AHP-ANP Spearman, Pearson of weights
from scipy.stats import spearmanr
print("Spearman AHP-ANP",spearmanr(w,anp)[0],"AHP-FAHP",spearmanr(w,fahp)[0],"max AHP-FAHP diff",np.abs(w-fahp).max())
rk=lambda x:(-x).argsort().argsort()+1
print("ranks AHP",dict(zip(names,rk(w))));print("ranks FAHP",dict(zip(names,rk(fahp))));print("ranks ANP",dict(zip(names,rk(anp))))
