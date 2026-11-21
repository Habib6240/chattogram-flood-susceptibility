import numpy as np,pandas as pd,glob,json
from PIL import Image
from scipy.stats import kendalltau
Image.MAX_IMAGE_PIXELS=None
A=glob.glob('/mnt/user-data/attachments/*/')[0];g=lambda p:glob.glob(A+p)[0]
out={}
# Eq12
m=np.asarray(Image.open(g('*b23da44e*')))==1
rd=lambda p:np.asarray(Image.open(g(p))).astype('float64')
rf,xg,cs=rd('*003c4b46*'),rd('*f285f228*'),rd('*6865cff3*')
d=np.abs(((rf+xg)/2)[m]-cs[m]);out['eq12_raster_maxdiff']=float(d.max());out['eq12_nodata_match']=bool(((cs==-9999)==(rf==-9999)).all())
z=np.load(g('*c2832f3a*'))
# OOF consensus event AUC is mean-based; already verified (S13)
# Eq15: grouped = sum of encoded
npz=np.load(g('*88a4296e*'),allow_pickle=True);out['npz_keys']=list(npz.files)
for k in npz.files: print(k,npz[k].shape)
json.dump(out,open('out/eq_rest_partial.json','w'))
print(out)
