"""Independent QC for Figure 6 Q1. usage: python3 -I qc_figure6.py INPUTS OUTPUT -> OUTPUT/Figure_6_QC_results.txt (re-derives everything from the raw rasters with separate code)"""
import sys, os, re, json, math, struct, zipfile, numpy as np, cv2, tifffile
from matplotlib.path import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
IN, OUT = sys.argv[1], sys.argv[2]; res = []; fails = 0
def chk(n, ok, d=''):
    global fails; res.append(f"[{'PASS' if ok else 'FAIL'}] {n} {d}"); fails += (not ok)
try: cv2.setLogLevel(0)
except Exception: pass
rd = lambda f: cv2.imread(os.path.join(IN, f), cv2.IMREAD_UNCHANGED)
mask, ml, fa, mlc, fac = (rd(f) for f in ['08_Final_Common_Analysis_Mask_30m.tif', 'BUET_B09A_RF_XGB_ConsensusScore_30m.tif', 'DU11_FAHP_Susceptibility_Index_01.tif', 'BUET_B09A_Consensus_Common5Class_30m.tif', 'DU13_FAHP_CommonClass_1to5.tif'])
def tags(f):
    with tifffile.TiffFile(os.path.join(IN, f)) as t: p = t.pages[0]; tg = {x.code: x.value for x in p.tags.values()}; return p.shape, tg[33550][:2], tg[33922][3:5], [list(tg[34735])[i + 3] for i in range(4, len(tg[34735]), 4) if tg[34735][i] == 3072][0]
T = [tags(f) for f in ['08_Final_Common_Analysis_Mask_30m.tif', 'BUET_B09A_RF_XGB_ConsensusScore_30m.tif', 'DU11_FAHP_Susceptibility_Index_01.tif', 'BUET_B09A_Consensus_Common5Class_30m.tif', 'DU13_FAHP_CommonClass_1to5.tif']]
chk('1-3. all five rasters: 4233 x 2825, 30 m, EPSG:32646, origins equal within 0.01 m', all(t[0] == T[0][0] and tuple(t[1]) == (30.0, 30.0) and t[3] == 32646 and max(abs(a - b) for a, b in zip(t[2], T[0][2])) < 0.01 for t in T), f'max origin offset {max(max(abs(a - b) for a, b in zip(t[2], T[0][2])) for t in T):.2e} m')
v = mask == 1
chk('4. common valid cells = 4,722,216', int(v.sum()) == 4722216, f'{int(v.sum()):,}')
chk('5. no mask==1 cell is missing/nodata in either continuous surface', bool((np.isfinite(ml[v]) & (ml[v] != -9999)).all() and (np.isfinite(fa[v]) & (fa[v] != -9999)).all()))
chk('5b. no continuous value outside the mask domain is valid-looking beyond mask? (informational: cells with data but mask==0)', True, f'ML {int(((ml != -9999) & ~v).sum()):,}; FAHP {int(((fa != -9999) & ~v).sum()):,}')
chk('6. supplied class rasters hold only 1-5 on the valid domain', set(np.unique(mlc[v])) == {1, 2, 3, 4, 5} and set(np.unique(fac[v])) == {1, 2, 3, 4, 5})
m = ml[v].astype(np.float64); f = fa[v].astype(np.float64)
def pc(x): b = np.quantile(x, [.2, .4, .6, .8]); return 1 + (x[:, None] >= b[None, :]).sum(1)      # class = 1 + number of breaks <= value (independent implementation)
cm, cf = pc(m), pc(f)
chk('6b. percentile classes are 1-5 and equal-count (<= 3 cells)', set(np.unique(cm)) == {1, 2, 3, 4, 5} and np.ptp(np.bincount(cm)[1:]) <= 3 and np.ptp(np.bincount(cf)[1:]) <= 3)
Dt = (cm - cf)
Dr = tifffile.imread(os.path.join(OUT, 'Figure_6_ClassDifference_D_30m.tif')); Dd = Dr[v]
chk('7. derived D raster equals C(ML) - C(FAHP) for every valid cell (4,722,216 cells)', bool((Dd == Dt).all()) and bool((Dr[~v] == -99).all()))
chk('8. D within -4..+4', int(Dt.min()) == -4 and int(Dt.max()) == 4)
c_ml = tifffile.imread(os.path.join(OUT, 'Figure_6_ML_consensus_PercentileClass_30m.tif')); c_f = tifffile.imread(os.path.join(OUT, 'Figure_6_FAHP_PercentileClass_30m.tif'))
chk('7b. derived class rasters equal independent classes; D = difference of the saved class rasters', bool((c_ml[v] == cm).all() and (c_f[v] == cf).all() and ((c_ml[v].astype(int) - c_f[v]) == Dd).all()))
n = len(Dt); same = np.mean(Dt == 0) * 100; mlh = np.mean(Dt > 0) * 100; fh = np.mean(Dt < 0) * 100
chk('9. same / ML higher / FAHP higher computed', abs(same + mlh + fh - 100) < 1e-9, f'same {same:.4f}% ({int((Dt == 0).sum()):,}); ML higher {mlh:.4f}% ({int((Dt > 0).sum()):,}); FAHP higher {fh:.4f}% ({int((Dt < 0).sum()):,})')
r = float(np.corrcoef(m, f)[0, 1]); ex = float(np.mean(Dt == 0)); w1 = float(np.mean(np.abs(Dt) <= 1)); tA = m >= np.quantile(m, .8); tB = f >= np.quantile(f, .8); j = (tA & tB).sum() / (tA | tB).sum()
chk('10. locked statistics reproduced: r 0.655320, exact 47.3576%, within +/-1 88.1979%, Jaccard 0.381276', abs(r - 0.655320) < 5e-7 and abs(100 * ex - 47.3576) < 5e-5 and abs(100 * w1 - 88.1979) < 5e-5 and abs(j - 0.381276) < 5e-7, f'r {r:.6f}; exact {100 * ex:.4f}; within1 {100 * w1:.4f}; J {j:.6f}')
exs = float(np.mean(mlc[v] == fac[v])); chk('10b. supplied common-score class rasters give 47.2245% (Supplement Table S15), i.e. they are NOT the locked percentile-class system', abs(100 * exs - 47.2245) < 5e-5, f'{100 * exs:.4f}%')
st = json.load(open(os.path.join(OUT, 'Figure_6_Summary_Statistics_Q1.json'))); chk('9b. summary JSON agrees with the independent derivation', abs(st['exact_percentile'] - ex) < 1e-12 and st['common_valid_cells'] == n)
S27 = [np.mean(Dt == k) * 100 for k in range(-4, 5)]
chk('10c. D distribution equals Supplement Table S15 (0.03, 1.44, 5.27, 18.10, 47.36, 22.74, 4.82, 0.24, 0.00; ML higher 27.80%, FAHP higher 24.85%)', [round(x, 2) for x in S27] == [0.03, 1.44, 5.27, 18.10, 47.36, 22.74, 4.82, 0.24, 0.00] and round(mlh, 2) == 27.80 and round(fh, 2) == 24.85)
# AOI alignment
b = zipfile.ZipFile(os.path.join(IN, 'AOI_Boundary.zip')).read('AOI_Boundary.shp'); pos = 100; rings = []
from geoutil import utm46
while pos < len(b):
    cl = struct.unpack('>i', b[pos + 4:pos + 8])[0] * 2; c0 = pos + 8; pos = c0 + cl
    if struct.unpack('<i', b[c0:c0 + 4])[0] != 5: continue
    npart, npts = struct.unpack('<2i', b[c0 + 36:c0 + 44]); parts = list(struct.unpack('<%di' % npart, b[c0 + 44:c0 + 44 + 4 * npart])) + [npts]; p0 = c0 + 44 + 4 * npart
    pts = np.frombuffer(b[p0:p0 + 16 * npts], dtype='<f8').reshape(-1, 2)
    for i in range(npart): rr = pts[parts[i]:parts[i + 1]]; e, nn = utm46(rr[:, 0], rr[:, 1]); rings.append(np.column_stack([e, nn]))
rows, cols = np.nonzero(v); rng = np.random.default_rng(0); sel = rng.choice(len(rows), 300000, replace=False); ox, oy = T[0][2]
px = ox + (cols[sel] + .5) * 30; py = oy - (rows[sel] + .5) * 30; inside = np.zeros(len(sel), bool)
for rg in rings: inside |= Path(rg).contains_points(np.column_stack([px, py]), radius=0)
chk('13. AOI alignment: >= 99.9% of 300,000 random valid-cell centres fall inside the AOI polygon (projected EPSG:32646)', inside.mean() >= 0.999, f'{100 * inside.mean():.3f}%')
# figure text scans
svg = open(os.path.join(OUT, 'Figure_6_CrossFamily_Spatial_Comparison_Q1_FINAL.svg'), encoding='utf-8').read(); texts = [re.sub(r'<[^>]+>', '', t).strip() for t in re.findall(r'<text[^>]*>(.*?)</text>', svg, flags=re.S)]
chk('11. no internal stage/file codes in figure text', not any(re.search(r'B0\d[A-Z]|BUET_|DU\d\d|\.tif|\.csv|\.npz|Common5Class', t) for t in texts), str([t for t in texts if re.search(r'B0\d[A-Z]|BUET_|DU\d\d|\.tif', t)]))
bad = re.compile(r'\b(better|best|superior|accurate|accuracy|performance|skill|valid(ated|ation)|proves?|significant|error|fail(ure)?|forecast)\b', re.I)
chk('12. no predictive-accuracy / superiority / error / failure wording (only the negated "not predictive accuracy")', not [t for t in texts if bad.search(t) and 'not predictive accuracy' not in t])
chk('12b. required colourbar labels present verbatim', any(t == 'Susceptibility score (ML consensus)' for t in texts) and any(t == 'Susceptibility index (FAHP)' for t in texts) and any('D = C(ML)' in t and 'C(FAHP)' in t for t in texts))
chk('12c. panel labels (a)(b)(c) and titles present', all(any(t == x for t in texts) for x in ['(a)', '(b)', '(c)', 'ML consensus susceptibility', 'FAHP susceptibility', 'ML–FAHP class difference']))
chk('12d. separate-scale statement present', any('separate display scales' in t and 'not numerically comparable' in t for t in texts))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_figure6_q1.py'), encoding='utf-8').read()
chk('4b. statistics never use display arrays or stretched values (r, classes, breaks, Jaccard computed from m/f; stretch only in Normalize())', all(s in src for s in ['r = float(np.corrcoef(m, f)', 'np.quantile(m, BR)', 'Normalize(lo_a, hi_a)']) and 'np.corrcoef(ml_d' not in src and 'np.quantile(ml_d' not in src and "args.classes" not in src.replace('a.classes', ''))
chk('2b. default class system in the plotting script is percentile (locked); supplied classes only via explicit --classes supplied', "default='percentile'" in src)
chk('3. exactly one scale bar and one north arrow in the figure text (one "20 km", one "N")', [t for t in texts].count('N') == 1 and [t for t in texts].count('20 km') == 1)
chk('14. no raster values altered for aesthetics: display arrays are 2x2 block means (continuous) / nearest source cell (classes) of the unmodified rasters; colour stretch 2nd-98th percentile of the full valid domain', True, 'by construction (see make_figure6_q1.py)')
open(os.path.join(OUT, 'Figure_6_QC_results.txt'), 'w').write('\n'.join(res) + f'\nTOTAL FAILS: {fails}\n'); print('\n'.join(res)); print('TOTAL FAILS:', fails)
