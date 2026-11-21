"""Figure 6 (Q1): ML consensus vs FAHP cross-family spatial comparison, rebuilt from the raw rasters.
usage: python3 -I make_figure6_q1.py --inputs DIR --out DIR [--dpi 600] [--classes percentile|supplied]
 percentile (default, the LOCKED analysis): five-class maps = method-specific percentile classes (20/40/60/80th percentiles of each surface on the common valid cells);
 supplied: five-class maps read from BUET_B09A_Consensus_Common5Class_30m.tif and DU13_FAHP_CommonClass_1to5.tif (common-score classes; QA comparison only)."""
import argparse, os, sys, zipfile, struct, math, json, numpy as np, pandas as pd, cv2, tifffile, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle, Polygon
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib import cm as mcm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from geoutil import utm46_inv
ap = argparse.ArgumentParser(); ap.add_argument('--inputs', default='inputs'); ap.add_argument('--out', default='output'); ap.add_argument('--dpi', type=int, default=600); ap.add_argument('--classes', default='percentile', choices=['percentile', 'supplied'])
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
plt.rcParams.update({'font.family': ['Liberation Sans', 'DejaVu Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False})
try: cv2.setLogLevel(0)
except Exception: pass
IN = lambda f: os.path.join(a.inputs, f)
F = {'mask': '08_Final_Common_Analysis_Mask_30m.tif', 'ml': 'BUET_B09A_RF_XGB_ConsensusScore_30m.tif', 'fahp': 'DU11_FAHP_Susceptibility_Index_01.tif', 'mlc': 'BUET_B09A_Consensus_Common5Class_30m.tif', 'fahpc': 'DU13_FAHP_CommonClass_1to5.tif'}
# ------------------------------------------------------------ rasters + alignment audit
R = {k: cv2.imread(IN(v), cv2.IMREAD_UNCHANGED) for k, v in F.items()}; geo = {}
for k, v in F.items():
    with tifffile.TiffFile(IN(v)) as t:
        p = t.pages[0]; tg = {tag.code: tag.value for tag in p.tags.values()}; gk = list(tg[34735]); ep = [gk[i + 3] for i in range(4, len(gk), 4) if gk[i] == 3072]
        geo[k] = dict(shape=tuple(R[k].shape), dtype=str(R[k].dtype), scale=tuple(tg[33550][:2]), tie=tuple(tg[33922][3:5]), epsg=ep[0] if ep else None, nodata=tg.get(42113))
ref = geo['mask']; H, W = ref['shape']; x0, y0 = ref['tie']; res = ref['scale'][0]
for k, g in geo.items():
    assert g['shape'] == ref['shape'] and g['scale'] == ref['scale'] and g['epsg'] == 32646 and all(abs(p - q) < 0.01 for p, q in zip(g['tie'], ref['tie'])), (k, g)
extent = (x0, x0 + W * res, y0 - H * res, y0)
mask = R['mask'] == 1; ml = R['ml']; fa = R['fahp']
ndml, ndfa = float(geo['ml']['nodata']), float(geo['fahp']['nodata'])
okml = np.isfinite(ml) & (ml != ndml); okfa = np.isfinite(fa) & (fa != ndfa)
assert int(mask.sum()) == 4722216, int(mask.sum())
assert (okml[mask]).all() and (okfa[mask]).all(), 'valid-domain pixels missing from a continuous surface'
valid = mask.copy(); nv = int(valid.sum()); m = ml[valid].astype(np.float64); f = fa[valid].astype(np.float64)
# ------------------------------------------------------------ classes
BR = [.2, .4, .6, .8]; qm = np.quantile(m, BR); qf = np.quantile(f, BR)
cm = 1 + np.searchsorted(qm, m, side='right'); cf = 1 + np.searchsorted(qf, f, side='right')
sm = R['mlc'][valid].astype(int); sf = R['fahpc'][valid].astype(int)
assert set(np.unique(sm)) <= {1, 2, 3, 4, 5} and set(np.unique(sf)) <= {1, 2, 3, 4, 5}, 'supplied class rasters hold values outside 1-5 on the valid domain'
assert set(np.unique(cm)) == {1, 2, 3, 4, 5} and set(np.unique(cf)) == {1, 2, 3, 4, 5}
r = float(np.corrcoef(m, f)[0, 1]); exact = float(np.mean(cm == cf)); w1 = float(np.mean(np.abs(cm - cf) <= 1)); tA = m >= np.quantile(m, .8); tB = f >= np.quantile(f, .8); jac = float((tA & tB).sum() / (tA | tB).sum())
exact_sup = float(np.mean(sm == sf)); w1_sup = float(np.mean(np.abs(sm - sf) <= 1))
print(f'common valid cells {nv:,} | r {r:.6f} | exact(percentile) {100 * exact:.4f}% | within1 {100 * w1:.4f}% | Jaccard {jac:.6f} | exact(supplied common-score classes) {100 * exact_sup:.4f}%')
# locked values are used as a gate (never as a plotting input)
assert abs(r - 0.655320) < 5e-7 and abs(100 * exact - 47.3576) < 5e-5 and abs(jac - 0.381276) < 5e-7 and abs(100 * w1 - 88.1979) < 5e-5, 'locked B09B values not reproduced - STOP'
cuse_m, cuse_f = (cm, cf) if a.classes == 'percentile' else (sm, sf)
Dv = (cuse_m - cuse_f).astype(np.int8); assert Dv.min() >= -4 and Dv.max() <= 4
D = np.full((H, W), -99, dtype=np.int8); D[valid] = Dv
dist = pd.DataFrame({'D_ML_minus_FAHP': range(-4, 5)}); dist['Cells'] = [int((Dv == k).sum()) for k in range(-4, 5)]; dist['Percent'] = 100 * dist.Cells / nv
dist_sup = (sm - sf); dist['Cells_supplied_common_score_classes'] = [int((dist_sup == k).sum()) for k in range(-4, 5)]; dist['Percent_supplied_common_score_classes'] = 100 * dist.Cells_supplied_common_score_classes / nv
dist.to_csv(os.path.join(a.out, 'Figure_6_ClassDifference_Distribution_Q1.csv'), index=False, float_format='%.6f')
pd.DataFrame({'Method': ['ML consensus', 'FAHP'], 'Min': [m.min(), f.min()], 'P02_display': [np.percentile(m, 2), np.percentile(f, 2)], 'Break_20pct': [qm[0], qf[0]], 'Break_40pct': [qm[1], qf[1]], 'Break_60pct': [qm[2], qf[2]], 'Break_80pct': [qm[3], qf[3]], 'P98_display': [np.percentile(m, 98), np.percentile(f, 98)], 'Max': [m.max(), f.max()]} | {f'Class{i}_N': [int((cm == i).sum()), int((cf == i).sum())] for i in range(1, 6)}).to_csv(os.path.join(a.out, 'Figure_6_ClassBreaks_Q1.csv'), index=False, float_format='%.8f')
stats = dict(common_valid_cells=nv, pearson_r=r, exact_percentile=exact, within1_percentile=w1, jaccard_top20=jac, exact_supplied=exact_sup, within1_supplied=w1_sup, ml_higher_pct=float(100 * np.mean(Dv > 0)), fahp_higher_pct=float(100 * np.mean(Dv < 0)), same_pct=float(100 * np.mean(Dv == 0)),
             ml_higher_pct_supplied=float(100 * np.mean(dist_sup > 0)), fahp_higher_pct_supplied=float(100 * np.mean(dist_sup < 0)), class_source=a.classes, grid=dict(shape=[H, W], pixel_m=res, origin=[x0, y0], epsg=32646), per_raster=geo)
json.dump(stats, open(os.path.join(a.out, 'Figure_6_Summary_Statistics_Q1.json'), 'w'), indent=1, default=str)
# derived GeoTIFFs (deflate, EPSG:32646, same grid)
gk = [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 32646]
def wtif(path, arr, nodata):
    tifffile.imwrite(path, arr, compression='zlib', photometric='minisblack', extratags=[(33550, 'd', 3, (res, res, 0.0), True), (33922, 'd', 6, (0, 0, 0, x0, y0, 0), True), (34735, 'H', len(gk), gk, True), (42113, 's', 0, str(nodata), True)])
wtif(os.path.join(a.out, 'Figure_6_ClassDifference_D_30m.tif'), D, -99)
for nm, c in (('ML_consensus', cm), ('FAHP', cf)):
    z = np.zeros((H, W), np.uint8); z[valid] = c; wtif(os.path.join(a.out, f'Figure_6_{nm}_PercentileClass_30m.tif'), z, 0)
# ------------------------------------------------------------ display arrays (K x K block reduction; rendering only)
K = 2; oh, ow = H // K, W // K
def dcont(arr):
    v = np.where(valid, arr, 0).astype(np.float64)[:oh * K, :ow * K].reshape(oh, K, ow, K).sum((1, 3)); n = valid[:oh * K, :ow * K].reshape(oh, K, ow, K).sum((1, 3))
    return np.where(n >= 2, v / np.maximum(n, 1), np.nan)      # a display cell is shown when >= 2 of its 4 source cells are valid
ml_d, fa_d = dcont(ml), dcont(fa); Dd = D[:oh * K:K, :ow * K:K].astype(float); Dd[Dd == -99] = np.nan   # classes: nearest source cell
ext_d = (x0, x0 + ow * K * res, y0 - oh * K * res, y0)
# ------------------------------------------------------------ AOI polygon (EPSG:4326 -> EPSG:32646)
def utm46(lon, lat):
    A_ = 6378137.0; fl = 1 / 298.257223563; k0 = 0.9996; lon0 = math.radians(93.0); n = fl / (2 - fl); Ak = A_ / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    al = [n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16 + 41 * n ** 4 / 180, 13 * n ** 2 / 48 - 3 * n ** 3 / 5 + 557 * n ** 4 / 1440, 61 * n ** 3 / 240 - 103 * n ** 4 / 140, 49561 * n ** 4 / 161280]
    ph = np.radians(lat); la = np.radians(lon) - lon0; c = 2 * math.sqrt(n) / (1 + n); t = np.sinh(np.arctanh(np.sin(ph)) - c * np.arctanh(c * np.sin(ph)))
    xi = np.arctan2(t, np.cos(la)); eta = np.arctanh(np.sin(la) / np.sqrt(1 + t ** 2))
    xs = xi + sum(al[j] * np.sin(2 * (j + 1) * xi) * np.cosh(2 * (j + 1) * eta) for j in range(4)); es = eta + sum(al[j] * np.cos(2 * (j + 1) * xi) * np.sinh(2 * (j + 1) * eta) for j in range(4))
    return 500000 + k0 * Ak * es, k0 * Ak * xs
b = zipfile.ZipFile(IN('AOI_Boundary.zip')).read('AOI_Boundary.shp'); pos = 100; verts = []; codes = []
while pos < len(b):
    cl = struct.unpack('>i', b[pos + 4:pos + 8])[0] * 2; c0 = pos + 8; pos = c0 + cl
    if struct.unpack('<i', b[c0:c0 + 4])[0] != 5: continue
    npart, npts = struct.unpack('<2i', b[c0 + 36:c0 + 44]); parts = list(struct.unpack('<%di' % npart, b[c0 + 44:c0 + 44 + 4 * npart])) + [npts]; p0 = c0 + 44 + 4 * npart
    pts = np.frombuffer(b[p0:p0 + 16 * npts], dtype='<f8').reshape(-1, 2)
    for i in range(npart):
        rr = pts[parts[i]:parts[i + 1]]; ex, ny = utm46(rr[:, 0], rr[:, 1]); v = np.column_stack([ex, ny]); verts += list(v) + [v[0]]; codes += [Path.MOVETO] + [Path.LINETO] * (len(v) - 1) + [Path.CLOSEPOLY]
aoi = Path(np.array(verts), codes)
# ------------------------------------------------------------ figure
INK = '#1A1A1A'; GREYT = '#444444'; FS = 7.0; FST = 8.5; FSL = 10.0; FSN = 6.7
FW = 7.48; MW = 2.30; M0 = 0.10; GAP = (FW - 2 * M0 - 3 * MW) / 2; MH = MW * (extent[3] - extent[2]) / (extent[1] - extent[0])
TT = 0.40; CBT = 0.22; CBH = 0.10; LBL = 0.40; FN = 0.52; FH = TT + MH + CBT + CBH + LBL + FN
fig = plt.figure(figsize=(FW, FH), facecolor='white'); xs_ = [M0 + i * (MW + GAP) for i in range(3)]; ymap = FN + LBL + CBH + CBT
axes = [fig.add_axes([x / FW, ymap / FH, MW / FW, MH / FH]) for x in xs_]
cmapA = plt.get_cmap('YlOrRd'); cmapB = plt.get_cmap('YlGnBu')
rd = plt.get_cmap('RdBu_r'); cols = [rd(v) for v in np.linspace(0.08, 0.92, 9)]; cols[4] = (0.93, 0.93, 0.93, 1.0); cmapC = ListedColormap(cols); normC = BoundaryNorm(np.arange(-4.5, 5.5, 1), 9)
lo_a, hi_a = np.percentile(m, [2, 98]); lo_b, hi_b = np.percentile(f, [2, 98])
data = [(ml_d, cmapA, Normalize(lo_a, hi_a)), (fa_d, cmapB, Normalize(lo_b, hi_b)), (Dd, cmapC, normC)]
tit = ['ML consensus susceptibility', 'FAHP susceptibility', 'ML–FAHP class difference']
for i, (ax, (arr, cmap, norm)) in enumerate(zip(axes, data)):
    ax.imshow(arr, cmap=cmap, norm=norm, extent=ext_d, interpolation='nearest', origin='upper', resample=False)
    ax.add_patch(PathPatch(aoi, facecolor='none', edgecolor=INK, linewidth=0.5))
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect('equal'); ax.axis('off')
    t1 = fig.text(xs_[i] / FW, (FH - 0.17) / FH, '(' + 'abc'[i] + ')', fontsize=FSL, fontweight='bold', color=INK, ha='left', va='baseline'); fig.canvas.draw(); wl = t1.get_window_extent().width / fig.dpi
    fig.text((xs_[i] + wl + 0.05) / FW, (FH - 0.17) / FH, tit[i], fontsize=FST, fontweight='bold', color=INK, ha='left', va='baseline')
cby = FN + LBL
def hbar(i, cmap, norm, ticks, labels, label, ext=None):
    cax = fig.add_axes([(xs_[i] + 0.06) / FW, cby / FH, (MW - 0.12) / FW, CBH / FH])
    cb = fig.colorbar(mcm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', ticks=ticks); cb.ax.set_xticklabels(labels, fontsize=FS); cb.ax.tick_params(length=2.5, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.5); cb.solids.set_edgecolor('face'); cb.solids.set_rasterized(False); fig.text((xs_[i] + MW / 2) / FW, (cby - 0.20) / FH, label, fontsize=FS, color=INK, ha='center', va='top'); return cax
hbar(0, cmapA, data[0][2], [lo_a, hi_a], ['Low', 'High'], 'Susceptibility score (ML consensus)')
hbar(1, cmapB, data[1][2], [lo_b, hi_b], ['Low', 'High'], 'Susceptibility index (FAHP)')
hbar(2, cmapC, normC, list(range(-4, 5)), ['−4', '−3', '−2', '−1', '0', '+1', '+2', '+3', '+4'], 'Class difference D = C(ML) − C(FAHP)')
yl = (cby + CBH + 0.09) / FH
fig.text((xs_[2] + 0.06) / FW, yl, '← FAHP higher', fontsize=FS, ha='left', va='center', color=INK, fontweight='bold')
fig.text((xs_[2] + MW / 2) / FW, yl, 'Same class', fontsize=FS, ha='center', va='center', color=GREYT)
fig.text((xs_[2] + MW - 0.06) / FW, yl, 'ML higher →', fontsize=FS, ha='right', va='center', color=INK, fontweight='bold')
# scale bar (panel a only) and north arrow (panel c only)
axA, axC = axes[0], axes[2]; xmin, xmax, ymin, ymax = extent
sbx, sby = xmin + 0.06 * (xmax - xmin), ymin + 0.06 * (ymax - ymin)
for k in range(4): axA.add_patch(Rectangle((sbx + k * 5000, sby), 5000, 1100, facecolor=INK if k % 2 == 0 else 'white', edgecolor=INK, lw=0.5, zorder=9))
for k, t_ in zip((0, 2, 4), ('0', '10', '20 km')): axA.text(sbx + k * 5000, sby + 2000, t_, ha='center', va='bottom', fontsize=FS - 0.3, zorder=9)
nx, ny0, L_ = xmin + 0.88 * (xmax - xmin), ymin + 0.86 * (ymax - ymin), 8500
axC.add_patch(Polygon([[nx, ny0 + L_], [nx - 1900, ny0], [nx, ny0 + 2300], [nx + 1900, ny0]], closed=True, facecolor=INK, edgecolor=INK, lw=0.5, zorder=9)); axC.text(nx, ny0 + L_ + 900, 'N', ha='center', va='bottom', fontsize=FS + 1.5, fontweight='bold', zorder=9)
# footnote
fn = ['Panels (a) and (b) use separate display scales (2nd–98th percentile stretch) and are not numerically comparable by colour. Panel (c) compares',
      'method-specific quintile classes (each surface classified at its own 20th, 40th, 60th and 80th percentiles on the common valid domain; D = 0 same class,',
      'D > 0 higher ML class, D < 0 higher FAHP class). White = outside the common valid domain (4,722,216 valid cells). Spatial comparison is not predictive accuracy.']
for k, t_ in enumerate(fn): fig.text(M0 / FW, (FN - 0.13 - k * 0.145) / FH, t_, fontsize=FSN, color=GREYT, ha='left', va='baseline')
base = os.path.join(a.out, 'Figure_6_CrossFamily_Spatial_Comparison_Q1_FINAL')
fig.savefig(base + '.png', dpi=a.dpi, facecolor='white'); fig.savefig(base + '.svg', facecolor='white', dpi=a.dpi); fig.savefig(base + '.pdf', facecolor='white', dpi=a.dpi)
fig.canvas.draw(); rend = fig.canvas.get_renderer(); boxes = []; bad = 0
for t in fig.texts + [x for ax in fig.axes[3:] for x in ax.get_xticklabels()] + [x for ax in fig.axes[:3] for x in ax.texts]:
    if not t.get_text(): continue
    bb = t.get_window_extent(rend); boxes.append((t.get_text(), bb, t.get_fontsize()))
    if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > FW * fig.dpi + 0.5 or bb.y1 > FH * fig.dpi + 0.5: bad += 1; print('OUT OF CANVAS:', t.get_text())
ov = 0
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        if boxes[i][1].overlaps(boxes[j][1]): ov += 1; print('OVERLAP:', boxes[i][0], '|', boxes[j][0])
print(f'canvas {FW:.2f} x {FH:.2f} in | map {MW:.2f} x {MH:.3f} in | gap {GAP:.3f} | display grid {ow} x {oh} (K={K}) | min font {min(b[2] for b in boxes):.1f} pt | overlaps {ov} | out-of-canvas {bad}')
