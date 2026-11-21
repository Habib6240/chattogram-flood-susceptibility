#!/usr/bin/env python3
"""Supplementary Figure S1: the 13 final conditioning factors on the common 30 m analysis domain.

Usage:  python build_figure_s1.py --data DIR [--out DIR]
DIR must contain the 13 final factor GeoTIFFs, 08_Final_Common_Analysis_Mask_30m.tif, 07A_Selected13_Factor_Audit.csv,
07B_Final_Mask_Summary.csv and AOI_Boundary.zip (shapefile, WGS 84). Requires numpy, pandas, Pillow and matplotlib only
(GeoTIFF headers are read directly; the shapefile is parsed and projected to UTM zone 46N without GIS libraries).
Read-only: no raster is modified, and no value is estimated. Writes figure files and QA helper tables."""
import os, sys, io, math, struct, zipfile, argparse
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, FancyArrow
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
args = ap.parse_args(); D, OUT = args.data, args.out
os.makedirs(os.path.join(OUT, "helper_tables"), exist_ok=True)

# ------------------------------------------------------------------ factor register (display order follows Table S3)
FACT = [  # key, title, file stem, kind, unit label, cmap
 ("Rainfall", "Rainfall", "01_Rainfall_CHIRPSv3_MAP_1991_2020_Chattogram_30m_STACK", "cont", "mm yr\u207b\u00b9", "Blues"),
 ("Elevation", "Elevation", "DEM_Clipped_UTM46N", "cont", "m", "cividis"),
 ("Slope", "Slope", "Slope_deg_30m", "cont", "degrees", "magma_r"),
 ("Drainage_Density", "Drainage density", "Drainage_Density_Chittagong_30m_STACK", "rel", "", "YlGnBu"),
 ("Distance_River", "Distance to river", "05_Distance_to_River_HydroRIVERS_UPLAND25_Chattogram_30m_STACK", "cont", "m", "viridis"),
 ("TWI", "TWI", "TWI_Chittagong_30m_STACK", "cont", "", "PuBuGn"),
 ("LULC", "LULC", "02_LULC_ESA_WorldCover_2021_Chattogram_30m_STACK", "cat", "", None),
 ("HSG", "HSG", "04_Soil_HYSOGs250m_Chattogram_30m_STACK", "cat", "", None),
 ("NDVI", "NDVI", "03_NDVI_Sentinel2_Median_2021_Chattogram_30m_STACK", "cont", "", "YlGn"),
 ("Distance_Road", "Distance to road", "06_Distance_to_Road_OSM2021_Chattogram_30m_STACK", "cont", "m", "viridis"),
 ("Profile_Curvature", "Profile curvature", "Profile_Curvature_30m", "relsym", "", "PuOr"),
 ("Tangential_Curvature", "Tangential curvature", "Tangential_Curvature_30m", "relsym", "", "PuOr"),
 ("Aspect", "Aspect", "Aspect_deg_30_FLATFIX", "aspect", "degrees", "twilight"),
]
NODATA = {"Rainfall": -9999, "Elevation": -9999, "Slope": -9999, "Drainage_Density": -9999, "Distance_River": -9999, "TWI": -9999, "LULC": 0, "HSG": 255, "NDVI": -9999, "Distance_Road": -9999,
          "Profile_Curvature": np.nan, "Tangential_Curvature": np.nan, "Aspect": -9999}
LULC = {10: ("Tree cover", "#006400"), 20: ("Shrubland", "#ffbb22"), 30: ("Grassland", "#ffff4c"), 40: ("Cropland", "#f096ff"), 50: ("Built-up", "#fa0000"), 60: ("Bare / sparse vegetation", "#b4b4b4"),
        90: ("Herbaceous wetland", "#0096a0"), 95: ("Mangroves", "#00cf75")}
HSG = {3: ("C", "#fee391"), 13: ("C/D", "#fe9929"), 4: ("D", "#cc4c02"), 14: ("D/D", "#662506")}

def rd(stem): return np.array(Image.open(os.path.join(D, stem + ".tif")))

def geoinfo(path):
    """Read pixel scale, tie point, raster type and EPSG from the GeoTIFF header (tags 33550, 33922, 34735)."""
    im = Image.open(path); t = im.tag_v2
    ps = t.get(33550); tp = t.get(33922); gk = t.get(34735); keys = {}
    for i in range(gk[3]):
        k, loc, cnt, val = gk[4 + 4 * i: 8 + 4 * i]; keys[k] = val
    rtype = keys.get(1025); epsg = keys.get(3072)
    x0, y0 = tp[3], tp[4]
    if rtype == 2:  # PixelIsPoint: the tie point is a pixel centre
        x0 -= ps[0] / 2; y0 += ps[1] / 2
    return dict(width=im.size[0], height=im.size[1], px=ps[0], py=ps[1], x0=x0, y0=y0, raster_type="PixelIsPoint" if rtype == 2 else "PixelIsArea", epsg=epsg)

# ------------------------------------------------------------------ AOI shapefile -> UTM 46N (pure Python)
def read_shp(zpath):
    z = zipfile.ZipFile(zpath); b = z.read("AOI_Boundary.shp"); parts_out = []
    pos = 100
    while pos < len(b):
        n, = struct.unpack(">i", b[pos + 4:pos + 8]); rec = b[pos + 8: pos + 8 + n * 2]; pos += 8 + n * 2
        st, = struct.unpack("<i", rec[:4])
        if st != 5: continue
        npart, npt = struct.unpack("<ii", rec[36:44]); parts = list(struct.unpack(f"<{npart}i", rec[44:44 + 4 * npart])) + [npt]
        pts = np.frombuffer(rec[44 + 4 * npart: 44 + 4 * npart + 16 * npt], dtype="<f8").reshape(-1, 2)
        for i in range(npart): parts_out.append(pts[parts[i]:parts[i + 1]])
    return parts_out

def utm46(lon, lat):
    a = 6378137.0; f = 1 / 298.257223563; k0 = 0.9996; lon0 = math.radians(93.0)
    n = f / (2 - f); A = a / (1 + n) * (1 + n**2 / 4 + n**4 / 64)
    al = [n / 2 - 2 * n**2 / 3 + 5 * n**3 / 16, 13 * n**2 / 48 - 3 * n**3 / 5, 61 * n**3 / 240]
    phi = np.radians(lat); lam = np.radians(lon) - lon0
    e = math.sqrt(f * (2 - f)); t = np.sinh(np.arctanh(np.sin(phi)) - e * np.arctanh(e * np.sin(phi)))
    xi = np.arctan2(t, np.cos(lam)); eta = np.arctanh(np.sin(lam) / np.sqrt(1 + t**2))
    X = eta.copy(); Y = xi.copy()
    for j, c in enumerate(al, 1):
        X += c * np.cos(2 * j * xi) * np.sinh(2 * j * eta); Y += c * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    return 500000 + k0 * A * X, k0 * A * Y

# ------------------------------------------------------------------ read mask, grid facts
mask = rd("08_Final_Common_Analysis_Mask_30m")
g0 = geoinfo(os.path.join(D, "08_Final_Common_Analysis_Mask_30m.tif")); H, W = mask.shape
audit = pd.read_csv(os.path.join(D, "07A_Selected13_Factor_Audit.csv")); summ = pd.read_csv(os.path.join(D, "07B_Final_Mask_Summary.csv")).set_index("Metric").Value
inm = mask == 1
rows_grid = []; rows_valid = []; rows_stats = []; rows_cls = []; sub = {}; STR = 3
for key, title, stem, kind, unit, cmap in FACT:
    gi = geoinfo(os.path.join(D, stem + ".tif")); a = rd(stem)
    nd = NODATA[key]
    valid = np.isfinite(a) if (isinstance(nd, float) and np.isnan(nd)) else (a != nd)
    if key in ("Rainfall", "Elevation", "Slope", "Drainage_Density", "Distance_River", "TWI", "NDVI", "Distance_Road", "Aspect", "Profile_Curvature", "Tangential_Curvature"):
        valid &= np.isfinite(a)
    nvalid_file = int(valid.sum()); nvalid_audit_basis = int((valid & (a != 80)).sum()) if key == "LULC" else nvalid_file
    ar = audit[audit.Factor == stem].iloc[0]
    rows_grid.append(dict(Factor=title, File=stem + ".tif", Width=gi["width"], Height=gi["height"], Pixel_X_m=gi["px"], Pixel_Y_m=gi["py"], Raster_type=gi["raster_type"], EPSG=gi["epsg"], UL_X=round(gi["x0"], 4), UL_Y=round(gi["y0"], 4),
                          dX_vs_mask_m=round(gi["x0"] - g0["x0"], 6), dY_vs_mask_m=round(gi["y0"] - g0["y0"], 6), Audit_extent_diff_Xmin_m=ar.Extent_Diff_Xmin_m, Audit_extent_diff_Ymin_m=ar.Extent_Diff_Ymin_m,
                          Same_CRS=gi["epsg"] == g0["epsg"] == 32646, Same_cell=(gi["px"], gi["py"]) == (30.0, 30.0), Same_dims=(gi["width"], gi["height"]) == (W, H),
                          Compatible_extent=abs(gi["x0"] - g0["x0"]) < 1e-3 and abs(gi["y0"] - g0["y0"]) < 1e-3))
    rows_valid.append(dict(Factor=title, File=stem + ".tif", NoData_value=nd, Valid_pixels_file=nvalid_file, Valid_pixels_audit_basis=nvalid_audit_basis, Valid_pixels_audit_csv=int(ar.Valid_Pixels), Match_audit=nvalid_audit_basis == int(ar.Valid_Pixels),
                           Valid_inside_final_mask=int((valid & inm).sum()), Mask_cells=int(inm.sum()), All_mask_cells_valid=int((valid & inm).sum()) == int(inm.sum()), Note=("Audit basis excludes WorldCover class 80 (open water)" if key == "LULC" else "")))
    v = a[valid & inm]
    if kind in ("cont", "rel", "relsym"):
        lo, hi = np.percentile(v, [2, 98]); rows_stats.append(dict(Factor=title, Kind=kind, N=int(v.size), Min=float(v.min()), P02=float(lo), Median=float(np.median(v)), P98=float(hi), Max=float(v.max()), Mean=float(v.mean())))
    elif kind == "aspect":
        rows_stats.append(dict(Factor=title, Kind=kind, N=int(v.size), Min=float(v.min()), P02=float("nan"), Median=float(np.median(v)), P98=float("nan"), Max=float(v.max()), Mean=float(v.mean())))
    else:
        u, c = np.unique(v, return_counts=True)
        for uu, cc in zip(u, c): rows_cls.append(dict(Factor=title, Code=int(uu), Class=(LULC.get(int(uu), HSG.get(int(uu), ("unlisted",)))[0] if key == "LULC" else HSG.get(int(uu), ("unlisted",))[0]), Cells=int(cc)))
        rows_stats.append(dict(Factor=title, Kind=kind, N=int(v.size), Min=float(v.min()), P02=float("nan"), Median=float(np.median(v)), P98=float("nan"), Max=float(v.max()), Mean=float("nan")))
    sub[key] = (a[::STR, ::STR].copy(), valid[::STR, ::STR].copy())
    del a, valid
# mask logic
mask_rows = [dict(Check="Mask encoding", Value="unique values " + str(sorted(np.unique(mask).tolist())), Expected="1 usable, 0 excluded", Result="PASS" if set(np.unique(mask).tolist()) <= {0, 1} else "FAIL"),
             dict(Check="Final valid cells", Value=int(inm.sum()), Expected=int(summ["Final_Common_Valid_Pixels"]), Result="PASS" if int(inm.sum()) == int(summ["Final_Common_Valid_Pixels"]) else "FAIL"),
             dict(Check="Analysis area km2 (cells x 900 m2)", Value=round(inm.sum() * 900 / 1e6, 4), Expected=float(summ["Final_Analysis_Area_km2"]), Result="PASS" if abs(inm.sum() * 900 / 1e6 - float(summ["Final_Analysis_Area_km2"])) < 1e-3 else "FAIL"),
             dict(Check="Excluded terrestrial cells", Value=int(summ["Terrestrial_AOI_Pixels"]) - int(inm.sum()), Expected=int(summ["Excluded_Terrestrial_Pixels"]), Result="PASS" if int(summ["Terrestrial_AOI_Pixels"]) - int(inm.sum()) == int(summ["Excluded_Terrestrial_Pixels"]) else "FAIL"),
             dict(Check="Factors valid on every mask cell", Value=f"{sum(r['All_mask_cells_valid'] for r in rows_valid)} of 13", Expected="13 of 13", Result="PASS" if all(r["All_mask_cells_valid"] for r in rows_valid) else "FAIL"),
             dict(Check="Grid checks passed (audit CSV)", Value=int(summ["Grid_Checks_Passed"]), Expected=13, Result="PASS" if int(summ["Grid_Checks_Passed"]) == 13 else "FAIL")]
# AOI polygon rasterisation on the 30 m grid
parts = read_shp(os.path.join(D, "AOI_Boundary.zip")); polys = []
for p in parts:
    x, y = utm46(p[:, 0], p[:, 1]); polys.append(np.column_stack([x, y]))
def area(p): return 0.5 * np.sum(p[:-1, 0] * p[1:, 1] - p[1:, 0] * p[:-1, 1])
img = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(img)
outer = [p for p in polys if area(p) < 0]; holes = [p for p in polys if area(p) > 0]   # shapefile: outer rings clockwise
for p in outer: dr.polygon([((x - g0["x0"]) / 30.0, (g0["y0"] - y) / 30.0) for x, y in p], fill=1)
for p in holes: dr.polygon([((x - g0["x0"]) / 30.0, (g0["y0"] - y) / 30.0) for x, y in p], fill=0)
aoi = np.array(img) == 1
bx = np.concatenate([p[:, 0] for p in polys]); by = np.concatenate([p[:, 1] for p in polys])
ext = (g0["x0"], g0["x0"] + W * 30.0, g0["y0"] - H * 30.0, g0["y0"])
aoi_rows = [dict(Check="AOI rings read (outer / hole)", Value=f"{len(outer)} / {len(holes)}", Expected="-", Result="INFO"),
            dict(Check="AOI projected bounding box vs raster extent (m)", Value=f"x {bx.min():.0f}-{bx.max():.0f}; y {by.min():.0f}-{by.max():.0f}", Expected=f"x {ext[0]:.0f}-{ext[1]:.0f}; y {ext[2]:.0f}-{ext[3]:.0f}", Result="INFO"),
            dict(Check="Mask cells inside the rasterised AOI", Value=f"{int((inm & aoi).sum()):,} of {int(inm.sum()):,} ({(inm & aoi).sum() / inm.sum() * 100:.3f}%)", Expected=">= 99.9%", Result="PASS" if (inm & aoi).sum() / inm.sum() > 0.999 else "NOTE"),
            dict(Check="Rasterised AOI cells (AOI polygon at 30 m)", Value=int(aoi.sum()), Expected="file valid cells 4,984,270 (all factors share this footprint)", Result="INFO")]
pd.DataFrame(rows_grid).to_csv(os.path.join(OUT, "helper_tables", "grid_alignment_check.csv"), index=False)
pd.DataFrame(rows_valid).to_csv(os.path.join(OUT, "helper_tables", "valid_pixel_check.csv"), index=False)
pd.DataFrame(rows_stats).to_csv(os.path.join(OUT, "helper_tables", "display_statistics.csv"), index=False)
pd.DataFrame(rows_cls).to_csv(os.path.join(OUT, "helper_tables", "categorical_class_counts.csv"), index=False)
pd.DataFrame(mask_rows + aoi_rows).to_csv(os.path.join(OUT, "helper_tables", "mask_and_aoi_check.csv"), index=False)
pd.DataFrame([dict(Panel=chr(97 + i), Factor=t, File=s + ".tif", Kind=k, Unit=u or "not stated / relative scale", Colour_ramp=(c or "categorical")) for i, (key, t, s, k, u, c) in enumerate(FACT)]).to_csv(os.path.join(OUT, "helper_tables", "panel_file_map.csv"), index=False)
ST = {r["Factor"]: r for r in rows_stats}

# ------------------------------------------------------------------ figure
FW, FH = 7.2, 7.9
fig = plt.figure(figsize=(FW, FH), dpi=300)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "svg.fonttype": "path"})
colw = (FW - 0.16) / 5; mw = colw - 0.14; mh = mw * H / W; rowh = mh + 0.62
AOIL = [((p[:, 0] - ext[0]) / 1000.0, (p[:, 1] - ext[2]) / 1000.0) for p in polys]
EXT = [0, (ext[1] - ext[0]) / 1000.0, 0, (ext[3] - ext[2]) / 1000.0]
def rect(r, c, kind):
    x = 0.08 + c * colw + 0.07; ytop = FH - 0.12 - r * rowh
    if kind == "title": return [x / FW, (ytop - 0.17) / FH, mw / FW, 0.17 / FH]
    if kind == "map": return [x / FW, (ytop - 0.17 - mh) / FH, mw / FW, mh / FH]
    if kind == "cbar": return [(x + 0.08) / FW, (ytop - 0.17 - mh - 0.14) / FH, (mw - 0.16) / FW, 0.07 / FH]
def mapax(r, c):
    ax = fig.add_axes(rect(r, c, "map")); ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3]); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    return ax
def draw_aoi(ax):
    for x, y in AOIL: ax.plot(x, y, color="#111", lw=0.35, zorder=5)
def title(r, c, s):
    ax = fig.add_axes(rect(r, c, "title")); ax.axis("off"); ax.text(0, 0.1, s, fontsize=7, fontweight="bold", va="bottom", ha="left")
def cbar(r, c, cmap, vlo, vhi, lo_lab, hi_lab, cyc=False):
    ax = fig.add_axes(rect(r, c, "cbar")); g = np.linspace(0, 1, 256)[None, :]; ax.imshow(g, aspect="auto", cmap=cmap); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.3)
    ax.text(0, -0.9, lo_lab, ha="left", va="top", fontsize=5.8, transform=ax.transAxes); ax.text(1, -0.9, hi_lab, ha="right", va="top", fontsize=5.8, transform=ax.transAxes)
sm_mask = inm[::STR, ::STR]
def fmt(v, dec): return f"{v:,.{dec}f}"
dec = {"Rainfall": 0, "Elevation": 0, "Slope": 1, "TWI": 1, "NDVI": 2, "Distance_River": 0, "Distance_Road": 0}
for i, (key, ttl, stem, kind, unit, cmap) in enumerate(FACT):
    r, c = divmod(i, 5); a, valid = sub[key]; show = valid & sm_mask
    title(r, c, f"{chr(97 + i)}  {ttl}")
    ax = mapax(r, c)
    if kind == "cat":
        cl = LULC if key == "LULC" else HSG; codes = sorted(cl); cols = [cl[k][1] for k in codes]
        idx = np.full(a.shape, np.nan); 
        for j, k in enumerate(codes): idx[(a == k) & show] = j
        ax.imshow(np.ma.masked_invalid(idx), cmap=ListedColormap(cols), vmin=-0.5, vmax=len(codes) - 0.5, extent=EXT, origin="upper", interpolation="nearest", zorder=2)
        ax2 = fig.add_axes(rect(r, c, "cbar")); ax2.axis("off"); ax2.text(0.5, 0.2, "categorical (legend in panel n)", ha="center", va="top", fontsize=5.8)
    elif kind == "aspect":
        flat = (a == -1) & show; cont = show & (a >= 0)
        ax.imshow(np.ma.masked_where(~cont, a), cmap="twilight", vmin=0, vmax=360, extent=EXT, origin="upper", interpolation="nearest", zorder=2)
        ax.imshow(np.ma.masked_where(~flat, np.ones(a.shape)), cmap=ListedColormap(["#808080"]), extent=EXT, origin="upper", interpolation="nearest", zorder=3)
        cbar(r, c, "twilight", 0, 360, "0\u00b0", "360\u00b0 (flat: grey)")
    else:
        s = ST[ttl]; lo, hi = s["P02"], s["P98"]
        if kind == "relsym": m_ = max(abs(lo), abs(hi)); lo, hi = -m_, m_
        ax.imshow(np.ma.masked_where(~show, a), cmap=cmap, vmin=lo, vmax=hi, extent=EXT, origin="upper", interpolation="nearest", zorder=2)
        if kind == "cont": cbar(r, c, cmap, lo, hi, fmt(lo, dec[key]), fmt(hi, dec[key]) + (" " + unit if unit else ""))
        else: cbar(r, c, cmap, lo, hi, "Low", "High")
    draw_aoi(ax)
# spare cells: row 2, columns 3 and 4
def spare_axes(c):
    ax = fig.add_axes(rect(2, c, "map")); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); return ax
axn = spare_axes(3)
axn.text(0, 1.0, "n  Reading guide", fontsize=7, fontweight="bold", va="top")
# legends
yy = 0.93
axn.text(0, yy, "LULC (ESA WorldCover 2021)", fontsize=5.8, fontweight="bold", va="top"); yy -= 0.05
for k in sorted(LULC):
    axn.add_patch(Rectangle((0, yy - 0.032), 0.09, 0.03, color=LULC[k][1], lw=0)); axn.text(0.12, yy - 0.017, LULC[k][0], fontsize=5.6, va="center"); yy -= 0.046
yy -= 0.02
axn.text(0, yy, "HSG (hydrologic soil group)", fontsize=5.8, fontweight="bold", va="top"); yy -= 0.045
for k in sorted(HSG, key=lambda z: ["C", "C/D", "D", "D/D"].index(HSG[z][0])):
    axn.add_patch(Rectangle((0, yy - 0.032), 0.09, 0.03, color=HSG[k][1], lw=0)); axn.text(0.12, yy - 0.017, HSG[k][0], fontsize=5.6, va="center"); yy -= 0.046
yy -= 0.02
axn.add_patch(Rectangle((0, yy - 0.032), 0.09, 0.03, color="#808080", lw=0)); axn.text(0.12, yy - 0.017, "Aspect: flat terrain", fontsize=5.6, va="center"); yy -= 0.05
axn.plot([0, 0.09], [yy - 0.017, yy - 0.017], color="#111", lw=0.6); axn.text(0.12, yy - 0.017, "Area-of-interest boundary", fontsize=5.6, va="center"); yy -= 0.046
axn.add_patch(Rectangle((0, yy - 0.032), 0.09, 0.03, facecolor="white", edgecolor="#999", lw=0.4)); axn.text(0.12, yy - 0.017, "Outside the analysis domain", fontsize=5.6, va="center")
# scale bar, north arrow, notes
axs = fig.add_axes(rect(2, 4, "map")); axs.set_xlim(EXT[0], EXT[1]); axs.set_ylim(0, (EXT[3] - EXT[2])); axs.axis("off")
axs.text(0, EXT[3] - EXT[2], "o  Map notes", fontsize=7, fontweight="bold", va="top", ha="left")
Hk = EXT[3] - EXT[2]
axs.annotate("", xy=(EXT[1] * 0.5, Hk * 0.80), xytext=(EXT[1] * 0.5, Hk * 0.68), arrowprops=dict(arrowstyle="-|>", color="#111", lw=0.9)); axs.text(EXT[1] * 0.5, Hk * 0.81, "N", ha="center", va="bottom", fontsize=7, fontweight="bold")
axs.text(EXT[1] * 0.5, Hk * 0.655, "grid north", ha="center", va="top", fontsize=5.4)
axs.plot([0, 10], [Hk * 0.52, Hk * 0.52], color="#111", lw=1.6, solid_capstyle="butt"); axs.plot([0, 0], [Hk * 0.50, Hk * 0.54], color="#111", lw=0.8); axs.plot([10, 10], [Hk * 0.50, Hk * 0.54], color="#111", lw=0.8)
axs.text(0, Hk * 0.48, "0", fontsize=5.8, ha="center", va="top"); axs.text(12, Hk * 0.48, "10 km", fontsize=5.8, ha="left", va="top")
note = ("UTM zone 46N (EPSG:32646),\n30 m grid; all panels share\nthe same valid domain\n(4,722,216 cells).\n\nContinuous factors: 2nd\u201398th\npercentile display stretch\n(not the MCDA P05\u2013P95\nbounds); end colours hold\nvalues beyond it. Low/High\npanels use a relative scale.\nAspect: full 0\u2013360\u00b0 range.\nInputs only; no result\nis implied.")
axs.text(0, Hk * 0.38, note, fontsize=5.5, va="top", ha="left", linespacing=1.25)
for ext_ in ("png", "svg"):
    fig.savefig(os.path.join(OUT, f"Figure_S1_Conditioning_Factors_v1.{ext_}"), dpi=300, facecolor="white")
print("figure written; mask rows:", [r["Result"] for r in mask_rows], "aoi:", [r["Result"] for r in aoi_rows])
