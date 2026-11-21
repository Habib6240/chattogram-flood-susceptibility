#!/usr/bin/env python3
"""Step 1. Prepare the river network from the OFFICIAL HydroRIVERS v1.0 download.

Input : HydroRIVERS_v10_as.shp (Asia; download from https://www.hydrosheds.org/products/hydrorivers under its licence)
Output: <out>.npz (segments in EPSG:32646) -- a private working file, do not redistribute (HydroSHEDS licence).

Method (as in the manuscript): keep reaches with UPLAND_SKM >= 25 km2; reproject to EPSG:32646; keep only reaches within
(lattice bounding box + buffer). The buffer must exceed the largest river distance (14.03 km in the original data); default 25 km.
Requires geopandas, pyogrio/fiona, shapely, pyproj, pandas, numpy.  NOT executed in the authoring session (the official file
could not be downloaded there): see the regeneration note for what was and was not validated.
"""
import argparse, gzip, csv, sys, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--hydrorivers-shp", required=True); ap.add_argument("--master", required=True,
    help="public master CSV (.csv or .csv.gz) with X,Y columns")
ap.add_argument("--out", required=True); ap.add_argument("--min-upland-skm", type=float, default=25.0)
ap.add_argument("--buffer-m", type=float, default=25000.0); ap.add_argument("--also-gpkg", default=None)
a = ap.parse_args()
import geopandas as gpd; from pyproj import Transformer; from shapely.geometry import box
opener = gzip.open if a.master.endswith(".gz") else open
with opener(a.master, "rt", newline="") as f:
    r = csv.DictReader(f); xs, ys = [], []
    for row in r: xs.append(float(row["X"])); ys.append(float(row["Y"]))
x0, x1, y0, y1 = min(xs) - a.buffer_m, max(xs) + a.buffer_m, min(ys) - a.buffer_m, max(ys) + a.buffer_m
t = Transformer.from_crs(32646, 4326, always_xy=True)
lon, lat = t.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
bbox = (min(lon) - 0.1, min(lat) - 0.1, max(lon) + 0.1, max(lat) + 0.1)      # generous geographic pre-filter
g = gpd.read_file(a.hydrorivers_shp, bbox=bbox)
assert g.crs is not None and g.crs.to_epsg() == 4326, "HydroRIVERS is expected in EPSG:4326"
g = g[g["UPLAND_SKM"] >= a.min_upland_skm].to_crs(32646)
g = g[g.intersects(box(x0, y0, x1, y1))]
print("reaches kept:", len(g), "| UPLAND_SKM range", g["UPLAND_SKM"].min(), g["UPLAND_SKM"].max())
segs = []
for geom in g.geometry:
    for ln in (geom.geoms if geom.geom_type == "MultiLineString" else [geom]):
        c = np.asarray(ln.coords)[:, :2]; segs.append(np.hstack([c[:-1], c[1:]]))
np.savez_compressed(a.out, segments=np.vstack(segs), epsg=32646, min_upland_skm=a.min_upland_skm, n_reaches=len(g))
print("segments:", sum(len(s) for s in segs), "->", a.out)
if a.also_gpkg: g.to_file(a.also_gpkg, driver="GPKG")
