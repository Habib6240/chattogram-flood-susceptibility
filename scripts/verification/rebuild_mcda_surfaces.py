"""Recompute the AHP, FAHP and ANP susceptibility surfaces from the DU09 standardized rasters
and the archived weight tables, and compare them with the archived DU10/DU11/DU12 rasters.
Also recompute the Elevation standardized raster from the DEM with the archived P05/P95.

Usage: python rebuild_mcda_surfaces.py --du-tif DIR --du-csv DIR --mask 08_Final_Common_Analysis_Mask_30m.tif --dem DEM_Clipped_UTM46N.tif --out OUT.csv
Requires numpy, pandas, rasterio. Inputs are not redistributed in the software record (see README)."""
import argparse, numpy as np, pandas as pd, rasterio
ap = argparse.ArgumentParser()
for a in ('du-tif', 'du-csv', 'mask', 'dem', 'out'): ap.add_argument('--' + a, required=True)
g = ap.parse_args()
mask = rasterio.open(g.mask).read(1) == 1
crit = pd.read_csv(f'{g.du_csv}/DU10_AHP_Weights_and_Consistency.csv').iloc[:13]
names = list(crit.Criterion)
def ser(f, col):
    d = pd.read_csv(f'{g.du_csv}/{f}'); d = d[d.Criterion.isin(names)]
    return pd.Series(pd.to_numeric(d[col]).values, d.Criterion.values)
W = {'AHP': pd.Series(crit.Weight.astype(float).values, names),
     'FAHP': ser('DU11_FAHP_Weights.csv', 'Final_Normalized_Weight'),
     'ANP': ser('DU12_ANP_Final_Weights.csv', 'ANP_Weight')}
acc = {k: np.zeros(mask.shape, np.float64) for k in W}
for c in names:
    r = rasterio.open(f'{g.du_tif}/DU09_{c}_STD01.tif'); a = r.read(1).astype(np.float64)
    a = np.where(a == r.nodata, np.nan, a)
    for k in W: acc[k] += W[k][c] * a
rows = []
for k, f in (('AHP', 'DU10_AHP_Susceptibility_Index_01.tif'), ('FAHP', 'DU11_FAHP_Susceptibility_Index_01.tif'), ('ANP', 'DU12_ANP_Susceptibility_Index_01.tif')):
    a = rasterio.open(f'{g.du_tif}/{f}').read(1).astype(np.float64); d = np.abs(acc[k][mask] - a[mask])
    rows.append((k, int(mask.sum()), float(W[k].sum()), float(np.nanmax(d)), float(np.nanmean(d)), int(np.isnan(d).sum())))
st = pd.read_csv(f'{g.du_csv}/DU09_Standardization_Rules.csv').set_index('Criterion')
lo, hi = float(st.loc['Elevation', 'P05_or_Lower_Control']), float(st.loc['Elevation', 'P95_or_Upper_Control'])
dem = rasterio.open(g.dem).read(1).astype(np.float64)
calc = np.clip((hi - dem) / (hi - lo), 0, 1)
arc = rasterio.open(f'{g.du_tif}/DU09_Elevation_STD01.tif').read(1).astype(np.float64)
d = np.abs(calc[mask] - arc[mask]); rows.append(('Elevation_STD01_from_DEM', int(mask.sum()), np.nan, float(d.max()), float(d.mean()), 0))
pd.DataFrame(rows, columns=['Surface', 'Cells_in_mask', 'Weight_sum', 'Max_abs_difference', 'Mean_abs_difference', 'NaN_cells']).to_csv(g.out, index=False)
print(pd.DataFrame(rows).to_string())
