from qgis.core import QgsProject
from osgeo import gdal
import numpy as np

ASPECT = "Aspect_deg_30"
SLOPE = "Slope_deg_30m"
LULC = "02_LULC_ESA_WorldCover_2021_Chattogram_30m_STACK"

def read_layer(name):
    lyr = QgsProject.instance().mapLayersByName(name)
    if not lyr:
        raise Exception("Layer not found: " + name)

    ds = gdal.Open(lyr[0].source())
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(float)
    nd = band.GetNoDataValue()

    valid = np.isfinite(arr)

    if nd is not None:
        if np.isnan(nd):
            valid &= ~np.isnan(arr)
        else:
            valid &= (arr != nd)

    return arr, valid

aspect, aspect_valid = read_layer(ASPECT)
slope, slope_valid = read_layer(SLOPE)
lulc, lulc_valid = read_layer(LULC)

# terrestrial domain
terrestrial = (
    lulc_valid &
    (lulc != 0) &
    (lulc != 80)
)

aspect_missing_land = (
    terrestrial &
    (~aspect_valid)
)

# only locations where slope itself is valid
check = (
    aspect_missing_land &
    slope_valid
)

vals = slope[check]

print("\n========== ASPECT MISSING-PIXEL QC ==========")
print("Terrestrial pixels with missing Aspect :", 
      f"{np.count_nonzero(aspect_missing_land):,}")
print("Of these with valid Slope              :", 
      f"{len(vals):,}")

if len(vals) > 0:
    print(f"Slope minimum                           : {np.min(vals):.6f} deg")
    print(f"Slope median                            : {np.median(vals):.6f} deg")
    print(f"Slope mean                              : {np.mean(vals):.6f} deg")
    print(f"Slope maximum                           : {np.max(vals):.6f} deg")

    print("\nSlope <= 0.1 deg :", 
          f"{np.count_nonzero(vals <= 0.1):,}",
          f"({100*np.mean(vals <= 0.1):.2f}%)")

    print("Slope <= 0.5 deg :", 
          f"{np.count_nonzero(vals <= 0.5):,}",
          f"({100*np.mean(vals <= 0.5):.2f}%)")

    print("Slope <= 1.0 deg :", 
          f"{np.count_nonzero(vals <= 1.0):,}",
          f"({100*np.mean(vals <= 1.0):.2f}%)")

print("=============================================")