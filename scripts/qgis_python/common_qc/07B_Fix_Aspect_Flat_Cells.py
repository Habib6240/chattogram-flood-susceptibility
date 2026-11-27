from qgis.core import QgsProject
from osgeo import gdal
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

ASPECT_NAME = "Aspect_deg_30"
SLOPE_NAME = "Slope_deg_30m"
LULC_NAME = "02_LULC_ESA_WorldCover_2021_Chattogram_30m_STACK"

OUTPUT_PATH = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
    r"\Aspect_deg_30_FLATFIX.tif"
)

OUTPUT_NODATA = -9999.0
FLAT_CODE = -1.0


# ============================================================
# HELPER
# ============================================================

def open_layer(name):

    layers = QgsProject.instance().mapLayersByName(name)

    if not layers:
        raise Exception("Layer not found: " + name)

    ds = gdal.Open(layers[0].source())

    if ds is None:
        raise Exception("Could not open: " + name)

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float32)

    nd = band.GetNoDataValue()

    valid = np.isfinite(arr)

    if nd is not None:

        if np.isnan(nd):
            valid &= ~np.isnan(arr)

        else:
            valid &= (arr != nd)

    return ds, arr, valid


# ============================================================
# LOAD
# ============================================================

aspect_ds, aspect, aspect_valid = open_layer(ASPECT_NAME)
slope_ds, slope, slope_valid = open_layer(SLOPE_NAME)
lulc_ds, lulc, lulc_valid = open_layer(LULC_NAME)


# ============================================================
# TERRESTRIAL DOMAIN
# ============================================================

terrestrial = (
    lulc_valid
    &
    (lulc != 0)
    &
    (lulc != 80)
)


# ============================================================
# IDENTIFY TRUE FLAT CELLS
# ============================================================

flat_cells = (
    terrestrial
    &
    (~aspect_valid)
    &
    slope_valid
    &
    np.isclose(slope, 0.0, atol=1e-10)
)

n_flat = np.count_nonzero(flat_cells)


# ============================================================
# CREATE CORRECTED ASPECT
# ============================================================

corrected = np.full(
    aspect.shape,
    OUTPUT_NODATA,
    dtype=np.float32
)

# retain original valid aspect
corrected[aspect_valid] = aspect[aspect_valid]

# assign explicit Flat code
corrected[flat_cells] = FLAT_CODE


# ============================================================
# EXPORT
# ============================================================

driver = gdal.GetDriverByName("GTiff")

out = driver.Create(
    OUTPUT_PATH,
    aspect_ds.RasterXSize,
    aspect_ds.RasterYSize,
    1,
    gdal.GDT_Float32,
    options=["COMPRESS=LZW"]
)

out.SetGeoTransform(
    aspect_ds.GetGeoTransform()
)

out.SetProjection(
    aspect_ds.GetProjection()
)

band = out.GetRasterBand(1)

band.WriteArray(corrected)
band.SetNoDataValue(OUTPUT_NODATA)
band.FlushCache()

out = None


# ============================================================
# QC
# ============================================================

valid_after = (
    np.isfinite(corrected)
    &
    (corrected != OUTPUT_NODATA)
)

remaining_missing_land = (
    terrestrial
    &
    (~valid_after)
)

print("\n========== ASPECT FLAT-CELL CORRECTION ==========")

print(
    "Original missing terrestrial :",
    f"{np.count_nonzero(terrestrial & (~aspect_valid)):,}"
)

print(
    "Flat cells assigned -1       :",
    f"{n_flat:,}"
)

print(
    "Remaining missing terrestrial:",
    f"{np.count_nonzero(remaining_missing_land):,}"
)

print(
    "Output                       :",
    OUTPUT_PATH
)

print("\nEncoding:")
print("0–360 deg = directional aspect")
print("-1        = Flat terrain")
print("-9999     = NoData")

print("\nFINAL STATUS: ASPECT FLAT CORRECTION COMPLETE")
print("================================================")