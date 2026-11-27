from qgis.core import QgsProject
from osgeo import gdal
import numpy as np
import os

GROUP_NAME = "07_FINAL_FACTORS_PreScreening"

# Output
output_path = r"D:\Research\Conference Paper\Fully New Work\QGIS & Here Generated File\07_Common_Analysis_Mask_30m.tif"

root = QgsProject.instance().layerTreeRoot()
group = root.findGroup(GROUP_NAME)

if group is None:
    raise Exception("Group not found: " + GROUP_NAME)

layers = [n.layer() for n in group.findLayers()]

if len(layers) != 14:
    raise Exception(f"Expected 14 rasters, found {len(layers)}")

# Reference raster
ref = QgsProject.instance().mapLayersByName("DEM_Clipped_UTM46N")[0]
ref_ds = gdal.Open(ref.source())

cols = ref_ds.RasterXSize
rows = ref_ds.RasterYSize
gt = ref_ds.GetGeoTransform()
proj = ref_ds.GetProjection()

# Start with all pixels valid
common_valid = np.ones((rows, cols), dtype=bool)

lulc_arr = None

print("\n=========== COMMON VALID-DATA AUDIT ===========")

for lyr in layers:

    ds = gdal.Open(lyr.source())
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()

    nodata = band.GetNoDataValue()

    # Generic validity
    valid = np.isfinite(arr)

    if nodata is not None:
        if np.isnan(nodata):
            valid &= ~np.isnan(arr)
        else:
            valid &= (arr != nodata)

    # LULC special handling
    if lyr.name().startswith("02_LULC"):
        lulc_arr = arr.copy()

        # 0 = NoData/outside analytical source
        valid &= (arr != 0)

        # 80 = Permanent Water Bodies
        valid &= (arr != 80)

    common_valid &= valid

    print(
        f"{lyr.name():65s} "
        f"Valid pixels = {np.count_nonzero(valid):,}"
    )

# ---------- Coverage statistics ----------

if lulc_arr is None:
    raise Exception("LULC layer not found.")

aoi_pixels = np.count_nonzero(lulc_arr != 0)

terrestrial_pixels = np.count_nonzero(
    (lulc_arr != 0) &
    (lulc_arr != 80)
)

common_pixels = np.count_nonzero(common_valid)

coverage_aoi = 100.0 * common_pixels / aoi_pixels
coverage_land = 100.0 * common_pixels / terrestrial_pixels

pixel_area_km2 = (30 * 30) / 1_000_000

print("\n================ SUMMARY ================")
print(f"AOI valid pixels including water : {aoi_pixels:,}")
print(f"Terrestrial AOI pixels           : {terrestrial_pixels:,}")
print(f"Common valid pixels              : {common_pixels:,}")
print(f"Common valid area                : {common_pixels * pixel_area_km2:,.2f} km²")
print(f"Coverage of total AOI            : {coverage_aoi:.2f}%")
print(f"Coverage of terrestrial AOI      : {coverage_land:.2f}%")

# ---------- Export mask ----------

driver = gdal.GetDriverByName("GTiff")

out = driver.Create(
    output_path,
    cols,
    rows,
    1,
    gdal.GDT_Byte,
    options=["COMPRESS=LZW"]
)

out.SetGeoTransform(gt)
out.SetProjection(proj)

out_band = out.GetRasterBand(1)

# 1 = usable analysis pixel, 0 = excluded
mask_array = common_valid.astype(np.uint8)

out_band.WriteArray(mask_array)
out_band.SetNoDataValue(0)
out_band.FlushCache()

out = None
ref_ds = None

print("\nMask saved:")
print(output_path)
print("\nFINAL: 1 = common terrestrial valid pixel; 0 = excluded")
print("==========================================")


from qgis.core import QgsProject
from osgeo import gdal
import numpy as np

GROUP_NAME = "07_FINAL_FACTORS_PreScreening"

root = QgsProject.instance().layerTreeRoot()
group = root.findGroup(GROUP_NAME)

layers = [n.layer() for n in group.findLayers()]

# Find LULC
lulc = None
for lyr in layers:
    if lyr.name().startswith("02_LULC"):
        lulc = lyr
        break

if lulc is None:
    raise Exception("LULC not found")

ds = gdal.Open(lulc.source())
b = ds.GetRasterBand(1)
lulc_arr = b.ReadAsArray()
lulc_nd = b.GetNoDataValue()

terrestrial = np.isfinite(lulc_arr)

if lulc_nd is not None:
    terrestrial &= (lulc_arr != lulc_nd)

terrestrial &= (lulc_arr != 0)
terrestrial &= (lulc_arr != 80)

n_land = np.count_nonzero(terrestrial)

print("\n========= TERRESTRIAL COVERAGE BY FACTOR =========")

for lyr in layers:

    ds = gdal.Open(lyr.source())
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    nd = band.GetNoDataValue()

    valid = np.isfinite(arr)

    if nd is not None:
        if np.isnan(nd):
            valid &= ~np.isnan(arr)
        else:
            valid &= (arr != nd)

    valid_land = valid & terrestrial

    n_valid = np.count_nonzero(valid_land)
    n_missing = n_land - n_valid
    pct = 100.0 * n_valid / n_land
    area_missing = n_missing * 0.0009

    print(
        f"{lyr.name():65s} "
        f"Coverage={pct:6.2f}%  "
        f"Missing={n_missing:8,d} px  "
        f"({area_missing:7.2f} km2)"
    )

print("\nTerrestrial pixels:", f"{n_land:,}")
print("==================================================")