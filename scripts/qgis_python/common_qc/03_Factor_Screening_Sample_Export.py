from qgis.core import QgsProject
from osgeo import gdal
import numpy as np
import csv
import os

# ============================================================
# SETTINGS
# ============================================================

GROUP_NAME = "07_FINAL_FACTORS_PreScreening"

MASK_NAME = "07_CandidateFactors_CommonValidMask_30m"

OUTPUT_CSV = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
    r"\Factor_Screening_Results\00_Samples"
    r"\03_Factor_Screening_Sample_1km.csv"
)

# Approx. 1 km systematic spacing on 30 m grid
STEP = 33


# ============================================================
# CONTINUOUS FACTORS ONLY
# ============================================================

factor_names = [
    "DEM_Clipped_UTM46N",
    "Slope_deg_30m",
    "TRI_Riley_30m",
    "Profile_Curvature_30m",
    "Tangential_Curvature_30m",
    "TWI_Chittagong_30m_STACK",
    "Drainage_Density_Chittagong_30m_STACK",
    "01_Rainfall_CHIRPSv3_MAP_1991_2020_Chattogram_30m_STACK",
    "03_NDVI_Sentinel2_Median_2021_Chattogram_30m_STACK",
    "05_Distance_to_River_HydroRIVERS_UPLAND25_Chattogram_30m_STACK",
    "06_Distance_to_Road_OSM2021_Chattogram_30m_STACK"
]


# ============================================================
# LOAD MASK
# ============================================================

mask_layers = QgsProject.instance().mapLayersByName(MASK_NAME)

if not mask_layers:
    raise Exception("Mask layer not found: " + MASK_NAME)

mask_ds = gdal.Open(mask_layers[0].source())
mask = mask_ds.GetRasterBand(1).ReadAsArray()

rows = mask_ds.RasterYSize
cols = mask_ds.RasterXSize
gt = mask_ds.GetGeoTransform()


# ============================================================
# LOAD FACTOR ARRAYS
# ============================================================

factor_arrays = {}

for name in factor_names:

    layers = QgsProject.instance().mapLayersByName(name)

    if not layers:
        raise Exception("Factor not found: " + name)

    ds = gdal.Open(layers[0].source())
    band = ds.GetRasterBand(1)

    arr = band.ReadAsArray().astype(np.float64)
    nd = band.GetNoDataValue()

    valid = np.isfinite(arr)

    if nd is not None:
        if np.isnan(nd):
            valid &= ~np.isnan(arr)
        else:
            valid &= (arr != nd)

    factor_arrays[name] = (arr, valid)


# ============================================================
# SYSTEMATIC SPATIAL SAMPLING
# ============================================================

samples = []

for r in range(STEP // 2, rows, STEP):

    for c in range(STEP // 2, cols, STEP):

        # Candidate-factor common mask
        if mask[r, c] != 1:
            continue

        values = []
        all_valid = True

        for name in factor_names:

            arr, valid = factor_arrays[name]

            if not valid[r, c]:
                all_valid = False
                break

            values.append(float(arr[r, c]))

        if not all_valid:
            continue

        # Pixel-center coordinates
        x = gt[0] + (c + 0.5) * gt[1]
        y = gt[3] + (r + 0.5) * gt[5]

        samples.append(
            [r, c, x, y] + values
        )


# ============================================================
# EXPORT CSV
# ============================================================

header = [
    "row",
    "col",
    "x_utm",
    "y_utm"
] + factor_names

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(samples)


print("\n================ SCREENING SAMPLE ================")
print("Systematic spacing :", STEP, "pixels")
print("Approx spacing     :", STEP * 30, "m")
print("Number of samples  :", len(samples))
print("Output CSV         :", OUTPUT_CSV)
print("==================================================")