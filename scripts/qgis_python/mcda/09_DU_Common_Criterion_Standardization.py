from qgis.core import QgsProject
from osgeo import gdal
import numpy as np
import os
import csv

# ============================================================
# PATHS
# ============================================================

BASE = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
)

STATS_CSV = os.path.join(
    BASE,
    "12_DU_MCDA",
    "00_Input_Standardization_QC",
    "08A_Continuous_Factor_Statistics.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "12_DU_MCDA",
    "01_Standardized_Criteria"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MASK_NAME = "08_Final_Common_Analysis_Mask_30m"

NODATA = -9999.0


# ============================================================
# CONTINUOUS FACTOR DIRECTIONS
#
# increasing:
#     larger raw value = larger flood susceptibility
#
# decreasing:
#     smaller raw value = larger flood susceptibility
# ============================================================

CONTINUOUS = {

    "DEM_Clipped_UTM46N":
        ("Elevation", "decreasing"),

    "Slope_deg_30m":
        ("Slope", "decreasing"),

    "Profile_Curvature_30m":
        ("Profile_Curvature", "decreasing"),

    "Tangential_Curvature_30m":
        ("Tangential_Curvature", "decreasing"),

    "TWI_Chittagong_30m_STACK":
        ("TWI", "increasing"),

    "Drainage_Density_Chittagong_30m_STACK":
        ("Drainage_Density", "increasing"),

    "01_Rainfall_CHIRPSv3_MAP_1991_2020_Chattogram_30m_STACK":
        ("Rainfall", "increasing"),

    "03_NDVI_Sentinel2_Median_2021_Chattogram_30m_STACK":
        ("NDVI", "decreasing"),

    "05_Distance_to_River_HydroRIVERS_UPLAND25_Chattogram_30m_STACK":
        ("Distance_River", "decreasing"),

    "06_Distance_to_Road_OSM2021_Chattogram_30m_STACK":
        ("Distance_Road", "decreasing")
}


# ============================================================
# CATEGORICAL SUSCEPTIBILITY SCORES
# 0 = lowest relative susceptibility
# 1 = highest relative susceptibility
# ============================================================

LULC_SCORES = {
    10: 0.10,   # Tree cover
    20: 0.30,   # Shrubland
    30: 0.40,   # Grassland
    40: 0.70,   # Cropland
    50: 1.00,   # Built-up
    60: 0.60,   # Bare / sparse vegetation
    90: 1.00,   # Herbaceous wetland
    95: 0.80    # Mangroves
}

HSG_SCORES = {
    3:  0.50,   # C
    4:  1.00,   # D
    13: 0.75,   # C/D
    14: 1.00    # D/D
}

# Bangladesh analogue:
# NE lowest; E low; SE moderate;
# SW highest; NW high.
# Flat given maximum because ponding potential is highest.
ASPECT_SCORES = {
    -1: 1.00,   # Flat
    "N":  0.20,
    "NE": 0.10,
    "E":  0.30,
    "SE": 0.50,
    "S":  0.75,
    "SW": 1.00,
    "W":  0.80,
    "NW": 0.70
}


# ============================================================
# HELPERS
# ============================================================

project = QgsProject.instance()


def get_layer(name):

    layers = project.mapLayersByName(name)

    if not layers:
        raise Exception("Layer not found: " + name)

    return layers[0]


def open_raster(name):

    lyr = get_layer(name)

    ds = gdal.Open(lyr.source())

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


def write_float_raster(
    path,
    array,
    reference_ds
):

    driver = gdal.GetDriverByName("GTiff")

    out = driver.Create(
        path,
        reference_ds.RasterXSize,
        reference_ds.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW"]
    )

    out.SetGeoTransform(
        reference_ds.GetGeoTransform()
    )

    out.SetProjection(
        reference_ds.GetProjection()
    )

    band = out.GetRasterBand(1)

    band.WriteArray(
        array.astype(np.float32)
    )

    band.SetNoDataValue(NODATA)

    band.FlushCache()

    out = None


# ============================================================
# READ FINAL ANALYSIS MASK
# ============================================================

mask_ds, mask_arr, mask_valid = open_raster(
    MASK_NAME
)

analysis_mask = (
    mask_valid &
    (mask_arr == 1)
)

n_analysis = np.count_nonzero(
    analysis_mask
)

print("\n========== DU STANDARDIZATION ==========")
print("Analysis pixels :", f"{n_analysis:,}")

if n_analysis != 4722216:
    raise Exception(
        "Final-mask pixel count mismatch."
    )


# ============================================================
# READ P05 / P95 FROM STEP 08 CSV
# ============================================================

bounds = {}

with open(
    STATS_CSV,
    "r",
    newline="",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        bounds[row["Factor"]] = (
            float(row["P05"]),
            float(row["P95"])
        )


# ============================================================
# ROBUST LINEAR MEMBERSHIP
# ============================================================

def standardize(
    arr,
    low,
    high,
    direction
):

    if high <= low:
        raise Exception(
            "Invalid P05/P95 bounds."
        )

    z = (
        (arr - low)
        /
        (high - low)
    )

    z = np.clip(
        z,
        0.0,
        1.0
    )

    if direction == "decreasing":
        z = 1.0 - z

    return z


# ============================================================
# 01. CONTINUOUS CRITERIA
# ============================================================

output_summary = []

for layer_name, info in CONTINUOUS.items():

    short_name, direction = info

    ds, arr, valid = open_raster(
        layer_name
    )

    if layer_name not in bounds:
        raise Exception(
            "Statistics not found for: "
            + layer_name
        )

    p05, p95 = bounds[
        layer_name
    ]

    standardized = standardize(
        arr,
        p05,
        p95,
        direction
    )

    output = np.full(
        arr.shape,
        NODATA,
        dtype=np.float32
    )

    use = (
        analysis_mask &
        valid
    )

    output[use] = standardized[use]

    path = os.path.join(
        OUTPUT_DIR,
        "DU09_" +
        short_name +
        "_STD01.tif"
    )

    write_float_raster(
        path,
        output,
        ds
    )

    vals = output[use]

    output_summary.append([
        short_name,
        "continuous_" + direction,
        p05,
        p95,
        float(np.min(vals)),
        float(np.mean(vals)),
        float(np.max(vals)),
        path
    ])

    print(
        f"{short_name:24s}"
        f" {direction:10s}"
        f" P05={p05:.6f}"
        f" P95={p95:.6f}"
        f" -> DONE"
    )


# ============================================================
# 02. LULC
# ============================================================

LULC_NAME = (
    "02_LULC_ESA_WorldCover_2021_"
    "Chattogram_30m_STACK"
)

lulc_ds, lulc, lulc_valid = open_raster(
    LULC_NAME
)

lulc_out = np.full(
    lulc.shape,
    NODATA,
    dtype=np.float32
)

for code, score in LULC_SCORES.items():

    select = (
        analysis_mask &
        lulc_valid &
        (lulc == code)
    )

    lulc_out[select] = score


if np.any(
    analysis_mask &
    (lulc_out == NODATA)
):
    raise Exception(
        "Unscored LULC class found "
        "inside final analysis mask."
    )


lulc_path = os.path.join(
    OUTPUT_DIR,
    "DU09_LULC_STD01.tif"
)

write_float_raster(
    lulc_path,
    lulc_out,
    lulc_ds
)

output_summary.append([
    "LULC",
    "categorical",
    "",
    "",
    np.min(lulc_out[analysis_mask]),
    np.mean(lulc_out[analysis_mask]),
    np.max(lulc_out[analysis_mask]),
    lulc_path
])

print("LULC                    categorical -> DONE")


# ============================================================
# 03. HSG
# ============================================================

HSG_NAME = (
    "04_Soil_HYSOGs250m_"
    "Chattogram_30m_STACK"
)

hsg_ds, hsg, hsg_valid = open_raster(
    HSG_NAME
)

hsg_out = np.full(
    hsg.shape,
    NODATA,
    dtype=np.float32
)

for code, score in HSG_SCORES.items():

    select = (
        analysis_mask &
        hsg_valid &
        (hsg == code)
    )

    hsg_out[select] = score


if np.any(
    analysis_mask &
    (hsg_out == NODATA)
):
    raise Exception(
        "Unscored HSG class found "
        "inside final analysis mask."
    )


hsg_path = os.path.join(
    OUTPUT_DIR,
    "DU09_HSG_STD01.tif"
)

write_float_raster(
    hsg_path,
    hsg_out,
    hsg_ds
)

output_summary.append([
    "HSG",
    "categorical",
    "",
    "",
    np.min(hsg_out[analysis_mask]),
    np.mean(hsg_out[analysis_mask]),
    np.max(hsg_out[analysis_mask]),
    hsg_path
])

print("HSG                     categorical -> DONE")


# ============================================================
# 04. ASPECT
# ============================================================

ASPECT_NAME = "Aspect_deg_30_FLATFIX"

aspect_ds, aspect, aspect_valid = open_raster(
    ASPECT_NAME
)

aspect_out = np.full(
    aspect.shape,
    NODATA,
    dtype=np.float32
)


# Flat
flat = (
    analysis_mask &
    aspect_valid &
    np.isclose(
        aspect,
        -1.0,
        atol=1e-6
    )
)

aspect_out[flat] = ASPECT_SCORES[-1]


# Directional cells
directional = (
    analysis_mask &
    aspect_valid &
    (~flat)
)

deg = np.mod(
    aspect,
    360.0
)


direction_masks = {

    "N":
        directional &
        (
            (deg >= 337.5) |
            (deg < 22.5)
        ),

    "NE":
        directional &
        (deg >= 22.5) &
        (deg < 67.5),

    "E":
        directional &
        (deg >= 67.5) &
        (deg < 112.5),

    "SE":
        directional &
        (deg >= 112.5) &
        (deg < 157.5),

    "S":
        directional &
        (deg >= 157.5) &
        (deg < 202.5),

    "SW":
        directional &
        (deg >= 202.5) &
        (deg < 247.5),

    "W":
        directional &
        (deg >= 247.5) &
        (deg < 292.5),

    "NW":
        directional &
        (deg >= 292.5) &
        (deg < 337.5)
}


for direction, mask in direction_masks.items():

    aspect_out[mask] = (
        ASPECT_SCORES[
            direction
        ]
    )


if np.any(
    analysis_mask &
    (aspect_out == NODATA)
):
    raise Exception(
        "Unscored Aspect pixel found "
        "inside final analysis mask."
    )


aspect_path = os.path.join(
    OUTPUT_DIR,
    "DU09_Aspect_STD01.tif"
)

write_float_raster(
    aspect_path,
    aspect_out,
    aspect_ds
)

output_summary.append([
    "Aspect",
    "categorical_directional",
    "",
    "",
    np.min(aspect_out[analysis_mask]),
    np.mean(aspect_out[analysis_mask]),
    np.max(aspect_out[analysis_mask]),
    aspect_path
])

print("Aspect                  categorical -> DONE")


# ============================================================
# 05. SAVE STANDARDIZATION RULES
# ============================================================

RULES_CSV = os.path.join(
    OUTPUT_DIR,
    "DU09_Standardization_Rules.csv"
)

with open(
    RULES_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Criterion",
        "Transformation",
        "P05_or_Lower_Control",
        "P95_or_Upper_Control",
        "Output_Min",
        "Output_Mean",
        "Output_Max",
        "Output_Raster"
    ])

    writer.writerows(
        output_summary
    )


# ============================================================
# FINAL
# ============================================================

print("\n==============================================")
print("Standardized criteria :", len(output_summary))
print("Expected              : 13")
print("Output directory      :", OUTPUT_DIR)
print("Rules CSV             :", RULES_CSV)

if len(output_summary) == 13:
    print(
        "FINAL STATUS: "
        "DU STEP 09 STANDARDIZATION PASS"
    )
else:
    print(
        "FINAL STATUS: CHECK REQUIRED"
    )

print("==============================================")