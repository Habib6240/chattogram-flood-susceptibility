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

OUTPUT_DIR = os.path.join(
    BASE,
    "10_QC_AUDIT",
    "09A_Internal_NoData_Culprit_Audit"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "09A_Factor_NoData_Culprit_Summary.csv"
)

SIGNATURE_CSV = os.path.join(
    OUTPUT_DIR,
    "09B_NoData_Overlap_Signatures.csv"
)

COUNT_RASTER = os.path.join(
    OUTPUT_DIR,
    "09C_Missing_Factor_Count_30m.tif"
)

CULPRIT_RASTER = os.path.join(
    OUTPUT_DIR,
    "09D_NoData_Primary_Culprit_30m.tif"
)


# ============================================================
# ORIGINAL 13 SELECTED FACTORS
# DO NOT use standardized DU09 rasters here.
# ============================================================

FACTORS = [

    ("Elevation",
     "DEM_Clipped_UTM46N"),

    ("Slope",
     "Slope_deg_30m"),

    ("Aspect",
     "Aspect_deg_30_FLATFIX"),

    ("Profile_Curvature",
     "Profile_Curvature_30m"),

    ("Tangential_Curvature",
     "Tangential_Curvature_30m"),

    ("TWI",
     "TWI_Chittagong_30m_STACK"),

    ("Drainage_Density",
     "Drainage_Density_Chittagong_30m_STACK"),

    ("Rainfall",
     "01_Rainfall_CHIRPSv3_MAP_1991_2020_Chattogram_30m_STACK"),

    ("LULC",
     "02_LULC_ESA_WorldCover_2021_Chattogram_30m_STACK"),

    ("NDVI",
     "03_NDVI_Sentinel2_Median_2021_Chattogram_30m_STACK"),

    ("HSG",
     "04_Soil_HYSOGs250m_Chattogram_30m_STACK"),

    ("Distance_River",
     "05_Distance_to_River_HydroRIVERS_UPLAND25_Chattogram_30m_STACK"),

    ("Distance_Road",
     "06_Distance_to_Road_OSM2021_Chattogram_30m_STACK")
]


FINAL_MASK_NAME = (
    "08_Final_Common_Analysis_Mask_30m"
)

LULC_NAME = (
    "02_LULC_ESA_WorldCover_2021_"
    "Chattogram_30m_STACK"
)


# ============================================================
# HELPERS
# ============================================================

project = QgsProject.instance()


def get_layer(name):

    layers = project.mapLayersByName(name)

    if not layers:
        raise Exception(
            "QGIS layer not found: " + name
        )

    return layers[0]


def read_raster(name):

    layer = get_layer(name)

    ds = gdal.Open(
        layer.source()
    )

    if ds is None:
        raise Exception(
            "Cannot open: " + name
        )

    band = ds.GetRasterBand(1)

    arr = band.ReadAsArray()

    nd = band.GetNoDataValue()

    valid = np.isfinite(
        arr.astype(np.float64)
    )

    if nd is not None:

        if np.isnan(nd):

            valid &= ~np.isnan(arr)

        else:

            valid &= (
                arr != nd
            )

    return (
        ds,
        arr,
        valid
    )


def write_byte(
    path,
    array,
    reference_ds,
    nodata=255
):

    driver = gdal.GetDriverByName(
        "GTiff"
    )

    out = driver.Create(
        path,
        reference_ds.RasterXSize,
        reference_ds.RasterYSize,
        1,
        gdal.GDT_Byte,
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
        array.astype(np.uint8)
    )

    band.SetNoDataValue(
        nodata
    )

    band.FlushCache()

    out = None


# ============================================================
# 01. DEFINE TERRESTRIAL AOI
#
# Raw ESA WorldCover is used because:
# - it covers the AOI
# - class 80 = permanent water
#
# Terrestrial = valid LULC AND class != 80
# ============================================================

lulc_ds, lulc, lulc_valid = read_raster(
    LULC_NAME
)

terrestrial = (
    lulc_valid
    &
    (lulc != 80)
)

n_terrestrial = int(
    np.count_nonzero(
        terrestrial
    )
)

print(
    "\n========== INTERNAL NODATA AUDIT =========="
)

print(
    "Terrestrial pixels :",
    f"{n_terrestrial:,}"
)


# ============================================================
# 02. READ FINAL COMMON MASK
# ============================================================

mask_ds, mask_arr, _ = read_raster(
    FINAL_MASK_NAME
)

final_valid = (
    mask_arr == 1
)

excluded = (
    terrestrial
    &
    (~final_valid)
)

n_excluded = int(
    np.count_nonzero(
        excluded
    )
)

print(
    "Final excluded px  :",
    f"{n_excluded:,}"
)

print(
    "Excluded area      :",
    f"{n_excluded * 900 / 1e6:.2f} km2"
)


# ============================================================
# 03. AUDIT ALL 13 ORIGINAL FACTORS
# ============================================================

shape = lulc.shape

missing_stack = []

summary_rows = []


for code, (
    short_name,
    layer_name
) in enumerate(
    FACTORS,
    start=1
):

    ds, arr, valid = read_raster(
        layer_name
    )

    if arr.shape != shape:

        raise Exception(
            "Shape mismatch: " +
            layer_name
        )

    missing = (
        terrestrial
        &
        (~valid)
    )

    missing_stack.append(
        missing
    )

    missing_count = int(
        np.count_nonzero(
            missing
        )
    )

    missing_area = (
        missing_count
        *
        900
        /
        1e6
    )

    missing_pct_terrestrial = (
        missing_count
        /
        n_terrestrial
        *
        100
    )

    missing_inside_final_exclusion = int(
        np.count_nonzero(
            missing
            &
            excluded
        )
    )

    contribution_pct = (
        missing_inside_final_exclusion
        /
        n_excluded
        *
        100
        if n_excluded > 0
        else 0
    )

    summary_rows.append([
        code,
        short_name,
        layer_name,
        missing_count,
        missing_area,
        missing_pct_terrestrial,
        missing_inside_final_exclusion,
        contribution_pct
    ])

    print(
        f"{code:2d}. "
        f"{short_name:24s}"
        f" missing={missing_count:7,d}"
        f" ({missing_pct_terrestrial:6.3f}%)"
    )


# ============================================================
# 04. COMBINED MISSING-FACTOR COUNT
# ============================================================

missing_array = np.stack(
    missing_stack,
    axis=0
)

missing_count_per_pixel = np.sum(
    missing_array,
    axis=0
).astype(
    np.uint8
)


# Union of all factor missing pixels
union_missing = (
    terrestrial
    &
    (
        missing_count_per_pixel > 0
    )
)

n_union = int(
    np.count_nonzero(
        union_missing
    )
)


print(
    "\nUnion missing px    :",
    f"{n_union:,}"
)


# ============================================================
# 05. FINAL MASK REPRODUCTION QC
#
# If the final common mask was built correctly from these
# 13 factors, union_missing should equal final excluded pixels.
# ============================================================

difference = np.logical_xor(
    union_missing,
    excluded
)

n_difference = int(
    np.count_nonzero(
        difference
    )
)


print(
    "Mask mismatch px   :",
    f"{n_difference:,}"
)


if n_difference == 0:

    mask_status = "PASS"

else:

    mask_status = "CHECK_REQUIRED"


print(
    "Mask reproduction  :",
    mask_status
)


# ============================================================
# 06. EXACT OVERLAP SIGNATURE
#
# Each factor gets one bit.
#
# Example:
# only HSG missing -> one unique signature
# slope + curvature missing -> another signature
# ============================================================

signature = np.zeros(
    shape,
    dtype=np.uint16
)


for i, missing in enumerate(
    missing_stack
):

    signature[
        missing
    ] |= (
        1 << i
    )


sig_values = signature[
    excluded
]


unique_sig, sig_counts = np.unique(
    sig_values,
    return_counts=True
)


signature_rows = []


for sig, count in zip(
    unique_sig,
    sig_counts
):

    if sig == 0:

        names = [
            "NO_FACTOR_MISSING_CHECK_MASK_LOGIC"
        ]

    else:

        names = []

        for i, (
            short_name,
            _
        ) in enumerate(
            FACTORS
        ):

            if sig & (
                1 << i
            ):

                names.append(
                    short_name
                )

    area_km2 = (
        int(count)
        *
        900
        /
        1e6
    )

    pct_excluded = (
        int(count)
        /
        n_excluded
        *
        100
        if n_excluded > 0
        else 0
    )

    signature_rows.append([
        int(sig),
        " + ".join(names),
        int(count),
        area_km2,
        pct_excluded
    ])


signature_rows.sort(
    key=lambda x: x[2],
    reverse=True
)


# ============================================================
# 07. PRIMARY CULPRIT RASTER
#
# 0   = no missing factor
# 1-13 = exactly one missing factor
# 250 = multiple factors missing
# 255 = outside terrestrial domain
# ============================================================

culprit = np.full(
    shape,
    255,
    dtype=np.uint8
)

culprit[
    terrestrial
] = 0


for i, missing in enumerate(
    missing_stack,
    start=1
):

    unique_missing = (
        missing
        &
        (
            missing_count_per_pixel == 1
        )
    )

    culprit[
        unique_missing
    ] = i


multiple = (
    terrestrial
    &
    (
        missing_count_per_pixel > 1
    )
)

culprit[
    multiple
] = 250


# ============================================================
# 08. MISSING COUNT RASTER
#
# 0 = all 13 factors available
# 1...13 = number of missing factors
# 255 = outside terrestrial domain
# ============================================================

count_output = np.full(
    shape,
    255,
    dtype=np.uint8
)

count_output[
    terrestrial
] = missing_count_per_pixel[
    terrestrial
]


# ============================================================
# 09. SAVE RASTERS
# ============================================================

write_byte(
    COUNT_RASTER,
    count_output,
    lulc_ds
)

write_byte(
    CULPRIT_RASTER,
    culprit,
    lulc_ds
)


# ============================================================
# 10. SAVE FACTOR SUMMARY
# ============================================================

with open(
    SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Factor_Code",
        "Factor",
        "QGIS_Layer_Name",
        "Missing_Terrestrial_Pixels",
        "Missing_Area_km2",
        "Missing_Percent_Terrestrial",
        "Missing_Pixels_Inside_Final_Exclusion",
        "Percent_of_Final_Excluded_Pixels_Touched"
    ])

    writer.writerows(
        summary_rows
    )

    writer.writerow([])

    writer.writerow([
        "QC_Metric",
        "Value"
    ])

    writer.writerow([
        "Terrestrial_Pixels",
        n_terrestrial
    ])

    writer.writerow([
        "Final_Excluded_Pixels",
        n_excluded
    ])

    writer.writerow([
        "Union_Missing_Pixels",
        n_union
    ])

    writer.writerow([
        "Mask_Reproduction_Mismatch_Pixels",
        n_difference
    ])

    writer.writerow([
        "Mask_Reproduction_Status",
        mask_status
    ])


# ============================================================
# 11. SAVE OVERLAP SIGNATURES
# ============================================================

with open(
    SIGNATURE_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Signature_Code",
        "Missing_Factor_Combination",
        "Pixel_Count",
        "Area_km2",
        "Percent_of_Final_Excluded_Area"
    ])

    writer.writerows(
        signature_rows
    )


# ============================================================
# 12. FINAL CONSOLE SUMMARY
# ============================================================

print(
    "\n========== TOP NODATA CONTRIBUTORS =========="
)

sorted_summary = sorted(
    summary_rows,
    key=lambda x: x[3],
    reverse=True
)


for row in sorted_summary:

    if row[3] > 0:

        print(
            f"{row[1]:24s}"
            f" {row[3]:7,d} px"
            f"  {row[4]:7.2f} km2"
            f"  {row[5]:6.3f}% terrestrial"
        )


print(
    "\n========== TOP OVERLAP SIGNATURES =========="
)


for row in signature_rows[:10]:

    print(
        f"{row[1]:55s}"
        f" {row[2]:7,d} px"
        f" ({row[4]:6.2f}% of excluded)"
    )


print(
    "\nOutputs:"
)

print(
    SUMMARY_CSV
)

print(
    SIGNATURE_CSV
)

print(
    COUNT_RASTER
)

print(
    CULPRIT_RASTER
)

print(
    "\nFINAL STATUS: "
    + (
        "09A NODATA CULPRIT AUDIT PASS"
        if mask_status == "PASS"
        else
        "09A CHECK REQUIRED"
    )
)