from qgis.core import QgsProject, QgsRasterLayer
from osgeo import gdal
import numpy as np
import os
import csv

# ============================================================
# PROJECT SETTINGS
# ============================================================

BASE_DIR = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
)

GROUP_NAME = "07_FINAL_FACTORS_Selected"
REFERENCE_NAME = "DEM_Clipped_UTM46N"

EXPECTED_COUNT = 13

# ------------------------------------------------------------
# Clean output structure
# ------------------------------------------------------------

QC_DIR = os.path.join(
    BASE_DIR,
    "10_QC_AUDIT",
    "07_Final_Selected_Factor_Mask_Audit"
)

FINAL_INPUT_DIR = os.path.join(
    BASE_DIR,
    "11_FINAL_ANALYSIS_INPUTS"
)

os.makedirs(QC_DIR, exist_ok=True)
os.makedirs(FINAL_INPUT_DIR, exist_ok=True)

AUDIT_CSV = os.path.join(
    QC_DIR,
    "07A_Selected13_Factor_Audit.csv"
)

SUMMARY_CSV = os.path.join(
    QC_DIR,
    "07B_Final_Mask_Summary.csv"
)

MASK_OUTPUT = os.path.join(
    FINAL_INPUT_DIR,
    "08_Final_Common_Analysis_Mask_30m.tif"
)

# ------------------------------------------------------------
# Numerical tolerances
# ------------------------------------------------------------

EXTENT_TOL = 0.01     # metres
PIXEL_TOL = 1e-8


# ============================================================
# EXPECTED FINAL SELECTED FACTORS
# ============================================================

EXPECTED_FACTORS = [
    "DEM_Clipped_UTM46N",
    "Slope_deg_30m",
    "Aspect_deg_30_FLATFIX",
    "Profile_Curvature_30m",
    "Tangential_Curvature_30m",
    "TWI_Chittagong_30m_STACK",
    "Drainage_Density_Chittagong_30m_STACK",
    "01_Rainfall_CHIRPSv3_MAP_1991_2020_Chattogram_30m_STACK",
    "02_LULC_ESA_WorldCover_2021_Chattogram_30m_STACK",
    "03_NDVI_Sentinel2_Median_2021_Chattogram_30m_STACK",
    "04_Soil_HYSOGs250m_Chattogram_30m_STACK",
    "05_Distance_to_River_HydroRIVERS_UPLAND25_Chattogram_30m_STACK",
    "06_Distance_to_Road_OSM2021_Chattogram_30m_STACK"
]


# ============================================================
# FIND GROUP
# ============================================================

project = QgsProject.instance()
root = project.layerTreeRoot()

group = root.findGroup(GROUP_NAME)

if group is None:
    raise Exception(
        "Group not found: " + GROUP_NAME
    )

layers = [
    node.layer()
    for node in group.findLayers()
    if isinstance(node.layer(), QgsRasterLayer)
]

layer_dict = {
    lyr.name(): lyr
    for lyr in layers
}


# ============================================================
# FACTOR COUNT + NAME CHECK
# ============================================================

print("\n========== FINAL SELECTED FACTOR CHECK ==========")

print(
    "Raster count :",
    len(layers)
)

print(
    "Expected     :",
    EXPECTED_COUNT
)

if len(layers) != EXPECTED_COUNT:
    raise Exception(
        f"Expected {EXPECTED_COUNT} rasters, "
        f"found {len(layers)}"
    )

missing_factors = [
    name
    for name in EXPECTED_FACTORS
    if name not in layer_dict
]

unexpected_factors = [
    name
    for name in layer_dict
    if name not in EXPECTED_FACTORS
]

if missing_factors:
    raise Exception(
        "Missing selected factors: "
        + ", ".join(missing_factors)
    )

if unexpected_factors:
    raise Exception(
        "Unexpected raster(s) in selected group: "
        + ", ".join(unexpected_factors)
    )

if "TRI_Riley_30m" in layer_dict:
    raise Exception(
        "TRI is still present in final selected group."
    )

print("Factor-name check : PASS")


# ============================================================
# REFERENCE GRID
# ============================================================

ref_layers = project.mapLayersByName(
    REFERENCE_NAME
)

if not ref_layers:
    raise Exception(
        "Reference raster not found: "
        + REFERENCE_NAME
    )

ref = ref_layers[0]

ref_ds = gdal.Open(
    ref.source()
)

if ref_ds is None:
    raise Exception(
        "Cannot open reference raster."
    )

rows = ref_ds.RasterYSize
cols = ref_ds.RasterXSize

ref_extent = ref.extent()

ref_px_x = ref.rasterUnitsPerPixelX()
ref_px_y = ref.rasterUnitsPerPixelY()


# ============================================================
# INITIALIZE COMMON VALID MASK
# ============================================================

common_valid = np.ones(
    (rows, cols),
    dtype=bool
)

lulc_arr = None
lulc_valid_original = None

audit_rows = []

grid_ok_count = 0


# ============================================================
# 13-FACTOR GRID + VALIDITY AUDIT
# ============================================================

print(
    "\n========== 13-FACTOR GRID / VALIDITY AUDIT =========="
)

for name in EXPECTED_FACTORS:

    lyr = layer_dict[name]

    extent = lyr.extent()

    # --------------------------------------------------------
    # Grid comparison
    # --------------------------------------------------------

    same_crs = (
        lyr.crs().authid()
        ==
        ref.crs().authid()
    )

    same_width = (
        lyr.width()
        ==
        ref.width()
    )

    same_height = (
        lyr.height()
        ==
        ref.height()
    )

    same_pixel = (
        abs(
            lyr.rasterUnitsPerPixelX()
            - ref_px_x
        ) < PIXEL_TOL
        and
        abs(
            lyr.rasterUnitsPerPixelY()
            - ref_px_y
        ) < PIXEL_TOL
    )

    dx_min = abs(
        extent.xMinimum()
        - ref_extent.xMinimum()
    )

    dy_min = abs(
        extent.yMinimum()
        - ref_extent.yMinimum()
    )

    dx_max = abs(
        extent.xMaximum()
        - ref_extent.xMaximum()
    )

    dy_max = abs(
        extent.yMaximum()
        - ref_extent.yMaximum()
    )

    same_extent = (
        dx_min < EXTENT_TOL
        and
        dy_min < EXTENT_TOL
        and
        dx_max < EXTENT_TOL
        and
        dy_max < EXTENT_TOL
    )

    grid_ok = (
        same_crs
        and same_width
        and same_height
        and same_pixel
        and same_extent
    )

    if not grid_ok:
        raise Exception(
            "GRID MISMATCH: " + name
        )

    grid_ok_count += 1

    # --------------------------------------------------------
    # Read raster
    # --------------------------------------------------------

    ds = gdal.Open(
        lyr.source()
    )

    if ds is None:
        raise Exception(
            "Could not open raster: " + name
        )

    band = ds.GetRasterBand(1)

    arr = band.ReadAsArray()

    nodata = band.GetNoDataValue()

    valid = np.isfinite(arr)

    if nodata is not None:

        if np.isnan(nodata):

            valid &= ~np.isnan(arr)

        else:

            valid &= (
                arr != nodata
            )

    # --------------------------------------------------------
    # LULC defines terrestrial domain
    # --------------------------------------------------------

    if name.startswith(
        "02_LULC_ESA_WorldCover"
    ):

        lulc_arr = arr.copy()

        lulc_valid_original = (
            valid.copy()
        )

        # Class 80 = permanent water
        valid &= (
            arr != 80
        )

        # defensive check
        valid &= (
            arr != 0
        )

    # --------------------------------------------------------
    # Add to common mask
    # --------------------------------------------------------

    common_valid &= valid

    valid_count = np.count_nonzero(
        valid
    )

    # --------------------------------------------------------
    # Store audit row
    # --------------------------------------------------------

    audit_rows.append([
        name,
        lyr.crs().authid(),
        lyr.width(),
        lyr.height(),
        lyr.rasterUnitsPerPixelX(),
        lyr.rasterUnitsPerPixelY(),
        nodata,
        dx_min,
        dy_min,
        dx_max,
        dy_max,
        valid_count,
        "OK" if grid_ok else "MISMATCH"
    ])

    print(
        f"{name:68s}"
        f" GRID=OK"
        f"  Valid={valid_count:,}"
    )


# ============================================================
# CHECK LULC
# ============================================================

if lulc_arr is None:
    raise Exception(
        "LULC raster not found."
    )


# ============================================================
# DEFINE TERRESTRIAL AOI
# ============================================================

terrestrial = (
    lulc_valid_original
    &
    (lulc_arr != 0)
    &
    (lulc_arr != 80)
)

n_land = np.count_nonzero(
    terrestrial
)

# Make absolutely sure final valid pixels are terrestrial
common_valid &= terrestrial

n_common = np.count_nonzero(
    common_valid
)

n_excluded = (
    n_land
    - n_common
)

coverage = (
    100.0
    * n_common
    / n_land
)

pixel_area_km2 = (
    abs(ref_px_x * ref_px_y)
    / 1_000_000
)

analysis_area_km2 = (
    n_common
    * pixel_area_km2
)

excluded_area_km2 = (
    n_excluded
    * pixel_area_km2
)


# ============================================================
# WRITE DETAILED AUDIT CSV
# ============================================================

with open(
    AUDIT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Factor",
        "CRS",
        "Width",
        "Height",
        "Pixel_X_m",
        "Pixel_Y_m",
        "NoData",
        "Extent_Diff_Xmin_m",
        "Extent_Diff_Ymin_m",
        "Extent_Diff_Xmax_m",
        "Extent_Diff_Ymax_m",
        "Valid_Pixels",
        "Grid_Check"
    ])

    writer.writerows(
        audit_rows
    )


# ============================================================
# EXPORT FINAL COMMON ANALYSIS MASK
# ============================================================

driver = gdal.GetDriverByName(
    "GTiff"
)

out_ds = driver.Create(
    MASK_OUTPUT,
    cols,
    rows,
    1,
    gdal.GDT_Byte,
    options=[
        "COMPRESS=LZW"
    ]
)

if out_ds is None:
    raise Exception(
        "Could not create final mask."
    )

out_ds.SetGeoTransform(
    ref_ds.GetGeoTransform()
)

out_ds.SetProjection(
    ref_ds.GetProjection()
)

out_band = out_ds.GetRasterBand(1)

mask_array = common_valid.astype(
    np.uint8
)

out_band.WriteArray(
    mask_array
)

# 1 = usable analytical pixel
# 0 = excluded / outside modelling domain
out_band.SetNoDataValue(0)

out_band.FlushCache()

out_ds = None


# ============================================================
# WRITE SUMMARY CSV
# ============================================================

summary_rows = [
    ["Selected_Factor_Count", len(layers)],
    ["Grid_Checks_Passed", grid_ok_count],
    ["Reference_CRS", ref.crs().authid()],
    ["Raster_Width", cols],
    ["Raster_Height", rows],
    ["Pixel_Size_m", abs(ref_px_x)],
    ["Terrestrial_AOI_Pixels", n_land],
    ["Final_Common_Valid_Pixels", n_common],
    ["Excluded_Terrestrial_Pixels", n_excluded],
    ["Final_Analysis_Area_km2", analysis_area_km2],
    ["Excluded_Terrestrial_Area_km2", excluded_area_km2],
    ["Terrestrial_Coverage_percent", coverage],
    ["Final_Mask", MASK_OUTPUT],
    [
        "Mask_Encoding",
        "1=usable terrestrial analysis pixel; "
        "0=excluded/NoData"
    ],
    [
        "Final_Status",
        "PASS"
    ]
]

with open(
    SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Metric",
        "Value"
    ])

    writer.writerows(
        summary_rows
    )


# ============================================================
# FINAL CONSOLE REPORT
# ============================================================

print(
    "\n================ FINAL SUMMARY ================"
)

print(
    "Selected factors          :",
    len(layers)
)

print(
    "Grid checks passed        :",
    grid_ok_count
)

print(
    "Terrestrial AOI pixels    :",
    f"{n_land:,}"
)

print(
    "Final common valid pixels :",
    f"{n_common:,}"
)

print(
    "Excluded terrestrial px   :",
    f"{n_excluded:,}"
)

print(
    "Excluded terrestrial area :",
    f"{excluded_area_km2:,.2f} km²"
)

print(
    "Final analysis area       :",
    f"{analysis_area_km2:,.2f} km²"
)

print(
    "Terrestrial coverage      :",
    f"{coverage:.2f}%"
)

print("\nDetailed QC:")
print(AUDIT_CSV)

print("\nSummary:")
print(SUMMARY_CSV)

print("\nFinal analysis mask:")
print(MASK_OUTPUT)

print(
    "\nFINAL STATUS: "
    "SELECTED-FACTOR STACK AND FINAL MASK PASS"
)

print(
    "================================================"
)

ref_ds = None