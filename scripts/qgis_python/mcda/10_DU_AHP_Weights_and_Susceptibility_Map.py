import os
import csv
import numpy as np
from osgeo import gdal

# ============================================================
# SETTINGS
# ============================================================

BASE = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
)

INPUT_DIR = os.path.join(
    BASE,
    "12_DU_MCDA",
    "01_Standardized_Criteria"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "12_DU_MCDA",
    "02_AHP"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

NODATA = -9999.0


# ============================================================
# CRITERIA ORDER
#
# IMPORTANT:
# This exact order corresponds to the AHP matrix below.
# ============================================================

CRITERIA = [
    "Rainfall",
    "Elevation",
    "Slope",
    "Drainage_Density",
    "Distance_River",
    "TWI",
    "LULC",
    "HSG",
    "NDVI",
    "Distance_Road",
    "Profile_Curvature",
    "Tangential_Curvature",
    "Aspect"
]


# ============================================================
# STANDARDIZED RASTERS
# ============================================================

FILES = {

    "Rainfall":
        "DU09_Rainfall_STD01.tif",

    "Elevation":
        "DU09_Elevation_STD01.tif",

    "Slope":
        "DU09_Slope_STD01.tif",

    "Drainage_Density":
        "DU09_Drainage_Density_STD01.tif",

    "Distance_River":
        "DU09_Distance_River_STD01.tif",

    "TWI":
        "DU09_TWI_STD01.tif",

    "LULC":
        "DU09_LULC_STD01.tif",

    "HSG":
        "DU09_HSG_STD01.tif",

    "NDVI":
        "DU09_NDVI_STD01.tif",

    "Distance_Road":
        "DU09_Distance_Road_STD01.tif",

    "Profile_Curvature":
        "DU09_Profile_Curvature_STD01.tif",

    "Tangential_Curvature":
        "DU09_Tangential_Curvature_STD01.tif",

    "Aspect":
        "DU09_Aspect_STD01.tif"
}


# ============================================================
# LITERATURE-INFORMED AHP PAIRWISE MATRIX
#
# Saaty scale:
# 1 = equal importance
# 3 = moderate
# 5 = strong
# 7 = very strong
# 9 = extreme
# 2,4,6,8 = intermediate judgments
#
# Reciprocal values occur below diagonal.
# ============================================================

A = np.array([

    # R     E     S     DD    DR    TWI   LU    HSG   NDVI  Road  PC    TC    Asp

    [1,    2,    3,    4,    4,    5,    5,    6,    6,    7,    7,    8,    9],

    [1/2,  1,    2,    3,    4,    4,    5,    5,    6,    6,    7,    7,    8],

    [1/3,  1/2,  1,    2,    3,    4,    4,    5,    5,    6,    6,    7,    7],

    [1/4,  1/3,  1/2,  1,    2,    3,    4,    4,    5,    5,    6,    6,    7],

    [1/4,  1/4,  1/3,  1/2,  1,    2,    3,    4,    4,    5,    5,    6,    6],

    [1/5,  1/4,  1/4,  1/3,  1/2,  1,    2,    3,    4,    4,    5,    5,    6],

    [1/5,  1/5,  1/4,  1/4,  1/3,  1/2,  1,    2,    3,    4,    4,    5,    5],

    [1/6,  1/5,  1/5,  1/4,  1/4,  1/3,  1/2,  1,    2,    3,    4,    4,    5],

    [1/6,  1/6,  1/5,  1/5,  1/4,  1/4,  1/3,  1/2,  1,    2,    3,    4,    4],

    [1/7,  1/6,  1/6,  1/5,  1/5,  1/4,  1/4,  1/3,  1/2,  1,    2,    3,    4],

    [1/7,  1/7,  1/6,  1/6,  1/5,  1/5,  1/4,  1/4,  1/3,  1/2,  1,    2,    3],

    [1/8,  1/7,  1/7,  1/6,  1/6,  1/5,  1/5,  1/4,  1/4,  1/3,  1/2,  1,    2],

    [1/9,  1/8,  1/7,  1/7,  1/6,  1/6,  1/5,  1/5,  1/4,  1/4,  1/3,  1/2,  1]

], dtype=np.float64)


# ============================================================
# MATRIX QC
# ============================================================

n = len(CRITERIA)

if A.shape != (n, n):
    raise Exception(
        "AHP matrix must be 13 x 13."
    )


# Diagonal must equal 1
if not np.allclose(
    np.diag(A),
    1.0
):
    raise Exception(
        "AHP diagonal check FAILED."
    )


# Reciprocal condition:
# a_ij * a_ji = 1
reciprocal_check = A * A.T

if not np.allclose(
    reciprocal_check,
    np.ones_like(A),
    atol=1e-10
):
    raise Exception(
        "AHP reciprocal check FAILED."
    )


print(
    "\n========== AHP MATRIX QC =========="
)

print(
    "Matrix size       :",
    A.shape
)

print(
    "Diagonal check    : PASS"
)

print(
    "Reciprocal check  : PASS"
)


# ============================================================
# PRINCIPAL EIGENVECTOR WEIGHTS
# ============================================================

eigenvalues, eigenvectors = np.linalg.eig(
    A
)

principal_index = np.argmax(
    eigenvalues.real
)

lambda_max = float(
    eigenvalues[
        principal_index
    ].real
)


weights = np.abs(
    eigenvectors[
        :,
        principal_index
    ].real
)

weights = (
    weights
    /
    weights.sum()
)


# ============================================================
# CONSISTENCY CHECK
# ============================================================

CI = (
    lambda_max - n
) / (
    n - 1
)


# Saaty Random Index
# n = 13
RI = 1.56


CR = (
    CI / RI
)


print(
    "\n========== AHP CONSISTENCY =========="
)

print(
    f"Lambda max : {lambda_max:.6f}"
)

print(
    f"CI         : {CI:.6f}"
)

print(
    f"RI         : {RI:.6f}"
)

print(
    f"CR         : {CR:.6f}"
)


if CR >= 0.10:

    raise Exception(
        "AHP consistency FAILED: CR >= 0.10"
    )


print(
    "Consistency : PASS (CR < 0.10)"
)


# ============================================================
# WEIGHT TABLE
# ============================================================

print(
    "\n========== AHP WEIGHTS =========="
)

for criterion, weight in zip(
    CRITERIA,
    weights
):

    print(
        f"{criterion:24s}"
        f" {weight:.6f}"
        f"  ({weight*100:.2f}%)"
    )


# ============================================================
# OUTPUT FILES
# ============================================================

MATRIX_CSV = os.path.join(
    OUTPUT_DIR,
    "DU10_AHP_Pairwise_Matrix.csv"
)

WEIGHTS_CSV = os.path.join(
    OUTPUT_DIR,
    "DU10_AHP_Weights_and_Consistency.csv"
)

MAP_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "DU10_AHP_Susceptibility_Index_01.tif"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "DU10_AHP_Map_Summary.csv"
)


# ============================================================
# SAVE PAIRWISE MATRIX
# ============================================================

with open(
    MATRIX_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        ["Criterion"] +
        CRITERIA
    )

    for i, criterion in enumerate(
        CRITERIA
    ):

        writer.writerow(
            [criterion] +
            list(A[i, :])
        )


# ============================================================
# SAVE WEIGHTS + CONSISTENCY
# ============================================================

with open(
    WEIGHTS_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Criterion",
        "Weight",
        "Weight_Percent"
    ])

    for criterion, weight in zip(
        CRITERIA,
        weights
    ):

        writer.writerow([
            criterion,
            weight,
            weight * 100
        ])

    writer.writerow([])
    writer.writerow([
        "Consistency_Metric",
        "Value"
    ])

    writer.writerow([
        "n",
        n
    ])

    writer.writerow([
        "Lambda_Max",
        lambda_max
    ])

    writer.writerow([
        "CI",
        CI
    ])

    writer.writerow([
        "RI",
        RI
    ])

    writer.writerow([
        "CR",
        CR
    ])

    writer.writerow([
        "Consistency_Status",
        "PASS" if CR < 0.10 else "FAIL"
    ])


# ============================================================
# PREPARE WEIGHTED OVERLAY
# ============================================================

reference_ds = None

susceptibility = None

common_valid = None


for i, criterion in enumerate(
    CRITERIA
):

    path = os.path.join(
        INPUT_DIR,
        FILES[
            criterion
        ]
    )

    if not os.path.exists(path):

        raise Exception(
            "Missing standardized raster: "
            + path
        )


    ds = gdal.Open(
        path
    )

    if ds is None:

        raise Exception(
            "Cannot open raster: "
            + path
        )


    band = ds.GetRasterBand(1)

    arr = band.ReadAsArray().astype(
        np.float32
    )

    nodata = band.GetNoDataValue()


    valid = np.isfinite(
        arr
    )


    if nodata is not None:

        if np.isnan(nodata):

            valid &= ~np.isnan(
                arr
            )

        else:

            valid &= (
                arr != nodata
            )


    # First raster initializes output geometry
    if reference_ds is None:

        reference_ds = ds

        rows = ds.RasterYSize
        cols = ds.RasterXSize

        susceptibility = np.zeros(
            (rows, cols),
            dtype=np.float64
        )

        common_valid = valid.copy()


    else:

        if (
            ds.RasterXSize != cols
            or
            ds.RasterYSize != rows
        ):

            raise Exception(
                "Raster dimension mismatch: "
                + criterion
            )

        common_valid &= valid


    susceptibility[
        valid
    ] += (
        weights[i]
        *
        arr[valid]
    )


    print(
        f"Overlay added : "
        f"{criterion}"
    )


# ============================================================
# FINAL AHP INDEX
# ============================================================

final_index = np.full(
    susceptibility.shape,
    NODATA,
    dtype=np.float32
)

final_index[
    common_valid
] = susceptibility[
    common_valid
].astype(
    np.float32
)


valid_values = final_index[
    common_valid
]


# ============================================================
# SANITY CHECK
# ============================================================

if np.min(valid_values) < -1e-6:

    raise Exception(
        "AHP index below 0 detected."
    )


if np.max(valid_values) > 1.000001:

    raise Exception(
        "AHP index above 1 detected."
    )


# ============================================================
# EXPORT AHP RASTER
# ============================================================

driver = gdal.GetDriverByName(
    "GTiff"
)


out = driver.Create(
    MAP_OUTPUT,
    reference_ds.RasterXSize,
    reference_ds.RasterYSize,
    1,
    gdal.GDT_Float32,
    options=[
        "COMPRESS=LZW"
    ]
)


out.SetGeoTransform(
    reference_ds.GetGeoTransform()
)

out.SetProjection(
    reference_ds.GetProjection()
)


out_band = out.GetRasterBand(1)

out_band.WriteArray(
    final_index
)

out_band.SetNoDataValue(
    NODATA
)

out_band.FlushCache()

out = None


# ============================================================
# MAP STATISTICS
# ============================================================

percentiles = np.percentile(
    valid_values,
    [
        5,
        25,
        50,
        75,
        95
    ]
)


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

    writer.writerow([
        "Valid_Pixels",
        len(valid_values)
    ])

    writer.writerow([
        "Minimum",
        float(np.min(valid_values))
    ])

    writer.writerow([
        "Mean",
        float(np.mean(valid_values))
    ])

    writer.writerow([
        "SD",
        float(
            np.std(
                valid_values,
                ddof=1
            )
        )
    ])

    writer.writerow([
        "P05",
        percentiles[0]
    ])

    writer.writerow([
        "P25",
        percentiles[1]
    ])

    writer.writerow([
        "Median",
        percentiles[2]
    ])

    writer.writerow([
        "P75",
        percentiles[3]
    ])

    writer.writerow([
        "P95",
        percentiles[4]
    ])

    writer.writerow([
        "Maximum",
        float(np.max(valid_values))
    ])

    writer.writerow([
        "CR",
        CR
    ])


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n========== AHP MAP SUMMARY =========="
)

print(
    "Valid pixels :",
    f"{len(valid_values):,}"
)

print(
    f"Minimum      : {np.min(valid_values):.6f}"
)

print(
    f"Mean         : {np.mean(valid_values):.6f}"
)

print(
    f"Median       : {np.median(valid_values):.6f}"
)

print(
    f"Maximum      : {np.max(valid_values):.6f}"
)

print(
    "\nPairwise matrix:"
)

print(
    MATRIX_CSV
)

print(
    "\nWeights:"
)

print(
    WEIGHTS_CSV
)

print(
    "\nAHP susceptibility map:"
)

print(
    MAP_OUTPUT
)

print(
    "\nSummary:"
)

print(
    SUMMARY_CSV
)

print(
    "\nFINAL STATUS: DU STEP 10 AHP PASS"
)