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

AHP_DIR = os.path.join(
    BASE,
    "12_DU_MCDA",
    "02_AHP"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "12_DU_MCDA",
    "03_FAHP"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

NODATA = -9999.0


# ============================================================
# CRITERIA ORDER
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
# ORIGINAL CRISP AHP MATRIX
#
# We fuzzify the SAME judgments used in Step 10.
# ============================================================

A = np.array([

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
        "Matrix dimension must be 13 x 13."
    )

if not np.allclose(
    np.diag(A),
    1.0
):
    raise Exception(
        "Diagonal check FAILED."
    )

if not np.allclose(
    A * A.T,
    np.ones_like(A),
    atol=1e-10
):
    raise Exception(
        "Reciprocal check FAILED."
    )


print(
    "\n========== FAHP INPUT MATRIX QC =========="
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
# TRIANGULAR FUZZY NUMBER SCALE
#
# Crisp Saaty judgment k:
#
# 1 -> (1,1,1)
# 2 -> (1,2,3)
# 3 -> (2,3,4)
# ...
# 8 -> (7,8,9)
# 9 -> (8,9,9)
#
# Reciprocal:
# (l,m,u)^-1 = (1/u, 1/m, 1/l)
# ============================================================

FUZZY_SCALE = {

    1: (1.0, 1.0, 1.0),

    2: (1.0, 2.0, 3.0),

    3: (2.0, 3.0, 4.0),

    4: (3.0, 4.0, 5.0),

    5: (4.0, 5.0, 6.0),

    6: (5.0, 6.0, 7.0),

    7: (6.0, 7.0, 8.0),

    8: (7.0, 8.0, 9.0),

    9: (8.0, 9.0, 9.0)

}


def reciprocal_tfn(tfn):

    l, m, u = tfn

    return (
        1.0 / u,
        1.0 / m,
        1.0 / l
    )


def crisp_to_tfn(value):

    if np.isclose(
        value,
        1.0
    ):

        return FUZZY_SCALE[1]


    if value > 1:

        k = int(
            round(value)
        )

        if k not in FUZZY_SCALE:
            raise Exception(
                "Unsupported Saaty value: "
                + str(value)
            )

        return FUZZY_SCALE[k]


    reciprocal_value = 1.0 / value

    k = int(
        round(reciprocal_value)
    )

    if k not in FUZZY_SCALE:
        raise Exception(
            "Unsupported reciprocal value: "
            + str(value)
        )

    return reciprocal_tfn(
        FUZZY_SCALE[k]
    )


# ============================================================
# BUILD FUZZY MATRIX
#
# Shape:
# 13 x 13 x 3
#
# Last axis:
# [lower, modal, upper]
# ============================================================

F = np.zeros(
    (n, n, 3),
    dtype=np.float64
)


for i in range(n):

    for j in range(n):

        F[i, j, :] = crisp_to_tfn(
            A[i, j]
        )


# ============================================================
# FUZZY RECIPROCAL QC
# ============================================================

for i in range(n):

    for j in range(n):

        l1, m1, u1 = F[i, j]

        l2, m2, u2 = F[j, i]

        expected = np.array([
            1.0 / u1,
            1.0 / m1,
            1.0 / l1
        ])

        actual = np.array([
            l2,
            m2,
            u2
        ])

        if not np.allclose(
            expected,
            actual,
            atol=1e-10
        ):

            raise Exception(
                "Fuzzy reciprocal QC FAILED."
            )


print(
    "Fuzzy reciprocal QC: PASS"
)


# ============================================================
# BUCKLEY FUZZY GEOMETRIC MEAN
#
# r_i =
# (
#   product_j l_ij ^ (1/n),
#   product_j m_ij ^ (1/n),
#   product_j u_ij ^ (1/n)
# )
# ============================================================

fuzzy_geometric_means = np.zeros(
    (n, 3),
    dtype=np.float64
)


for i in range(n):

    fuzzy_geometric_means[
        i,
        0
    ] = np.prod(
        F[i, :, 0]
    ) ** (
        1.0 / n
    )


    fuzzy_geometric_means[
        i,
        1
    ] = np.prod(
        F[i, :, 1]
    ) ** (
        1.0 / n
    )


    fuzzy_geometric_means[
        i,
        2
    ] = np.prod(
        F[i, :, 2]
    ) ** (
        1.0 / n
    )


# ============================================================
# SUM OF FUZZY GEOMETRIC MEANS
# ============================================================

sum_l = np.sum(
    fuzzy_geometric_means[:, 0]
)

sum_m = np.sum(
    fuzzy_geometric_means[:, 1]
)

sum_u = np.sum(
    fuzzy_geometric_means[:, 2]
)


# Inverse of fuzzy sum:
#
# (sum_l, sum_m, sum_u)^-1
#
# =
#
# (1/sum_u, 1/sum_m, 1/sum_l)

inverse_sum = np.array([
    1.0 / sum_u,
    1.0 / sum_m,
    1.0 / sum_l
])


# ============================================================
# FUZZY WEIGHTS
# ============================================================

fuzzy_weights = np.zeros(
    (n, 3),
    dtype=np.float64
)


for i in range(n):

    fuzzy_weights[
        i,
        0
    ] = (
        fuzzy_geometric_means[i, 0]
        *
        inverse_sum[0]
    )


    fuzzy_weights[
        i,
        1
    ] = (
        fuzzy_geometric_means[i, 1]
        *
        inverse_sum[1]
    )


    fuzzy_weights[
        i,
        2
    ] = (
        fuzzy_geometric_means[i, 2]
        *
        inverse_sum[2]
    )


# ============================================================
# DEFUZZIFICATION
#
# Centroid:
#
# C_i = (l + m + u) / 3
# ============================================================

crisp_fuzzy_weights = np.mean(
    fuzzy_weights,
    axis=1
)


# Final normalization
weights = (
    crisp_fuzzy_weights
    /
    np.sum(
        crisp_fuzzy_weights
    )
)


print(
    "\n========== FAHP WEIGHTS =========="
)


for criterion, fw, weight in zip(
    CRITERIA,
    fuzzy_weights,
    weights
):

    print(
        f"{criterion:24s}"
        f" L={fw[0]:.6f}"
        f" M={fw[1]:.6f}"
        f" U={fw[2]:.6f}"
        f" -> {weight:.6f}"
        f" ({weight*100:.2f}%)"
    )


if not np.isclose(
    np.sum(weights),
    1.0,
    atol=1e-10
):

    raise Exception(
        "FAHP weights do not sum to 1."
    )


print(
    "\nWeight sum :",
    np.sum(weights)
)

print(
    "Weight normalization: PASS"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

FUZZY_MATRIX_CSV = os.path.join(
    OUTPUT_DIR,
    "DU11_FAHP_Fuzzy_Pairwise_Matrix_Long.csv"
)

WEIGHTS_CSV = os.path.join(
    OUTPUT_DIR,
    "DU11_FAHP_Weights.csv"
)

COMPARE_CSV = os.path.join(
    OUTPUT_DIR,
    "DU11_FAHP_vs_AHP_Weights.csv"
)

MAP_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "DU11_FAHP_Susceptibility_Index_01.tif"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "DU11_FAHP_Map_Summary.csv"
)


# ============================================================
# SAVE FUZZY MATRIX
# ============================================================

with open(
    FUZZY_MATRIX_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Criterion_Row",
        "Criterion_Column",
        "Original_Crisp_Value",
        "Fuzzy_L",
        "Fuzzy_M",
        "Fuzzy_U"
    ])

    for i in range(n):

        for j in range(n):

            writer.writerow([
                CRITERIA[i],
                CRITERIA[j],
                A[i, j],
                F[i, j, 0],
                F[i, j, 1],
                F[i, j, 2]
            ])


# ============================================================
# SAVE FAHP WEIGHTS
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
        "GeometricMean_L",
        "GeometricMean_M",
        "GeometricMean_U",
        "FuzzyWeight_L",
        "FuzzyWeight_M",
        "FuzzyWeight_U",
        "Defuzzified_PreNormalized",
        "Final_Normalized_Weight",
        "Weight_Percent"
    ])

    for i in range(n):

        writer.writerow([
            CRITERIA[i],

            fuzzy_geometric_means[
                i,
                0
            ],

            fuzzy_geometric_means[
                i,
                1
            ],

            fuzzy_geometric_means[
                i,
                2
            ],

            fuzzy_weights[
                i,
                0
            ],

            fuzzy_weights[
                i,
                1
            ],

            fuzzy_weights[
                i,
                2
            ],

            crisp_fuzzy_weights[i],

            weights[i],

            weights[i] * 100
        ])


# ============================================================
# READ AHP WEIGHTS FOR DIRECT COMPARISON
# ============================================================

AHP_WEIGHTS_FILE = os.path.join(
    AHP_DIR,
    "DU10_AHP_Weights_and_Consistency.csv"
)


ahp_weights = {}


with open(
    AHP_WEIGHTS_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as f:

    reader = csv.reader(f)

    header = next(reader)

    for row in reader:

        if len(row) < 3:
            break

        criterion = row[0]

        if criterion in CRITERIA:

            ahp_weights[
                criterion
            ] = float(
                row[1]
            )


# ============================================================
# SAVE AHP vs FAHP WEIGHT COMPARISON
# ============================================================

with open(
    COMPARE_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Criterion",
        "AHP_Weight",
        "FAHP_Weight",
        "Difference_FAHP_minus_AHP",
        "Absolute_Difference"
    ])

    for i, criterion in enumerate(
        CRITERIA
    ):

        ahp_w = ahp_weights[
            criterion
        ]

        fahp_w = weights[i]

        writer.writerow([
            criterion,
            ahp_w,
            fahp_w,
            fahp_w - ahp_w,
            abs(
                fahp_w - ahp_w
            )
        ])


# ============================================================
# RASTER WEIGHTED OVERLAY
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
            "Could not open raster: "
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

        if np.isnan(
            nodata
        ):

            valid &= ~np.isnan(
                arr
            )

        else:

            valid &= (
                arr != nodata
            )


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
        "Overlay added :",
        criterion
    )


# ============================================================
# FINAL FAHP INDEX
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
# RANGE QC
# ============================================================

if np.min(
    valid_values
) < -1e-6:

    raise Exception(
        "FAHP index below 0."
    )


if np.max(
    valid_values
) > 1.000001:

    raise Exception(
        "FAHP index above 1."
    )


# ============================================================
# EXPORT FAHP RASTER
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


band = out.GetRasterBand(1)


band.WriteArray(
    final_index
)


band.SetNoDataValue(
    NODATA
)


band.FlushCache()


out = None


# ============================================================
# SUMMARY STATISTICS
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
        float(
            np.min(
                valid_values
            )
        )
    ])


    writer.writerow([
        "Mean",
        float(
            np.mean(
                valid_values
            )
        )
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
        float(
            np.max(
                valid_values
            )
        )
    ])


# ============================================================
# FINAL CONSOLE REPORT
# ============================================================

print(
    "\n========== FAHP MAP SUMMARY =========="
)


print(
    "Valid pixels :",
    f"{len(valid_values):,}"
)


print(
    f"Minimum      : "
    f"{np.min(valid_values):.6f}"
)


print(
    f"Mean         : "
    f"{np.mean(valid_values):.6f}"
)


print(
    f"Median       : "
    f"{np.median(valid_values):.6f}"
)


print(
    f"Maximum      : "
    f"{np.max(valid_values):.6f}"
)


print(
    "\nFAHP output:"
)

print(
    MAP_OUTPUT
)


print(
    "\nFAHP weights:"
)

print(
    WEIGHTS_CSV
)


print(
    "\nAHP vs FAHP:"
)

print(
    COMPARE_CSV
)


print(
    "\nFINAL STATUS: DU STEP 11 FAHP PASS"
)