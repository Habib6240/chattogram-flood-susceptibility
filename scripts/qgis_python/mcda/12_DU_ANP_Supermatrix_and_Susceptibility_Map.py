import os
import csv
import numpy as np
from osgeo import gdal


# ============================================================
# 01. PATHS
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

FAHP_DIR = os.path.join(
    BASE,
    "12_DU_MCDA",
    "03_FAHP"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "12_DU_MCDA",
    "04_ANP"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

NODATA = -9999.0



# ============================================================
# 02. CRITERIA
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


INDEX = {
    c: i
    for i, c in enumerate(CRITERIA)
}


n = len(CRITERIA)



# ============================================================
# 03. STANDARDIZED RASTER FILES
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
# 04. SAME LITERATURE-INFORMED PAIRWISE JUDGMENTS
#
# Reusing the Step-10 AHP pairwise judgments deliberately.
#
# ANP difference will therefore originate primarily from
# network dependence / feedback rather than a new ranking.
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
# 05. AHP MATRIX QC
# ============================================================

if A.shape != (n, n):

    raise Exception(
        "Pairwise matrix must be 13 x 13."
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
        "Reciprocal matrix check FAILED."
    )


print(
    "\n========== ANP BASE MATRIX QC =========="
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
# 06. ANP DEPENDENCY NETWORK
#
# Key = influencing/source criterion.
#
# Values = criteria directly influenced by that source.
#
# This is a ONE-CLUSTER ANP network of conditioning factors.
#
# No self-loops are used.
# ============================================================

NETWORK = {


    # Rainfall directly modifies catchment wetness/runoff
    # response and long-term drainage response.

    "Rainfall": [

        "TWI",

        "Drainage_Density"

    ],


    # Terrain elevation controls gradients,
    # drainage organization, wetness distribution and
    # topographic curvature; orographic rainfall coupling
    # is also represented.

    "Elevation": [

        "Rainfall",

        "Slope",

        "TWI",

        "Drainage_Density",

        "Distance_River",

        "Profile_Curvature",

        "Tangential_Curvature"

    ],


    # Slope controls runoff concentration, wetness,
    # drainage response and terrain morphology.
    # Orographic precipitation and vegetation response
    # provide feedback pathways.

    "Slope": [

        "Rainfall",

        "TWI",

        "Drainage_Density",

        "Profile_Curvature",

        "Tangential_Curvature",

        "NDVI"

    ],


    # Drainage-network organization is coupled to
    # wetness, river proximity and terrain.

    "Drainage_Density": [

        "TWI",

        "Distance_River",

        "Elevation",

        "Slope"

    ],


    # River proximity interacts with local wetness,
    # drainage organization, lowland topography and LULC.

    "Distance_River": [

        "TWI",

        "Drainage_Density",

        "LULC",

        "Elevation"

    ],


    # TWI interacts with hydrological concentration,
    # vegetation, soil infiltration and river proximity.

    "TWI": [

        "Drainage_Density",

        "LULC",

        "NDVI",

        "HSG",

        "Distance_River"

    ],


    # LULC controls vegetation response, runoff,
    # road-related disturbance and infiltration behaviour.

    "LULC": [

        "NDVI",

        "TWI",

        "Distance_Road",

        "HSG",

        "Drainage_Density"

    ],


    # Hydrologic soil group controls infiltration/runoff,
    # which interacts with wetness and drainage response.

    "HSG": [

        "TWI",

        "LULC",

        "Drainage_Density"

    ],


    # NDVI interacts strongly with land cover and
    # terrain-controlled wetness/exposure.

    "NDVI": [

        "LULC",

        "TWI",

        "Slope",

        "Aspect"

    ],


    # Road proximity acts through surface modification,
    # vegetation/land cover and drainage disruption.

    "Distance_Road": [

        "LULC",

        "NDVI",

        "Drainage_Density",

        "TWI"

    ],


    # Profile curvature represents downslope
    # convergence/divergence and therefore couples
    # to slope, wetness and drainage.

    "Profile_Curvature": [

        "Slope",

        "TWI",

        "Drainage_Density",

        "Elevation"

    ],


    # Tangential curvature represents contour-normal
    # convergence/divergence and interacts with terrain,
    # wetness and exposure.

    "Tangential_Curvature": [

        "Slope",

        "TWI",

        "Drainage_Density",

        "Aspect"

    ],


    # Aspect controls exposure, vegetation and
    # orographic rainfall differences.

    "Aspect": [

        "Rainfall",

        "Slope",

        "NDVI",

        "LULC",

        "TWI"

    ]

}



# ============================================================
# 07. RANDOM INDEX TABLE
# ============================================================

RI = {

    1: 0.00,

    2: 0.00,

    3: 0.58,

    4: 0.90,

    5: 1.12,

    6: 1.24,

    7: 1.32,

    8: 1.41,

    9: 1.45,

    10: 1.49,

    11: 1.51,

    12: 1.48,

    13: 1.56

}



# ============================================================
# 08. PRINCIPAL EIGENVECTOR FUNCTION
# ============================================================

def principal_priority(matrix):

    eigenvalues, eigenvectors = np.linalg.eig(
        matrix
    )


    principal_index = np.argmax(
        eigenvalues.real
    )


    lambda_max = float(
        eigenvalues[
            principal_index
        ].real
    )


    vector = np.abs(
        eigenvectors[
            :,
            principal_index
        ].real
    )


    vector = (
        vector /
        vector.sum()
    )


    return vector, lambda_max



# ============================================================
# 09. LOCAL CONSISTENCY FUNCTION
# ============================================================

def consistency_ratio(
    matrix,
    lambda_max
):

    m = matrix.shape[0]


    if m <= 2:

        return (
            0.0,
            0.0
        )


    CI = (
        lambda_max - m
    ) / (
        m - 1
    )


    CR = (
        CI /
        RI[m]
    )


    return (
        CI,
        CR
    )



# ============================================================
# 10. BUILD COLUMN-STOCHASTIC SUPERMATRIX
#
# Each source criterion represents one column.
#
# For the criteria influenced by that source:
#
# 1. Extract their corresponding pairwise submatrix.
# 2. Calculate local priority vector.
# 3. Place local priorities in the supermatrix column.
#
# Since this is a single-cluster ANP and every column
# sums to 1, this matrix is already column-stochastic.
# ============================================================

W = np.zeros(
    (n, n),
    dtype=np.float64
)


local_records = []


print(
    "\n========== ANP LOCAL PRIORITIES =========="
)


for source in CRITERIA:


    if source not in NETWORK:

        raise Exception(
            "Missing network definition for "
            + source
        )


    targets = NETWORK[
        source
    ]


    if len(targets) == 0:

        raise Exception(
            "No outgoing dependency from "
            + source
        )


    target_indices = [
        INDEX[t]
        for t in targets
    ]


    # --------------------------------------------------------
    # Pairwise comparison among nodes influenced
    # by this source criterion.
    #
    # We restrict the original judgment structure
    # to the relevant connected elements.
    # --------------------------------------------------------

    local_matrix = A[
        np.ix_(
            target_indices,
            target_indices
        )
    ]


    local_weights, lambda_max = principal_priority(
        local_matrix
    )


    CI, CR = consistency_ratio(
        local_matrix,
        lambda_max
    )


    if CR >= 0.10:

        raise Exception(
            "Local ANP consistency FAILED for "
            + source
            + f" : CR={CR:.6f}"
        )


    source_index = INDEX[
        source
    ]


    for target, target_index, weight in zip(
        targets,
        target_indices,
        local_weights
    ):


        W[
            target_index,
            source_index
        ] = weight


        local_records.append([

            source,

            target,

            weight,

            len(targets),

            lambda_max,

            CI,

            CR

        ])


    print(
        f"{source:24s}"
        f" connections={len(targets):2d}"
        f" CR={CR:.6f}"
        f" -> PASS"
    )



# ============================================================
# 11. SUPERMATRIX QC
# ============================================================

column_sums = W.sum(
    axis=0
)


if not np.allclose(
    column_sums,
    np.ones(n),
    atol=1e-10
):

    raise Exception(
        "Supermatrix is not column stochastic."
    )


if np.any(
    W < 0
):

    raise Exception(
        "Negative supermatrix value detected."
    )


print(
    "\n========== SUPERMATRIX QC =========="
)

print(
    "Minimum column sum :",
    column_sums.min()
)

print(
    "Maximum column sum :",
    column_sums.max()
)

print(
    "Column stochastic  : PASS"
)



# ============================================================
# 12. LIMIT SUPERMATRIX
#
# ANP final priorities are obtained by raising the
# weighted/column-stochastic supermatrix to successive
# powers until convergence.
# ============================================================

LIMIT = W.copy()


TOLERANCE = 1e-12

MAX_ITERATIONS = 10000


converged = False


for iteration in range(
    1,
    MAX_ITERATIONS + 1
):


    NEXT = LIMIT @ W


    difference = np.max(
        np.abs(
            NEXT - LIMIT
        )
    )


    LIMIT = NEXT


    if difference < TOLERANCE:

        converged = True

        break



if not converged:

    raise Exception(
        "Limit supermatrix failed to converge."
    )



# ============================================================
# 13. LIMIT-MATRIX COLUMN EQUALITY QC
# ============================================================

column_spread = np.max(
    np.ptp(
        LIMIT,
        axis=1
    )
)


if column_spread > 1e-8:

    raise Exception(
        "Limit supermatrix columns did not converge "
        "to common priorities."
    )


weights = LIMIT[
    :,
    0
].copy()


weights = (
    weights /
    weights.sum()
)



if not np.isclose(
    weights.sum(),
    1.0,
    atol=1e-10
):

    raise Exception(
        "Final ANP weights do not sum to 1."
    )


print(
    "\n========== LIMIT SUPERMATRIX =========="
)

print(
    "Iterations       :",
    iteration
)

print(
    f"Final difference : {difference:.3e}"
)

print(
    f"Column spread    : {column_spread:.3e}"
)

print(
    "Convergence      : PASS"
)



# ============================================================
# 14. FINAL ANP WEIGHTS
# ============================================================

print(
    "\n========== FINAL ANP WEIGHTS =========="
)


for criterion, weight in zip(
    CRITERIA,
    weights
):


    print(
        f"{criterion:24s}"
        f" {weight:.6f}"
        f" ({weight*100:.2f}%)"
    )



# ============================================================
# 15. OUTPUT PATHS
# ============================================================

NETWORK_CSV = os.path.join(
    OUTPUT_DIR,
    "DU12_ANP_Network_Links.csv"
)


LOCAL_CSV = os.path.join(
    OUTPUT_DIR,
    "DU12_ANP_Local_Priorities_and_CR.csv"
)


SUPER_CSV = os.path.join(
    OUTPUT_DIR,
    "DU12_ANP_Column_Stochastic_Supermatrix.csv"
)


LIMIT_CSV = os.path.join(
    OUTPUT_DIR,
    "DU12_ANP_Limit_Supermatrix.csv"
)


WEIGHTS_CSV = os.path.join(
    OUTPUT_DIR,
    "DU12_ANP_Final_Weights.csv"
)


COMPARE_CSV = os.path.join(
    OUTPUT_DIR,
    "DU12_AHP_FAHP_ANP_Weight_Comparison.csv"
)


MAP_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "DU12_ANP_Susceptibility_Index_01.tif"
)


SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "DU12_ANP_Map_Summary.csv"
)



# ============================================================
# 16. SAVE NETWORK LINKS
# ============================================================

with open(
    NETWORK_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Source_Criterion",

        "Influenced_Criterion"

    ])


    for source in CRITERIA:


        for target in NETWORK[
            source
        ]:


            writer.writerow([

                source,

                target

            ])



# ============================================================
# 17. SAVE LOCAL PRIORITIES + CR
# ============================================================

with open(
    LOCAL_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Source_Criterion",

        "Influenced_Criterion",

        "Local_Priority",

        "Number_of_Connected_Elements",

        "Local_Lambda_Max",

        "Local_CI",

        "Local_CR"

    ])


    writer.writerows(
        local_records
    )



# ============================================================
# 18. SAVE SUPERMATRIX FUNCTION
# ============================================================

def save_matrix(
    path,
    matrix
):


    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:


        writer = csv.writer(
            f
        )


        writer.writerow(
            ["Criterion"] +
            CRITERIA
        )


        for i, criterion in enumerate(
            CRITERIA
        ):


            writer.writerow(
                [criterion] +
                list(
                    matrix[
                        i,
                        :
                    ]
                )
            )



save_matrix(
    SUPER_CSV,
    W
)


save_matrix(
    LIMIT_CSV,
    LIMIT
)



# ============================================================
# 19. SAVE FINAL ANP WEIGHTS
# ============================================================

with open(
    WEIGHTS_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Criterion",

        "ANP_Weight",

        "ANP_Weight_Percent"

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
        "Metric",
        "Value"
    ])


    writer.writerow([
        "Limit_Iterations",
        iteration
    ])


    writer.writerow([
        "Final_Difference",
        difference
    ])


    writer.writerow([
        "Limit_Column_Spread",
        column_spread
    ])



# ============================================================
# 20. READ PREVIOUS AHP WEIGHTS
# ============================================================

AHP_FILE = os.path.join(
    AHP_DIR,
    "DU10_AHP_Weights_and_Consistency.csv"
)


ahp_weights = {}


with open(
    AHP_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as f:


    reader = csv.reader(
        f
    )


    next(
        reader
    )


    for row in reader:


        if len(row) < 3:

            break


        if row[0] in CRITERIA:


            ahp_weights[
                row[0]
            ] = float(
                row[1]
            )



# ============================================================
# 21. READ PREVIOUS FAHP WEIGHTS
# ============================================================

FAHP_FILE = os.path.join(
    FAHP_DIR,
    "DU11_FAHP_Weights.csv"
)


fahp_weights = {}


with open(
    FAHP_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as f:


    reader = csv.DictReader(
        f
    )


    for row in reader:


        criterion = row[
            "Criterion"
        ]


        if criterion in CRITERIA:


            fahp_weights[
                criterion
            ] = float(
                row[
                    "Final_Normalized_Weight"
                ]
            )



# ============================================================
# 22. SAVE 3-METHOD WEIGHT COMPARISON
# ============================================================

with open(
    COMPARE_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Criterion",

        "AHP_Weight",

        "FAHP_Weight",

        "ANP_Weight",

        "ANP_minus_AHP",

        "ANP_minus_FAHP"

    ])


    for i, criterion in enumerate(
        CRITERIA
    ):


        writer.writerow([

            criterion,

            ahp_weights[
                criterion
            ],

            fahp_weights[
                criterion
            ],

            weights[i],

            weights[i]
            - ahp_weights[
                criterion
            ],

            weights[i]
            - fahp_weights[
                criterion
            ]

        ])



# ============================================================
# 23. ANP WEIGHTED RASTER OVERLAY
# ============================================================

susceptibility = None

common_valid = None


rows = None

cols = None

geotransform = None

projection = None


for i, criterion in enumerate(
    CRITERIA
):


    path = os.path.join(
        INPUT_DIR,
        FILES[
            criterion
        ]
    )


    if not os.path.exists(
        path
    ):


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


    band = ds.GetRasterBand(
        1
    )


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


    if susceptibility is None:


        rows = ds.RasterYSize

        cols = ds.RasterXSize


        geotransform = ds.GetGeoTransform()

        projection = ds.GetProjection()


        susceptibility = np.zeros(
            (
                rows,
                cols
            ),
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


    ds = None



# ============================================================
# 24. FINAL ANP INDEX
# ============================================================

final_index = np.full(
    (
        rows,
        cols
    ),
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
# 25. RANGE QC
# ============================================================

if np.min(
    valid_values
) < -1e-6:


    raise Exception(
        "ANP susceptibility below zero."
    )


if np.max(
    valid_values
) > 1.000001:


    raise Exception(
        "ANP susceptibility above one."
    )



# ============================================================
# 26. VALID PIXEL QC
# ============================================================

if len(
    valid_values
) != 4722216:


    raise Exception(
        "ANP valid-pixel count mismatch."
    )



# ============================================================
# 27. EXPORT ANP RASTER
# ============================================================

driver = gdal.GetDriverByName(
    "GTiff"
)


out = driver.Create(

    MAP_OUTPUT,

    cols,

    rows,

    1,

    gdal.GDT_Float32,

    options=[
        "COMPRESS=LZW"
    ]

)


out.SetGeoTransform(
    geotransform
)


out.SetProjection(
    projection
)


out_band = out.GetRasterBand(
    1
)


out_band.WriteArray(
    final_index
)


out_band.SetNoDataValue(
    NODATA
)


out_band.FlushCache()


out = None



# ============================================================
# 28. MAP SUMMARY
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


    writer = csv.writer(
        f
    )


    writer.writerow([
        "Metric",
        "Value"
    ])


    writer.writerow([
        "Valid_Pixels",
        len(
            valid_values
        )
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
# 29. FINAL REPORT
# ============================================================

print(
    "\n========== ANP MAP SUMMARY =========="
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
    "\nNetwork:"
)

print(
    NETWORK_CSV
)


print(
    "\nLocal priorities:"
)

print(
    LOCAL_CSV
)


print(
    "\nSupermatrix:"
)

print(
    SUPER_CSV
)


print(
    "\nLimit supermatrix:"
)

print(
    LIMIT_CSV
)


print(
    "\nWeights:"
)

print(
    WEIGHTS_CSV
)


print(
    "\nAHP / FAHP / ANP comparison:"
)

print(
    COMPARE_CSV
)


print(
    "\nANP susceptibility raster:"
)

print(
    MAP_OUTPUT
)


print(
    "\nFINAL STATUS: DU STEP 12 ANP PASS"
)