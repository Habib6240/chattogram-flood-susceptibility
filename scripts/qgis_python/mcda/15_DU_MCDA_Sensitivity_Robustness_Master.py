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

STD_DIR = os.path.join(
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

ANP_DIR = os.path.join(
    BASE,
    "12_DU_MCDA",
    "04_ANP"
)

VALIDATION_DIR = os.path.join(
    BASE,
    "12_DU_MCDA",
    "06_Sentinel1_Validation"
)

S1_FILE = os.path.join(
    BASE,
    "13_BUET_ML",
    "01_Sentinel1_Event_Masks",
    "BUET_B03_E04_2021_Jul_FloodMask30m.tif"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "12_DU_MCDA",
    "07_Sensitivity_Robustness"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 02. INPUT MAPS
# ============================================================

AHP_MAP = os.path.join(
    AHP_DIR,
    "DU10_AHP_Susceptibility_Index_01.tif"
)

FAHP_MAP = os.path.join(
    FAHP_DIR,
    "DU11_FAHP_Susceptibility_Index_01.tif"
)

ANP_MAP = os.path.join(
    ANP_DIR,
    "DU12_ANP_Susceptibility_Index_01.tif"
)


# ============================================================
# 03. WEIGHT FILES
# ============================================================

AHP_WEIGHT_FILE = os.path.join(
    AHP_DIR,
    "DU10_AHP_Weights_and_Consistency.csv"
)

FAHP_WEIGHT_FILE = os.path.join(
    FAHP_DIR,
    "DU11_FAHP_Weights.csv"
)

ANP_WEIGHT_FILE = os.path.join(
    ANP_DIR,
    "DU12_ANP_Final_Weights.csv"
)


# ============================================================
# 04. STEP-14 OFFSET RESULTS
# ============================================================

OFFSET_FILE = os.path.join(
    VALIDATION_DIR,
    "DU14_300m_Systematic_Offset_AUC.csv"
)


# ============================================================
# 05. OUTPUT FILES
# ============================================================

PERTURB_CSV = os.path.join(
    OUTPUT_DIR,
    "DU15_Weight_Perturbation_PlusMinus20pct.csv"
)

LOFO_CSV = os.path.join(
    OUTPUT_DIR,
    "DU15_Leave_One_Factor_Out.csv"
)

ANP_LINK_CSV = os.path.join(
    OUTPUT_DIR,
    "DU15_ANP_Link_Removal_Sensitivity.csv"
)

OFFSET_COMPARE_CSV = os.path.join(
    OUTPUT_DIR,
    "DU15_Offset_Paired_Method_Robustness.csv"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "DU15_Sensitivity_Robustness_Summary.csv"
)


# ============================================================
# 06. CRITERIA ORDER
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


n = len(
    CRITERIA
)


INDEX = {
    criterion: i
    for i, criterion in enumerate(
        CRITERIA
    )
}


# ============================================================
# 07. STANDARDIZED RASTERS
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
# 08. READ RASTER
# ============================================================

def read_raster(path):

    if not os.path.exists(
        path
    ):

        raise Exception(
            "Missing raster:\n"
            + path
        )


    ds = gdal.Open(
        path
    )


    if ds is None:

        raise Exception(
            "Cannot open raster:\n"
            + path
        )


    band = ds.GetRasterBand(
        1
    )


    arr = band.ReadAsArray()


    nodata = band.GetNoDataValue()


    valid = np.isfinite(
        arr.astype(
            np.float64
        )
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


    return (
        ds,
        arr,
        valid
    )


# ============================================================
# 09. LOAD MAPS + S1
# ============================================================

ahp_ds, ahp_map, ahp_valid = read_raster(
    AHP_MAP
)

fahp_ds, fahp_map, fahp_valid = read_raster(
    FAHP_MAP
)

anp_ds, anp_map, anp_valid = read_raster(
    ANP_MAP
)

s1_ds, s1, s1_valid_raw = read_raster(
    S1_FILE
)


s1_valid = (
    s1_valid_raw
    &
    (
        (s1 == 0)
        |
        (s1 == 1)
    )
)


validation_mask = (
    ahp_valid
    &
    fahp_valid
    &
    anp_valid
    &
    s1_valid
)


# ============================================================
# 10. REPRESENTATIVE SPATIALLY THINNED SAMPLE
#
# Same deterministic offset used in Step 14:
#
# 10 pixels × 30 m = 300 m
#
# row offset = 5
# col offset = 5
# ============================================================

STEP = 10
ROW_OFFSET = 5
COL_OFFSET = 5


sample_mask = np.zeros(
    validation_mask.shape,
    dtype=bool
)


sample_mask[
    ROW_OFFSET::STEP,
    COL_OFFSET::STEP
] = True


sample_mask &= validation_mask


y = s1[
    sample_mask
].astype(
    np.uint8
)


sample_n = len(
    y
)


flood_n = int(
    np.count_nonzero(
        y == 1
    )
)


nonflood_n = int(
    np.count_nonzero(
        y == 0
    )
)


print(
    "\n========== DU15 SENSITIVITY SAMPLE =========="
)

print(
    "Spacing        : 300 m"
)

print(
    "Grid offset    : row 5 / col 5"
)

print(
    "Sample N       :",
    f"{sample_n:,}"
)

print(
    "Flood N        :",
    f"{flood_n:,}"
)

print(
    "Non-flood N    :",
    f"{nonflood_n:,}"
)


if (
    flood_n == 0
    or
    nonflood_n == 0
):

    raise Exception(
        "Sensitivity sample lacks both classes."
    )


# ============================================================
# 11. FAST EXACT ROC-AUC
#
# Same manual trapezoidal approach as Step 14.
# ============================================================

def calculate_auc(
    y_true,
    scores
):

    y_true = y_true.astype(
        np.uint8
    )

    scores = scores.astype(
        np.float64
    )


    order = np.argsort(
        -scores,
        kind="mergesort"
    )


    sorted_y = y_true[
        order
    ]

    sorted_scores = scores[
        order
    ]


    distinct = np.where(
        np.diff(
            sorted_scores
        ) != 0
    )[0]


    threshold_indices = np.r_[
        distinct,
        len(
            sorted_scores
        ) - 1
    ]


    cumulative_tp = np.cumsum(
        sorted_y,
        dtype=np.int64
    )


    tp = cumulative_tp[
        threshold_indices
    ].astype(
        np.float64
    )


    selected = (
        threshold_indices + 1
    ).astype(
        np.float64
    )


    fp = (
        selected
        -
        tp
    )


    P = float(
        np.count_nonzero(
            y_true == 1
        )
    )


    N = float(
        np.count_nonzero(
            y_true == 0
        )
    )


    tpr = np.r_[
        0.0,
        tp / P
    ]


    fpr = np.r_[
        0.0,
        fp / N
    ]


    delta_fpr = (
        fpr[1:]
        -
        fpr[:-1]
    )


    mean_tpr = (
        tpr[1:]
        +
        tpr[:-1]
    ) * 0.5


    return float(
        np.sum(
            delta_fpr
            *
            mean_tpr
        )
    )


# ============================================================
# 12. READ AHP WEIGHTS
# ============================================================

def read_ahp_weights():

    result = {}


    with open(
        AHP_WEIGHT_FILE,
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


            if len(
                row
            ) < 3:

                break


            if row[0] in CRITERIA:

                result[
                    row[0]
                ] = float(
                    row[1]
                )


    return np.array(
        [
            result[c]
            for c in CRITERIA
        ],
        dtype=np.float64
    )


# ============================================================
# 13. READ FAHP WEIGHTS
# ============================================================

def read_fahp_weights():

    result = {}


    with open(
        FAHP_WEIGHT_FILE,
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


                result[
                    criterion
                ] = float(
                    row[
                        "Final_Normalized_Weight"
                    ]
                )


    return np.array(
        [
            result[c]
            for c in CRITERIA
        ],
        dtype=np.float64
    )


# ============================================================
# 14. READ ANP WEIGHTS
# ============================================================

def read_anp_weights():

    result = {}


    with open(
        ANP_WEIGHT_FILE,
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


            if len(
                row
            ) < 3:

                break


            if row[0] in CRITERIA:


                result[
                    row[0]
                ] = float(
                    row[1]
                )


    return np.array(
        [
            result[c]
            for c in CRITERIA
        ],
        dtype=np.float64
    )


AHP_WEIGHTS = read_ahp_weights()

FAHP_WEIGHTS = read_fahp_weights()

ANP_WEIGHTS = read_anp_weights()


METHOD_WEIGHTS = {

    "AHP":
        AHP_WEIGHTS,

    "FAHP":
        FAHP_WEIGHTS,

    "ANP":
        ANP_WEIGHTS

}


for name, weights in METHOD_WEIGHTS.items():

    if not np.isclose(
        weights.sum(),
        1.0,
        atol=1e-8
    ):

        raise Exception(
            name +
            " weights do not sum to 1."
        )


print(
    "\nWeight-file QC : PASS"
)


# ============================================================
# 15. EXTRACT 13 STANDARDIZED CRITERIA
# AT THE 300 m SAMPLE LOCATIONS
# ============================================================

X = np.zeros(
    (
        sample_n,
        n
    ),
    dtype=np.float32
)


print(
    "\n========== EXTRACTING CRITERIA =========="
)


for j, criterion in enumerate(
    CRITERIA
):


    path = os.path.join(
        STD_DIR,
        FILES[
            criterion
        ]
    )


    ds, arr, valid = read_raster(
        path
    )


    if not np.all(
        valid[
            sample_mask
        ]
    ):

        raise Exception(
            "Missing standardized values in sensitivity sample: "
            +
            criterion
        )


    X[
        :,
        j
    ] = arr[
        sample_mask
    ].astype(
        np.float32
    )


    print(
        criterion,
        "-> extracted"
    )


    ds = None


# ============================================================
# 16. BASELINE MAP REPRODUCTION QC
#
# X @ weights should reproduce values in saved maps.
# ============================================================

saved_sample_scores = {

    "AHP":
        ahp_map[
            sample_mask
        ].astype(
            np.float64
        ),

    "FAHP":
        fahp_map[
            sample_mask
        ].astype(
            np.float64
        ),

    "ANP":
        anp_map[
            sample_mask
        ].astype(
            np.float64
        )

}


baseline_auc = {}


print(
    "\n========== BASELINE REPRODUCTION QC =========="
)


for method, weights in METHOD_WEIGHTS.items():


    reconstructed = (
        X.astype(
            np.float64
        )
        @
        weights
    )


    saved = saved_sample_scores[
        method
    ]


    rmse = float(
        np.sqrt(
            np.mean(
                (
                    reconstructed
                    -
                    saved
                ) ** 2
            )
        )
    )


    max_error = float(
        np.max(
            np.abs(
                reconstructed
                -
                saved
            )
        )
    )


    auc = calculate_auc(
        y,
        reconstructed
    )


    baseline_auc[
        method
    ] = auc


    print(
        f"{method:4s}"
        f" RMSE={rmse:.10f}"
        f" MaxError={max_error:.10f}"
        f" AUC={auc:.6f}"
    )


    if max_error > 1e-5:

        raise Exception(
            method +
            " baseline reconstruction failed."
        )


print(
    "Baseline reproduction : PASS"
)


# ============================================================
# 17. ±20% ONE-AT-A-TIME WEIGHT PERTURBATION
#
# Target weight multiplied by 0.8 or 1.2.
# All weights then renormalized to sum to 1.
# ============================================================

PERTURBATIONS = {

    "-20%":
        0.80,

    "+20%":
        1.20

}


perturb_rows = []


print(
    "\n========== ±20% WEIGHT PERTURBATION =========="
)


for method, original_weights in METHOD_WEIGHTS.items():


    for i, criterion in enumerate(
        CRITERIA
    ):


        for label, multiplier in PERTURBATIONS.items():


            w = original_weights.copy()


            original_target = w[
                i
            ]


            w[
                i
            ] *= multiplier


            w /= w.sum()


            scores = (
                X
                @
                w
            )


            auc = calculate_auc(
                y,
                scores
            )


            delta_auc = (
                auc
                -
                baseline_auc[
                    method
                ]
            )


            perturb_rows.append([

                method,

                criterion,

                label,

                original_target,

                w[i],

                baseline_auc[
                    method
                ],

                auc,

                delta_auc,

                abs(
                    delta_auc
                )

            ])


with open(
    PERTURB_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Method",

        "Perturbed_Criterion",

        "Perturbation",

        "Original_Weight",

        "Perturbed_Normalized_Weight",

        "Baseline_AUC",

        "Scenario_AUC",

        "Delta_AUC",

        "Absolute_Delta_AUC"

    ])


    writer.writerows(
        perturb_rows
    )


# ============================================================
# 18. LEAVE-ONE-FACTOR-OUT
#
# Removed factor weight = 0.
# Remaining weights renormalized.
# ============================================================

lofo_rows = []


print(
    "\n========== LEAVE-ONE-FACTOR-OUT =========="
)


for method, original_weights in METHOD_WEIGHTS.items():


    for i, criterion in enumerate(
        CRITERIA
    ):


        w = original_weights.copy()


        removed_weight = w[
            i
        ]


        w[
            i
        ] = 0.0


        w /= w.sum()


        scores = (
            X
            @
            w
        )


        auc = calculate_auc(
            y,
            scores
        )


        delta_auc = (
            auc
            -
            baseline_auc[
                method
            ]
        )


        lofo_rows.append([

            method,

            criterion,

            removed_weight,

            baseline_auc[
                method
            ],

            auc,

            delta_auc,

            abs(
                delta_auc
            )

        ])


with open(
    LOFO_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Method",

        "Removed_Criterion",

        "Removed_Original_Weight",

        "Baseline_AUC",

        "LeaveOneOut_AUC",

        "Delta_AUC",

        "Absolute_Delta_AUC"

    ])


    writer.writerows(
        lofo_rows
    )


# ============================================================
# 19. PAIRWISE ROBUSTNESS ACROSS ALL 100
# STEP-14 SYSTEMATIC OFFSETS
# ============================================================

offset_data = []


with open(
    OFFSET_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as f:


    reader = csv.DictReader(
        f
    )


    for row in reader:


        offset_data.append({

            "AHP":
                float(
                    row[
                        "AHP_AUC"
                    ]
                ),

            "FAHP":
                float(
                    row[
                        "FAHP_AUC"
                    ]
                ),

            "ANP":
                float(
                    row[
                        "ANP_AUC"
                    ]
                )

        })


if len(
    offset_data
) != 100:

    raise Exception(
        "Expected exactly 100 Step-14 offsets."
    )


PAIRWISE = [

    (
        "FAHP",
        "AHP"
    ),

    (
        "ANP",
        "AHP"
    ),

    (
        "FAHP",
        "ANP"
    )

]


offset_compare_rows = []


print(
    "\n========== PAIRED 300 m OFFSET ROBUSTNESS =========="
)


for method1, method2 in PAIRWISE:


    differences = np.array(
        [
            row[
                method1
            ]
            -
            row[
                method2
            ]
            for row in offset_data
        ],
        dtype=np.float64
    )


    wins = float(
        np.mean(
            differences > 0
        )
        *
        100
    )


    ties = float(
        np.mean(
            np.isclose(
                differences,
                0,
                atol=1e-12
            )
        )
        *
        100
    )


    p025 = float(
        np.percentile(
            differences,
            2.5
        )
    )


    p975 = float(
        np.percentile(
            differences,
            97.5
        )
    )


    mean_diff = float(
        np.mean(
            differences
        )
    )


    median_diff = float(
        np.median(
            differences
        )
    )


    offset_compare_rows.append([

        method1,

        method2,

        mean_diff,

        median_diff,

        p025,

        p975,

        wins,

        ties

    ])


    print(
        f"{method1:4s} - {method2:4s}"
        f" mean ΔAUC={mean_diff:.6f}"
        f" empirical 2.5–97.5%=[{p025:.6f}, {p975:.6f}]"
        f" wins={wins:.1f}%"
    )


with open(
    OFFSET_COMPARE_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Method_1",

        "Method_2",

        "Mean_Paired_AUC_Difference",

        "Median_Paired_AUC_Difference",

        "P02_5_Paired_Difference",

        "P97_5_Paired_Difference",

        "Method1_Win_Percent",

        "Tie_Percent"

    ])


    writer.writerows(
        offset_compare_rows
    )


# ============================================================
# 20. ANP NETWORK
#
# Same network as Step 12.
# ============================================================

NETWORK = {


    "Rainfall": [

        "TWI",

        "Drainage_Density"

    ],


    "Elevation": [

        "Rainfall",

        "Slope",

        "TWI",

        "Drainage_Density",

        "Distance_River",

        "Profile_Curvature",

        "Tangential_Curvature"

    ],


    "Slope": [

        "Rainfall",

        "TWI",

        "Drainage_Density",

        "Profile_Curvature",

        "Tangential_Curvature",

        "NDVI"

    ],


    "Drainage_Density": [

        "TWI",

        "Distance_River",

        "Elevation",

        "Slope"

    ],


    "Distance_River": [

        "TWI",

        "Drainage_Density",

        "LULC",

        "Elevation"

    ],


    "TWI": [

        "Drainage_Density",

        "LULC",

        "NDVI",

        "HSG",

        "Distance_River"

    ],


    "LULC": [

        "NDVI",

        "TWI",

        "Distance_Road",

        "HSG",

        "Drainage_Density"

    ],


    "HSG": [

        "TWI",

        "LULC",

        "Drainage_Density"

    ],


    "NDVI": [

        "LULC",

        "TWI",

        "Slope",

        "Aspect"

    ],


    "Distance_Road": [

        "LULC",

        "NDVI",

        "Drainage_Density",

        "TWI"

    ],


    "Profile_Curvature": [

        "Slope",

        "TWI",

        "Drainage_Density",

        "Elevation"

    ],


    "Tangential_Curvature": [

        "Slope",

        "TWI",

        "Drainage_Density",

        "Aspect"

    ],


    "Aspect": [

        "Rainfall",

        "Slope",

        "NDVI",

        "LULC",

        "TWI"

    ]

}


# ============================================================
# 21. SAME BASE AHP JUDGMENT MATRIX USED IN STEP 12
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
# 22. LOCAL PRIORITY FUNCTION
# ============================================================

def principal_priority(
    matrix
):

    if matrix.shape == (
        1,
        1
    ):

        return np.array(
            [1.0],
            dtype=np.float64
        )


    eigenvalues, eigenvectors = np.linalg.eig(
        matrix
    )


    idx = np.argmax(
        eigenvalues.real
    )


    vector = np.abs(
        eigenvectors[
            :,
            idx
        ].real
    )


    return (
        vector
        /
        vector.sum()
    )


# ============================================================
# 23. BUILD ANP SUPERMATRIX
# ============================================================

def build_supermatrix(
    network
):

    W = np.zeros(
        (
            n,
            n
        ),
        dtype=np.float64
    )


    for source in CRITERIA:


        targets = network[
            source
        ]


        if len(
            targets
        ) == 0:

            return None


        target_indices = [
            INDEX[
                target
            ]
            for target in targets
        ]


        local_matrix = A[
            np.ix_(
                target_indices,
                target_indices
            )
        ]


        local_weights = principal_priority(
            local_matrix
        )


        source_index = INDEX[
            source
        ]


        for target_index, weight in zip(
            target_indices,
            local_weights
        ):


            W[
                target_index,
                source_index
            ] = weight


    if not np.allclose(
        W.sum(
            axis=0
        ),
        np.ones(
            n
        ),
        atol=1e-10
    ):

        return None


    return W


# ============================================================
# 24. LIMIT SUPERMATRIX
# ============================================================

def limit_weights(
    W,
    tolerance=1e-12,
    max_iterations=10000
):

    LIMIT = W.copy()


    for iteration in range(
        1,
        max_iterations + 1
    ):


        NEXT = (
            LIMIT
            @
            W
        )


        difference = float(
            np.max(
                np.abs(
                    NEXT
                    -
                    LIMIT
                )
            )
        )


        LIMIT = NEXT


        if difference < tolerance:


            spread = float(
                np.max(
                    np.ptp(
                        LIMIT,
                        axis=1
                    )
                )
            )


            if spread > 1e-8:

                return (
                    None,
                    iteration,
                    difference,
                    spread
                )


            weights = LIMIT[
                :,
                0
            ].copy()


            weights /= weights.sum()


            return (
                weights,
                iteration,
                difference,
                spread
            )


    return (
        None,
        max_iterations,
        difference,
        np.nan
    )


# ============================================================
# 25. BASE NETWORK REPRODUCTION QC
# ============================================================

baseline_W = build_supermatrix(
    NETWORK
)


baseline_network_weights, _, _, _ = (
    limit_weights(
        baseline_W
    )
)


if baseline_network_weights is None:

    raise Exception(
        "Baseline ANP network did not converge."
    )


anp_weight_error = float(
    np.max(
        np.abs(
            baseline_network_weights
            -
            ANP_WEIGHTS
        )
    )
)


print(
    "\n========== ANP NETWORK REPRODUCTION =========="
)

print(
    f"Max weight error : {anp_weight_error:.10e}"
)


if anp_weight_error > 1e-6:

    raise Exception(
        "ANP Step-12 network could not be reproduced."
    )


print(
    "ANP network reproduction : PASS"
)


# ============================================================
# 26. REMOVE EACH ANP LINK ONE AT A TIME
# ============================================================

link_rows = []


total_links = sum(
    len(
        targets
    )
    for targets in NETWORK.values()
)


print(
    "\n========== ANP LINK-REMOVAL SENSITIVITY =========="
)

print(
    "Total dependency links :",
    total_links
)


for source in CRITERIA:


    for target in NETWORK[
        source
    ]:


        modified_network = {

            key:
                list(
                    values
                )

            for key, values in NETWORK.items()
        }


        modified_network[
            source
        ].remove(
            target
        )


        W = build_supermatrix(
            modified_network
        )


        if W is None:


            link_rows.append([

                source,

                target,

                "FAILED_SUPERMATRIX",

                "",

                "",

                "",

                "",

                ""

            ])


            continue


        (
            new_weights,
            iterations,
            difference,
            spread
        ) = limit_weights(
            W
        )


        if new_weights is None:


            link_rows.append([

                source,

                target,

                "FAILED_CONVERGENCE",

                iterations,

                difference,

                spread,

                "",

                ""

            ])


            continue


        scores = (
            X
            @
            new_weights
        )


        auc = calculate_auc(
            y,
            scores
        )


        delta_auc = (
            auc
            -
            baseline_auc[
                "ANP"
            ]
        )


        weight_shift_l1 = float(
            np.sum(
                np.abs(
                    new_weights
                    -
                    ANP_WEIGHTS
                )
            )
        )


        link_rows.append([

            source,

            target,

            "PASS",

            iterations,

            difference,

            spread,

            auc,

            delta_auc,

            weight_shift_l1

        ])


with open(
    ANP_LINK_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Source_Criterion",

        "Removed_Link_Target",

        "Status",

        "Limit_Iterations",

        "Final_Difference",

        "Limit_Column_Spread",

        "Scenario_AUC",

        "Delta_AUC",

        "L1_Weight_Vector_Shift"

    ])


    writer.writerows(
        link_rows
    )


# ============================================================
# 27. CONSOLE SUMMARY — WEIGHT PERTURBATION
# ============================================================

print(
    "\n========== MOST SENSITIVE ±20% WEIGHT CASES =========="
)


for method in [
    "AHP",
    "FAHP",
    "ANP"
]:


    rows = [
        row
        for row in perturb_rows
        if row[0] == method
    ]


    worst = max(
        rows,
        key=lambda row: row[8]
    )


    max_abs_delta = worst[
        8
    ]


    print(
        f"{method:4s}"
        f" max |ΔAUC|={max_abs_delta:.6f}"
        f" from {worst[1]} {worst[2]}"
        f" -> AUC={worst[6]:.6f}"
    )


# ============================================================
# 28. CONSOLE SUMMARY — LEAVE-ONE-OUT
# ============================================================

print(
    "\n========== MOST INFLUENTIAL FACTOR REMOVALS =========="
)


for method in [
    "AHP",
    "FAHP",
    "ANP"
]:


    rows = [
        row
        for row in lofo_rows
        if row[0] == method
    ]


    worst_drop = min(
        rows,
        key=lambda row: row[5]
    )


    print(
        f"{method:4s}"
        f" largest AUC drop:"
        f" remove {worst_drop[1]}"
        f"  ΔAUC={worst_drop[5]:.6f}"
        f"  AUC={worst_drop[4]:.6f}"
    )


# ============================================================
# 29. CONSOLE SUMMARY — ANP LINK REMOVAL
# ============================================================

passed_links = [
    row
    for row in link_rows
    if row[2] == "PASS"
]


failed_links = [
    row
    for row in link_rows
    if row[2] != "PASS"
]


print(
    "\n========== ANP LINK-REMOVAL SUMMARY =========="
)

print(
    "Successful scenarios :",
    len(
        passed_links
    )
)

print(
    "Failed scenarios     :",
    len(
        failed_links
    )
)


if len(
    passed_links
) > 0:


    worst_link = min(
        passed_links,
        key=lambda row: row[7]
    )


    best_link = max(
        passed_links,
        key=lambda row: row[7]
    )


    max_abs_link = max(
        passed_links,
        key=lambda row: abs(
            row[7]
        )
    )


    print(
        "Largest AUC decrease:"
    )

    print(
        f"  Remove {worst_link[0]} -> {worst_link[1]}"
        f"  ΔAUC={worst_link[7]:.6f}"
        f"  AUC={worst_link[6]:.6f}"
    )


    print(
        "Largest AUC increase:"
    )

    print(
        f"  Remove {best_link[0]} -> {best_link[1]}"
        f"  ΔAUC={best_link[7]:.6f}"
        f"  AUC={best_link[6]:.6f}"
    )


    print(
        "Largest absolute link effect:"
    )

    print(
        f"  {max_abs_link[0]} -> {max_abs_link[1]}"
        f"  |ΔAUC|={abs(max_abs_link[7]):.6f}"
    )


# ============================================================
# 30. SAVE MASTER SUMMARY
# ============================================================

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
        "Section",
        "Metric",
        "Value"
    ])


    writer.writerow([
        "Sensitivity_Sample",
        "Sample_N",
        sample_n
    ])


    writer.writerow([
        "Sensitivity_Sample",
        "Flood_N",
        flood_n
    ])


    writer.writerow([
        "Sensitivity_Sample",
        "NonFlood_N",
        nonflood_n
    ])


    for method in [
        "AHP",
        "FAHP",
        "ANP"
    ]:


        writer.writerow([
            "Baseline",
            method + "_AUC_300m_Representative",
            baseline_auc[
                method
            ]
        ])


        p_rows = [
            row
            for row in perturb_rows
            if row[0] == method
        ]


        maximum_perturbation = max(
            row[8]
            for row in p_rows
        )


        writer.writerow([
            "Weight_Perturbation",
            method + "_Maximum_Absolute_Delta_AUC",
            maximum_perturbation
        ])


        l_rows = [
            row
            for row in lofo_rows
            if row[0] == method
        ]


        worst_drop = min(
            row[5]
            for row in l_rows
        )


        writer.writerow([
            "Leave_One_Factor_Out",
            method + "_Largest_AUC_Drop",
            worst_drop
        ])


    writer.writerow([
        "ANP_Network",
        "Total_Links",
        total_links
    ])


    writer.writerow([
        "ANP_Network",
        "Successful_Link_Removal_Scenarios",
        len(
            passed_links
        )
    ])


    writer.writerow([
        "ANP_Network",
        "Failed_Link_Removal_Scenarios",
        len(
            failed_links
        )
    ])


    if len(
        passed_links
    ) > 0:


        writer.writerow([
            "ANP_Network",
            "Largest_Absolute_Link_Delta_AUC",
            max(
                abs(
                    row[7]
                )
                for row in passed_links
            )
        ])


# ============================================================
# 31. FINAL STATUS
# ============================================================

print(
    "\n=============================================="
)

print(
    "DU STEP 15 SENSITIVITY / ROBUSTNESS COMPLETE"
)

print(
    "\nOutputs:"
)

print(
    PERTURB_CSV
)

print(
    LOFO_CSV
)

print(
    ANP_LINK_CSV
)

print(
    OFFSET_COMPARE_CSV
)

print(
    SUMMARY_CSV
)

print(
    "\nFINAL STATUS: DU STEP 15 PASS"
)

print(
    "=============================================="
)