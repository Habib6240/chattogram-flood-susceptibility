import os
import csv
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from xgboost import (
    XGBClassifier,
    DMatrix
)

import joblib

import matplotlib.pyplot as plt


# ============================================================
# B08
# XGBOOST TREE-SHAP EXPLAINABILITY
#
# PURPOSE
# ------------------------------------------------------------
# 1. Fit final explanatory XGBoost model using all clean
#    four-event composite samples.
#
# 2. Use native XGBoost TreeSHAP:
#
#       booster.predict(..., pred_contribs=True)
#
# 3. Aggregate encoded variables back to the original
#    13 conceptual conditioning factors.
#
# 4. Produce:
#    - global SHAP importance
#    - encoded-feature importance
#    - geographically distributed explanation sample
#    - SHAP additivity QC
#    - publication-ready importance bar figure
#
# IMPORTANT
# ------------------------------------------------------------
# B05-B07 remain the validation evidence.
#
# This full-data model is used for explanation only.
# Training performance is NOT reported as validation.
# ============================================================


# ============================================================
# 01. PATHS
# ============================================================

BASE = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
)


INPUT_NPZ = os.path.join(
    BASE,
    "13_BUET_ML",
    "02_ML_Dataset",
    "BUET_B04_ML_Master_150m.npz"
)


B06_FOLD_FILE = os.path.join(
    BASE,
    "13_BUET_ML",
    "04_Spatial_CV",
    "BUET_B06_5km_Block_Spatial_Folds.csv"
)


OUTPUT_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "06_Explainability"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


GLOBAL_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B08_XGB_SHAP_Global_13Factor_Importance.csv"
)


ENCODED_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B08_XGB_SHAP_Encoded_Feature_Importance.csv"
)


SAMPLE_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B08_XGB_SHAP_Explanation_Sample.csv"
)


SAMPLE_NPZ = os.path.join(
    OUTPUT_DIR,
    "BUET_B08_XGB_SHAP_Explanation_Sample.npz"
)


QC_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B08_SHAP_QC.csv"
)


FIGURE_PNG = os.path.join(
    OUTPUT_DIR,
    "BUET_B08_XGB_SHAP_Global_Importance.png"
)


MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "BUET_B08_XGB_FullComposite_Model.joblib"
)


PREPROCESSOR_FILE = os.path.join(
    OUTPUT_DIR,
    "BUET_B08_XGB_FullComposite_Preprocessor.joblib"
)


# ============================================================
# 02. LOAD B04 DATA
# ============================================================

if not os.path.exists(
    INPUT_NPZ
):

    raise Exception(
        "B04 dataset not found:\n"
        +
        INPUT_NPZ
    )


if not os.path.exists(
    B06_FOLD_FILE
):

    raise Exception(
        "B06 fold file not found:\n"
        +
        B06_FOLD_FILE
    )


D = np.load(
    INPUT_NPZ,
    allow_pickle=False
)


# ============================================================
# 03. RAW ML FEATURE MATRIX
# ============================================================

X_ALL = np.column_stack([

    D["Elevation"],

    D["Slope"],

    D["Aspect_sin"],

    D["Aspect_cos"],

    D["Aspect_flat"],

    D["Profile_Curvature"],

    D["Tangential_Curvature"],

    D["TWI"],

    D["Drainage_Density"],

    D["Rainfall"],

    D["LULC"],

    D["NDVI"],

    D["HSG"],

    D["Distance_River"],

    D["Distance_Road"]

]).astype(
    np.float32
)


Y_ALL = D[
    "Composite_AnyFlood_All4Valid"
].astype(
    np.int8
)


BLOCK_ALL = D[
    "Block_5km"
].astype(
    np.int64
)


X_COORD_ALL = D[
    "X"
].astype(
    np.float64
)


Y_COORD_ALL = D[
    "Y"
].astype(
    np.float64
)


# ============================================================
# 04. CLEAN COMPOSITE DOMAIN
# ============================================================

VALID = (
    Y_ALL >= 0
)


ORIGINAL_INDEX = np.where(
    VALID
)[0]


X = X_ALL[
    VALID
]


y = Y_ALL[
    VALID
]


blocks = BLOCK_ALL[
    VALID
]


x_coord = X_COORD_ALL[
    VALID
]


y_coord = Y_COORD_ALL[
    VALID
]


N = len(
    y
)


POS = int(
    np.count_nonzero(
        y == 1
    )
)


NEG = int(
    np.count_nonzero(
        y == 0
    )
)


print(
    "\n========== B08 EXPLANATION DATASET =========="
)

print(
    "Rows       :",
    f"{N:,}"
)

print(
    "Flood      :",
    f"{POS:,}"
)

print(
    "Non-flood  :",
    f"{NEG:,}"
)

print(
    "Flood %    :",
    f"{100 * POS / N:.3f}%"
)


# ============================================================
# 05. REUSE B06 GEOGRAPHIC FOLDS
# ============================================================

BLOCK_TO_FOLD = {}


with open(
    B06_FOLD_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as f:


    reader = csv.DictReader(
        f
    )


    for row in reader:


        BLOCK_TO_FOLD[
            int(
                row["Block_5km"]
            )
        ] = int(
            row["Spatial_Fold"]
        )


sample_fold = np.array(

    [
        BLOCK_TO_FOLD[
            int(
                block
            )
        ]

        for block in blocks
    ],

    dtype=np.uint8

)


print(
    "Spatial folds:",
    sorted(
        np.unique(
            sample_fold
        ).tolist()
    )
)


# ============================================================
# 06. FEATURE TYPES
# ============================================================

NUMERIC_INDICES = [

    0,      # Elevation

    1,      # Slope

    2,      # Aspect sin

    3,      # Aspect cos

    4,      # Aspect flat

    5,      # Profile curvature

    6,      # Tangential curvature

    7,      # TWI

    8,      # Drainage density

    9,      # Rainfall

    11,     # NDVI

    13,     # Distance river

    14      # Distance road

]


CATEGORICAL_INDICES = [

    10,     # LULC

    12      # HSG

]


NUMERIC_NAMES = [

    "Elevation",

    "Slope",

    "Aspect_sin",

    "Aspect_cos",

    "Aspect_flat",

    "Profile_Curvature",

    "Tangential_Curvature",

    "TWI",

    "Drainage_Density",

    "Rainfall",

    "NDVI",

    "Distance_River",

    "Distance_Road"

]


# ============================================================
# 07. ONE-HOT PREPROCESSOR
# ============================================================

try:

    encoder = OneHotEncoder(

        handle_unknown="ignore",

        sparse_output=False,

        dtype=np.float32

    )

except TypeError:

    encoder = OneHotEncoder(

        handle_unknown="ignore",

        sparse=False,

        dtype=np.float32

    )


preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",

            "passthrough",

            NUMERIC_INDICES
        ),

        (
            "categorical",

            encoder,

            CATEGORICAL_INDICES
        )

    ],

    remainder="drop",

    sparse_threshold=0.0

)


print(
    "\n========== FEATURE ENCODING =========="
)


X_ENCODED = preprocessor.fit_transform(
    X
).astype(
    np.float32
)


print(
    "Raw columns     :",
    X.shape[1]
)

print(
    "Encoded columns :",
    X_ENCODED.shape[1]
)


# ============================================================
# 08. EXTRACT CATEGORY VALUES
# ============================================================

fitted_encoder = (
    preprocessor
    .named_transformers_[
        "categorical"
    ]
)


LULC_CATEGORIES = (
    fitted_encoder
    .categories_[0]
)


HSG_CATEGORIES = (
    fitted_encoder
    .categories_[1]
)


print(
    "LULC categories:",
    LULC_CATEGORIES.tolist()
)

print(
    "HSG categories :",
    HSG_CATEGORIES.tolist()
)


# ============================================================
# 09. BUILD ENCODED FEATURE NAMES
# ============================================================

ENCODED_NAMES = list(
    NUMERIC_NAMES
)


for value in LULC_CATEGORIES:


    ENCODED_NAMES.append(
        "LULC_"
        +
        str(
            int(
                value
            )
        )
    )


for value in HSG_CATEGORIES:


    ENCODED_NAMES.append(
        "HSG_"
        +
        str(
            int(
                value
            )
        )
    )


if len(
    ENCODED_NAMES
) != X_ENCODED.shape[1]:

    raise Exception(
        "Encoded feature-name mismatch."
    )


# ============================================================
# 10. FINAL EXPLANATORY XGBOOST MODEL
#
# Same architecture as B05-B07.
#
# Fit on all clean composite data because:
#
# - validation is already completed independently
# - B08 objective is interpretation, not validation
# ============================================================

SCALE_POS_WEIGHT = (
    NEG
    /
    POS
)


print(
    "\n========== FULL-DATA EXPLANATORY XGBOOST =========="
)

print(
    "scale_pos_weight:",
    f"{SCALE_POS_WEIGHT:.6f}"
)


MODEL = XGBClassifier(

    n_estimators=600,

    learning_rate=0.05,

    max_depth=5,

    min_child_weight=2,

    subsample=0.80,

    colsample_bytree=0.80,

    reg_lambda=1.0,

    reg_alpha=0.0,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=
        SCALE_POS_WEIGHT,

    tree_method="hist",

    n_jobs=-1,

    random_state=42

)


print(
    "Training..."
)


MODEL.fit(
    X_ENCODED,
    y
)


print(
    "Training complete."
)


joblib.dump(
    MODEL,
    MODEL_FILE,
    compress=3
)


joblib.dump(
    preprocessor,
    PREPROCESSOR_FILE,
    compress=3
)


# ============================================================
# 11. GEOGRAPHICALLY DISTRIBUTED SHAP SAMPLE
#
# 4,000 samples per B06 geographic fold.
#
# Total target = 20,000.
#
# Sampling remains proportional to natural class prevalence
# within each fold because we do not artificially balance it.
# ============================================================

RNG = np.random.default_rng(
    42
)


PER_FOLD = 4000


explain_indices = []


print(
    "\n========== SHAP SAMPLE =========="
)


for fold in range(
    1,
    6
):


    candidates = np.where(
        sample_fold
        ==
        fold
    )[0]


    take_n = min(
        PER_FOLD,
        len(
            candidates
        )
    )


    chosen = RNG.choice(

        candidates,

        size=take_n,

        replace=False

    )


    explain_indices.extend(
        chosen.tolist()
    )


    fold_y = y[
        chosen
    ]


    print(
        f"Fold {fold}:"
        f" N={take_n:,}"
        f" flood={np.count_nonzero(fold_y == 1):,}"
        f" flood%={100 * np.mean(fold_y == 1):.2f}%"
    )


explain_indices = np.array(
    explain_indices,
    dtype=np.int64
)


X_EXPLAIN = X_ENCODED[
    explain_indices
]


Y_EXPLAIN = y[
    explain_indices
]


print(
    "Total SHAP sample:",
    f"{len(explain_indices):,}"
)


# ============================================================
# 12. NATIVE XGBOOST TREE-SHAP
#
# pred_contribs returns:
#
# [SHAP feature 1 ... SHAP feature p, BIAS]
#
# Contributions are on raw margin / log-odds scale.
# ============================================================

print(
    "\n========== TREE-SHAP =========="
)

print(
    "Calculating exact tree contributions..."
)


BOOSTER = MODEL.get_booster()


DMAT = DMatrix(
    X_EXPLAIN
)


CONTRIB = BOOSTER.predict(

    DMAT,

    pred_contribs=True

)


if CONTRIB.shape[1] != (
    X_EXPLAIN.shape[1]
    +
    1
):

    raise Exception(
        "Unexpected SHAP contribution shape."
    )


SHAP_VALUES = CONTRIB[
    :,
    :-1
]


SHAP_BIAS = CONTRIB[
    :,
    -1
]


print(
    "SHAP matrix:",
    SHAP_VALUES.shape
)


# ============================================================
# 13. SHAP ADDITIVITY QC
#
# Sum(SHAP contributions + bias)
# must reproduce raw XGBoost margin.
# ============================================================

MARGIN = BOOSTER.predict(

    DMAT,

    output_margin=True

)


RECONSTRUCTED_MARGIN = np.sum(
    CONTRIB,
    axis=1
)


MAX_MARGIN_ERROR = float(
    np.max(
        np.abs(
            MARGIN
            -
            RECONSTRUCTED_MARGIN
        )
    )
)


PROBABILITY = MODEL.predict_proba(
    X_EXPLAIN
)[
    :,
    1
]


SIGMOID_MARGIN = (
    1.0
    /
    (
        1.0
        +
        np.exp(
            -MARGIN
        )
    )
)


MAX_PROBABILITY_ERROR = float(
    np.max(
        np.abs(
            PROBABILITY
            -
            SIGMOID_MARGIN
        )
    )
)


print(
    "\n========== SHAP ADDITIVITY QC =========="
)

print(
    "Max margin reconstruction error:",
    f"{MAX_MARGIN_ERROR:.10e}"
)

print(
    "Max sigmoid probability error  :",
    f"{MAX_PROBABILITY_ERROR:.10e}"
)


if MAX_MARGIN_ERROR > 1e-4:

    raise Exception(
        "SHAP additivity QC failed."
    )


if MAX_PROBABILITY_ERROR > 1e-5:

    raise Exception(
        "Probability reconstruction QC failed."
    )


print(
    "SHAP additivity : PASS"
)


# ============================================================
# 14. ENCODED-FEATURE GLOBAL IMPORTANCE
# ============================================================

ENCODED_MEAN_ABS = np.mean(

    np.abs(
        SHAP_VALUES
    ),

    axis=0

)


ENCODED_MEAN_SIGNED = np.mean(

    SHAP_VALUES,

    axis=0

)


encoded_order = np.argsort(
    -ENCODED_MEAN_ABS
)


with open(
    ENCODED_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Rank",

        "Encoded_Feature",

        "Mean_Absolute_SHAP",

        "Mean_Signed_SHAP"

    ])


    for rank, idx in enumerate(
        encoded_order,
        start=1
    ):


        writer.writerow([

            rank,

            ENCODED_NAMES[
                idx
            ],

            float(
                ENCODED_MEAN_ABS[
                    idx
                ]
            ),

            float(
                ENCODED_MEAN_SIGNED[
                    idx
                ]
            )

        ])


# ============================================================
# 15. GROUP ENCODED FEATURES BACK TO 13 FACTORS
#
# IMPORTANT:
#
# Group contribution is calculated ROW-WISE:
#
#     grouped SHAP =
#     sum(component SHAP values)
#
# Then:
#
#     global importance =
#     mean(abs(grouped SHAP))
#
# This avoids artificially inflating importance simply because
# a categorical variable has many dummy columns.
# ============================================================

LULC_START = len(
    NUMERIC_NAMES
)


LULC_INDICES = list(
    range(
        LULC_START,
        LULC_START
        +
        len(
            LULC_CATEGORIES
        )
    )
)


HSG_START = (
    LULC_START
    +
    len(
        LULC_CATEGORIES
    )
)


HSG_INDICES = list(
    range(
        HSG_START,
        HSG_START
        +
        len(
            HSG_CATEGORIES
        )
    )
)


FACTOR_GROUPS = {

    "Elevation":
        [0],

    "Slope":
        [1],

    "Aspect":
        [2, 3, 4],

    "Profile_Curvature":
        [5],

    "Tangential_Curvature":
        [6],

    "TWI":
        [7],

    "Drainage_Density":
        [8],

    "Rainfall":
        [9],

    "LULC":
        LULC_INDICES,

    "NDVI":
        [10],

    "HSG":
        HSG_INDICES,

    "Distance_River":
        [11],

    "Distance_Road":
        [12]

}


FACTOR_ORDER = [

    "Elevation",

    "Slope",

    "Aspect",

    "Profile_Curvature",

    "Tangential_Curvature",

    "TWI",

    "Drainage_Density",

    "Rainfall",

    "LULC",

    "NDVI",

    "HSG",

    "Distance_River",

    "Distance_Road"

]


GROUPED_SHAP = {}


for factor in FACTOR_ORDER:


    indices = FACTOR_GROUPS[
        factor
    ]


    GROUPED_SHAP[
        factor
    ] = np.sum(

        SHAP_VALUES[
            :,
            indices
        ],

        axis=1

    )


# ============================================================
# 16. GLOBAL 13-FACTOR IMPORTANCE
# ============================================================

global_rows = []


for factor in FACTOR_ORDER:


    values = GROUPED_SHAP[
        factor
    ]


    mean_abs = float(
        np.mean(
            np.abs(
                values
            )
        )
    )


    mean_signed = float(
        np.mean(
            values
        )
    )


    global_rows.append([

        factor,

        mean_abs,

        mean_signed

    ])


TOTAL_IMPORTANCE = sum(
    row[1]
    for row in global_rows
)


global_rows.sort(

    key=lambda row:
        row[1],

    reverse=True

)


print(
    "\n========== GLOBAL 13-FACTOR SHAP RANKING =========="
)


final_global_rows = []


for rank, row in enumerate(
    global_rows,
    start=1
):


    factor = row[0]

    mean_abs = row[1]

    mean_signed = row[2]


    relative = (

        100
        *
        mean_abs
        /
        TOTAL_IMPORTANCE

    )


    final_global_rows.append([

        rank,

        factor,

        mean_abs,

        mean_signed,

        relative

    ])


    print(
        f"{rank:2d}. "
        f"{factor:24s}"
        f" mean|SHAP|={mean_abs:.6f}"
        f" relative={relative:6.2f}%"
    )


with open(
    GLOBAL_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Rank",

        "Factor",

        "Mean_Absolute_Grouped_SHAP",

        "Mean_Signed_Grouped_SHAP",

        "Relative_Importance_Percent"

    ])


    writer.writerows(
        final_global_rows
    )


# ============================================================
# 17. RAW FACTOR VALUES FOR EXPLANATION SAMPLE
# ============================================================

original_sample_index = ORIGINAL_INDEX[
    explain_indices
]


RAW_FACTOR = {}


RAW_FACTOR[
    "Elevation"
] = D[
    "Elevation"
][
    original_sample_index
]


RAW_FACTOR[
    "Slope"
] = D[
    "Slope"
][
    original_sample_index
]


# Reconstruct aspect angle from sin/cos.
aspect_sin = D[
    "Aspect_sin"
][
    original_sample_index
]


aspect_cos = D[
    "Aspect_cos"
][
    original_sample_index
]


aspect_flat = D[
    "Aspect_flat"
][
    original_sample_index
]


aspect_angle = np.degrees(

    np.arctan2(
        aspect_sin,
        aspect_cos
    )

)


aspect_angle[
    aspect_angle < 0
] += 360.0


aspect_angle[
    aspect_flat == 1
] = -1.0


RAW_FACTOR[
    "Aspect"
] = aspect_angle


for factor in [

    "Profile_Curvature",

    "Tangential_Curvature",

    "TWI",

    "Drainage_Density",

    "Rainfall",

    "LULC",

    "NDVI",

    "HSG",

    "Distance_River",

    "Distance_Road"

]:


    RAW_FACTOR[
        factor
    ] = D[
        factor
    ][
        original_sample_index
    ]


# ============================================================
# 18. SAVE EXPLANATION SAMPLE CSV
#
# Contains raw factor value + grouped SHAP contribution.
# Useful for dependence plots later.
# ============================================================

with open(
    SAMPLE_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    header = [

        "X",

        "Y",

        "Spatial_Fold",

        "Observed_Composite_Label",

        "Predicted_Probability",

        "SHAP_Bias"

    ]


    for factor in FACTOR_ORDER:


        header.append(
            factor
            +
            "_Value"
        )


        header.append(
            factor
            +
            "_SHAP"
        )


    writer.writerow(
        header
    )


    for i in range(
        len(
            explain_indices
        )
    ):


        row = [

            float(
                x_coord[
                    explain_indices[
                        i
                    ]
                ]
            ),

            float(
                y_coord[
                    explain_indices[
                        i
                    ]
                ]
            ),

            int(
                sample_fold[
                    explain_indices[
                        i
                    ]
                ]
            ),

            int(
                Y_EXPLAIN[
                    i
                ]
            ),

            float(
                PROBABILITY[
                    i
                ]
            ),

            float(
                SHAP_BIAS[
                    i
                ]
            )

        ]


        for factor in FACTOR_ORDER:


            row.append(
                float(
                    RAW_FACTOR[
                        factor
                    ][
                        i
                    ]
                )
            )


            row.append(
                float(
                    GROUPED_SHAP[
                        factor
                    ][
                        i
                    ]
                )
            )


        writer.writerow(
            row
        )


# ============================================================
# 19. SAVE COMPRESSED NUMERIC SHAP DATA
# ============================================================

np.savez_compressed(

    SAMPLE_NPZ,

    Explain_Index=
        explain_indices,

    Observed_Label=
        Y_EXPLAIN,

    Predicted_Probability=
        PROBABILITY.astype(
            np.float32
        ),

    SHAP_Bias=
        SHAP_BIAS.astype(
            np.float32
        ),

    Encoded_SHAP=
        SHAP_VALUES.astype(
            np.float32
        ),

    Grouped_SHAP=
        np.column_stack(
            [
                GROUPED_SHAP[
                    factor
                ]

                for factor in FACTOR_ORDER
            ]
        ).astype(
            np.float32
        )

)


# ============================================================
# 20. SAVE QC
# ============================================================

with open(
    QC_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([
        "QC_Metric",
        "Value"
    ])


    writer.writerows([

        [
            "Composite_Model_Rows",
            N
        ],

        [
            "Encoded_Feature_Count",
            X_ENCODED.shape[1]
        ],

        [
            "Conceptual_Factor_Count",
            13
        ],

        [
            "SHAP_Sample_N",
            len(
                explain_indices
            )
        ],

        [
            "Sampling_Design",
            "4000 random samples per B06 geographic fold"
        ],

        [
            "SHAP_Method",
            "Native XGBoost TreeSHAP pred_contribs"
        ],

        [
            "SHAP_Scale",
            "Raw model margin / log-odds"
        ],

        [
            "Maximum_Margin_Reconstruction_Error",
            MAX_MARGIN_ERROR
        ],

        [
            "Maximum_Probability_Reconstruction_Error",
            MAX_PROBABILITY_ERROR
        ],

        [
            "Additivity_Status",
            "PASS"
        ]

    ])


# ============================================================
# 21. GLOBAL IMPORTANCE BAR FIGURE
# ============================================================

plot_rows = list(
    reversed(
        final_global_rows
    )
)


plot_names = [
    row[1]
    for row in plot_rows
]


plot_values = [
    row[2]
    for row in plot_rows
]


plt.figure(
    figsize=(
        8,
        6.5
    )
)


plt.barh(
    plot_names,
    plot_values
)


plt.xlabel(
    "Mean absolute grouped SHAP value"
)


plt.ylabel(
    "Conditioning factor"
)


plt.title(
    "XGBoost global factor importance (TreeSHAP)"
)


plt.tight_layout()


plt.savefig(
    FIGURE_PNG,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 22. FINAL
# ============================================================

print(
    "\n=============================================="
)

print(
    "BUET B08 XGBOOST TREE-SHAP COMPLETE"
)

print(
    "=============================================="
)

print(
    "\nOutputs:"
)

print(
    GLOBAL_CSV
)

print(
    ENCODED_CSV
)

print(
    SAMPLE_CSV
)

print(
    SAMPLE_NPZ
)

print(
    QC_CSV
)

print(
    FIGURE_PNG
)

print(
    MODEL_FILE
)

print(
    PREPROCESSOR_FILE
)

print(
    "\nFINAL STATUS: BUET B08 PASS"
)

print(
    "=============================================="
)