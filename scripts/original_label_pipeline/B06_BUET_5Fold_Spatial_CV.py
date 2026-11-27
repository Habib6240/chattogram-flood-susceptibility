import os
import csv
import sys
import numpy as np

import sklearn

from sklearn.cluster import KMeans

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    brier_score_loss
)

from xgboost import XGBClassifier


# ============================================================
# B06
# 5-FOLD GEOGRAPHICALLY STRUCTURED SPATIAL CV
#
# Composite target:
# 1 = flooded in >=1 of four events
# 0 = observed non-flood in all four events
#
# Spatial units:
# existing 5-km blocks from B04
#
# Fold construction:
# K-means clustering of 5-km block centroids into
# five geographically compact regions.
#
# IMPORTANT:
# - preprocessing fitted within each training fold
# - class weights calculated within each training fold
# - threshold determined using TRAINING data only
# - test fold remains untouched
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

OUTPUT_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "04_Spatial_CV"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


FOLD_ASSIGNMENT_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B06_5km_Block_Spatial_Folds.csv"
)

FOLD_PERFORMANCE_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B06_Fold_Performance.csv"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B06_Spatial_CV_Summary.csv"
)

OOF_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B06_OutOfFold_Predictions.csv"
)

PAIRED_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B06_RF_vs_XGB_Paired_Fold_Comparison.csv"
)


# ============================================================
# 02. LOAD B04 DATA
# ============================================================

if not os.path.exists(
    INPUT_NPZ
):

    raise Exception(
        "B04 NPZ not found:\n"
        +
        INPUT_NPZ
    )


D = np.load(
    INPUT_NPZ,
    allow_pickle=False
)


# ============================================================
# 03. RAW FEATURE MATRIX
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


RASTER_ROW_ALL = D[
    "Raster_Row"
].astype(
    np.int32
)


RASTER_COL_ALL = D[
    "Raster_Column"
].astype(
    np.int32
)


# ============================================================
# 04. CLEAN COMPOSITE DOMAIN
# ============================================================

VALID = (
    Y_ALL >= 0
)


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


raster_row = RASTER_ROW_ALL[
    VALID
]


raster_col = RASTER_COL_ALL[
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
    "\n========== B06 DATASET =========="
)

print(
    "Rows             :",
    f"{N:,}"
)

print(
    "Flood            :",
    f"{POS:,}"
)

print(
    "Non-flood        :",
    f"{NEG:,}"
)

print(
    "Flood %          :",
    f"{100 * POS / N:.3f}%"
)

print(
    "Unique 5km blocks:",
    len(
        np.unique(
            blocks
        )
    )
)


# ============================================================
# 05. FEATURE COLUMN TYPES
# ============================================================

NUMERIC_INDICES = [

    0,      # Elevation

    1,      # Slope

    2,      # Aspect sin

    3,      # Aspect cos

    4,      # Flat

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


# ============================================================
# 06. CALCULATE 5-KM BLOCK CENTROIDS
# ============================================================

unique_blocks = np.unique(
    blocks
)


block_centroid_x = []

block_centroid_y = []

block_n = []

block_flood_n = []


for block in unique_blocks:


    mask = (
        blocks == block
    )


    block_centroid_x.append(
        np.mean(
            x_coord[
                mask
            ]
        )
    )


    block_centroid_y.append(
        np.mean(
            y_coord[
                mask
            ]
        )
    )


    block_n.append(
        np.count_nonzero(
            mask
        )
    )


    block_flood_n.append(
        np.count_nonzero(
            y[
                mask
            ]
            ==
            1
        )
    )


block_centroid_x = np.array(
    block_centroid_x,
    dtype=np.float64
)


block_centroid_y = np.array(
    block_centroid_y,
    dtype=np.float64
)


block_n = np.array(
    block_n,
    dtype=np.int64
)


block_flood_n = np.array(
    block_flood_n,
    dtype=np.int64
)


block_xy = np.column_stack([

    block_centroid_x,

    block_centroid_y

])


# ============================================================
# 07. CREATE FIVE GEOGRAPHIC FOLDS
#
# K-means on block centroids.
#
# Random state fixed for reproducibility.
# ============================================================

N_FOLDS = 5


kmeans = KMeans(

    n_clusters=N_FOLDS,

    random_state=42,

    n_init=50

)


raw_fold = kmeans.fit_predict(
    block_xy
)


# ------------------------------------------------------------
# Re-number clusters deterministically from west to east
# based on cluster centroid X.
#
# This prevents arbitrary KMeans labels from changing the
# presentation order.
# ------------------------------------------------------------

cluster_centers = (
    kmeans.cluster_centers_
)


west_to_east_order = np.argsort(
    cluster_centers[
        :,
        0
    ]
)


label_map = {}


for new_fold, old_label in enumerate(
    west_to_east_order,
    start=1
):

    label_map[
        int(
            old_label
        )
    ] = new_fold


block_fold = np.array(
    [
        label_map[
            int(
                value
            )
        ]

        for value in raw_fold
    ],
    dtype=np.uint8
)


BLOCK_TO_FOLD = {

    int(
        block
    ):
        int(
            fold
        )

    for block, fold in zip(
        unique_blocks,
        block_fold
    )

}


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


# ============================================================
# 08. FOLD QC
# ============================================================

print(
    "\n========== GEOGRAPHIC FOLD QC =========="
)


fold_qc = []


for fold in range(
    1,
    N_FOLDS + 1
):


    test_mask = (
        sample_fold
        ==
        fold
    )


    fold_y = y[
        test_mask
    ]


    fn = len(
        fold_y
    )


    fp = int(
        np.count_nonzero(
            fold_y == 1
        )
    )


    fneg = int(
        np.count_nonzero(
            fold_y == 0
        )
    )


    fold_blocks = np.unique(
        blocks[
            test_mask
        ]
    )


    prevalence = (
        fp
        /
        fn
    )


    fold_qc.append([

        fold,

        len(
            fold_blocks
        ),

        fn,

        fp,

        fneg,

        prevalence

    ])


    print(
        f"Fold {fold}:"
        f" blocks={len(fold_blocks):3d}"
        f" N={fn:7,d}"
        f" flood={fp:6,d}"
        f" nonflood={fneg:7,d}"
        f" flood%={100 * prevalence:6.2f}"
    )


    if fp < 500:

        raise Exception(
            f"Fold {fold} has too few flood observations."
        )


    if fneg < 500:

        raise Exception(
            f"Fold {fold} has too few non-flood observations."
        )


# ============================================================
# 09. SAVE BLOCK/FOLD ASSIGNMENTS
# ============================================================

with open(
    FOLD_ASSIGNMENT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Block_5km",

        "Centroid_X",

        "Centroid_Y",

        "Spatial_Fold",

        "Sample_N",

        "Flood_N",

        "NonFlood_N",

        "Flood_Percent"

    ])


    for i in range(
        len(
            unique_blocks
        )
    ):


        bn = int(
            block_n[
                i
            ]
        )


        bp = int(
            block_flood_n[
                i
            ]
        )


        writer.writerow([

            int(
                unique_blocks[
                    i
                ]
            ),

            float(
                block_centroid_x[
                    i
                ]
            ),

            float(
                block_centroid_y[
                    i
                ]
            ),

            int(
                block_fold[
                    i
                ]
            ),

            bn,

            bp,

            bn - bp,

            100
            *
            bp
            /
            bn

        ])


# ============================================================
# 10. PREPROCESSOR FACTORY
#
# Must be re-created and re-fitted in each fold.
# ============================================================

def make_preprocessor():


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


    return ColumnTransformer(

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


# ============================================================
# 11. MODEL FACTORIES
# ============================================================

def make_rf():


    return RandomForestClassifier(

        n_estimators=500,

        max_features="sqrt",

        min_samples_leaf=2,

        class_weight="balanced_subsample",

        bootstrap=True,

        n_jobs=-1,

        random_state=42

    )


def make_xgb(
    scale_pos_weight
):


    return XGBClassifier(

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
            scale_pos_weight,

        tree_method="hist",

        n_jobs=-1,

        random_state=42

    )


# ============================================================
# 12. TRAIN-DERIVED YOUDEN THRESHOLD
#
# Threshold is estimated only from TRAINING predictions.
#
# Test labels are never used for threshold selection.
# ============================================================

def training_youden_threshold(
    y_train,
    p_train
):


    fpr, tpr, thresholds = roc_curve(

        y_train,

        p_train

    )


    j = (
        tpr
        -
        fpr
    )


    index = int(
        np.argmax(
            j
        )
    )


    return float(
        thresholds[
            index
        ]
    )


# ============================================================
# 13. TEST METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probability,
    threshold
):


    prediction = (
        probability
        >=
        threshold
    ).astype(
        np.uint8
    )


    auc = roc_auc_score(
        y_true,
        probability
    )


    ap = average_precision_score(
        y_true,
        probability
    )


    tn, fp, fn, tp = confusion_matrix(

        y_true,

        prediction,

        labels=[
            0,
            1
        ]

    ).ravel()


    sensitivity = recall_score(

        y_true,

        prediction,

        zero_division=0

    )


    specificity = (

        tn
        /
        (
            tn
            +
            fp
        )

        if (
            tn
            +
            fp
        ) > 0

        else np.nan

    )


    precision = precision_score(

        y_true,

        prediction,

        zero_division=0

    )


    f1 = f1_score(

        y_true,

        prediction,

        zero_division=0

    )


    balanced_accuracy = (
        balanced_accuracy_score(

            y_true,

            prediction

        )
    )


    accuracy = accuracy_score(

        y_true,

        prediction

    )


    brier = brier_score_loss(

        y_true,

        probability

    )


    return {

        "ROC_AUC":
            float(
                auc
            ),

        "Average_Precision":
            float(
                ap
            ),

        "Threshold":
            float(
                threshold
            ),

        "Sensitivity":
            float(
                sensitivity
            ),

        "Specificity":
            float(
                specificity
            ),

        "Precision":
            float(
                precision
            ),

        "F1":
            float(
                f1
            ),

        "Balanced_Accuracy":
            float(
                balanced_accuracy
            ),

        "Accuracy":
            float(
                accuracy
            ),

        "Brier":
            float(
                brier
            ),

        "TP":
            int(
                tp
            ),

        "TN":
            int(
                tn
            ),

        "FP":
            int(
                fp
            ),

        "FN":
            int(
                fn
            )

    }


# ============================================================
# 14. OUT-OF-FOLD ARRAYS
# ============================================================

RF_OOF = np.full(
    N,
    np.nan,
    dtype=np.float32
)


XGB_OOF = np.full(
    N,
    np.nan,
    dtype=np.float32
)


fold_rows = []


# ============================================================
# 15. FIVE-FOLD SPATIAL CV
# ============================================================

for fold in range(
    1,
    N_FOLDS + 1
):


    print(
        "\n=============================================="
    )

    print(
        f"SPATIAL FOLD {fold} / {N_FOLDS}"
    )

    print(
        "=============================================="
    )


    test_idx = np.where(
        sample_fold
        ==
        fold
    )[0]


    train_idx = np.where(
        sample_fold
        !=
        fold
    )[0]


    X_train_raw = X[
        train_idx
    ]


    X_test_raw = X[
        test_idx
    ]


    y_train = y[
        train_idx
    ]


    y_test = y[
        test_idx
    ]


    train_blocks = np.unique(
        blocks[
            train_idx
        ]
    )


    test_blocks = np.unique(
        blocks[
            test_idx
        ]
    )


    block_overlap = np.intersect1d(

        train_blocks,

        test_blocks

    )


    if len(
        block_overlap
    ) != 0:

        raise Exception(
            f"Fold {fold}: spatial block leakage."
        )


    train_pos = int(
        np.count_nonzero(
            y_train == 1
        )
    )


    train_neg = int(
        np.count_nonzero(
            y_train == 0
        )
    )


    test_pos = int(
        np.count_nonzero(
            y_test == 1
        )
    )


    test_neg = int(
        np.count_nonzero(
            y_test == 0
        )
    )


    print(
        "Train N       :",
        f"{len(y_train):,}"
    )

    print(
        "Test N        :",
        f"{len(y_test):,}"
    )

    print(
        "Train blocks  :",
        len(
            train_blocks
        )
    )

    print(
        "Test blocks   :",
        len(
            test_blocks
        )
    )

    print(
        "Block overlap :",
        len(
            block_overlap
        )
    )

    print(
        "Train flood % :",
        f"{100 * train_pos / len(y_train):.3f}%"
    )

    print(
        "Test flood %  :",
        f"{100 * test_pos / len(y_test):.3f}%"
    )


    # --------------------------------------------------------
    # Preprocessing fitted ONLY on training fold
    # --------------------------------------------------------

    preprocessor = make_preprocessor()


    X_train = preprocessor.fit_transform(

        X_train_raw

    ).astype(
        np.float32
    )


    X_test = preprocessor.transform(

        X_test_raw

    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print(
        "\nRF training..."
    )


    rf = make_rf()


    rf.fit(

        X_train,

        y_train

    )


    rf_train_prob = rf.predict_proba(

        X_train

    )[
        :,
        1
    ]


    rf_threshold = training_youden_threshold(

        y_train,

        rf_train_prob

    )


    rf_test_prob = rf.predict_proba(

        X_test

    )[
        :,
        1
    ]


    RF_OOF[
        test_idx
    ] = rf_test_prob.astype(
        np.float32
    )


    rf_result = calculate_metrics(

        y_test,

        rf_test_prob,

        rf_threshold

    )


    print(
        f"RF  AUC={rf_result['ROC_AUC']:.6f}"
        f" AP={rf_result['Average_Precision']:.6f}"
        f" BA={rf_result['Balanced_Accuracy']:.6f}"
        f" F1={rf_result['F1']:.6f}"
    )


    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    scale_pos_weight = (

        train_neg
        /
        train_pos

    )


    print(
        "XGB training..."
    )


    xgb = make_xgb(

        scale_pos_weight

    )


    xgb.fit(

        X_train,

        y_train

    )


    xgb_train_prob = xgb.predict_proba(

        X_train

    )[
        :,
        1
    ]


    xgb_threshold = training_youden_threshold(

        y_train,

        xgb_train_prob

    )


    xgb_test_prob = xgb.predict_proba(

        X_test

    )[
        :,
        1
    ]


    XGB_OOF[
        test_idx
    ] = xgb_test_prob.astype(
        np.float32
    )


    xgb_result = calculate_metrics(

        y_test,

        xgb_test_prob,

        xgb_threshold

    )


    print(
        f"XGB AUC={xgb_result['ROC_AUC']:.6f}"
        f" AP={xgb_result['Average_Precision']:.6f}"
        f" BA={xgb_result['Balanced_Accuracy']:.6f}"
        f" F1={xgb_result['F1']:.6f}"
    )


    # --------------------------------------------------------
    # Save fold rows
    # --------------------------------------------------------

    for model_name, result in [

        (
            "Random_Forest",
            rf_result
        ),

        (
            "XGBoost",
            xgb_result
        )

    ]:


        fold_rows.append([

            fold,

            model_name,

            len(
                y_train
            ),

            len(
                y_test
            ),

            len(
                train_blocks
            ),

            len(
                test_blocks
            ),

            train_pos,

            train_neg,

            test_pos,

            test_neg,

            100
            *
            test_pos
            /
            len(
                y_test
            ),

            result[
                "ROC_AUC"
            ],

            result[
                "Average_Precision"
            ],

            result[
                "Threshold"
            ],

            result[
                "Sensitivity"
            ],

            result[
                "Specificity"
            ],

            result[
                "Precision"
            ],

            result[
                "F1"
            ],

            result[
                "Balanced_Accuracy"
            ],

            result[
                "Accuracy"
            ],

            result[
                "Brier"
            ],

            result[
                "TP"
            ],

            result[
                "TN"
            ],

            result[
                "FP"
            ],

            result[
                "FN"
            ]

        ])


# ============================================================
# 16. OOF COMPLETENESS QC
# ============================================================

if np.any(
    ~np.isfinite(
        RF_OOF
    )
):

    raise Exception(
        "RF OOF prediction coverage incomplete."
    )


if np.any(
    ~np.isfinite(
        XGB_OOF
    )
):

    raise Exception(
        "XGB OOF prediction coverage incomplete."
    )


print(
    "\n========== OOF COVERAGE QC =========="
)

print(
    "RF OOF coverage  : PASS"
)

print(
    "XGB OOF coverage : PASS"
)


# ============================================================
# 17. SAVE FOLD PERFORMANCE
# ============================================================

headers = [

    "Fold",

    "Model",

    "Train_N",

    "Test_N",

    "Train_Block_N",

    "Test_Block_N",

    "Train_Flood_N",

    "Train_NonFlood_N",

    "Test_Flood_N",

    "Test_NonFlood_N",

    "Test_Flood_Percent",

    "ROC_AUC",

    "Average_Precision",

    "Train_Derived_Threshold",

    "Sensitivity",

    "Specificity",

    "Precision",

    "F1",

    "Balanced_Accuracy",

    "Accuracy",

    "Brier_Score",

    "TP",

    "TN",

    "FP",

    "FN"

]


with open(
    FOLD_PERFORMANCE_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow(
        headers
    )


    writer.writerows(
        fold_rows
    )


# ============================================================
# 18. MODEL SUMMARY
# ============================================================

summary_rows = []


print(
    "\n========== 5-FOLD SPATIAL CV SUMMARY =========="
)


for model_name in [

    "Random_Forest",

    "XGBoost"

]:


    model_rows = [

        row

        for row in fold_rows

        if row[1]
        ==
        model_name

    ]


    auc_values = np.array(
        [
            row[11]

            for row in model_rows
        ],
        dtype=np.float64
    )


    ap_values = np.array(
        [
            row[12]

            for row in model_rows
        ],
        dtype=np.float64
    )


    ba_values = np.array(
        [
            row[18]

            for row in model_rows
        ],
        dtype=np.float64
    )


    f1_values = np.array(
        [
            row[17]

            for row in model_rows
        ],
        dtype=np.float64
    )


    if model_name == "Random_Forest":

        oof_prob = RF_OOF

    else:

        oof_prob = XGB_OOF


    pooled_auc = roc_auc_score(

        y,

        oof_prob

    )


    pooled_ap = average_precision_score(

        y,

        oof_prob

    )


    summary_rows.append([

        model_name,

        float(
            np.mean(
                auc_values
            )
        ),

        float(
            np.std(
                auc_values,
                ddof=1
            )
        ),

        float(
            np.min(
                auc_values
            )
        ),

        float(
            np.max(
                auc_values
            )
        ),

        float(
            np.mean(
                ap_values
            )
        ),

        float(
            np.std(
                ap_values,
                ddof=1
            )
        ),

        float(
            np.mean(
                ba_values
            )
        ),

        float(
            np.std(
                ba_values,
                ddof=1
            )
        ),

        float(
            np.mean(
                f1_values
            )
        ),

        float(
            np.std(
                f1_values,
                ddof=1
            )
        ),

        float(
            pooled_auc
        ),

        float(
            pooled_ap
        )

    ])


    print(
        f"{model_name:15s}"
        f" mean AUC={np.mean(auc_values):.6f}"
        f" SD={np.std(auc_values, ddof=1):.6f}"
        f" min={np.min(auc_values):.6f}"
        f" max={np.max(auc_values):.6f}"
        f" pooled OOF AUC={pooled_auc:.6f}"
        f" pooled OOF AP={pooled_ap:.6f}"
    )


# ============================================================
# 19. PAIRED RF-vs-XGB FOLD COMPARISON
# ============================================================

rf_auc = np.array(
    [
        row[11]

        for row in fold_rows

        if row[1]
        ==
        "Random_Forest"
    ],
    dtype=np.float64
)


xgb_auc = np.array(
    [
        row[11]

        for row in fold_rows

        if row[1]
        ==
        "XGBoost"
    ],
    dtype=np.float64
)


rf_ap = np.array(
    [
        row[12]

        for row in fold_rows

        if row[1]
        ==
        "Random_Forest"
    ],
    dtype=np.float64
)


xgb_ap = np.array(
    [
        row[12]

        for row in fold_rows

        if row[1]
        ==
        "XGBoost"
    ],
    dtype=np.float64
)


delta_auc = (
    xgb_auc
    -
    rf_auc
)


delta_ap = (
    xgb_ap
    -
    rf_ap
)


with open(
    PAIRED_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Fold",

        "RF_AUC",

        "XGB_AUC",

        "XGB_minus_RF_AUC",

        "RF_AP",

        "XGB_AP",

        "XGB_minus_RF_AP"

    ])


    for i in range(
        N_FOLDS
    ):


        writer.writerow([

            i + 1,

            rf_auc[
                i
            ],

            xgb_auc[
                i
            ],

            delta_auc[
                i
            ],

            rf_ap[
                i
            ],

            xgb_ap[
                i
            ],

            delta_ap[
                i
            ]

        ])


print(
    "\n========== RF vs XGB PAIRED FOLDS =========="
)

print(
    "Mean XGB-RF AUC difference:",
    f"{np.mean(delta_auc):.6f}"
)

print(
    "XGB higher AUC folds      :",
    f"{np.count_nonzero(delta_auc > 0)}/{N_FOLDS}"
)

print(
    "RF higher AUC folds       :",
    f"{np.count_nonzero(delta_auc < 0)}/{N_FOLDS}"
)

print(
    "Mean XGB-RF AP difference :",
    f"{np.mean(delta_ap):.6f}"
)


# ============================================================
# 20. SAVE SUMMARY
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

        "Model",

        "Mean_ROC_AUC",

        "SD_ROC_AUC",

        "Min_ROC_AUC",

        "Max_ROC_AUC",

        "Mean_Average_Precision",

        "SD_Average_Precision",

        "Mean_Balanced_Accuracy",

        "SD_Balanced_Accuracy",

        "Mean_F1",

        "SD_F1",

        "Pooled_OOF_ROC_AUC",

        "Pooled_OOF_Average_Precision"

    ])


    writer.writerows(
        summary_rows
    )


# ============================================================
# 21. SAVE OOF PREDICTIONS
# ============================================================

with open(
    OOF_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Raster_Row",

        "Raster_Column",

        "X",

        "Y",

        "Block_5km",

        "Spatial_Fold",

        "Observed_Label",

        "RF_OOF_Probability",

        "XGB_OOF_Probability"

    ])


    for i in range(
        N
    ):


        writer.writerow([

            int(
                raster_row[
                    i
                ]
            ),

            int(
                raster_col[
                    i
                ]
            ),

            float(
                x_coord[
                    i
                ]
            ),

            float(
                y_coord[
                    i
                ]
            ),

            int(
                blocks[
                    i
                ]
            ),

            int(
                sample_fold[
                    i
                ]
            ),

            int(
                y[
                    i
                ]
            ),

            float(
                RF_OOF[
                    i
                ]
            ),

            float(
                XGB_OOF[
                    i
                ]
            )

        ])


# ============================================================
# 22. FINAL
# ============================================================

print(
    "\n=============================================="
)

print(
    "BUET B06 5-FOLD SPATIAL CV COMPLETE"
)

print(
    "=============================================="
)

print(
    "\nCreated:"
)

print(
    FOLD_ASSIGNMENT_CSV
)

print(
    FOLD_PERFORMANCE_CSV
)

print(
    SUMMARY_CSV
)

print(
    OOF_CSV
)

print(
    PAIRED_CSV
)

print(
    "\nFINAL STATUS: BUET B06 PASS"
)

print(
    "=============================================="
)