import os
import csv
import sys
import numpy as np


# ============================================================
# B05
# BUET RF + XGBOOST BASELINE
#
# PURPOSE
# ------------------------------------------------------------
# Composite multi-event target:
#
# 1 = flooded in >= 1 of four events
# 0 = valid in all four events and never flooded
#
# Evaluation:
# Spatial 5-km block holdout.
#
# IMPORTANT:
# This is a baseline/sanity-check stage.
# Formal spatial CV = B06.
# Event generalization = B07.
# ============================================================


# ============================================================
# 01. PACKAGE CHECK
# ============================================================

try:

    import sklearn

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import GroupShuffleSplit

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
        brier_score_loss,
        log_loss
    )

    import joblib

except ImportError as e:

    print(
        "\nERROR: scikit-learn/joblib is not available."
    )

    print(
        "Missing package:",
        e
    )

    raise


try:

    import xgboost

    from xgboost import XGBClassifier

except ImportError as e:

    print(
        "\nERROR: XGBoost is not installed in this Python environment."
    )

    print(
        "Do not substitute another model."
    )

    print(
        "Send me this error and I will give you the exact "
        "installation command for your QGIS Python."
    )

    raise


print(
    "\n========== SOFTWARE =========="
)

print(
    "Python       :",
    sys.version.split()[0]
)

print(
    "NumPy        :",
    np.__version__
)

print(
    "scikit-learn :",
    sklearn.__version__
)

print(
    "XGBoost      :",
    xgboost.__version__
)


# ============================================================
# 02. PATHS
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
    "03_Baseline_Models"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


PERFORMANCE_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B05_Baseline_Performance.csv"
)


PREDICTION_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B05_SpatialHoldout_Predictions.csv"
)


BLOCK_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B05_5km_Block_Holdout_Assignment.csv"
)


FEATURE_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B05_Encoded_Feature_Schema.csv"
)


RF_MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "BUET_B05_RF_Baseline.joblib"
)


XGB_MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "BUET_B05_XGB_Baseline.joblib"
)


PREPROCESSOR_FILE = os.path.join(
    OUTPUT_DIR,
    "BUET_B05_Feature_Preprocessor.joblib"
)


# ============================================================
# 03. INPUT CHECK
# ============================================================

if not os.path.exists(
    INPUT_NPZ
):

    raise Exception(
        "B04 dataset not found:\n"
        +
        INPUT_NPZ
    )


D = np.load(
    INPUT_NPZ,
    allow_pickle=False
)


# ============================================================
# 04. RAW FEATURE DEFINITIONS
#
# 13 conceptual factors become 15 raw ML columns because:
#
# Aspect -> sin + cos + flat
#
# LULC/HSG remain categorical until one-hot encoding.
# ============================================================

FEATURE_NAMES = [

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

    "LULC",

    "NDVI",

    "HSG",

    "Distance_River",

    "Distance_Road"

]


NUMERIC_INDICES = [

    0,      # Elevation
    1,      # Slope
    2,      # Aspect_sin
    3,      # Aspect_cos
    4,      # Aspect_flat
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
# 05. BUILD RAW FEATURE MATRIX
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


GROUP_ALL = D[
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


ROW_ALL = D[
    "Raster_Row"
].astype(
    np.int32
)


COL_ALL = D[
    "Raster_Column"
].astype(
    np.int32
)


# ============================================================
# 06. USE ONLY CLEAN COMPOSITE LABELS
#
# -1 = at least one event unavailable
#  0 = stable non-flood
#  1 = flooded >= 1 event
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


groups = GROUP_ALL[
    VALID
]


x_coord = X_COORD_ALL[
    VALID
]


y_coord = Y_COORD_ALL[
    VALID
]


rows = ROW_ALL[
    VALID
]


cols = COL_ALL[
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


PREVALENCE = (
    POS / N
)


print(
    "\n========== B05 DATASET =========="
)

print(
    "Rows            :",
    f"{N:,}"
)

print(
    "Positive        :",
    f"{POS:,}"
)

print(
    "Negative        :",
    f"{NEG:,}"
)

print(
    "Positive %      :",
    f"{PREVALENCE * 100:.3f}%"
)

print(
    "Unique 5km blocks:",
    len(
        np.unique(
            groups
        )
    )
)


if N != 169900:

    print(
        "WARNING: expected B04 clean composite rows = 169,900"
    )


# ============================================================
# 07. SPATIAL BLOCK HOLDOUT
#
# 80% blocks -> training
# 20% blocks -> testing
#
# NO BLOCK appears in both sets.
# ============================================================

splitter = GroupShuffleSplit(

    n_splits=1,

    test_size=0.20,

    random_state=42

)


train_idx, test_idx = next(

    splitter.split(
        X,
        y,
        groups=groups
    )

)


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


train_groups = groups[
    train_idx
]


test_groups = groups[
    test_idx
]


overlap = np.intersect1d(

    np.unique(
        train_groups
    ),

    np.unique(
        test_groups
    )

)


if len(
    overlap
) != 0:

    raise Exception(
        "SPATIAL LEAKAGE: 5 km block overlap detected."
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
    "\n========== 5 km SPATIAL HOLDOUT =========="
)

print(
    "Train rows      :",
    f"{len(y_train):,}"
)

print(
    "Test rows       :",
    f"{len(y_test):,}"
)

print(
    "Train blocks    :",
    len(
        np.unique(
            train_groups
        )
    )
)

print(
    "Test blocks     :",
    len(
        np.unique(
            test_groups
        )
    )
)

print(
    "Block overlap   :",
    len(
        overlap
    )
)

print(
    "Train flood %   :",
    f"{100 * train_pos / len(y_train):.3f}%"
)

print(
    "Test flood %    :",
    f"{100 * test_pos / len(y_test):.3f}%"
)


# ============================================================
# 08. SAVE BLOCK ASSIGNMENT
# ============================================================

all_blocks = np.unique(
    groups
)


train_block_set = set(
    np.unique(
        train_groups
    ).tolist()
)


test_block_set = set(
    np.unique(
        test_groups
    ).tolist()
)


with open(
    BLOCK_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Block_5km",

        "Split",

        "N",

        "Flood_N",

        "NonFlood_N",

        "Flood_Percent"

    ])


    for block in all_blocks:


        block_mask = (
            groups == block
        )


        block_y = y[
            block_mask
        ]


        bn = len(
            block_y
        )


        bp = int(
            np.count_nonzero(
                block_y == 1
            )
        )


        bneg = int(
            np.count_nonzero(
                block_y == 0
            )
        )


        if block in train_block_set:

            split_name = "TRAIN"

        elif block in test_block_set:

            split_name = "TEST"

        else:

            raise Exception(
                "Block assignment failure."
            )


        writer.writerow([

            int(
                block
            ),

            split_name,

            bn,

            bp,

            bneg,

            100 * bp / bn

        ])


# ============================================================
# 09. ONE-HOT ENCODING
#
# Numeric variables: passed through unchanged.
#
# Categorical:
# LULC
# HSG
#
# Encoder fitted ONLY on training data.
# ============================================================

try:

    encoder = OneHotEncoder(

        handle_unknown="ignore",

        sparse_output=False,

        dtype=np.float32

    )

except TypeError:

    # Compatibility with older scikit-learn.
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


print(
    "Raw feature columns :",
    X_train_raw.shape[1]
)

print(
    "Encoded columns     :",
    X_train.shape[1]
)


if not np.all(
    np.isfinite(
        X_train
    )
):

    raise Exception(
        "Non-finite values in encoded training matrix."
    )


if not np.all(
    np.isfinite(
        X_test
    )
):

    raise Exception(
        "Non-finite values in encoded test matrix."
    )


# ============================================================
# 10. FEATURE NAMES
# ============================================================

try:

    encoded_names = (
        preprocessor.get_feature_names_out()
    ).tolist()

except Exception:

    encoded_names = [

        "Encoded_Feature_"
        +
        str(
            i + 1
        )

        for i in range(
            X_train.shape[1]
        )

    ]


with open(
    FEATURE_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Encoded_Feature_Index",

        "Encoded_Feature_Name"

    ])


    for i, name in enumerate(
        encoded_names
    ):


        writer.writerow([

            i,

            name

        ])


# ============================================================
# 11. CLASS WEIGHT FOR XGBOOST
# ============================================================

SCALE_POS_WEIGHT = (

    train_neg
    /
    train_pos

)


print(
    "XGB scale_pos_weight:",
    f"{SCALE_POS_WEIGHT:.6f}"
)


# ============================================================
# 12. MODELS
#
# Parameters intentionally pre-specified and moderate.
#
# No validation-event tuning is being performed here.
# ============================================================

RF = RandomForestClassifier(

    n_estimators=500,

    max_features="sqrt",

    min_samples_leaf=2,

    class_weight="balanced_subsample",

    bootstrap=True,

    n_jobs=-1,

    random_state=42

)


XGB = XGBClassifier(

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

    scale_pos_weight=SCALE_POS_WEIGHT,

    tree_method="hist",

    n_jobs=-1,

    random_state=42

)


# ============================================================
# 13. METRIC FUNCTION
# ============================================================

def evaluate_model(
    name,
    y_true,
    probability
):


    auc = roc_auc_score(
        y_true,
        probability
    )


    ap = average_precision_score(
        y_true,
        probability
    )


    fpr, tpr, thresholds = roc_curve(
        y_true,
        probability
    )


    youden = (
        tpr
        -
        fpr
    )


    best_index = int(
        np.argmax(
            youden
        )
    )


    threshold = float(
        thresholds[
            best_index
        ]
    )


    prediction = (
        probability
        >=
        threshold
    ).astype(
        np.uint8
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


    accuracy = accuracy_score(
        y_true,
        prediction
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            prediction
        )
    )


    brier = brier_score_loss(
        y_true,
        probability
    )


    ll = log_loss(
        y_true,
        probability,
        labels=[
            0,
            1
        ]
    )


    return {

        "Model":
            name,

        "ROC_AUC":
            auc,

        "Average_Precision":
            ap,

        "Youden_Threshold":
            threshold,

        "Youden_J":
            float(
                youden[
                    best_index
                ]
            ),

        "Sensitivity":
            sensitivity,

        "Specificity":
            specificity,

        "Precision":
            precision,

        "F1":
            f1,

        "Accuracy":
            accuracy,

        "Balanced_Accuracy":
            balanced_accuracy,

        "Brier_Score":
            brier,

        "Log_Loss":
            ll,

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
# 14. RANDOM FOREST
# ============================================================

print(
    "\n========== RANDOM FOREST =========="
)

print(
    "Training..."
)


RF.fit(
    X_train,
    y_train
)


RF_PROB = RF.predict_proba(
    X_test
)[
    :,
    1
]


RF_RESULT = evaluate_model(

    "Random_Forest",

    y_test,

    RF_PROB

)


print(
    f"ROC-AUC          : {RF_RESULT['ROC_AUC']:.6f}"
)

print(
    f"Average Precision: {RF_RESULT['Average_Precision']:.6f}"
)

print(
    f"Balanced Accuracy: {RF_RESULT['Balanced_Accuracy']:.6f}"
)

print(
    f"F1               : {RF_RESULT['F1']:.6f}"
)


# ============================================================
# 15. XGBOOST
# ============================================================

print(
    "\n========== XGBOOST =========="
)

print(
    "Training..."
)


XGB.fit(
    X_train,
    y_train
)


XGB_PROB = XGB.predict_proba(
    X_test
)[
    :,
    1
]


XGB_RESULT = evaluate_model(

    "XGBoost",

    y_test,

    XGB_PROB

)


print(
    f"ROC-AUC          : {XGB_RESULT['ROC_AUC']:.6f}"
)

print(
    f"Average Precision: {XGB_RESULT['Average_Precision']:.6f}"
)

print(
    f"Balanced Accuracy: {XGB_RESULT['Balanced_Accuracy']:.6f}"
)

print(
    f"F1               : {XGB_RESULT['F1']:.6f}"
)


# ============================================================
# 16. SAVE PERFORMANCE
# ============================================================

RESULTS = [

    RF_RESULT,

    XGB_RESULT

]


metric_headers = [

    "Model",

    "ROC_AUC",

    "Average_Precision",

    "Youden_Threshold",

    "Youden_J",

    "Sensitivity",

    "Specificity",

    "Precision",

    "F1",

    "Accuracy",

    "Balanced_Accuracy",

    "Brier_Score",

    "Log_Loss",

    "TP",

    "TN",

    "FP",

    "FN"

]


with open(
    PERFORMANCE_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow(
        metric_headers
    )


    for result in RESULTS:


        writer.writerow([

            result[
                header
            ]

            for header in metric_headers

        ])


# ============================================================
# 17. SAVE TEST-SET PREDICTIONS
# ============================================================

with open(
    PREDICTION_CSV,
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

        "Observed_Composite_Label",

        "RF_Probability",

        "XGB_Probability"

    ])


    for local_i, global_i in enumerate(
        test_idx
    ):


        writer.writerow([

            int(
                rows[
                    global_i
                ]
            ),

            int(
                cols[
                    global_i
                ]
            ),

            float(
                x_coord[
                    global_i
                ]
            ),

            float(
                y_coord[
                    global_i
                ]
            ),

            int(
                groups[
                    global_i
                ]
            ),

            int(
                y[
                    global_i
                ]
            ),

            float(
                RF_PROB[
                    local_i
                ]
            ),

            float(
                XGB_PROB[
                    local_i
                ]
            )

        ])


# ============================================================
# 18. SAVE PREPROCESSOR + MODELS
# ============================================================

joblib.dump(
    preprocessor,
    PREPROCESSOR_FILE
)


joblib.dump(
    RF,
    RF_MODEL_FILE,
    compress=3
)


joblib.dump(
    XGB,
    XGB_MODEL_FILE,
    compress=3
)


# ============================================================
# 19. BASELINE RANKING
# ============================================================

ranking = sorted(

    RESULTS,

    key=lambda x:
        x[
            "ROC_AUC"
        ],

    reverse=True

)


print(
    "\n========== BASELINE SPATIAL-HOLDOUT RANKING =========="
)


for rank, result in enumerate(
    ranking,
    start=1
):


    print(
        f"{rank}. "
        f"{result['Model']:15s}"
        f" ROC-AUC={result['ROC_AUC']:.6f}"
        f" AP={result['Average_Precision']:.6f}"
    )


print(
    "\nChance/prevalence AP baseline:",
    f"{np.mean(y_test):.6f}"
)


# ============================================================
# 20. FINAL STATUS
# ============================================================

print(
    "\n=============================================="
)

print(
    "BUET B05 RF + XGBOOST BASELINE COMPLETE"
)

print(
    "=============================================="
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nCreated:"
)

print(
    "1. BUET_B05_Baseline_Performance.csv"
)

print(
    "2. BUET_B05_SpatialHoldout_Predictions.csv"
)

print(
    "3. BUET_B05_5km_Block_Holdout_Assignment.csv"
)

print(
    "4. BUET_B05_Encoded_Feature_Schema.csv"
)

print(
    "5. BUET_B05_RF_Baseline.joblib"
)

print(
    "6. BUET_B05_XGB_Baseline.joblib"
)

print(
    "7. BUET_B05_Feature_Preprocessor.joblib"
)

print(
    "\nFINAL STATUS: BUET B05 PASS"
)

print(
    "=============================================="
)