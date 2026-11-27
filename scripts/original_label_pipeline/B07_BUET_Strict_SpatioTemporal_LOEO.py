import os
import csv
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

from xgboost import XGBClassifier


# ============================================================
# B07
# STRICT SPATIO-TEMPORAL LEAVE-ONE-EVENT-OUT VALIDATION
#
# 4 held-out events × 5 spatial folds
#
# For every scenario:
#
# 1. One entire flood event is excluded from training labels.
# 2. One geographic 5-km fold is excluded from ALL training.
# 3. Training target = any flood in the OTHER 3 events.
# 4. Test target = observed label in held-out event.
#
# Primary metrics:
# ROC-AUC
# Average Precision
# Brier Score
#
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
    "05_Strict_LOEO"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

SCENARIO_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B07_LOEO_Scenario_Performance.csv"
)

EVENT_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B07_LOEO_Event_Summary.csv"
)

OVERALL_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B07_LOEO_Overall_Summary.csv"
)

PAIRED_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B07_RF_vs_XGB_Scenario_Comparison.csv"
)

PREDICTIONS_NPZ = os.path.join(
    OUTPUT_DIR,
    "BUET_B07_LOEO_OutOfFold_Predictions.npz"
)


# ============================================================
# 02. LOAD B04 DATA
# ============================================================

if not os.path.exists(INPUT_NPZ):
    raise Exception(
        "B04 master dataset not found:\n"
        + INPUT_NPZ
    )

if not os.path.exists(B06_FOLD_FILE):
    raise Exception(
        "B06 spatial-fold file not found:\n"
        + B06_FOLD_FILE
    )

D = np.load(
    INPUT_NPZ,
    allow_pickle=False
)


# ============================================================
# 03. RAW FEATURE MATRIX
# ============================================================

X = np.column_stack([

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


BLOCKS = D[
    "Block_5km"
].astype(
    np.int64
)


EVENT_LABELS = {

    "2019":
        D["Flood_2019"].astype(
            np.int8
        ),

    "2021":
        D["Flood_2021"].astype(
            np.int8
        ),

    "2023":
        D["Flood_2023"].astype(
            np.int8
        ),

    "2026":
        D["Flood_2026"].astype(
            np.int8
        )

}


EVENTS = [
    "2019",
    "2021",
    "2023",
    "2026"
]


N = X.shape[0]


print(
    "\n========== B07 MASTER DATASET =========="
)

print(
    "Rows:",
    f"{N:,}"
)

print(
    "5-km blocks:",
    len(
        np.unique(
            BLOCKS
        )
    )
)


# ============================================================
# 04. FEATURE TYPES
# ============================================================

NUMERIC_INDICES = [

    0,  # elevation
    1,  # slope
    2,  # aspect sin
    3,  # aspect cos
    4,  # flat
    5,  # profile curvature
    6,  # tangential curvature
    7,  # TWI
    8,  # drainage density
    9,  # rainfall
    11, # NDVI
    13, # distance river
    14  # distance road

]


CATEGORICAL_INDICES = [

    10, # LULC
    12  # HSG

]


# ============================================================
# 05. LOAD EXACT B06 SPATIAL FOLDS
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


missing_blocks = [

    int(block)

    for block in np.unique(
        BLOCKS
    )

    if int(block)
    not in BLOCK_TO_FOLD

]


if len(
    missing_blocks
) > 0:

    raise Exception(
        "Some B04 blocks are missing from B06 fold map."
    )


SPATIAL_FOLD = np.array(

    [
        BLOCK_TO_FOLD[
            int(block)
        ]

        for block in BLOCKS
    ],

    dtype=np.uint8

)


print(
    "\n========== B06 FOLD REUSE QC =========="
)

print(
    "Mapped rows:",
    f"{len(SPATIAL_FOLD):,}"
)

print(
    "Fold IDs:",
    sorted(
        np.unique(
            SPATIAL_FOLD
        ).tolist()
    )
)

print(
    "Status: PASS"
)


# ============================================================
# 06. PREPROCESSOR
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
# 07. MODEL FACTORIES
#
# Same fixed architecture used previously.
# No held-out-event tuning.
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
# 08. METRICS
# ============================================================

def calculate_metrics(
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


    brier = brier_score_loss(
        y_true,
        probability
    )


    flood_scores = probability[
        y_true == 1
    ]


    nonflood_scores = probability[
        y_true == 0
    ]


    return {

        "ROC_AUC":
            float(
                auc
            ),

        "Average_Precision":
            float(
                ap
            ),

        "Brier":
            float(
                brier
            ),

        "Flood_Mean_Probability":
            float(
                np.mean(
                    flood_scores
                )
            ),

        "NonFlood_Mean_Probability":
            float(
                np.mean(
                    nonflood_scores
                )
            ),

        "Probability_Separation":
            float(
                np.mean(
                    flood_scores
                )
                -
                np.mean(
                    nonflood_scores
                )
            )

    }


# ============================================================
# 09. CREATE PREDICTION STORAGE
#
# Each held-out event gets full spatial OOF prediction arrays.
# Only event-valid pixels receive predictions.
# ============================================================

RF_EVENT_PRED = {}

XGB_EVENT_PRED = {}


for event in EVENTS:


    RF_EVENT_PRED[
        event
    ] = np.full(
        N,
        np.nan,
        dtype=np.float32
    )


    XGB_EVENT_PRED[
        event
    ] = np.full(
        N,
        np.nan,
        dtype=np.float32
    )


# ============================================================
# 10. MAIN 4 × 5 VALIDATION LOOP
# ============================================================

scenario_rows = []


for held_event in EVENTS:


    print(
        "\n##################################################"
    )

    print(
        "HELD-OUT EVENT:",
        held_event
    )

    print(
        "##################################################"
    )


    training_events = [

        event

        for event in EVENTS

        if event != held_event

    ]


    print(
        "Training events:",
        ", ".join(
            training_events
        )
    )


    # --------------------------------------------------------
    # Training composite from the OTHER 3 events.
    #
    # Negative class is accepted only when all 3 labels
    # are valid and all are zero.
    # --------------------------------------------------------

    TRAIN_MATRIX = np.column_stack(

        [
            EVENT_LABELS[
                event
            ]

            for event in training_events
        ]

    )


    TRAIN_LABEL_VALID = np.all(
        TRAIN_MATRIX >= 0,
        axis=1
    )


    TRAIN_TARGET = np.full(
        N,
        -1,
        dtype=np.int8
    )


    TRAIN_TARGET[
        TRAIN_LABEL_VALID
    ] = np.any(

        TRAIN_MATRIX[
            TRAIN_LABEL_VALID
        ]
        ==
        1,

        axis=1

    ).astype(
        np.int8
    )


    TEST_TARGET = EVENT_LABELS[
        held_event
    ]


    print(
        "Valid 3-event training-label rows:",
        f"{np.count_nonzero(TRAIN_TARGET >= 0):,}"
    )

    print(
        "Valid held-event label rows:",
        f"{np.count_nonzero(TEST_TARGET >= 0):,}"
    )


    # ========================================================
    # FIVE SPATIAL FOLDS
    # ========================================================

    for fold in range(
        1,
        6
    ):


        print(
            "\n----------------------------------------------"
        )

        print(
            f"Event {held_event} | Spatial Fold {fold}/5"
        )

        print(
            "----------------------------------------------"
        )


        # ----------------------------------------------------
        # STRICT TRAIN
        #
        # - labels derived only from other 3 events
        # - current test spatial fold excluded entirely
        # ----------------------------------------------------

        train_mask = (

            (TRAIN_TARGET >= 0)

            &

            (SPATIAL_FOLD != fold)

        )


        # ----------------------------------------------------
        # STRICT TEST
        #
        # - held event only
        # - held geographic fold only
        # ----------------------------------------------------

        test_mask = (

            (TEST_TARGET >= 0)

            &

            (SPATIAL_FOLD == fold)

        )


        train_idx = np.where(
            train_mask
        )[0]


        test_idx = np.where(
            test_mask
        )[0]


        y_train = TRAIN_TARGET[
            train_idx
        ]


        y_test = TEST_TARGET[
            test_idx
        ]


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


        if (
            train_pos == 0
            or
            train_neg == 0
        ):

            raise Exception(
                "Training class missing."
            )


        if (
            test_pos == 0
            or
            test_neg == 0
        ):

            raise Exception(
                "Held-out test class missing."
            )


        # ----------------------------------------------------
        # Check strict block separation
        # ----------------------------------------------------

        train_blocks = np.unique(
            BLOCKS[
                train_idx
            ]
        )


        test_blocks = np.unique(
            BLOCKS[
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
                "STRICT spatial leakage detected."
            )


        print(
            "Train N:",
            f"{len(train_idx):,}",
            "| flood:",
            f"{train_pos:,}",
            "| flood%:",
            f"{100 * train_pos / len(train_idx):.2f}%"
        )


        print(
            "Test N :",
            f"{len(test_idx):,}",
            "| flood:",
            f"{test_pos:,}",
            "| flood%:",
            f"{100 * test_pos / len(test_idx):.2f}%"
        )


        print(
            "Block overlap:",
            len(
                block_overlap
            )
        )


        # ----------------------------------------------------
        # Preprocessing fitted only on training scenario
        # ----------------------------------------------------

        preprocessor = make_preprocessor()


        X_train = preprocessor.fit_transform(

            X[
                train_idx
            ]

        ).astype(
            np.float32
        )


        X_test = preprocessor.transform(

            X[
                test_idx
            ]

        ).astype(
            np.float32
        )


        # ====================================================
        # RANDOM FOREST
        # ====================================================

        print(
            "RF training..."
        )


        rf = make_rf()


        rf.fit(
            X_train,
            y_train
        )


        rf_prob = rf.predict_proba(
            X_test
        )[
            :,
            1
        ]


        RF_EVENT_PRED[
            held_event
        ][
            test_idx
        ] = rf_prob.astype(
            np.float32
        )


        rf_result = calculate_metrics(

            y_test,

            rf_prob

        )


        print(
            f"RF  AUC={rf_result['ROC_AUC']:.6f}"
            f" AP={rf_result['Average_Precision']:.6f}"
            f" Brier={rf_result['Brier']:.6f}"
        )


        # ====================================================
        # XGBOOST
        # ====================================================

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


        xgb_prob = xgb.predict_proba(
            X_test
        )[
            :,
            1
        ]


        XGB_EVENT_PRED[
            held_event
        ][
            test_idx
        ] = xgb_prob.astype(
            np.float32
        )


        xgb_result = calculate_metrics(

            y_test,

            xgb_prob

        )


        print(
            f"XGB AUC={xgb_result['ROC_AUC']:.6f}"
            f" AP={xgb_result['Average_Precision']:.6f}"
            f" Brier={xgb_result['Brier']:.6f}"
        )


        # ====================================================
        # RECORD BOTH MODELS
        # ====================================================

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


            scenario_rows.append([

                held_event,

                fold,

                model_name,

                ",".join(
                    training_events
                ),

                len(
                    train_idx
                ),

                train_pos,

                train_neg,

                100
                *
                train_pos
                /
                len(
                    train_idx
                ),

                len(
                    test_idx
                ),

                test_pos,

                test_neg,

                100
                *
                test_pos
                /
                len(
                    test_idx
                ),

                len(
                    train_blocks
                ),

                len(
                    test_blocks
                ),

                len(
                    block_overlap
                ),

                result[
                    "ROC_AUC"
                ],

                result[
                    "Average_Precision"
                ],

                result[
                    "Brier"
                ],

                result[
                    "Flood_Mean_Probability"
                ],

                result[
                    "NonFlood_Mean_Probability"
                ],

                result[
                    "Probability_Separation"
                ]

            ])


# ============================================================
# 11. SAVE SCENARIO RESULTS
# ============================================================

SCENARIO_HEADERS = [

    "Held_Out_Event",

    "Spatial_Fold",

    "Model",

    "Training_Events",

    "Train_N",

    "Train_Flood_N",

    "Train_NonFlood_N",

    "Train_Flood_Percent",

    "Test_N",

    "Test_Flood_N",

    "Test_NonFlood_N",

    "Test_Flood_Percent",

    "Train_Block_N",

    "Test_Block_N",

    "Block_Overlap_N",

    "ROC_AUC",

    "Average_Precision",

    "Brier_Score",

    "Flood_Mean_Probability",

    "NonFlood_Mean_Probability",

    "Probability_Separation"

]


with open(
    SCENARIO_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow(
        SCENARIO_HEADERS
    )


    writer.writerows(
        scenario_rows
    )


# ============================================================
# 12. EVENT-LEVEL POOLED OOF PERFORMANCE
# ============================================================

event_summary_rows = []


print(
    "\n=================================================="
)

print(
    "EVENT-LEVEL STRICT LOEO SUMMARY"
)

print(
    "=================================================="
)


for held_event in EVENTS:


    y_event = EVENT_LABELS[
        held_event
    ]


    valid = (
        y_event >= 0
    )


    for model_name, predictions in [

        (
            "Random_Forest",
            RF_EVENT_PRED[
                held_event
            ]
        ),

        (
            "XGBoost",
            XGB_EVENT_PRED[
                held_event
            ]
        )

    ]:


        if np.any(
            ~np.isfinite(
                predictions[
                    valid
                ]
            )
        ):

            raise Exception(
                held_event
                +
                " "
                +
                model_name
                +
                " incomplete OOF coverage."
            )


        pooled_result = calculate_metrics(

            y_event[
                valid
            ],

            predictions[
                valid
            ]

        )


        fold_rows = [

            row

            for row in scenario_rows

            if (
                row[0]
                ==
                held_event

                and

                row[2]
                ==
                model_name
            )

        ]


        auc_values = np.array(

            [
                row[15]
                for row in fold_rows
            ],

            dtype=np.float64

        )


        ap_values = np.array(

            [
                row[16]
                for row in fold_rows
            ],

            dtype=np.float64

        )


        event_summary_rows.append([

            held_event,

            model_name,

            int(
                np.count_nonzero(
                    valid
                )
            ),

            int(
                np.count_nonzero(
                    y_event[
                        valid
                    ]
                    ==
                    1
                )
            ),

            100
            *
            np.mean(
                y_event[
                    valid
                ]
                ==
                1
            ),

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

            pooled_result[
                "ROC_AUC"
            ],

            pooled_result[
                "Average_Precision"
            ],

            pooled_result[
                "Brier"
            ],

            pooled_result[
                "Probability_Separation"
            ]

        ])


        print(
            f"{held_event} {model_name:15s}"
            f" mean-fold AUC={np.mean(auc_values):.6f}"
            f" SD={np.std(auc_values, ddof=1):.6f}"
            f" pooled AUC={pooled_result['ROC_AUC']:.6f}"
            f" pooled AP={pooled_result['Average_Precision']:.6f}"
        )


# ============================================================
# 13. SAVE EVENT SUMMARY
# ============================================================

EVENT_HEADERS = [

    "Held_Out_Event",

    "Model",

    "Event_Valid_N",

    "Event_Flood_N",

    "Event_Flood_Percent",

    "Mean_Fold_ROC_AUC",

    "SD_Fold_ROC_AUC",

    "Min_Fold_ROC_AUC",

    "Max_Fold_ROC_AUC",

    "Mean_Fold_Average_Precision",

    "SD_Fold_Average_Precision",

    "Pooled_Event_ROC_AUC",

    "Pooled_Event_Average_Precision",

    "Pooled_Event_Brier",

    "Pooled_Probability_Separation"

]


with open(
    EVENT_SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow(
        EVENT_HEADERS
    )


    writer.writerows(
        event_summary_rows
    )


# ============================================================
# 14. OVERALL MACRO SUMMARY
#
# Equal weighting across the 4 held-out events.
#
# This prevents the largest event from dominating the
# overall headline result.
# ============================================================

overall_rows = []


print(
    "\n=================================================="
)

print(
    "OVERALL STRICT LOEO SUMMARY"
)

print(
    "=================================================="
)


for model_name in [

    "Random_Forest",

    "XGBoost"

]:


    model_event_rows = [

        row

        for row in event_summary_rows

        if row[1]
        ==
        model_name

    ]


    pooled_event_auc = np.array(

        [
            row[11]

            for row in model_event_rows
        ],

        dtype=np.float64

    )


    pooled_event_ap = np.array(

        [
            row[12]

            for row in model_event_rows
        ],

        dtype=np.float64

    )


    all_scenario_auc = np.array(

        [
            row[15]

            for row in scenario_rows

            if row[2]
            ==
            model_name
        ],

        dtype=np.float64

    )


    all_scenario_ap = np.array(

        [
            row[16]

            for row in scenario_rows

            if row[2]
            ==
            model_name
        ],

        dtype=np.float64

    )


    overall_rows.append([

        model_name,

        float(
            np.mean(
                pooled_event_auc
            )
        ),

        float(
            np.std(
                pooled_event_auc,
                ddof=1
            )
        ),

        float(
            np.min(
                pooled_event_auc
            )
        ),

        float(
            np.max(
                pooled_event_auc
            )
        ),

        float(
            np.mean(
                pooled_event_ap
            )
        ),

        float(
            np.std(
                pooled_event_ap,
                ddof=1
            )
        ),

        float(
            np.mean(
                all_scenario_auc
            )
        ),

        float(
            np.std(
                all_scenario_auc,
                ddof=1
            )
        ),

        float(
            np.min(
                all_scenario_auc
            )
        ),

        float(
            np.max(
                all_scenario_auc
            )
        ),

        float(
            np.mean(
                all_scenario_ap
            )
        ),

        float(
            np.std(
                all_scenario_ap,
                ddof=1
            )
        )

    ])


    print(
        f"{model_name:15s}"
        f" macro event AUC={np.mean(pooled_event_auc):.6f}"
        f" SD={np.std(pooled_event_auc, ddof=1):.6f}"
        f" scenario mean AUC={np.mean(all_scenario_auc):.6f}"
        f" scenario SD={np.std(all_scenario_auc, ddof=1):.6f}"
    )


# ============================================================
# 15. SAVE OVERALL SUMMARY
# ============================================================

OVERALL_HEADERS = [

    "Model",

    "Macro_Mean_Event_Pooled_AUC",

    "SD_Event_Pooled_AUC",

    "Minimum_Event_Pooled_AUC",

    "Maximum_Event_Pooled_AUC",

    "Macro_Mean_Event_Pooled_AP",

    "SD_Event_Pooled_AP",

    "Mean_20Scenario_AUC",

    "SD_20Scenario_AUC",

    "Minimum_20Scenario_AUC",

    "Maximum_20Scenario_AUC",

    "Mean_20Scenario_AP",

    "SD_20Scenario_AP"

]


with open(
    OVERALL_SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow(
        OVERALL_HEADERS
    )


    writer.writerows(
        overall_rows
    )


# ============================================================
# 16. RF-vs-XGB PAIRED SCENARIO COMPARISON
# ============================================================

paired_rows = []


for held_event in EVENTS:


    for fold in range(
        1,
        6
    ):


        rf_row = [

            row

            for row in scenario_rows

            if (
                row[0] == held_event
                and
                row[1] == fold
                and
                row[2] == "Random_Forest"
            )

        ][0]


        xgb_row = [

            row

            for row in scenario_rows

            if (
                row[0] == held_event
                and
                row[1] == fold
                and
                row[2] == "XGBoost"
            )

        ][0]


        paired_rows.append([

            held_event,

            fold,

            rf_row[15],

            xgb_row[15],

            xgb_row[15]
            -
            rf_row[15],

            rf_row[16],

            xgb_row[16],

            xgb_row[16]
            -
            rf_row[16]

        ])


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

        "Held_Out_Event",

        "Spatial_Fold",

        "RF_AUC",

        "XGB_AUC",

        "XGB_minus_RF_AUC",

        "RF_AP",

        "XGB_AP",

        "XGB_minus_RF_AP"

    ])


    writer.writerows(
        paired_rows
    )


delta_auc = np.array(

    [
        row[4]

        for row in paired_rows
    ],

    dtype=np.float64

)


delta_ap = np.array(

    [
        row[7]

        for row in paired_rows
    ],

    dtype=np.float64

)


print(
    "\n========== RF vs XGB: 20 PAIRED SCENARIOS =========="
)

print(
    "Mean XGB-RF AUC difference:",
    f"{np.mean(delta_auc):.6f}"
)

print(
    "XGB AUC wins:",
    f"{np.count_nonzero(delta_auc > 0)}/20"
)

print(
    "RF AUC wins :",
    f"{np.count_nonzero(delta_auc < 0)}/20"
)

print(
    "Mean XGB-RF AP difference :",
    f"{np.mean(delta_ap):.6f}"
)


# ============================================================
# 17. SAVE COMPRESSED PREDICTIONS
# ============================================================

np.savez_compressed(

    PREDICTIONS_NPZ,

    Spatial_Fold=
        SPATIAL_FOLD,

    Block_5km=
        BLOCKS,

    Flood_2019=
        EVENT_LABELS[
            "2019"
        ],

    Flood_2021=
        EVENT_LABELS[
            "2021"
        ],

    Flood_2023=
        EVENT_LABELS[
            "2023"
        ],

    Flood_2026=
        EVENT_LABELS[
            "2026"
        ],

    RF_2019=
        RF_EVENT_PRED[
            "2019"
        ],

    XGB_2019=
        XGB_EVENT_PRED[
            "2019"
        ],

    RF_2021=
        RF_EVENT_PRED[
            "2021"
        ],

    XGB_2021=
        XGB_EVENT_PRED[
            "2021"
        ],

    RF_2023=
        RF_EVENT_PRED[
            "2023"
        ],

    XGB_2023=
        XGB_EVENT_PRED[
            "2023"
        ],

    RF_2026=
        RF_EVENT_PRED[
            "2026"
        ],

    XGB_2026=
        XGB_EVENT_PRED[
            "2026"
        ]

)


# ============================================================
# 18. FINAL
# ============================================================

print(
    "\n=============================================="
)

print(
    "BUET B07 STRICT SPATIO-TEMPORAL LOEO COMPLETE"
)

print(
    "=============================================="
)

print(
    "\nCreated:"
)

print(
    SCENARIO_CSV
)

print(
    EVENT_SUMMARY_CSV
)

print(
    OVERALL_SUMMARY_CSV
)

print(
    PAIRED_CSV
)

print(
    PREDICTIONS_NPZ
)

print(
    "\nFINAL STATUS: BUET B07 PASS"
)

print(
    "=============================================="
)