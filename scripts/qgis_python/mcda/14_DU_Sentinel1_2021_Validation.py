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

AHP_FILE = os.path.join(
    BASE,
    "12_DU_MCDA",
    "02_AHP",
    "DU10_AHP_Susceptibility_Index_01.tif"
)

FAHP_FILE = os.path.join(
    BASE,
    "12_DU_MCDA",
    "03_FAHP",
    "DU11_FAHP_Susceptibility_Index_01.tif"
)

ANP_FILE = os.path.join(
    BASE,
    "12_DU_MCDA",
    "04_ANP",
    "DU12_ANP_Susceptibility_Index_01.tif"
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
    "06_Sentinel1_Validation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 02. OUTPUT FILES
# ============================================================

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "DU14_2021_Validation_Summary.csv"
)

SCORE_STATS_CSV = os.path.join(
    OUTPUT_DIR,
    "DU14_Flood_NonFlood_Score_Statistics.csv"
)

ROC_CSV = os.path.join(
    OUTPUT_DIR,
    "DU14_ROC_Curves_501Points.csv"
)

OFFSET_CSV = os.path.join(
    OUTPUT_DIR,
    "DU14_300m_Systematic_Offset_AUC.csv"
)

OFFSET_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "DU14_300m_AUC_Robustness_Summary.csv"
)

SAMPLE_CSV = os.path.join(
    OUTPUT_DIR,
    "DU14_Representative_300m_Validation_Sample.csv"
)


# ============================================================
# 03. READ RASTER
# ============================================================

def read_raster(path):

    if not os.path.exists(path):
        raise Exception(
            "Missing raster:\n" + path
        )

    ds = gdal.Open(path)

    if ds is None:
        raise Exception(
            "Cannot open raster:\n" + path
        )

    band = ds.GetRasterBand(1)

    arr = band.ReadAsArray()

    nodata = band.GetNoDataValue()

    valid = np.isfinite(
        arr.astype(np.float64)
    )

    if nodata is not None:

        if np.isnan(nodata):

            valid &= ~np.isnan(arr)

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
# 04. LOAD INPUT RASTERS
# ============================================================

ahp_ds, ahp, ahp_valid = read_raster(
    AHP_FILE
)

fahp_ds, fahp, fahp_valid = read_raster(
    FAHP_FILE
)

anp_ds, anp, anp_valid = read_raster(
    ANP_FILE
)

s1_ds, s1, s1_valid_raw = read_raster(
    S1_FILE
)


# ============================================================
# 05. SENTINEL-1 LABEL DOMAIN
#
# 0 = observed non-flood candidate
# 1 = detected flood
# 255 = NoData / invalid label domain
# ============================================================

s1_valid = (
    s1_valid_raw
    &
    (
        (s1 == 0)
        |
        (s1 == 1)
    )
)


# ============================================================
# 06. GRID QC
# ============================================================

ref_x = ahp_ds.RasterXSize
ref_y = ahp_ds.RasterYSize

ref_gt = ahp_ds.GetGeoTransform()
ref_projection = ahp_ds.GetProjection()


DATASETS = {

    "FAHP":
        fahp_ds,

    "ANP":
        anp_ds,

    "Sentinel1":
        s1_ds
}


for name, ds in DATASETS.items():

    if (
        ds.RasterXSize != ref_x
        or
        ds.RasterYSize != ref_y
    ):

        raise Exception(
            name +
            " dimension mismatch."
        )


    if not np.allclose(
        ds.GetGeoTransform(),
        ref_gt,
        atol=0.01
    ):

        raise Exception(
            name +
            " geotransform mismatch."
        )


print(
    "\n========== DU14 GRID QC =========="
)

print(
    "AHP / FAHP / ANP / S1 grid : PASS"
)


# ============================================================
# 07. COMMON VALIDATION DOMAIN
#
# A pixel is used only if valid in:
#
# AHP
# FAHP
# ANP
# Sentinel-1 flood inventory
# ============================================================

validation_mask = (
    ahp_valid
    &
    fahp_valid
    &
    anp_valid
    &
    s1_valid
)


n_validation = int(
    np.count_nonzero(
        validation_mask
    )
)


labels = s1[
    validation_mask
].astype(
    np.uint8
)


n_flood = int(
    np.count_nonzero(
        labels == 1
    )
)


n_nonflood = int(
    np.count_nonzero(
        labels == 0
    )
)


print(
    "\n========== VALIDATION DOMAIN =========="
)

print(
    "Valid pixels     :",
    f"{n_validation:,}"
)

print(
    "Flood pixels     :",
    f"{n_flood:,}"
)

print(
    "Non-flood pixels :",
    f"{n_nonflood:,}"
)

print(
    "Flood fraction   :",
    f"{100 * n_flood / n_validation:.3f}%"
)


if n_flood == 0:

    raise Exception(
        "No flood pixels found."
    )


if n_nonflood == 0:

    raise Exception(
        "No non-flood pixels found."
    )


# ============================================================
# 08. EXACT ROC ANALYSIS
#
# Higher susceptibility score should indicate higher
# likelihood of observed flood.
#
# AUC is calculated manually using trapezoidal integration.
#
# This avoids np.trapz / np.trapezoid compatibility issues.
# ============================================================

def roc_analysis(
    y_true,
    scores
):

    y_true = y_true.astype(
        np.uint8
    )

    scores = scores.astype(
        np.float64
    )


    # --------------------------------------------------------
    # Sort from highest score to lowest score
    # --------------------------------------------------------

    order = np.argsort(
        -scores,
        kind="mergesort"
    )


    y_sorted = y_true[
        order
    ]


    score_sorted = scores[
        order
    ]


    # --------------------------------------------------------
    # Find last location of every unique susceptibility score
    # --------------------------------------------------------

    distinct = np.where(
        np.diff(
            score_sorted
        ) != 0
    )[0]


    threshold_indices = np.r_[
        distinct,
        y_true.size - 1
    ]


    cumulative_tp = np.cumsum(
        y_sorted,
        dtype=np.int64
    )


    tp = cumulative_tp[
        threshold_indices
    ].astype(
        np.float64
    )


    total_predicted_positive = (
        threshold_indices + 1
    ).astype(
        np.float64
    )


    fp = (
        total_predicted_positive
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


    # --------------------------------------------------------
    # ROC coordinates
    # --------------------------------------------------------

    tpr = np.r_[
        0.0,
        tp / P
    ]


    fpr = np.r_[
        0.0,
        fp / N
    ]


    thresholds = np.r_[
        np.inf,
        score_sorted[
            threshold_indices
        ]
    ]


    # --------------------------------------------------------
    # MANUAL TRAPEZOIDAL AUC
    #
    # Integral:
    #
    # sum [
    #     (x_i - x_i-1)
    #     *
    #     (y_i + y_i-1)/2
    # ]
    #
    # x = FPR
    # y = TPR
    # --------------------------------------------------------

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


    auc = float(
        np.sum(
            delta_fpr
            *
            mean_tpr
        )
    )


    # --------------------------------------------------------
    # Youden J
    #
    # J = Sensitivity + Specificity - 1
    #   = TPR - FPR
    # --------------------------------------------------------

    youden = (
        tpr
        -
        fpr
    )


    best_idx = int(
        np.argmax(
            youden
        )
    )


    best_threshold = float(
        thresholds[
            best_idx
        ]
    )


    best_youden = float(
        youden[
            best_idx
        ]
    )


    return {

        "auc":
            auc,

        "fpr":
            fpr,

        "tpr":
            tpr,

        "thresholds":
            thresholds,

        "best_threshold":
            best_threshold,

        "best_youden":
            best_youden

    }


# ============================================================
# 09. FAST AUC FOR SPATIALLY THINNED SAMPLES
#
# Uses ROC function above for consistency.
# ============================================================

def calculate_auc(
    y_true,
    scores
):

    return roc_analysis(
        y_true,
        scores
    )[
        "auc"
    ]


# ============================================================
# 10. THRESHOLD-BASED PERFORMANCE METRICS
# ============================================================

def classification_metrics(
    y_true,
    scores,
    threshold
):

    predicted = (
        scores >= threshold
    ).astype(
        np.uint8
    )


    TP = int(
        np.count_nonzero(
            (y_true == 1)
            &
            (predicted == 1)
        )
    )


    TN = int(
        np.count_nonzero(
            (y_true == 0)
            &
            (predicted == 0)
        )
    )


    FP = int(
        np.count_nonzero(
            (y_true == 0)
            &
            (predicted == 1)
        )
    )


    FN = int(
        np.count_nonzero(
            (y_true == 1)
            &
            (predicted == 0)
        )
    )


    sensitivity = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else np.nan
    )


    specificity = (
        TN / (TN + FP)
        if (TN + FP) > 0
        else np.nan
    )


    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0
        else np.nan
    )


    accuracy = (
        (TP + TN)
        /
        (
            TP
            +
            TN
            +
            FP
            +
            FN
        )
    )


    if (
        np.isfinite(precision)
        and
        np.isfinite(sensitivity)
        and
        (
            precision
            +
            sensitivity
        ) > 0
    ):

        f1 = (
            2
            *
            precision
            *
            sensitivity
            /
            (
                precision
                +
                sensitivity
            )
        )

    else:

        f1 = np.nan


    balanced_accuracy = (
        sensitivity
        +
        specificity
    ) / 2


    return {

        "TP":
            TP,

        "TN":
            TN,

        "FP":
            FP,

        "FN":
            FN,

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
            balanced_accuracy

    }


# ============================================================
# 11. MCDA METHODS
# ============================================================

METHODS = {

    "AHP":
        ahp.astype(
            np.float32
        ),

    "FAHP":
        fahp.astype(
            np.float32
        ),

    "ANP":
        anp.astype(
            np.float32
        )
}


summary_rows = []

score_rows = []

roc_rows = []


# ============================================================
# 12. FULL-PIXEL VALIDATION
# ============================================================

print(
    "\n========== FULL-PIXEL VALIDATION =========="
)


for method, raster in METHODS.items():


    scores = raster[
        validation_mask
    ].astype(
        np.float32
    )


    roc = roc_analysis(
        labels,
        scores
    )


    threshold = roc[
        "best_threshold"
    ]


    metrics = classification_metrics(
        labels,
        scores,
        threshold
    )


    flood_scores = scores[
        labels == 1
    ]


    nonflood_scores = scores[
        labels == 0
    ]


    flood_mean = float(
        np.mean(
            flood_scores
        )
    )


    nonflood_mean = float(
        np.mean(
            nonflood_scores
        )
    )


    flood_median = float(
        np.median(
            flood_scores
        )
    )


    nonflood_median = float(
        np.median(
            nonflood_scores
        )
    )


    # --------------------------------------------------------
    # Score statistics
    # --------------------------------------------------------

    score_rows.append([

        method,

        "Flood",

        len(
            flood_scores
        ),

        flood_mean,

        flood_median,

        float(
            np.std(
                flood_scores,
                ddof=1
            )
        ),

        float(
            np.percentile(
                flood_scores,
                25
            )
        ),

        float(
            np.percentile(
                flood_scores,
                75
            )
        )

    ])


    score_rows.append([

        method,

        "NonFlood",

        len(
            nonflood_scores
        ),

        nonflood_mean,

        nonflood_median,

        float(
            np.std(
                nonflood_scores,
                ddof=1
            )
        ),

        float(
            np.percentile(
                nonflood_scores,
                25
            )
        ),

        float(
            np.percentile(
                nonflood_scores,
                75
            )
        )

    ])


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows.append([

        method,

        roc[
            "auc"
        ],

        threshold,

        roc[
            "best_youden"
        ],

        metrics[
            "Sensitivity"
        ],

        metrics[
            "Specificity"
        ],

        metrics[
            "Precision"
        ],

        metrics[
            "F1"
        ],

        metrics[
            "Accuracy"
        ],

        metrics[
            "Balanced_Accuracy"
        ],

        metrics[
            "TP"
        ],

        metrics[
            "TN"
        ],

        metrics[
            "FP"
        ],

        metrics[
            "FN"
        ],

        flood_mean,

        nonflood_mean,

        flood_median,

        nonflood_median,

        flood_mean
        -
        nonflood_mean

    ])


    print(
        f"{method:4s}"
        f"  AUC={roc['auc']:.6f}"
        f"  Threshold={threshold:.6f}"
        f"  Sens={metrics['Sensitivity']:.4f}"
        f"  Spec={metrics['Specificity']:.4f}"
        f"  F1={metrics['F1']:.4f}"
        f"  FloodMean={flood_mean:.4f}"
        f"  NonFloodMean={nonflood_mean:.4f}"
    )


    # --------------------------------------------------------
    # Compact ROC output
    #
    # Exact AUC uses complete ROC.
    # CSV stores <=501 representative points.
    # --------------------------------------------------------

    n_points = len(
        roc[
            "fpr"
        ]
    )


    sample_idx = np.unique(
        np.linspace(
            0,
            n_points - 1,
            min(
                501,
                n_points
            )
        ).astype(
            int
        )
    )


    for idx in sample_idx:


        threshold_value = (
            roc[
                "thresholds"
            ][idx]
        )


        if np.isinf(
            threshold_value
        ):

            threshold_output = ""

        else:

            threshold_output = float(
                threshold_value
            )


        roc_rows.append([

            method,

            int(idx),

            float(
                roc[
                    "fpr"
                ][idx]
            ),

            float(
                roc[
                    "tpr"
                ][idx]
            ),

            threshold_output

        ])


# ============================================================
# 13. SAVE FULL-PIXEL VALIDATION SUMMARY
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

        "Method",

        "ROC_AUC",

        "Youden_Optimal_Threshold",

        "Youden_J",

        "Sensitivity",

        "Specificity",

        "Precision",

        "F1",

        "Accuracy",

        "Balanced_Accuracy",

        "TP",

        "TN",

        "FP",

        "FN",

        "Flood_Mean_Score",

        "NonFlood_Mean_Score",

        "Flood_Median_Score",

        "NonFlood_Median_Score",

        "Flood_minus_NonFlood_Mean"

    ])


    writer.writerows(
        summary_rows
    )


# ============================================================
# 14. SAVE SCORE STATISTICS
# ============================================================

with open(
    SCORE_STATS_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Method",

        "Observed_Class",

        "N",

        "Mean",

        "Median",

        "SD",

        "P25",

        "P75"

    ])


    writer.writerows(
        score_rows
    )


# ============================================================
# 15. SAVE ROC CURVES
# ============================================================

with open(
    ROC_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Method",

        "Curve_Index",

        "False_Positive_Rate",

        "True_Positive_Rate",

        "Threshold"

    ])


    writer.writerows(
        roc_rows
    )


# ============================================================
# 16. SPATIALLY THINNED VALIDATION
#
# Raster resolution = 30 m
#
# STEP = 10 pixels
#
# 10 × 30 m = 300 m
#
# We test ALL 100 possible 10x10 grid offsets.
# ============================================================

STEP = 10


offset_rows = []


print(
    "\n========== 300 m SYSTEMATIC AUC ROBUSTNESS =========="
)


for row_offset in range(
    STEP
):


    for col_offset in range(
        STEP
    ):


        subset_mask = np.zeros(
            validation_mask.shape,
            dtype=bool
        )


        subset_mask[
            row_offset::STEP,
            col_offset::STEP
        ] = True


        subset_mask &= validation_mask


        y = s1[
            subset_mask
        ].astype(
            np.uint8
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


        if (
            flood_n == 0
            or
            nonflood_n == 0
        ):

            continue


        ahp_auc = calculate_auc(

            y,

            ahp[
                subset_mask
            ]

        )


        fahp_auc = calculate_auc(

            y,

            fahp[
                subset_mask
            ]

        )


        anp_auc = calculate_auc(

            y,

            anp[
                subset_mask
            ]

        )


        offset_rows.append([

            row_offset,

            col_offset,

            len(y),

            flood_n,

            nonflood_n,

            ahp_auc,

            fahp_auc,

            anp_auc

        ])


print(
    "Systematic offsets completed :",
    len(
        offset_rows
    )
)


# ============================================================
# 17. SAVE OFFSET RESULTS
# ============================================================

with open(
    OFFSET_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Row_Offset",

        "Column_Offset",

        "N",

        "Flood_N",

        "NonFlood_N",

        "AHP_AUC",

        "FAHP_AUC",

        "ANP_AUC"

    ])


    writer.writerows(
        offset_rows
    )


# ============================================================
# 18. OFFSET ROBUSTNESS SUMMARY
# ============================================================

if len(
    offset_rows
) == 0:

    raise Exception(
        "No systematic validation offsets completed."
    )


offset_array = np.array(

    [
        row[5:]
        for row in offset_rows
    ],

    dtype=np.float64

)


METHOD_NAMES = [

    "AHP",

    "FAHP",

    "ANP"

]


offset_summary_rows = []


print(
    "\n========== 300 m OFFSET SUMMARY =========="
)


for i, method in enumerate(
    METHOD_NAMES
):


    values = offset_array[
        :,
        i
    ]


    mean_auc = float(
        np.mean(
            values
        )
    )


    sd_auc = float(
        np.std(
            values,
            ddof=1
        )
    )


    min_auc = float(
        np.min(
            values
        )
    )


    median_auc = float(
        np.median(
            values
        )
    )


    max_auc = float(
        np.max(
            values
        )
    )


    offset_summary_rows.append([

        method,

        len(values),

        mean_auc,

        sd_auc,

        min_auc,

        median_auc,

        max_auc

    ])


    print(
        f"{method:4s}"
        f" mean={mean_auc:.6f}"
        f" SD={sd_auc:.6f}"
        f" min={min_auc:.6f}"
        f" median={median_auc:.6f}"
        f" max={max_auc:.6f}"
    )


with open(
    OFFSET_SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Method",

        "Number_of_Offsets",

        "Mean_AUC",

        "SD_AUC",

        "Minimum_AUC",

        "Median_AUC",

        "Maximum_AUC"

    ])


    writer.writerows(
        offset_summary_rows
    )


# ============================================================
# 19. REPRESENTATIVE 300 m SAMPLE
#
# Fixed offset:
#
# row = 5
# column = 5
# ============================================================

REP_ROW = 5
REP_COL = 5


rep_mask = np.zeros(
    validation_mask.shape,
    dtype=bool
)


rep_mask[
    REP_ROW::STEP,
    REP_COL::STEP
] = True


rep_mask &= validation_mask


rows_idx, cols_idx = np.where(
    rep_mask
)


rep_labels = s1[
    rep_mask
].astype(
    np.uint8
)


rep_ahp = ahp[
    rep_mask
]


rep_fahp = fahp[
    rep_mask
]


rep_anp = anp[
    rep_mask
]


with open(
    SAMPLE_CSV,
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

        "Observed_2021_Flood_Label",

        "AHP_Score",

        "FAHP_Score",

        "ANP_Score"

    ])


    for i in range(
        len(
            rep_labels
        )
    ):


        writer.writerow([

            int(
                rows_idx[i]
            ),

            int(
                cols_idx[i]
            ),

            int(
                rep_labels[i]
            ),

            float(
                rep_ahp[i]
            ),

            float(
                rep_fahp[i]
            ),

            float(
                rep_anp[i]
            )

        ])


# ============================================================
# 20. FINAL METHOD RANKING
#
# Ranking based on full-pixel ROC-AUC.
# Validation conclusion should later also consider
# 300 m spatially thinned robustness.
# ============================================================

ranking = sorted(

    [
        (
            row[0],
            row[1]
        )
        for row in summary_rows
    ],

    key=lambda x: x[1],

    reverse=True

)


print(
    "\n========== FULL-PIXEL AUC RANKING =========="
)


for rank, (
    method,
    auc
) in enumerate(
    ranking,
    start=1
):


    print(
        f"{rank}. "
        f"{method:4s}"
        f" AUC={auc:.6f}"
    )


# ============================================================
# 21. FINAL STATUS
# ============================================================

print(
    "\n=============================================="
)

print(
    "DU STEP 14 VALIDATION COMPLETE"
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
    "1. DU14_2021_Validation_Summary.csv"
)

print(
    "2. DU14_Flood_NonFlood_Score_Statistics.csv"
)

print(
    "3. DU14_ROC_Curves_501Points.csv"
)

print(
    "4. DU14_300m_Systematic_Offset_AUC.csv"
)

print(
    "5. DU14_300m_AUC_Robustness_Summary.csv"
)

print(
    "6. DU14_Representative_300m_Validation_Sample.csv"
)

print(
    "\nFINAL STATUS: DU STEP 14 PASS"
)

print(
    "=============================================="
)