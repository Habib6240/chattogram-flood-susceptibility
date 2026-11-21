import os
import csv
import numpy as np
from osgeo import gdal, osr
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# B09B — STRICT ML vs MCDA COMPARISON
#
# PURPOSE
# ------------------------------------------------------------
# A) Validation comparison:
#    Compare AHP, FAHP, ANP against STRICT B07U out-of-fold
#    RF, XGBoost, and RF-XGB consensus predictions on the same
#    held-out event labels.
#
#    This avoids the unfair comparison of MCDA against a
#    full-data ML model evaluated on its own training target.
#
# B) Full-domain spatial comparison:
#    Compare final B09A RF/XGB/consensus maps with AHP/FAHP/ANP
#    using continuous correlation and scale-invariant top-20%
#    overlap / percentile-class agreement.
#
# IMPORTANT
# ------------------------------------------------------------
# MCDA and ML scores have different score distributions.
# Therefore raw common thresholds are NOT imposed across
# the two method families.
# ============================================================


# ============================================================
# 01. PATHS
# ============================================================

BASE = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
)

B07U_OOF = os.path.join(
    BASE,
    "13_BUET_ML",
    "05B_Strict_LOEO_UrbanAware",
    "BUET_B07U_LOEO_OutOfFold_Predictions.npz",
)

B09A_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "07_Final_30m_Maps_UrbanAware",
)

RF_MAP = os.path.join(
    B09A_DIR,
    "BUET_B09A_RF_SusceptibilityScore_30m.tif",
)

XGB_MAP = os.path.join(
    B09A_DIR,
    "BUET_B09A_XGB_SusceptibilityScore_30m.tif",
)

CONSENSUS_MAP = os.path.join(
    B09A_DIR,
    "BUET_B09A_RF_XGB_ConsensusScore_30m.tif",
)

FINAL_MASK = os.path.join(
    BASE,
    "11_FINAL_ANALYSIS_INPUTS",
    "08_Final_Common_Analysis_Mask_30m.tif",
)

OUTPUT_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "08_ML_vs_MCDA_Comparison",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

EVENT_PERFORMANCE_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B09B_Strict_Event_Performance_All_Methods.csv",
)

MACRO_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B09B_Strict_Macro_Performance_All_Methods.csv",
)

PAIRWISE_CORRELATION_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B09B_FullDomain_Pairwise_Correlation.csv",
)

TOP20_OVERLAP_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B09B_Top20_Susceptibility_Overlap.csv",
)

PERCENTILE_CLASS_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B09B_Percentile5Class_Agreement.csv",
)

MCDA_DISCOVERY_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B09B_MCDA_Map_Discovery.csv",
)


# ============================================================
# 02. OPTIONAL MANUAL MCDA PATH OVERRIDES
#
# Leave as None unless automatic discovery reports ambiguity.
# ============================================================

AHP_OVERRIDE = None
FAHP_OVERRIDE = None
ANP_OVERRIDE = None


# ============================================================
# 03. DISCOVER FINAL DU CONTINUOUS MCDA MAPS
# ============================================================

def walk_tifs(root):

    results = []

    skip_dirs = {
        os.path.abspath(B09A_DIR),
        os.path.abspath(OUTPUT_DIR),
    }

    for current_root, dirs, files in os.walk(root):

        current_abs = os.path.abspath(current_root)

        if any(
            current_abs.startswith(skip)
            for skip in skip_dirs
        ):
            continue

        for filename in files:

            if filename.lower().endswith(
                (".tif", ".tiff")
            ):
                results.append(
                    os.path.join(
                        current_root,
                        filename,
                    )
                )

    return results


ALL_TIFS = walk_tifs(BASE)


def find_candidates(method):

    candidates = []

    for path in ALL_TIFS:

        name = os.path.basename(path).lower()

        # We want continuous susceptibility/index outputs,
        # not validation masks, class maps, or input criteria.
        if not (
            "suscept" in name
            or "index" in name
        ):
            continue

        if "class" in name:
            continue

        if method == "AHP":

            if (
                "ahp" in name
                and
                "fahp" not in name
            ):
                candidates.append(path)

        elif method == "FAHP":

            if "fahp" in name:
                candidates.append(path)

        elif method == "ANP":

            if "anp" in name:
                candidates.append(path)

    return candidates


def resolve_map(
    method,
    override,
):

    if override is not None:

        if not os.path.exists(override):
            raise Exception(
                method
                +
                " override does not exist:\n"
                +
                override
            )

        return override, [
            override
        ]

    candidates = find_candidates(
        method
    )

    # Prefer explicit DU/final susceptibility files.
    preferred = [
        p
        for p in candidates
        if (
            "du" in os.path.basename(p).lower()
            or "susceptibility" in os.path.basename(p).lower()
        )
    ]

    if len(preferred) == 1:
        return preferred[0], candidates

    if len(candidates) == 1:
        return candidates[0], candidates

    print(
        "\n========== "
        + method
        + " CANDIDATES =========="
    )

    for p in candidates:
        print(p)

    raise Exception(
        method
        +
        " map discovery is ambiguous. "
        "Set the corresponding *_OVERRIDE path at the top "
        "of this script and rerun."
    )


AHP_MAP, AHP_CANDIDATES = resolve_map(
    "AHP",
    AHP_OVERRIDE,
)

FAHP_MAP, FAHP_CANDIDATES = resolve_map(
    "FAHP",
    FAHP_OVERRIDE,
)

ANP_MAP, ANP_CANDIDATES = resolve_map(
    "ANP",
    ANP_OVERRIDE,
)


print(
    "\n========== RESOLVED MCDA MAPS =========="
)

print(
    "AHP :",
    AHP_MAP,
)

print(
    "FAHP:",
    FAHP_MAP,
)

print(
    "ANP :",
    ANP_MAP,
)


with open(
    MCDA_DISCOVERY_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "Method",
        "Resolved_Path",
        "Candidate_Count",
    ])

    writer.writerows([
        [
            "AHP",
            AHP_MAP,
            len(AHP_CANDIDATES),
        ],
        [
            "FAHP",
            FAHP_MAP,
            len(FAHP_CANDIDATES),
        ],
        [
            "ANP",
            ANP_MAP,
            len(ANP_CANDIDATES),
        ],
    ])


# ============================================================
# 04. RASTER HELPERS
# ============================================================

gdal.UseExceptions()


def same_projection(
    wkt1,
    wkt2,
):

    s1 = osr.SpatialReference()
    s2 = osr.SpatialReference()

    s1.ImportFromWkt(
        wkt1
    )

    s2.ImportFromWkt(
        wkt2
    )

    return bool(
        s1.IsSame(s2)
    )


def check_grid(
    ds,
    ref_ds,
    name,
):

    if (
        ds.RasterXSize
        !=
        ref_ds.RasterXSize
        or
        ds.RasterYSize
        !=
        ref_ds.RasterYSize
    ):
        raise Exception(
            name
            +
            ": dimension mismatch."
        )

    if not np.allclose(
        ds.GetGeoTransform(),
        ref_ds.GetGeoTransform(),
        atol=0.01,
    ):
        raise Exception(
            name
            +
            ": geotransform mismatch."
        )

    if not same_projection(
        ds.GetProjection(),
        ref_ds.GetProjection(),
    ):
        raise Exception(
            name
            +
            ": CRS mismatch."
        )


def read_valid_array(
    path,
    ref_ds,
    analysis_mask,
    name,
):

    ds = gdal.Open(
        path,
        gdal.GA_ReadOnly,
    )

    if ds is None:
        raise Exception(
            "Cannot open "
            +
            name
            +
            ":\n"
            +
            path
        )

    check_grid(
        ds,
        ref_ds,
        name,
    )

    band = ds.GetRasterBand(
        1
    )

    arr = band.ReadAsArray().astype(
        np.float32
    )

    nodata = band.GetNoDataValue()

    valid = analysis_mask.copy()

    if nodata is not None:

        if np.isnan(nodata):
            valid &= np.isfinite(arr)
        else:
            valid &= (
                arr != nodata
            )

    valid &= np.isfinite(
        arr
    )

    if np.count_nonzero(valid) != np.count_nonzero(analysis_mask):

        missing = (
            np.count_nonzero(analysis_mask)
            -
            np.count_nonzero(valid)
        )

        raise Exception(
            name
            +
            f": {missing:,} analysis pixels are missing."
        )

    values = arr[
        analysis_mask
    ].astype(
        np.float32
    )

    ds = None

    return values


# ============================================================
# 05. LOAD FULL-DOMAIN MAPS
# ============================================================

mask_ds = gdal.Open(
    FINAL_MASK,
    gdal.GA_ReadOnly,
)

if mask_ds is None:
    raise Exception(
        "Cannot open final analysis mask."
    )

mask_arr = mask_ds.GetRasterBand(
    1
).ReadAsArray()

ANALYSIS_MASK = (
    mask_arr == 1
)

VALID_N = int(
    np.count_nonzero(
        ANALYSIS_MASK
    )
)

print(
    "\n========== FULL-DOMAIN GRID QC =========="
)

print(
    "Valid pixels:",
    f"{VALID_N:,}"
)

if VALID_N != 4722216:

    print(
        "WARNING: expected 4,722,216 valid pixels."
    )


MAP_VALUES = {}

for method, path in [

    (
        "RF",
        RF_MAP,
    ),

    (
        "XGBoost",
        XGB_MAP,
    ),

    (
        "ML_Consensus",
        CONSENSUS_MAP,
    ),

    (
        "AHP",
        AHP_MAP,
    ),

    (
        "FAHP",
        FAHP_MAP,
    ),

    (
        "ANP",
        ANP_MAP,
    ),

]:

    MAP_VALUES[
        method
    ] = read_valid_array(
        path,
        mask_ds,
        ANALYSIS_MASK,
        method,
    )

    print(
        f"{method:12s}"
        f" min={np.min(MAP_VALUES[method]):.6f}"
        f" mean={np.mean(MAP_VALUES[method]):.6f}"
        f" max={np.max(MAP_VALUES[method]):.6f}"
    )

print(
    "Full-domain map QC: PASS"
)


# ============================================================
# 06. FULL-DOMAIN CONTINUOUS CORRELATION
# ============================================================

METHODS = [
    "RF",
    "XGBoost",
    "ML_Consensus",
    "AHP",
    "FAHP",
    "ANP",
]

correlation_rows = []


print(
    "\n========== FULL-DOMAIN SCORE CORRELATION =========="
)

for i in range(
    len(METHODS)
):

    for j in range(
        i + 1,
        len(METHODS)
    ):

        m1 = METHODS[i]
        m2 = METHODS[j]

        a = MAP_VALUES[m1]
        b = MAP_VALUES[m2]

        pearson = float(
            np.corrcoef(
                a,
                b,
            )[0, 1]
        )

        spearman = float(
            spearmanr(
                a,
                b,
            ).statistic
        )

        correlation_rows.append([
            m1,
            m2,
            pearson,
            spearman,
        ])

        print(
            f"{m1:12s} vs {m2:12s}"
            f" Pearson={pearson:.4f}"
            f" Spearman={spearman:.4f}"
        )


with open(
    PAIRWISE_CORRELATION_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "Method_1",
        "Method_2",
        "Pearson_r",
        "Spearman_rho",
    ])

    writer.writerows(
        correlation_rows
    )


# ============================================================
# 07. SCALE-INVARIANT TOP-20% OVERLAP
#
# Each method supplies its own top 20%.
# Jaccard measures spatial overlap of high-susceptibility zones.
# ============================================================

TOP20 = {}

for method in METHODS:

    threshold = float(
        np.quantile(
            MAP_VALUES[
                method
            ],
            0.80,
        )
    )

    TOP20[
        method
    ] = (
        MAP_VALUES[
            method
        ]
        >=
        threshold
    )


top20_rows = []


print(
    "\n========== TOP-20% SUSCEPTIBILITY OVERLAP =========="
)

for i in range(
    len(METHODS)
):

    for j in range(
        i + 1,
        len(METHODS)
    ):

        m1 = METHODS[i]
        m2 = METHODS[j]

        a = TOP20[m1]
        b = TOP20[m2]

        intersection = int(
            np.count_nonzero(
                a
                &
                b
            )
        )

        union = int(
            np.count_nonzero(
                a
                |
                b
            )
        )

        jaccard = (
            intersection
            /
            union
        )

        overlap_of_smaller = (
            intersection
            /
            min(
                np.count_nonzero(a),
                np.count_nonzero(b),
            )
        )

        top20_rows.append([
            m1,
            m2,
            intersection,
            union,
            jaccard,
            overlap_of_smaller,
        ])

        print(
            f"{m1:12s} vs {m2:12s}"
            f" Jaccard={jaccard:.4f}"
            f" sharedTop20={100*overlap_of_smaller:.2f}%"
        )


with open(
    TOP20_OVERLAP_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "Method_1",
        "Method_2",
        "Top20_Intersection_Pixels",
        "Top20_Union_Pixels",
        "Jaccard",
        "Shared_Percent_of_Smaller_Top20",
    ])

    writer.writerows(
        top20_rows
    )


# ============================================================
# 08. PERCENTILE 5-CLASS AGREEMENT
#
# This is rank-based and scale-invariant:
# every method is classified by its own quintile thresholds.
# ============================================================

def percentile_five_class(
    values,
):

    breaks = np.quantile(
        values,
        [
            0.20,
            0.40,
            0.60,
            0.80,
        ],
    )

    return (
        np.digitize(
            values,
            breaks,
            right=False,
        )
        +
        1
    ).astype(
        np.uint8
    )


PCLASS = {
    method:
        percentile_five_class(
            MAP_VALUES[
                method
            ]
        )
    for method in METHODS
}


pclass_rows = []


print(
    "\n========== PERCENTILE 5-CLASS AGREEMENT =========="
)

for i in range(
    len(METHODS)
):

    for j in range(
        i + 1,
        len(METHODS)
    ):

        m1 = METHODS[i]
        m2 = METHODS[j]

        c1 = PCLASS[m1]
        c2 = PCLASS[m2]

        exact = float(
            np.mean(
                c1 == c2
            )
        )

        within1 = float(
            np.mean(
                np.abs(
                    c1.astype(
                        np.int16
                    )
                    -
                    c2.astype(
                        np.int16
                    )
                )
                <=
                1
            )
        )

        pclass_rows.append([
            m1,
            m2,
            exact,
            within1,
        ])

        print(
            f"{m1:12s} vs {m2:12s}"
            f" exact={100*exact:.2f}%"
            f" within1={100*within1:.2f}%"
        )


with open(
    PERCENTILE_CLASS_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "Method_1",
        "Method_2",
        "Exact_5Class_Agreement",
        "Within_One_Class_Agreement",
    ])

    writer.writerows(
        pclass_rows
    )


# ============================================================
# 09. STRICT EVENT-LEVEL PERFORMANCE
#
# B07U RF/XGB predictions are genuinely held out in both event
# and geography.
#
# MCDA scores are sampled at the SAME rows and assessed against
# the SAME held-out event labels.
# ============================================================

OOF = np.load(
    B07U_OOF,
    allow_pickle=False,
)

ROWS = OOF[
    "Raster_Row"
].astype(
    np.int32
)

COLS = OOF[
    "Raster_Column"
].astype(
    np.int32
)


# Need full raster arrays only for MCDA row/column sampling.
MCDA_FULL = {}

for method, path in [
    ("AHP", AHP_MAP),
    ("FAHP", FAHP_MAP),
    ("ANP", ANP_MAP),
]:

    ds = gdal.Open(
        path,
        gdal.GA_ReadOnly,
    )

    check_grid(
        ds,
        mask_ds,
        method,
    )

    MCDA_FULL[
        method
    ] = ds.GetRasterBand(
        1
    ).ReadAsArray().astype(
        np.float32
    )

    ds = None


EVENTS = [
    "2019",
    "2021",
    "2023",
    "2026",
]

event_rows = []


print(
    "\n========== STRICT EVENT PERFORMANCE: ML vs MCDA =========="
)

for event in EVENTS:

    y = OOF[
        "Observed_"
        +
        event
    ].astype(
        np.int8
    )

    rf = OOF[
        "RF_"
        +
        event
    ].astype(
        np.float32
    )

    xgb = OOF[
        "XGB_"
        +
        event
    ].astype(
        np.float32
    )

    valid = (
        (y >= 0)
        &
        np.isfinite(
            rf
        )
        &
        np.isfinite(
            xgb
        )
    )

    yv = y[
        valid
    ]

    rf_v = rf[
        valid
    ]

    xgb_v = xgb[
        valid
    ]

    consensus_v = (
        (
            rf_v
            +
            xgb_v
        )
        /
        2.0
    ).astype(
        np.float32
    )

    valid_rows = ROWS[
        valid
    ]

    valid_cols = COLS[
        valid
    ]

    method_scores = {
        "Random_Forest":
            rf_v,

        "XGBoost":
            xgb_v,

        "ML_Consensus":
            consensus_v,

        "AHP":
            MCDA_FULL[
                "AHP"
            ][
                valid_rows,
                valid_cols
            ],

        "FAHP":
            MCDA_FULL[
                "FAHP"
            ][
                valid_rows,
                valid_cols
            ],

        "ANP":
            MCDA_FULL[
                "ANP"
            ][
                valid_rows,
                valid_cols
            ],
    }

    prevalence = float(
        np.mean(
            yv == 1
        )
    )

    print(
        "\nHeld event",
        event,
        "| N=",
        f"{len(yv):,}",
        "| prevalence=",
        f"{100*prevalence:.2f}%",
    )

    for method, score in method_scores.items():

        score = np.asarray(
            score,
            dtype=np.float32
        )

        if not np.all(
            np.isfinite(
                score
            )
        ):
            raise Exception(
                method
                +
                " contains non-finite sampled scores for "
                +
                event
            )

        auc = float(
            roc_auc_score(
                yv,
                score,
            )
        )

        ap = float(
            average_precision_score(
                yv,
                score,
            )
        )

        event_rows.append([
            event,
            method,
            len(yv),
            int(
                np.count_nonzero(
                    yv == 1
                )
            ),
            prevalence,
            auc,
            ap,
        ])

        print(
            f"  {method:14s}"
            f" AUC={auc:.6f}"
            f" AP={ap:.6f}"
        )


with open(
    EVENT_PERFORMANCE_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "Held_Event",
        "Method",
        "N",
        "Flood_N",
        "Flood_Prevalence",
        "ROC_AUC",
        "Average_Precision",
    ])

    writer.writerows(
        event_rows
    )


# ============================================================
# 10. MACRO EVENT SUMMARY
#
# Equal event weighting avoids largest-event domination.
# ============================================================

SUMMARY_METHODS = [
    "Random_Forest",
    "XGBoost",
    "ML_Consensus",
    "AHP",
    "FAHP",
    "ANP",
]

macro_rows = []


print(
    "\n========== STRICT MACRO EVENT SUMMARY =========="
)

for method in SUMMARY_METHODS:

    rows = [
        r
        for r in event_rows
        if r[1] == method
    ]

    aucs = np.array(
        [
            r[5]
            for r in rows
        ],
        dtype=float,
    )

    aps = np.array(
        [
            r[6]
            for r in rows
        ],
        dtype=float,
    )

    macro_rows.append([
        method,
        float(
            np.mean(
                aucs
            )
        ),
        float(
            np.std(
                aucs
            )
        ),
        float(
            np.min(
                aucs
            )
        ),
        float(
            np.max(
                aucs
            )
        ),
        float(
            np.mean(
                aps
            )
        ),
        float(
            np.std(
                aps
            )
        ),
    ])

    print(
        f"{method:14s}"
        f" macro AUC={np.mean(aucs):.6f}"
        f" SD={np.std(aucs):.6f}"
        f" macro AP={np.mean(aps):.6f}"
    )


with open(
    MACRO_SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "Method",
        "Macro_Event_AUC",
        "SD_Event_AUC",
        "Minimum_Event_AUC",
        "Maximum_Event_AUC",
        "Macro_Event_AP",
        "SD_Event_AP",
    ])

    writer.writerows(
        macro_rows
    )


# ============================================================
# 11. FINAL
# ============================================================

mask_ds = None

print(
    "\n=============================================="
)

print(
    "BUET B09B STRICT ML vs MCDA COMPARISON COMPLETE"
)

print(
    "=============================================="
)

print(
    "\nFINAL STATUS: BUET B09B PASS"
)

print(
    "=============================================="
)
