import os
import csv
import numpy as np


# ============================================================
# B08C
# LULC × EVENT FLOOD-LABEL AUDIT
#
# PURPOSE
# ------------------------------------------------------------
# Check whether the B08B "0% built-up flood" result is:
#
# A) only a SHAP-sample effect
#
# or
#
# B) systematic in the full Sentinel-1 label dataset.
#
# No model fitting.
# No raster regeneration.
# Pure label QC.
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
    "06_Explainability",
    "B08C_Label_Bias_Audit"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

FULL_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B08C_LULC_Event_Label_Audit.csv"
)

BUILTUP_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B08C_BuiltUp_Label_Summary.csv"
)

QC_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B08C_Label_Bias_QC.csv"
)


# ============================================================
# 02. LOAD DATA
# ============================================================

if not os.path.exists(INPUT_NPZ):

    raise Exception(
        "B04 dataset not found:\n"
        + INPUT_NPZ
    )


D = np.load(
    INPUT_NPZ,
    allow_pickle=False
)


LULC = D[
    "LULC"
].astype(
    np.int16
)


EVENTS = {

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


COMPOSITE = D[
    "Composite_AnyFlood_All4Valid"
].astype(
    np.int8
)


N = len(
    LULC
)


print(
    "\n========== B08C DATASET =========="
)

print(
    "Rows:",
    f"{N:,}"
)


# ============================================================
# 03. LULC CLASS LABELS
# ============================================================

LULC_NAMES = {

    10:
        "Tree cover",

    20:
        "Shrubland",

    30:
        "Grassland",

    40:
        "Cropland",

    50:
        "Built-up",

    60:
        "Bare / sparse vegetation",

    90:
        "Herbaceous wetland",

    95:
        "Mangroves"

}


codes = sorted(
    np.unique(
        LULC
    ).tolist()
)


print(
    "LULC codes:",
    codes
)


# ============================================================
# 04. EVENT TOTAL FLOOD COUNTS
#
# Needed to calculate how much of each event's observed
# flood inventory belongs to each LULC class.
# ============================================================

EVENT_TOTAL_FLOOD = {}


for event, labels in EVENTS.items():

    EVENT_TOTAL_FLOOD[
        event
    ] = int(
        np.count_nonzero(
            labels == 1
        )
    )


COMPOSITE_TOTAL_FLOOD = int(
    np.count_nonzero(
        COMPOSITE == 1
    )
)


# ============================================================
# 05. FULL LULC × EVENT AUDIT
# ============================================================

rows = []


print(
    "\n========== LULC × EVENT LABEL AUDIT =========="
)


for code in codes:


    class_mask = (
        LULC == code
    )


    class_n = int(
        np.count_nonzero(
            class_mask
        )
    )


    class_name = LULC_NAMES.get(
        int(code),
        "Unknown"
    )


    print(
        "\n",
        code,
        class_name,
        "N=",
        f"{class_n:,}"
    )


    # --------------------------------------------------------
    # Individual flood events
    # --------------------------------------------------------

    for event, labels in EVENTS.items():


        valid = (
            class_mask
            &
            (
                labels >= 0
            )
        )


        valid_n = int(
            np.count_nonzero(
                valid
            )
        )


        flood_n = int(
            np.count_nonzero(
                valid
                &
                (
                    labels == 1
                )
            )
        )


        nonflood_n = (
            valid_n
            -
            flood_n
        )


        flood_pct = (
            100
            *
            flood_n
            /
            valid_n

            if valid_n > 0

            else np.nan
        )


        flood_share = (
            100
            *
            flood_n
            /
            EVENT_TOTAL_FLOOD[
                event
            ]

            if EVENT_TOTAL_FLOOD[
                event
            ] > 0

            else np.nan
        )


        rows.append([

            int(code),

            class_name,

            event,

            class_n,

            valid_n,

            flood_n,

            nonflood_n,

            flood_pct,

            flood_share

        ])


        print(
            f"  {event}:"
            f" valid={valid_n:7,d}"
            f" flood={flood_n:6,d}"
            f" flood%={flood_pct:7.3f}"
        )


    # --------------------------------------------------------
    # Composite label
    # --------------------------------------------------------

    composite_valid = (
        class_mask
        &
        (
            COMPOSITE >= 0
        )
    )


    composite_valid_n = int(
        np.count_nonzero(
            composite_valid
        )
    )


    composite_flood_n = int(
        np.count_nonzero(
            composite_valid
            &
            (
                COMPOSITE == 1
            )
        )
    )


    composite_nonflood_n = (
        composite_valid_n
        -
        composite_flood_n
    )


    composite_pct = (
        100
        *
        composite_flood_n
        /
        composite_valid_n

        if composite_valid_n > 0

        else np.nan
    )


    composite_share = (
        100
        *
        composite_flood_n
        /
        COMPOSITE_TOTAL_FLOOD

        if COMPOSITE_TOTAL_FLOOD > 0

        else np.nan
    )


    rows.append([

        int(code),

        class_name,

        "Composite_AnyFlood",

        class_n,

        composite_valid_n,

        composite_flood_n,

        composite_nonflood_n,

        composite_pct,

        composite_share

    ])


    print(
        "  Composite:"
        f" valid={composite_valid_n:7,d}"
        f" flood={composite_flood_n:6,d}"
        f" flood%={composite_pct:7.3f}"
    )


# ============================================================
# 06. SAVE FULL TABLE
# ============================================================

with open(
    FULL_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "LULC_Code",

        "LULC_Class",

        "Label_Set",

        "Total_Class_Sample_N",

        "Valid_Label_N",

        "Flood_N",

        "NonFlood_N",

        "Flood_Percent_of_Valid_Class",

        "Percent_of_All_Flood_Pixels"

    ])


    writer.writerows(
        rows
    )


# ============================================================
# 07. BUILT-UP SPECIFIC AUDIT
# ============================================================

BUILT_CODE = 50


built_rows = [

    row

    for row in rows

    if row[0] == BUILT_CODE

]


print(
    "\n========== BUILT-UP CLASS AUDIT =========="
)


with open(
    BUILTUP_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Label_Set",

        "Valid_N",

        "Flood_N",

        "Flood_Percent"

    ])


    for row in built_rows:


        print(
            f"{row[2]:20s}"
            f" valid={row[4]:7,d}"
            f" flood={row[5]:6,d}"
            f" flood%={row[7]:8.4f}"
        )


        writer.writerow([

            row[2],

            row[4],

            row[5],

            row[7]

        ])


# ============================================================
# 08. DECISION LOGIC
# ============================================================

built_event_flood_counts = [

    row[5]

    for row in built_rows

    if row[2] != "Composite_AnyFlood"

]


built_composite_row = [

    row

    for row in built_rows

    if row[2] == "Composite_AnyFlood"

][0]


built_composite_flood = int(
    built_composite_row[
        5
    ]
)


built_composite_pct = float(
    built_composite_row[
        7
    ]
)


all_events_zero = all(

    value == 0

    for value in built_event_flood_counts

)


if all_events_zero:

    status = (
        "CRITICAL_CHECK: built-up has zero flood labels "
        "in every event"
    )

elif built_composite_pct < 1.0:

    status = (
        "CAUTION: built-up flood prevalence is below 1%; "
        "possible urban SAR under-detection"
    )

else:

    status = (
        "PASS: built-up contains positive flood labels; "
        "B08B zero was sample-specific"
    )


print(
    "\n========== B08C DECISION =========="
)

print(
    status
)


# ============================================================
# 09. SAVE QC SUMMARY
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
            "Master_Sample_N",
            N
        ],

        [
            "Composite_Flood_N",
            COMPOSITE_TOTAL_FLOOD
        ],

        [
            "BuiltUp_Composite_Flood_N",
            built_composite_flood
        ],

        [
            "BuiltUp_Composite_Flood_Percent",
            built_composite_pct
        ],

        [
            "BuiltUp_AllFourEvents_Zero",
            all_events_zero
        ],

        [
            "Decision",
            status
        ]

    ])


# ============================================================
# 10. FINAL
# ============================================================

print(
    "\n=============================================="
)

print(
    "BUET B08C LABEL-BIAS AUDIT COMPLETE"
)

print(
    "=============================================="
)

print(
    "\nOutputs:"
)

print(
    FULL_CSV
)

print(
    BUILTUP_CSV
)

print(
    QC_CSV
)

print(
    "\nFINAL STATUS: BUET B08C COMPLETE"
)

print(
    "=============================================="
)