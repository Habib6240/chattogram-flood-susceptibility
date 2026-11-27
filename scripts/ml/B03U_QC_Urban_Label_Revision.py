import os
import csv
import numpy as np
from osgeo import gdal


# ============================================================
# B03U CONSOLIDATED URBAN-LABEL REVISION QC
#
# PURPOSE
# ------------------------------------------------------------
# Compare original B03 masks with urban-aware B03U masks.
#
# Checks:
# 1. Exact grid compatibility
# 2. Valid-domain preservation
# 3. Old flood -> new non-flood must be ZERO
# 4. Quantify newly-added pixels
# 5. Audit new labels by LULC
# 6. Re-check built-up flood prevalence
# 7. Rebuild four-event composite label statistics
#
# No model fitting.
# ============================================================


# ============================================================
# 01. PATHS
# ============================================================

BASE = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
)


OLD_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "01_Sentinel1_Event_Masks"
)


NEW_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "01B_Sentinel1_Event_Masks_UrbanAware"
)


B04_NPZ = os.path.join(
    BASE,
    "13_BUET_ML",
    "02_ML_Dataset",
    "BUET_B04_ML_Master_150m.npz"
)


OUTPUT_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "01B_Sentinel1_Event_Masks_UrbanAware",
    "B03U_QC"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


EVENT_QC_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B03U_QC_Event_Revision.csv"
)


LULC_QC_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B03U_QC_New_Labels_By_LULC.csv"
)


BUILTUP_QC_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B03U_QC_BuiltUp_Prevalence.csv"
)


COMPOSITE_QC_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B03U_QC_Composite_Comparison.csv"
)


# ============================================================
# 02. EVENT FILES
# ============================================================

EVENTS = {

    "2019": {

        "old":
            "BUET_B03_E03_2019_Jul_FloodMask30m.tif",

        "new":
            "BUET_B03U_E03_2019_Jul_FloodMask30m.tif"

    },


    "2021": {

        "old":
            "BUET_B03_E04_2021_Jul_FloodMask30m.tif",

        "new":
            "BUET_B03U_E04_2021_Jul_FloodMask30m.tif"

    },


    "2023": {

        "old":
            "BUET_B03_E05_2023_Aug_FloodMask30m.tif",

        "new":
            "BUET_B03U_E05_2023_Aug_FloodMask30m.tif"

    },


    "2026": {

        "old":
            "BUET_B03_E07_2026_Jul_FloodMask30m.tif",

        "new":
            "BUET_B03U_E07_2026_Jul_FloodMask30m.tif"

    }

}


# ============================================================
# 03. LULC NAMES
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


# ============================================================
# 04. READ RASTER
# ============================================================

def read_raster(path):


    if not os.path.exists(
        path
    ):

        raise Exception(
            "Missing raster:\n"
            +
            path
        )


    ds = gdal.Open(
        path,
        gdal.GA_ReadOnly
    )


    if ds is None:

        raise Exception(
            "Cannot open:\n"
            +
            path
        )


    band = ds.GetRasterBand(
        1
    )


    arr = band.ReadAsArray()


    return (
        ds,
        arr
    )


# ============================================================
# 05. LOAD B04 SAMPLE
#
# This gives exact 150 m sample locations and WorldCover LULC.
# ============================================================

if not os.path.exists(
    B04_NPZ
):

    raise Exception(
        "B04 NPZ not found:\n"
        +
        B04_NPZ
    )


D = np.load(
    B04_NPZ,
    allow_pickle=False
)


ROWS = D[
    "Raster_Row"
].astype(
    np.int32
)


COLS = D[
    "Raster_Column"
].astype(
    np.int32
)


LULC = D[
    "LULC"
].astype(
    np.int16
)


N_SAMPLE = len(
    ROWS
)


print(
    "\n========== B03U QC SAMPLE =========="
)

print(
    "150 m sample N:",
    f"{N_SAMPLE:,}"
)


# ============================================================
# 06. STORAGE
# ============================================================

OLD_SAMPLE = {}

NEW_SAMPLE = {}

ADDED_SAMPLE = {}

event_rows = []


reference_ds = None


# ============================================================
# 07. EVENT-BY-EVENT FULL-RASTER QC
# ============================================================

print(
    "\n========== FULL-RASTER REVISION QC =========="
)


for event, files in EVENTS.items():


    old_path = os.path.join(
        OLD_DIR,
        files[
            "old"
        ]
    )


    new_path = os.path.join(
        NEW_DIR,
        files[
            "new"
        ]
    )


    old_ds, old = read_raster(
        old_path
    )


    new_ds, new = read_raster(
        new_path
    )


    # --------------------------------------------------------
    # Grid QC
    # --------------------------------------------------------

    if reference_ds is None:

        reference_ds = old_ds


    if (
        old_ds.RasterXSize
        !=
        new_ds.RasterXSize
        or
        old_ds.RasterYSize
        !=
        new_ds.RasterYSize
    ):

        raise Exception(
            event
            +
            ": dimension mismatch."
        )


    if not np.allclose(

        old_ds.GetGeoTransform(),

        new_ds.GetGeoTransform(),

        atol=0.01

    ):

        raise Exception(
            event
            +
            ": geotransform mismatch."
        )


    if old_ds.GetProjection() != new_ds.GetProjection():

        raise Exception(
            event
            +
            ": projection mismatch."
        )


    # --------------------------------------------------------
    # Valid domains
    # --------------------------------------------------------

    old_valid = (
        (old == 0)
        |
        (old == 1)
    )


    new_valid = (
        (new == 0)
        |
        (new == 1)
    )


    valid_domain_mismatch = int(
        np.count_nonzero(
            old_valid
            !=
            new_valid
        )
    )


    both_valid = (
        old_valid
        &
        new_valid
    )


    # --------------------------------------------------------
    # Revision transitions
    # --------------------------------------------------------

    old_flood = (
        both_valid
        &
        (old == 1)
    )


    new_flood = (
        both_valid
        &
        (new == 1)
    )


    added = (
        both_valid
        &
        (old == 0)
        &
        (new == 1)
    )


    removed = (
        both_valid
        &
        (old == 1)
        &
        (new == 0)
    )


    old_n = int(
        np.count_nonzero(
            old_flood
        )
    )


    new_n = int(
        np.count_nonzero(
            new_flood
        )
    )


    added_n = int(
        np.count_nonzero(
            added
        )
    )


    removed_n = int(
        np.count_nonzero(
            removed
        )
    )


    # Exact 30 x 30 m grid:
    PIXEL_AREA_KM2 = 900.0 / 1e6


    old_area = (
        old_n
        *
        PIXEL_AREA_KM2
    )


    new_area = (
        new_n
        *
        PIXEL_AREA_KM2
    )


    added_area = (
        added_n
        *
        PIXEL_AREA_KM2
    )


    increase_pct = (

        100
        *
        added_n
        /
        old_n

        if old_n > 0

        else np.nan

    )


    event_rows.append([

        event,

        valid_domain_mismatch,

        int(
            np.count_nonzero(
                both_valid
            )
        ),

        old_n,

        new_n,

        added_n,

        removed_n,

        old_area,

        new_area,

        added_area,

        increase_pct

    ])


    print(
        f"{event}:"
        f" valid-domain mismatch={valid_domain_mismatch:,}"
        f" | old flood={old_n:,}"
        f" | new flood={new_n:,}"
        f" | added={added_n:,}"
        f" | removed={removed_n:,}"
        f" | increase={increase_pct:.2f}%"
    )


    # --------------------------------------------------------
    # Sample at B04 locations
    # --------------------------------------------------------

    old_s = old[
        ROWS,
        COLS
    ]


    new_s = new[
        ROWS,
        COLS
    ]


    OLD_SAMPLE[
        event
    ] = old_s.astype(
        np.int16
    )


    NEW_SAMPLE[
        event
    ] = new_s.astype(
        np.int16
    )


    ADDED_SAMPLE[
        event
    ] = (
        (old_s == 0)
        &
        (new_s == 1)
    )


    old_ds = None
    new_ds = None

    del old
    del new


# ============================================================
# 08. SAVE EVENT QC
# ============================================================

with open(
    EVENT_QC_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Event",

        "Valid_Domain_Mismatch_Pixels",

        "Common_Valid_Pixels",

        "Old_Flood_Pixels",

        "New_Flood_Pixels",

        "Added_0to1_Pixels",

        "Removed_1to0_Pixels",

        "Old_Flood_Area_km2",

        "New_Flood_Area_km2",

        "Added_Area_km2",

        "Added_Percent_of_Old_Flood"

    ])


    writer.writerows(
        event_rows
    )


# ============================================================
# 09. NEW LABELS BY LULC
# ============================================================

lulc_rows = []


print(
    "\n========== NEW URBAN-AWARE LABELS BY LULC =========="
)


for event in EVENTS.keys():


    added = ADDED_SAMPLE[
        event
    ]


    total_added = int(
        np.count_nonzero(
            added
        )
    )


    print(
        "\nEvent",
        event,
        "| total added sample labels:",
        f"{total_added:,}"
    )


    for code in sorted(
        np.unique(
            LULC
        )
    ):


        class_mask = (
            LULC
            ==
            code
        )


        class_n = int(
            np.count_nonzero(
                class_mask
            )
        )


        added_n = int(
            np.count_nonzero(
                added
                &
                class_mask
            )
        )


        percent_of_added = (

            100
            *
            added_n
            /
            total_added

            if total_added > 0

            else np.nan

        )


        percent_of_class = (

            100
            *
            added_n
            /
            class_n

            if class_n > 0

            else np.nan

        )


        name = LULC_NAMES.get(
            int(code),
            "Unknown"
        )


        lulc_rows.append([

            event,

            int(code),

            name,

            class_n,

            added_n,

            percent_of_added,

            percent_of_class

        ])


        if added_n > 0:

            print(
                f"  {int(code):3d}"
                f" {name:25s}"
                f" added={added_n:6,d}"
                f" | {percent_of_added:6.2f}% of additions"
                f" | {percent_of_class:6.2f}% of class"
            )


with open(
    LULC_QC_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Event",

        "LULC_Code",

        "LULC_Class",

        "Class_Sample_N",

        "Added_Label_N",

        "Percent_of_All_New_Labels",

        "Percent_of_LULC_Class_Added"

    ])


    writer.writerows(
        lulc_rows
    )


# ============================================================
# 10. BUILT-UP EVENT PREVALENCE
# ============================================================

built_mask = (
    LULC == 50
)


built_rows = []


print(
    "\n========== BUILT-UP EVENT AUDIT =========="
)


for event in EVENTS.keys():


    old = OLD_SAMPLE[
        event
    ]


    new = NEW_SAMPLE[
        event
    ]


    old_valid = (
        built_mask
        &
        (
            (old == 0)
            |
            (old == 1)
        )
    )


    new_valid = (
        built_mask
        &
        (
            (new == 0)
            |
            (new == 1)
        )
    )


    old_valid_n = int(
        np.count_nonzero(
            old_valid
        )
    )


    new_valid_n = int(
        np.count_nonzero(
            new_valid
        )
    )


    old_flood_n = int(
        np.count_nonzero(
            old_valid
            &
            (old == 1)
        )
    )


    new_flood_n = int(
        np.count_nonzero(
            new_valid
            &
            (new == 1)
        )
    )


    added_n = int(
        np.count_nonzero(
            built_mask
            &
            ADDED_SAMPLE[
                event
            ]
        )
    )


    old_pct = (

        100
        *
        old_flood_n
        /
        old_valid_n

        if old_valid_n > 0

        else np.nan

    )


    new_pct = (

        100
        *
        new_flood_n
        /
        new_valid_n

        if new_valid_n > 0

        else np.nan

    )


    built_rows.append([

        event,

        old_valid_n,

        old_flood_n,

        old_pct,

        new_valid_n,

        new_flood_n,

        new_pct,

        added_n

    ])


    print(
        f"{event}:"
        f" old={old_flood_n:,}/{old_valid_n:,}"
        f" ({old_pct:.3f}%)"
        f" -> new={new_flood_n:,}/{new_valid_n:,}"
        f" ({new_pct:.3f}%)"
        f" | added built-up={added_n:,}"
    )


with open(
    BUILTUP_QC_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Event",

        "Old_BuiltUp_Valid_N",

        "Old_BuiltUp_Flood_N",

        "Old_BuiltUp_Flood_Percent",

        "New_BuiltUp_Valid_N",

        "New_BuiltUp_Flood_N",

        "New_BuiltUp_Flood_Percent",

        "Added_BuiltUp_N"

    ])


    writer.writerows(
        built_rows
    )


# ============================================================
# 11. OLD + NEW FOUR-EVENT COMPOSITES
# ============================================================

event_names = list(
    EVENTS.keys()
)


old_matrix = np.column_stack(

    [
        OLD_SAMPLE[
            event
        ]

        for event in event_names
    ]

)


new_matrix = np.column_stack(

    [
        NEW_SAMPLE[
            event
        ]

        for event in event_names
    ]

)


old_all_valid = np.all(
    (
        (old_matrix == 0)
        |
        (old_matrix == 1)
    ),
    axis=1
)


new_all_valid = np.all(
    (
        (new_matrix == 0)
        |
        (new_matrix == 1)
    ),
    axis=1
)


old_composite = np.full(
    N_SAMPLE,
    -1,
    dtype=np.int8
)


new_composite = np.full(
    N_SAMPLE,
    -1,
    dtype=np.int8
)


old_composite[
    old_all_valid
] = np.any(

    old_matrix[
        old_all_valid
    ]
    ==
    1,

    axis=1

).astype(
    np.int8
)


new_composite[
    new_all_valid
] = np.any(

    new_matrix[
        new_all_valid
    ]
    ==
    1,

    axis=1

).astype(
    np.int8
)


def summarize_composite(
    label,
    name
):


    valid = (
        label >= 0
    )


    flood = (
        label == 1
    )


    return [

        name,

        int(
            np.count_nonzero(
                valid
            )
        ),

        int(
            np.count_nonzero(
                flood
            )
        ),

        100
        *
        np.count_nonzero(
            flood
        )
        /
        np.count_nonzero(
            valid
        )

    ]


old_summary = summarize_composite(
    old_composite,
    "Original_B03"
)


new_summary = summarize_composite(
    new_composite,
    "UrbanAware_B03U"
)


# ============================================================
# 12. BUILT-UP COMPOSITE
# ============================================================

old_built_valid = (
    built_mask
    &
    (
        old_composite >= 0
    )
)


new_built_valid = (
    built_mask
    &
    (
        new_composite >= 0
    )
)


old_built_flood = int(
    np.count_nonzero(
        built_mask
        &
        (
            old_composite == 1
        )
    )
)


new_built_flood = int(
    np.count_nonzero(
        built_mask
        &
        (
            new_composite == 1
        )
    )
)


old_built_n = int(
    np.count_nonzero(
        old_built_valid
    )
)


new_built_n = int(
    np.count_nonzero(
        new_built_valid
    )
)


old_built_pct = (
    100
    *
    old_built_flood
    /
    old_built_n
)


new_built_pct = (
    100
    *
    new_built_flood
    /
    new_built_n
)


print(
    "\n========== FOUR-EVENT COMPOSITE COMPARISON =========="
)

print(
    f"Original B03 :"
    f" valid={old_summary[1]:,}"
    f" flood={old_summary[2]:,}"
    f" flood%={old_summary[3]:.3f}"
)

print(
    f"Urban B03U   :"
    f" valid={new_summary[1]:,}"
    f" flood={new_summary[2]:,}"
    f" flood%={new_summary[3]:.3f}"
)

print(
    "\nBuilt-up composite:"
)

print(
    f"Original:"
    f" {old_built_flood:,}/{old_built_n:,}"
    f" = {old_built_pct:.3f}%"
)

print(
    f"B03U    :"
    f" {new_built_flood:,}/{new_built_n:,}"
    f" = {new_built_pct:.3f}%"
)


with open(
    COMPOSITE_QC_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Dataset",

        "All4_Valid_N",

        "Composite_Flood_N",

        "Composite_Flood_Percent",

        "BuiltUp_Valid_N",

        "BuiltUp_Composite_Flood_N",

        "BuiltUp_Composite_Flood_Percent"

    ])


    writer.writerow([

        old_summary[0],

        old_summary[1],

        old_summary[2],

        old_summary[3],

        old_built_n,

        old_built_flood,

        old_built_pct

    ])


    writer.writerow([

        new_summary[0],

        new_summary[1],

        new_summary[2],

        new_summary[3],

        new_built_n,

        new_built_flood,

        new_built_pct

    ])


# ============================================================
# 13. EVENT-REPEAT AUDIT FOR NEW BUILT-UP LABELS
# ============================================================

added_matrix = np.column_stack(

    [
        ADDED_SAMPLE[
            event
        ]

        for event in event_names
    ]

)


built_added_count = np.sum(

    added_matrix[
        built_mask
    ],

    axis=1

)


print(
    "\n========== BUILT-UP NEW-LABEL REPEAT COUNT =========="
)


for count in range(
    0,
    5
):


    n = int(
        np.count_nonzero(
            built_added_count
            ==
            count
        )
    )


    print(
        f"Added in {count} events:"
        f" {n:,} built-up samples"
    )


# ============================================================
# 14. LOCK CONDITIONS
# ============================================================

total_removed = sum(
    row[6]
    for row in event_rows
)


total_domain_mismatch = sum(
    row[1]
    for row in event_rows
)


print(
    "\n========== B03U LOCK QC =========="
)

print(
    "Total old-flood pixels removed :",
    f"{total_removed:,}"
)

print(
    "Total valid-domain mismatches  :",
    f"{total_domain_mismatch:,}"
)


if (
    total_removed == 0
    and
    total_domain_mismatch == 0
):

    print(
        "Technical revision consistency: PASS"
    )

else:

    print(
        "Technical revision consistency: FAIL"
    )


print(
    "\nFINAL STATUS: B03U QC COMPLETE"
)