import os
import csv
import numpy as np
from osgeo import gdal, osr


# ============================================================
# B04
# BUET MULTI-EVENT ML MASTER DATASET
#
# PURPOSE
# ------------------------------------------------------------
# 1 spatial pixel = 1 dataset row
#
# Static predictors:
#     13 conditioning factors
#
# Multi-event labels:
#     Flood_2019
#     Flood_2021
#     Flood_2023
#     Flood_2026
#
# Sampling:
#     systematic 150 m candidate grid
#     = every 5th pixel on 30 m raster
#
# Outputs:
#     CSV
#     compressed NPZ
#     QC summary
#     event summary
#     feature metadata
# ============================================================


# ============================================================
# 01. ROOT PATH
# ============================================================

BASE = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
)


# ============================================================
# 02. OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "02_ML_Dataset"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


MASTER_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B04_ML_Master_150m.csv"
)

MASTER_NPZ = os.path.join(
    OUTPUT_DIR,
    "BUET_B04_ML_Master_150m.npz"
)

QC_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B04_Dataset_QC.csv"
)

EVENT_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B04_Event_Label_Summary.csv"
)

FEATURE_METADATA_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B04_Feature_Definitions.csv"
)

EVENT_METADATA_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B04_Event_Metadata.csv"
)


# ============================================================
# 03. FINAL ANALYSIS MASK
# ============================================================

FINAL_MASK_FILE = os.path.join(
    BASE,
    "11_FINAL_ANALYSIS_INPUTS",
    "08_Final_Common_Analysis_Mask_30m.tif"
)


# ============================================================
# 04. SENTINEL-1 EVENT MASKS
# ============================================================

S1_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "01_Sentinel1_Event_Masks"
)


EVENT_FILES = {

    "Flood_2019":
        os.path.join(
            S1_DIR,
            "BUET_B03_E03_2019_Jul_FloodMask30m.tif"
        ),

    "Flood_2021":
        os.path.join(
            S1_DIR,
            "BUET_B03_E04_2021_Jul_FloodMask30m.tif"
        ),

    "Flood_2023":
        os.path.join(
            S1_DIR,
            "BUET_B03_E05_2023_Aug_FloodMask30m.tif"
        ),

    "Flood_2026":
        os.path.join(
            S1_DIR,
            "BUET_B03_E07_2026_Jul_FloodMask30m.tif"
        )
}


# ============================================================
# 05. FACTOR FILE NAMES
#
# Script automatically searches BASE recursively.
#
# If duplicate exact filenames exist, script will stop rather
# than silently selecting the wrong raster.
# ============================================================

FACTOR_FILENAMES = {

    "Elevation":
        "DEM_Clipped_UTM46N.tif",

    "Slope":
        "Slope_deg_30m.tif",

    "Aspect":
        "Aspect_deg_30_FLATFIX.tif",

    "Profile_Curvature":
        "Profile_Curvature_30m.tif",

    "Tangential_Curvature":
        "Tangential_Curvature_30m.tif",

    "TWI":
        "TWI_Chittagong_30m_STACK.tif",

    "Drainage_Density":
        "Drainage_Density_Chittagong_30m_STACK.tif",

    "Rainfall":
        (
            "01_Rainfall_CHIRPSv3_MAP_1991_2020_"
            "Chattogram_30m_STACK.tif"
        ),

    "LULC":
        (
            "02_LULC_ESA_WorldCover_2021_"
            "Chattogram_30m_STACK.tif"
        ),

    "NDVI":
        (
            "03_NDVI_Sentinel2_Median_2021_"
            "Chattogram_30m_STACK.tif"
        ),

    "HSG":
        (
            "04_Soil_HYSOGs250m_"
            "Chattogram_30m_STACK.tif"
        ),

    "Distance_River":
        (
            "05_Distance_to_River_HydroRIVERS_"
            "UPLAND25_Chattogram_30m_STACK.tif"
        ),

    "Distance_Road":
        (
            "06_Distance_to_Road_OSM2021_"
            "Chattogram_30m_STACK.tif"
        )
}


# ============================================================
# 06. FACTOR ORDER
# ============================================================

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


# ============================================================
# 07. BUILD FILE INDEX
# ============================================================

print(
    "\n========== B04 FILE DISCOVERY =========="
)

print(
    "Indexing TIFF files under:"
)

print(
    BASE
)


tif_index = {}


for root, dirs, files in os.walk(
    BASE
):

    for filename in files:

        if filename.lower().endswith(
            (
                ".tif",
                ".tiff"
            )
        ):

            key = filename.lower()

            tif_index.setdefault(
                key,
                []
            ).append(
                os.path.join(
                    root,
                    filename
                )
            )


def find_unique_raster(
    filename
):

    matches = tif_index.get(
        filename.lower(),
        []
    )


    if len(
        matches
    ) == 0:

        raise Exception(
            "\nRaster not found:\n"
            +
            filename
        )


    if len(
        matches
    ) > 1:

        print(
            "\nDUPLICATE FILE FOUND:"
        )

        print(
            filename
        )

        for path in matches:

            print(
                "  ",
                path
            )


        raise Exception(
            "Duplicate exact raster filename. "
            "Remove/rename duplicate or specify exact path."
        )


    return matches[0]


FACTOR_PATHS = {}


for factor in FACTOR_ORDER:

    path = find_unique_raster(
        FACTOR_FILENAMES[
            factor
        ]
    )

    FACTOR_PATHS[
        factor
    ] = path

    print(
        f"{factor:24s} -> {path}"
    )


# ============================================================
# 08. BASIC FILE EXISTENCE QC
# ============================================================

required_direct_files = [
    FINAL_MASK_FILE
] + list(
    EVENT_FILES.values()
)


for path in required_direct_files:

    if not os.path.exists(
        path
    ):

        raise Exception(
            "\nRequired file missing:\n"
            +
            path
        )


print(
    "\nDirect-path file QC : PASS"
)


# ============================================================
# 09. RASTER READER
# ============================================================

def read_raster(
    path
):

    ds = gdal.Open(
        path,
        gdal.GA_ReadOnly
    )


    if ds is None:

        raise Exception(
            "Cannot open raster:\n"
            +
            path
        )


    band = ds.GetRasterBand(
        1
    )


    arr = band.ReadAsArray()


    nodata = band.GetNoDataValue()


    numeric = arr.astype(
        np.float64
    )


    valid = np.isfinite(
        numeric
    )


    if nodata is not None:

        if np.isnan(
            nodata
        ):

            valid &= ~np.isnan(
                numeric
            )

        else:

            valid &= (
                arr != nodata
            )


    return (
        ds,
        arr,
        valid,
        nodata
    )


# ============================================================
# 10. GRID COMPARISON FUNCTION
# ============================================================

def same_projection(
    wkt1,
    wkt2
):

    srs1 = osr.SpatialReference()

    srs2 = osr.SpatialReference()


    srs1.ImportFromWkt(
        wkt1
    )

    srs2.ImportFromWkt(
        wkt2
    )


    return bool(
        srs1.IsSame(
            srs2
        )
    )


def check_grid(
    ds,
    ref_ds,
    name
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
            " raster dimensions mismatch."
        )


    if not np.allclose(
        ds.GetGeoTransform(),
        ref_ds.GetGeoTransform(),
        atol=0.01
    ):

        raise Exception(
            name
            +
            " geotransform mismatch."
        )


    if not same_projection(
        ds.GetProjection(),
        ref_ds.GetProjection()
    ):

        raise Exception(
            name
            +
            " CRS mismatch."
        )


# ============================================================
# 11. READ FINAL MASK
# ============================================================

mask_ds, mask_arr, mask_valid, mask_nodata = (
    read_raster(
        FINAL_MASK_FILE
    )
)


analysis_mask = (
    mask_arr == 1
)


analysis_pixels = int(
    np.count_nonzero(
        analysis_mask
    )
)


print(
    "\n========== ANALYSIS MASK =========="
)

print(
    "Analysis pixels :",
    f"{analysis_pixels:,}"
)

print(
    "Expected approx : 4,722,216"
)


# ============================================================
# 12. SYSTEMATIC SAMPLING GRID
#
# Raster = 30 m
#
# 5 pixels x 30 m = 150 m
#
# Central offset inside each 5x5 cell:
#
# row offset = 2
# col offset = 2
# ============================================================

SAMPLE_STEP = 5

ROW_OFFSET = 2

COL_OFFSET = 2


systematic_grid = np.zeros(
    analysis_mask.shape,
    dtype=bool
)


systematic_grid[
    ROW_OFFSET::SAMPLE_STEP,
    COL_OFFSET::SAMPLE_STEP
] = True


sample_mask = (
    analysis_mask
    &
    systematic_grid
)


sample_rows, sample_cols = np.where(
    sample_mask
)


N = len(
    sample_rows
)


print(
    "\n========== 150 m SYSTEMATIC SAMPLE =========="
)

print(
    "Step              : 5 pixels"
)

print(
    "Ground spacing    : 150 m"
)

print(
    "Offset            : row 2 / col 2"
)

print(
    "Sample pixels     :",
    f"{N:,}"
)


if N < 10000:

    raise Exception(
        "Unexpectedly small sample."
    )


# ============================================================
# 13. SAMPLE COORDINATES
# ============================================================

gt = mask_ds.GetGeoTransform()


# Pixel center coordinates
X_COORD = (
    gt[0]
    +
    (
        sample_cols
        +
        0.5
    )
    *
    gt[1]
    +
    (
        sample_rows
        +
        0.5
    )
    *
    gt[2]
)


Y_COORD = (
    gt[3]
    +
    (
        sample_cols
        +
        0.5
    )
    *
    gt[4]
    +
    (
        sample_rows
        +
        0.5
    )
    *
    gt[5]
)


# ============================================================
# 14. SPATIAL BLOCK IDS
#
# Primary later:
#     5 km block CV
#
# Sensitivity:
#     3 km block CV
# ============================================================

xmin = gt[0]

ytop = gt[3]


block3_col = np.floor(
    (
        X_COORD
        -
        xmin
    )
    /
    3000.0
).astype(
    np.int32
)


block3_row = np.floor(
    (
        ytop
        -
        Y_COORD
    )
    /
    3000.0
).astype(
    np.int32
)


BLOCK_3KM = (
    block3_row.astype(
        np.int64
    )
    *
    10000
    +
    block3_col.astype(
        np.int64
    )
)


block5_col = np.floor(
    (
        X_COORD
        -
        xmin
    )
    /
    5000.0
).astype(
    np.int32
)


block5_row = np.floor(
    (
        ytop
        -
        Y_COORD
    )
    /
    5000.0
).astype(
    np.int32
)


BLOCK_5KM = (
    block5_row.astype(
        np.int64
    )
    *
    10000
    +
    block5_col.astype(
        np.int64
    )
)


print(
    "Unique 3 km blocks:",
    len(
        np.unique(
            BLOCK_3KM
        )
    )
)

print(
    "Unique 5 km blocks:",
    len(
        np.unique(
            BLOCK_5KM
        )
    )
)


# ============================================================
# 15. EXTRACT STATIC FACTORS
# ============================================================

DATA = {}


print(
    "\n========== EXTRACTING STATIC FACTORS =========="
)


for factor in FACTOR_ORDER:


    path = FACTOR_PATHS[
        factor
    ]


    ds, arr, valid, nodata = read_raster(
        path
    )


    check_grid(
        ds,
        mask_ds,
        factor
    )


    # Final common mask should guarantee validity
    # for all 13 factors.
    invalid_in_sample = int(
        np.count_nonzero(
            ~valid[
                sample_mask
            ]
        )
    )


    if invalid_in_sample > 0:

        raise Exception(
            factor
            +
            f" has {invalid_in_sample} invalid "
            "pixels inside B04 sample."
        )


    values = arr[
        sample_mask
    ]


    if factor in [
        "LULC",
        "HSG"
    ]:

        DATA[
            factor
        ] = values.astype(
            np.int16
        )

    else:

        DATA[
            factor
        ] = values.astype(
            np.float32
        )


    print(
        f"{factor:24s}"
        f" N={len(values):,}"
        f" min={np.min(values):.6f}"
        f" max={np.max(values):.6f}"
    )


    ds = None

    del arr

    del valid


# ============================================================
# 16. ASPECT TRANSFORMATION
#
# Raw:
#     -1 = flat
#      0..360 = direction
#
# ML representation:
#
# non-flat:
#     sin(theta)
#     cos(theta)
#     flat = 0
#
# flat:
#     sin = 0
#     cos = 0
#     flat = 1
# ============================================================

aspect = DATA[
    "Aspect"
].astype(
    np.float64
)


ASPECT_FLAT = (
    aspect == -1
).astype(
    np.uint8
)


ASPECT_SIN = np.zeros(
    N,
    dtype=np.float32
)


ASPECT_COS = np.zeros(
    N,
    dtype=np.float32
)


nonflat = (
    aspect >= 0
)


theta = np.deg2rad(
    aspect[
        nonflat
    ]
)


ASPECT_SIN[
    nonflat
] = np.sin(
    theta
).astype(
    np.float32
)


ASPECT_COS[
    nonflat
] = np.cos(
    theta
).astype(
    np.float32
)


print(
    "\n========== ASPECT ENCODING =========="
)

print(
    "Flat samples    :",
    f"{np.count_nonzero(ASPECT_FLAT == 1):,}"
)

print(
    "Non-flat samples:",
    f"{np.count_nonzero(ASPECT_FLAT == 0):,}"
)


# ============================================================
# 17. READ FOUR EVENT LABELS
#
# Encoding in B04 master dataset:
#
#  1 = flood
#  0 = observed non-flood candidate
# -1 = invalid / unavailable event label
# ============================================================

EVENT_LABELS = {}


event_summary_rows = []


print(
    "\n========== EXTRACTING EVENT LABELS =========="
)


for event_name, event_path in EVENT_FILES.items():


    ds, arr, valid_raw, nodata = read_raster(
        event_path
    )


    check_grid(
        ds,
        mask_ds,
        event_name
    )


    valid_label = (
        valid_raw
        &
        (
            (arr == 0)
            |
            (arr == 1)
        )
    )


    labels = np.full(
        N,
        -1,
        dtype=np.int8
    )


    sample_event_values = arr[
        sample_mask
    ]


    sample_event_valid = valid_label[
        sample_mask
    ]


    labels[
        sample_event_valid
    ] = sample_event_values[
        sample_event_valid
    ].astype(
        np.int8
    )


    EVENT_LABELS[
        event_name
    ] = labels


    valid_n = int(
        np.count_nonzero(
            labels >= 0
        )
    )


    flood_n = int(
        np.count_nonzero(
            labels == 1
        )
    )


    nonflood_n = int(
        np.count_nonzero(
            labels == 0
        )
    )


    invalid_n = int(
        np.count_nonzero(
            labels == -1
        )
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


    event_summary_rows.append([

        event_name,

        valid_n,

        flood_n,

        nonflood_n,

        invalid_n,

        flood_pct

    ])


    print(
        f"{event_name:12s}"
        f" valid={valid_n:8,d}"
        f" flood={flood_n:7,d}"
        f" nonflood={nonflood_n:8,d}"
        f" invalid={invalid_n:6,d}"
        f" flood%={flood_pct:6.3f}"
    )


    ds = None

    del arr

    del valid_raw

    del valid_label


# ============================================================
# 18. EVENT-MATRIX DERIVED VARIABLES
# ============================================================

EVENT_MATRIX = np.column_stack([

    EVENT_LABELS[
        "Flood_2019"
    ],

    EVENT_LABELS[
        "Flood_2021"
    ],

    EVENT_LABELS[
        "Flood_2023"
    ],

    EVENT_LABELS[
        "Flood_2026"
    ]

]).astype(
    np.int8
)


VALID_EVENT_COUNT = np.sum(
    EVENT_MATRIX >= 0,
    axis=1
).astype(
    np.uint8
)


FLOOD_EVENT_COUNT = np.sum(
    EVENT_MATRIX == 1,
    axis=1
).astype(
    np.uint8
)


# ============================================================
# 19. FOUR-EVENT COMPOSITE LABEL
#
# Only assigned if ALL four event labels are valid.
#
# 1 = flooded in >=1 event
# 0 = non-flood in all four events
# -1 = at least one event unavailable
#
# This is useful for B05 baseline.
#
# LOEO in B07 will NOT use the held-out event
# when constructing its training labels.
# ============================================================

ALL4_VALID = (
    VALID_EVENT_COUNT == 4
)


COMPOSITE_ANY4 = np.full(
    N,
    -1,
    dtype=np.int8
)


COMPOSITE_ANY4[
    ALL4_VALID
] = (
    FLOOD_EVENT_COUNT[
        ALL4_VALID
    ]
    > 0
).astype(
    np.int8
)


all4_n = int(
    np.count_nonzero(
        COMPOSITE_ANY4 >= 0
    )
)


any_flood_n = int(
    np.count_nonzero(
        COMPOSITE_ANY4 == 1
    )
)


stable_nonflood_n = int(
    np.count_nonzero(
        COMPOSITE_ANY4 == 0
    )
)


print(
    "\n========== FOUR-EVENT COMPOSITE LABEL =========="
)

print(
    "All-4-event valid pixels :",
    f"{all4_n:,}"
)

print(
    "Flooded >=1 event        :",
    f"{any_flood_n:,}"
)

print(
    "Non-flood in all 4       :",
    f"{stable_nonflood_n:,}"
)

print(
    "Composite flood fraction :",
    f"{100 * any_flood_n / all4_n:.3f}%"
)


# ============================================================
# 20. FINAL ML FEATURE QC
# ============================================================

CONTINUOUS_FEATURES = [

    "Elevation",

    "Slope",

    "Profile_Curvature",

    "Tangential_Curvature",

    "TWI",

    "Drainage_Density",

    "Rainfall",

    "NDVI",

    "Distance_River",

    "Distance_Road"

]


for feature in CONTINUOUS_FEATURES:

    if not np.all(
        np.isfinite(
            DATA[
                feature
            ]
        )
    ):

        raise Exception(
            feature
            +
            " contains non-finite ML values."
        )


valid_lulc_codes = np.unique(
    DATA[
        "LULC"
    ]
)


valid_hsg_codes = np.unique(
    DATA[
        "HSG"
    ]
)


print(
    "\n========== CATEGORICAL QC =========="
)

print(
    "LULC codes:",
    valid_lulc_codes.tolist()
)

print(
    "HSG codes :",
    valid_hsg_codes.tolist()
)


# ============================================================
# 21. SAVE EVENT SUMMARY
# ============================================================

with open(
    EVENT_SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Event",

        "Valid_Sample_N",

        "Flood_N",

        "NonFlood_N",

        "Invalid_N",

        "Flood_Percent_of_Valid"

    ])


    writer.writerows(
        event_summary_rows
    )


# ============================================================
# 22. EVENT METADATA
# ============================================================

with open(
    EVENT_METADATA_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "Event",

        "Observation_Type",

        "Pre_Date",

        "Observed_Date",

        "Sensor_Track",

        "Interpretation"

    ])


    writer.writerows([

        [
            "2019",
            "DURING_EVENT",
            "2019-07-02",
            "2019-07-14",
            "S1A_ASC_R41",
            "Core event"
        ],

        [
            "2021",
            "DURING_EVENT",
            "2021-07-17",
            "2021-07-29",
            "S1A_DESC_R77",
            "Core event"
        ],

        [
            "2023",
            "POST_PEAK_PERSISTENT_INUNDATION",
            "2023-07-31",
            "2023-08-12",
            "S1A_DESC_R77",
            "Interpret as persistent post-peak inundation"
        ],

        [
            "2026",
            "DURING_EVENT",
            "2026-06-28",
            "2026-07-10",
            "S1D_DESC_R77",
            "Core event"
        ]

    ])


# ============================================================
# 23. FEATURE METADATA
# ============================================================

feature_rows = [

    [
        "Elevation",
        "Continuous",
        "Raw",
        "m",
        "DEM elevation"
    ],

    [
        "Slope",
        "Continuous",
        "Raw",
        "degree",
        "Terrain slope"
    ],

    [
        "Aspect_sin",
        "Continuous_circular",
        "Derived_from_Aspect",
        "-1 to 1",
        "sin(aspect)"
    ],

    [
        "Aspect_cos",
        "Continuous_circular",
        "Derived_from_Aspect",
        "-1 to 1",
        "cos(aspect)"
    ],

    [
        "Aspect_flat",
        "Binary",
        "Derived_from_Aspect",
        "0/1",
        "1 for flat terrain"
    ],

    [
        "Profile_Curvature",
        "Continuous",
        "Raw",
        "raster units",
        "Profile curvature"
    ],

    [
        "Tangential_Curvature",
        "Continuous",
        "Raw",
        "raster units",
        "Tangential curvature"
    ],

    [
        "TWI",
        "Continuous",
        "Raw",
        "dimensionless",
        "Topographic wetness index"
    ],

    [
        "Drainage_Density",
        "Continuous",
        "Raw",
        "source raster units",
        "Drainage density"
    ],

    [
        "Rainfall",
        "Continuous",
        "Raw",
        "mm/year",
        "1991-2020 CHIRPS mean annual precipitation"
    ],

    [
        "LULC",
        "Categorical",
        "Raw_class_code",
        "ESA code",
        "One-hot encode in B05"
    ],

    [
        "NDVI",
        "Continuous",
        "Raw",
        "-1 to 1",
        "Sentinel-2 median NDVI, 2021"
    ],

    [
        "HSG",
        "Categorical",
        "Raw_class_code",
        "HYSOG code",
        "One-hot encode in B05"
    ],

    [
        "Distance_River",
        "Continuous",
        "Raw",
        "m",
        "Distance to selected HydroRIVERS network"
    ],

    [
        "Distance_Road",
        "Continuous",
        "Raw",
        "m",
        "Distance to selected OSM road network"
    ]

]


with open(
    FEATURE_METADATA_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.writer(
        f
    )


    writer.writerow([

        "ML_Feature",

        "Variable_Type",

        "Encoding",

        "Unit_or_Code",

        "Notes"

    ])


    writer.writerows(
        feature_rows
    )


# ============================================================
# 24. SAVE COMPRESSED NPZ
#
# This is the FAST file used by B05-B08.
# ============================================================

print(
    "\n========== SAVING NPZ =========="
)


np.savez_compressed(

    MASTER_NPZ,

    Raster_Row=
        sample_rows.astype(
            np.int32
        ),

    Raster_Column=
        sample_cols.astype(
            np.int32
        ),

    X=
        X_COORD.astype(
            np.float64
        ),

    Y=
        Y_COORD.astype(
            np.float64
        ),

    Block_3km=
        BLOCK_3KM,

    Block_5km=
        BLOCK_5KM,

    Elevation=
        DATA[
            "Elevation"
        ],

    Slope=
        DATA[
            "Slope"
        ],

    Aspect_sin=
        ASPECT_SIN,

    Aspect_cos=
        ASPECT_COS,

    Aspect_flat=
        ASPECT_FLAT,

    Profile_Curvature=
        DATA[
            "Profile_Curvature"
        ],

    Tangential_Curvature=
        DATA[
            "Tangential_Curvature"
        ],

    TWI=
        DATA[
            "TWI"
        ],

    Drainage_Density=
        DATA[
            "Drainage_Density"
        ],

    Rainfall=
        DATA[
            "Rainfall"
        ],

    LULC=
        DATA[
            "LULC"
        ],

    NDVI=
        DATA[
            "NDVI"
        ],

    HSG=
        DATA[
            "HSG"
        ],

    Distance_River=
        DATA[
            "Distance_River"
        ],

    Distance_Road=
        DATA[
            "Distance_Road"
        ],

    Flood_2019=
        EVENT_LABELS[
            "Flood_2019"
        ],

    Flood_2021=
        EVENT_LABELS[
            "Flood_2021"
        ],

    Flood_2023=
        EVENT_LABELS[
            "Flood_2023"
        ],

    Flood_2026=
        EVENT_LABELS[
            "Flood_2026"
        ],

    Valid_Event_Count=
        VALID_EVENT_COUNT,

    Flood_Event_Count=
        FLOOD_EVENT_COUNT,

    Composite_AnyFlood_All4Valid=
        COMPOSITE_ANY4

)


print(
    "NPZ saved."
)


# ============================================================
# 25. SAVE HUMAN-READABLE CSV
# ============================================================

print(
    "\n========== SAVING MASTER CSV =========="
)

print(
    "This may take a little longer than NPZ..."
)


headers = [

    "Raster_Row",

    "Raster_Column",

    "X",

    "Y",

    "Block_3km",

    "Block_5km",

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

    "Distance_Road",

    "Flood_2019",

    "Flood_2021",

    "Flood_2023",

    "Flood_2026",

    "Valid_Event_Count",

    "Flood_Event_Count",

    "Composite_AnyFlood_All4Valid"

]


with open(
    MASTER_CSV,
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


    for i in range(
        N
    ):


        writer.writerow([

            int(
                sample_rows[
                    i
                ]
            ),

            int(
                sample_cols[
                    i
                ]
            ),

            float(
                X_COORD[
                    i
                ]
            ),

            float(
                Y_COORD[
                    i
                ]
            ),

            int(
                BLOCK_3KM[
                    i
                ]
            ),

            int(
                BLOCK_5KM[
                    i
                ]
            ),

            float(
                DATA[
                    "Elevation"
                ][i]
            ),

            float(
                DATA[
                    "Slope"
                ][i]
            ),

            float(
                ASPECT_SIN[
                    i
                ]
            ),

            float(
                ASPECT_COS[
                    i
                ]
            ),

            int(
                ASPECT_FLAT[
                    i
                ]
            ),

            float(
                DATA[
                    "Profile_Curvature"
                ][i]
            ),

            float(
                DATA[
                    "Tangential_Curvature"
                ][i]
            ),

            float(
                DATA[
                    "TWI"
                ][i]
            ),

            float(
                DATA[
                    "Drainage_Density"
                ][i]
            ),

            float(
                DATA[
                    "Rainfall"
                ][i]
            ),

            int(
                DATA[
                    "LULC"
                ][i]
            ),

            float(
                DATA[
                    "NDVI"
                ][i]
            ),

            int(
                DATA[
                    "HSG"
                ][i]
            ),

            float(
                DATA[
                    "Distance_River"
                ][i]
            ),

            float(
                DATA[
                    "Distance_Road"
                ][i]
            ),

            int(
                EVENT_LABELS[
                    "Flood_2019"
                ][i]
            ),

            int(
                EVENT_LABELS[
                    "Flood_2021"
                ][i]
            ),

            int(
                EVENT_LABELS[
                    "Flood_2023"
                ][i]
            ),

            int(
                EVENT_LABELS[
                    "Flood_2026"
                ][i]
            ),

            int(
                VALID_EVENT_COUNT[
                    i
                ]
            ),

            int(
                FLOOD_EVENT_COUNT[
                    i
                ]
            ),

            int(
                COMPOSITE_ANY4[
                    i
                ]
            )

        ])


        if (
            (i + 1)
            %
            50000
            ==
            0
        ):

            print(
                f"CSV rows written: {i + 1:,}"
            )


print(
    f"CSV rows written: {N:,}"
)


# ============================================================
# 26. DATASET QC SUMMARY
# ============================================================

qc_rows = [

    [
        "Reference_CRS",
        "EPSG:32646"
    ],

    [
        "Reference_Resolution_m",
        30
    ],

    [
        "Original_Common_Analysis_Pixels",
        analysis_pixels
    ],

    [
        "Systematic_Sampling_Step_Pixels",
        SAMPLE_STEP
    ],

    [
        "Systematic_Sampling_Spacing_m",
        SAMPLE_STEP * 30
    ],

    [
        "Sampling_Row_Offset",
        ROW_OFFSET
    ],

    [
        "Sampling_Column_Offset",
        COL_OFFSET
    ],

    [
        "B04_Master_Rows",
        N
    ],

    [
        "Unique_3km_Blocks",
        len(
            np.unique(
                BLOCK_3KM
            )
        )
    ],

    [
        "Unique_5km_Blocks",
        len(
            np.unique(
                BLOCK_5KM
            )
        )
    ],

    [
        "All4Event_Valid_Rows",
        all4_n
    ],

    [
        "AnyFlood_All4Valid_Rows",
        any_flood_n
    ],

    [
        "StableNonFlood_All4Valid_Rows",
        stable_nonflood_n
    ],

    [
        "ML_Conceptual_Factor_Count",
        13
    ],

    [
        "ML_Feature_Column_Count",
        15
    ],

    [
        "Aspect_Encoding",
        "sin + cos + flat_indicator"
    ],

    [
        "Categorical_Handling",
        "LULC/HSG stored as codes; one-hot encoding deferred to B05"
    ],

    [
        "Dataset_Status",
        "PASS"
    ]

]


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


    writer.writerows(
        qc_rows
    )


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print(
    "\n=============================================="
)

print(
    "BUET B04 MULTI-EVENT ML DATASET COMPLETE"
)

print(
    "=============================================="
)

print(
    "\nMaster rows         :",
    f"{N:,}"
)

print(
    "All-4-event valid   :",
    f"{all4_n:,}"
)

print(
    "Any flood >=1 event :",
    f"{any_flood_n:,}"
)

print(
    "Stable non-flood    :",
    f"{stable_nonflood_n:,}"
)

print(
    "\nOutputs:"
)

print(
    MASTER_NPZ
)

print(
    MASTER_CSV
)

print(
    QC_CSV
)

print(
    EVENT_SUMMARY_CSV
)

print(
    EVENT_METADATA_CSV
)

print(
    FEATURE_METADATA_CSV
)

print(
    "\nFINAL STATUS: BUET B04 PASS"
)