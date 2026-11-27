import os
import csv
import gc
import numpy as np
from osgeo import gdal, osr

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
import joblib

# ============================================================
# B09A — FINAL 30 m RF/XGBOOST SUSCEPTIBILITY MAPPING
#
# FINAL LOCKED DESIGN
# ------------------------------------------------------------
# - Labels: B03U urban-aware Sentinel-1 observations
# - Training dataset: B04U, all four-event-valid samples
# - XGBoost: exact full-data B08U explanatory model
# - Preprocessor: exact full-data B08U preprocessor
# - RF: trained on the same full B04U encoded dataset
# - Predictions: complete 30 m common analysis domain
#
# OUTPUTS
# ------------------------------------------------------------
# 1. RF continuous susceptibility score
# 2. XGBoost continuous susceptibility score
# 3. RF-XGB consensus = arithmetic mean
# 4. Inter-model disagreement = |RF - XGB|
# 5. RF common 5-class map
# 6. XGB common 5-class map
# 7. Consensus common 5-class map
# 8. Common class breaks + map statistics + agreement QC
#
# IMPORTANT TERMINOLOGY
# ------------------------------------------------------------
# The 0-1 outputs are called "susceptibility scores".
# They are NOT claimed as calibrated flood probabilities because
# class weighting was used during model fitting.
#
# No MCDA comparison is performed in B09A.
# That follows in B09B after the final ML maps are locked.
# ============================================================


# ============================================================
# 01. PATHS
# ============================================================

BASE = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
)

B04U_NPZ = os.path.join(
    BASE,
    "13_BUET_ML",
    "02B_ML_Dataset_UrbanAware",
    "BUET_B04U_ML_Master_150m.npz",
)

B08U_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "06B_Explainability_UrbanAware",
)

XGB_MODEL_FILE = os.path.join(
    B08U_DIR,
    "BUET_B08U_XGB_FullComposite_Model.joblib",
)

PREPROCESSOR_FILE = os.path.join(
    B08U_DIR,
    "BUET_B08U_XGB_FullComposite_Preprocessor.joblib",
)

FINAL_MASK_FILE = os.path.join(
    BASE,
    "11_FINAL_ANALYSIS_INPUTS",
    "08_Final_Common_Analysis_Mask_30m.tif",
)

OUTPUT_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "07_Final_30m_Maps_UrbanAware",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

RF_MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "BUET_B09A_RF_FullComposite_Model.joblib",
)

RF_SCORE_FILE = os.path.join(
    OUTPUT_DIR,
    "BUET_B09A_RF_SusceptibilityScore_30m.tif",
)

XGB_SCORE_FILE = os.path.join(
    OUTPUT_DIR,
    "BUET_B09A_XGB_SusceptibilityScore_30m.tif",
)

CONSENSUS_SCORE_FILE = os.path.join(
    OUTPUT_DIR,
    "BUET_B09A_RF_XGB_ConsensusScore_30m.tif",
)

DISAGREEMENT_FILE = os.path.join(
    OUTPUT_DIR,
    "BUET_B09A_RF_XGB_AbsoluteDisagreement_30m.tif",
)

RF_CLASS_FILE = os.path.join(
    OUTPUT_DIR,
    "BUET_B09A_RF_Common5Class_30m.tif",
)

XGB_CLASS_FILE = os.path.join(
    OUTPUT_DIR,
    "BUET_B09A_XGB_Common5Class_30m.tif",
)

CONSENSUS_CLASS_FILE = os.path.join(
    OUTPUT_DIR,
    "BUET_B09A_Consensus_Common5Class_30m.tif",
)

BREAKS_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B09A_Common_5Class_Breaks.csv",
)

STATS_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B09A_Final_Map_Statistics.csv",
)

AGREEMENT_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B09A_RF_XGB_Agreement_QC.csv",
)

CLASS_AREA_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B09A_Class_Area_Summary.csv",
)

MODEL_METADATA_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B09A_Model_Metadata.csv",
)


# ============================================================
# 02. FACTOR FILES
#
# Exact canonical names used in B04U.
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
        "01_Rainfall_CHIRPSv3_MAP_1991_2020_Chattogram_30m_STACK.tif",

    "LULC":
        "02_LULC_ESA_WorldCover_2021_Chattogram_30m_STACK.tif",

    "NDVI":
        "03_NDVI_Sentinel2_Median_2021_Chattogram_30m_STACK.tif",

    "HSG":
        "04_Soil_HYSOGs250m_Chattogram_30m_STACK.tif",

    "Distance_River":
        "05_Distance_to_River_HydroRIVERS_UPLAND25_Chattogram_30m_STACK.tif",

    "Distance_Road":
        "06_Distance_to_Road_OSM2021_Chattogram_30m_STACK.tif",
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
    "Distance_Road",
]


# ============================================================
# 03. FILE DISCOVERY
# ============================================================

print(
    "\n========== B09A FILE DISCOVERY =========="
)

for required in [
    B04U_NPZ,
    XGB_MODEL_FILE,
    PREPROCESSOR_FILE,
    FINAL_MASK_FILE,
]:
    if not os.path.exists(required):
        raise Exception(
            "Required file missing:\n" + required
        )

tif_index = {}

for root, dirs, files in os.walk(BASE):

    # Avoid re-indexing B09 outputs while script is being rerun.
    if os.path.abspath(root).startswith(
        os.path.abspath(OUTPUT_DIR)
    ):
        continue

    for filename in files:

        if filename.lower().endswith(
            (".tif", ".tiff")
        ):
            tif_index.setdefault(
                filename.lower(),
                []
            ).append(
                os.path.join(
                    root,
                    filename,
                )
            )

def find_unique_raster(filename):

    matches = tif_index.get(
        filename.lower(),
        [],
    )

    if len(matches) == 0:
        raise Exception(
            "Raster not found:\n" + filename
        )

    if len(matches) > 1:

        print(
            "\nDUPLICATE EXACT FILENAME:",
            filename,
        )

        for p in matches:
            print(" ", p)

        raise Exception(
            "Duplicate canonical raster filename. "
            "Keep/use one canonical copy."
        )

    return matches[0]

FACTOR_PATHS = {
    factor:
        find_unique_raster(
            FACTOR_FILENAMES[factor]
        )
    for factor in FACTOR_ORDER
}

for factor in FACTOR_ORDER:
    print(
        f"{factor:24s} -> "
        f"{FACTOR_PATHS[factor]}"
    )


# ============================================================
# 04. GDAL HELPERS
# ============================================================

gdal.UseExceptions()

NODATA_FLOAT = -9999.0
NODATA_CLASS = 0

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

def create_float_raster(
    path,
    ref_ds,
    band_description,
):

    driver = gdal.GetDriverByName(
        "GTiff"
    )

    ds = driver.Create(
        path,
        ref_ds.RasterXSize,
        ref_ds.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "PREDICTOR=3",
            "BIGTIFF=IF_SAFER",
        ],
    )

    ds.SetGeoTransform(
        ref_ds.GetGeoTransform()
    )

    ds.SetProjection(
        ref_ds.GetProjection()
    )

    band = ds.GetRasterBand(
        1
    )

    band.SetNoDataValue(
        NODATA_FLOAT
    )

    band.SetDescription(
        band_description
    )

    return ds

def create_byte_raster(
    path,
    ref_ds,
    band_description,
):

    driver = gdal.GetDriverByName(
        "GTiff"
    )

    ds = driver.Create(
        path,
        ref_ds.RasterXSize,
        ref_ds.RasterYSize,
        1,
        gdal.GDT_Byte,
        options=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "BIGTIFF=IF_SAFER",
        ],
    )

    ds.SetGeoTransform(
        ref_ds.GetGeoTransform()
    )

    ds.SetProjection(
        ref_ds.GetProjection()
    )

    band = ds.GetRasterBand(
        1
    )

    band.SetNoDataValue(
        NODATA_CLASS
    )

    band.SetDescription(
        band_description
    )

    return ds


# ============================================================
# 05. LOAD REFERENCE MASK + FACTOR DATASETS
# ============================================================

mask_ds = gdal.Open(
    FINAL_MASK_FILE,
    gdal.GA_ReadOnly,
)

if mask_ds is None:
    raise Exception(
        "Cannot open final analysis mask."
    )

WIDTH = mask_ds.RasterXSize
HEIGHT = mask_ds.RasterYSize

mask_band = mask_ds.GetRasterBand(
    1
)

factor_ds = {}
factor_band = {}

for factor in FACTOR_ORDER:

    ds = gdal.Open(
        FACTOR_PATHS[factor],
        gdal.GA_ReadOnly,
    )

    if ds is None:
        raise Exception(
            "Cannot open factor:\n"
            +
            FACTOR_PATHS[factor]
        )

    check_grid(
        ds,
        mask_ds,
        factor,
    )

    factor_ds[factor] = ds
    factor_band[factor] = ds.GetRasterBand(
        1
    )

print(
    "\nGrid:",
    WIDTH,
    "x",
    HEIGHT,
)

print(
    "CRS/grid QC: PASS"
)


# ============================================================
# 06. LOAD B04U TRAINING DATA
# ============================================================

D = np.load(
    B04U_NPZ,
    allow_pickle=False,
)

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

VALID_TRAIN = (
    Y_ALL >= 0
)

X_TRAIN_RAW = X_ALL[
    VALID_TRAIN
]

Y_TRAIN = Y_ALL[
    VALID_TRAIN
]

POS = int(
    np.count_nonzero(
        Y_TRAIN == 1
    )
)

NEG = int(
    np.count_nonzero(
        Y_TRAIN == 0
    )
)

print(
    "\n========== FINAL TRAINING DATA =========="
)

print(
    "Rows      :",
    f"{len(Y_TRAIN):,}"
)

print(
    "Flood     :",
    f"{POS:,}"
)

print(
    "Non-flood :",
    f"{NEG:,}"
)

print(
    "Flood %   :",
    f"{100 * POS / len(Y_TRAIN):.3f}%"
)


# ============================================================
# 07. LOAD LOCKED B08U PREPROCESSOR + XGBOOST MODEL
# ============================================================

print(
    "\n========== LOCKED XGBOOST / PREPROCESSOR =========="
)

PREPROCESSOR = joblib.load(
    PREPROCESSOR_FILE
)

XGB_MODEL = joblib.load(
    XGB_MODEL_FILE
)

X_TRAIN = PREPROCESSOR.transform(
    X_TRAIN_RAW
).astype(
    np.float32
)

print(
    "Raw training columns :",
    X_TRAIN_RAW.shape[1]
)

print(
    "Encoded columns      :",
    X_TRAIN.shape[1]
)

if X_TRAIN.shape[1] != 25:
    raise Exception(
        "Expected 25 encoded columns."
    )


# ============================================================
# 08. TRAIN FINAL FULL-DATA RANDOM FOREST
#
# Same architecture as B05U/B06U/B07U.
# ============================================================

print(
    "\n========== FINAL FULL-DATA RANDOM FOREST =========="
)

RF_MODEL = RandomForestClassifier(

    n_estimators=500,

    max_features="sqrt",

    min_samples_leaf=2,

    class_weight="balanced_subsample",

    bootstrap=True,

    n_jobs=-1,

    random_state=42

)

print(
    "Training RF..."
)

RF_MODEL.fit(
    X_TRAIN,
    Y_TRAIN,
)

print(
    "RF training complete."
)

joblib.dump(
    RF_MODEL,
    RF_MODEL_FILE,
    compress=3,
)


# ============================================================
# 09. TRAINING-DOMAIN CONSISTENCY QC
#
# This is NOT validation.
# It simply ensures both final models produce finite scores.
# ============================================================

rf_train_score = RF_MODEL.predict_proba(
    X_TRAIN
)[:, 1]

xgb_train_score = XGB_MODEL.predict_proba(
    X_TRAIN
)[:, 1]

if not np.all(
    np.isfinite(
        rf_train_score
    )
):
    raise Exception(
        "Non-finite RF training scores."
    )

if not np.all(
    np.isfinite(
        xgb_train_score
    )
):
    raise Exception(
        "Non-finite XGB training scores."
    )

print(
    "Training-score QC: PASS"
)

del X_TRAIN
gc.collect()


# ============================================================
# 10. CREATE OUTPUT RASTERS
# ============================================================

rf_ds = create_float_raster(
    RF_SCORE_FILE,
    mask_ds,
    "RF susceptibility score",
)

xgb_ds = create_float_raster(
    XGB_SCORE_FILE,
    mask_ds,
    "XGBoost susceptibility score",
)

consensus_ds = create_float_raster(
    CONSENSUS_SCORE_FILE,
    mask_ds,
    "RF-XGB mean consensus susceptibility score",
)

disagreement_ds = create_float_raster(
    DISAGREEMENT_FILE,
    mask_ds,
    "Absolute RF-XGB score disagreement",
)

rf_band_out = rf_ds.GetRasterBand(1)
xgb_band_out = xgb_ds.GetRasterBand(1)
consensus_band_out = consensus_ds.GetRasterBand(1)
disagreement_band_out = disagreement_ds.GetRasterBand(1)


# ============================================================
# 11. COUNT VALID PIXELS
# ============================================================

VALID_PIXEL_COUNT = 0

COUNT_BLOCK_ROWS = 256

for yoff in range(
    0,
    HEIGHT,
    COUNT_BLOCK_ROWS,
):

    ysize = min(
        COUNT_BLOCK_ROWS,
        HEIGHT - yoff,
    )

    block = mask_band.ReadAsArray(
        0,
        yoff,
        WIDTH,
        ysize,
    )

    VALID_PIXEL_COUNT += int(
        np.count_nonzero(
            block == 1
        )
    )

print(
    "\nFull 30 m valid pixels:",
    f"{VALID_PIXEL_COUNT:,}"
)

if VALID_PIXEL_COUNT != 4722216:

    print(
        "WARNING: expected 4,722,216 valid pixels."
    )


# ============================================================
# 12. CHUNKED FULL-DOMAIN PREDICTION
#
# Keep only valid RF/XGB scores in RAM for quantile/stat QC.
# 4.72M float32 values ~18 MB/model.
# ============================================================

RF_VALID = np.empty(
    VALID_PIXEL_COUNT,
    dtype=np.float32,
)

XGB_VALID = np.empty(
    VALID_PIXEL_COUNT,
    dtype=np.float32,
)

cursor = 0

CHUNK_ROWS = 96

print(
    "\n========== FULL 30 m PREDICTION =========="
)

for yoff in range(
    0,
    HEIGHT,
    CHUNK_ROWS,
):

    ysize = min(
        CHUNK_ROWS,
        HEIGHT - yoff,
    )

    mask_block = mask_band.ReadAsArray(
        0,
        yoff,
        WIDTH,
        ysize,
    )

    valid = (
        mask_block == 1
    )

    n_valid = int(
        np.count_nonzero(
            valid
        )
    )

    rf_out = np.full(
        (ysize, WIDTH),
        NODATA_FLOAT,
        dtype=np.float32,
    )

    xgb_out = np.full(
        (ysize, WIDTH),
        NODATA_FLOAT,
        dtype=np.float32,
    )

    consensus_out = np.full(
        (ysize, WIDTH),
        NODATA_FLOAT,
        dtype=np.float32,
    )

    disagreement_out = np.full(
        (ysize, WIDTH),
        NODATA_FLOAT,
        dtype=np.float32,
    )

    if n_valid > 0:

        values = {}

        for factor in FACTOR_ORDER:

            arr = factor_band[
                factor
            ].ReadAsArray(
                0,
                yoff,
                WIDTH,
                ysize,
            )

            values[
                factor
            ] = arr[
                valid
            ]

        aspect = values[
            "Aspect"
        ].astype(
            np.float64
        )

        aspect_flat = (
            aspect == -1
        ).astype(
            np.float32
        )

        aspect_sin = np.zeros(
            n_valid,
            dtype=np.float32,
        )

        aspect_cos = np.zeros(
            n_valid,
            dtype=np.float32,
        )

        nonflat = (
            aspect >= 0
        )

        theta = np.deg2rad(
            aspect[
                nonflat
            ]
        )

        aspect_sin[
            nonflat
        ] = np.sin(
            theta
        ).astype(
            np.float32
        )

        aspect_cos[
            nonflat
        ] = np.cos(
            theta
        ).astype(
            np.float32
        )

        X_RAW = np.column_stack([

            values["Elevation"],

            values["Slope"],

            aspect_sin,

            aspect_cos,

            aspect_flat,

            values["Profile_Curvature"],

            values["Tangential_Curvature"],

            values["TWI"],

            values["Drainage_Density"],

            values["Rainfall"],

            values["LULC"],

            values["NDVI"],

            values["HSG"],

            values["Distance_River"],

            values["Distance_Road"]

        ]).astype(
            np.float32
        )

        if not np.all(
            np.isfinite(
                X_RAW
            )
        ):
            raise Exception(
                f"Non-finite factor value in rows "
                f"{yoff}:{yoff + ysize}"
            )

        X_ENC = PREPROCESSOR.transform(
            X_RAW
        ).astype(
            np.float32
        )

        rf_score = RF_MODEL.predict_proba(
            X_ENC
        )[:, 1].astype(
            np.float32
        )

        xgb_score = XGB_MODEL.predict_proba(
            X_ENC
        )[:, 1].astype(
            np.float32
        )

        consensus_score = (
            (
                rf_score
                +
                xgb_score
            )
            /
            2.0
        ).astype(
            np.float32
        )

        disagreement = np.abs(
            rf_score
            -
            xgb_score
        ).astype(
            np.float32
        )

        rf_out[
            valid
        ] = rf_score

        xgb_out[
            valid
        ] = xgb_score

        consensus_out[
            valid
        ] = consensus_score

        disagreement_out[
            valid
        ] = disagreement

        RF_VALID[
            cursor:
            cursor + n_valid
        ] = rf_score

        XGB_VALID[
            cursor:
            cursor + n_valid
        ] = xgb_score

        cursor += n_valid

        del X_RAW
        del X_ENC
        del rf_score
        del xgb_score
        del consensus_score
        del disagreement
        del values

    rf_band_out.WriteArray(
        rf_out,
        0,
        yoff,
    )

    xgb_band_out.WriteArray(
        xgb_out,
        0,
        yoff,
    )

    consensus_band_out.WriteArray(
        consensus_out,
        0,
        yoff,
    )

    disagreement_band_out.WriteArray(
        disagreement_out,
        0,
        yoff,
    )

    del mask_block
    del valid
    del rf_out
    del xgb_out
    del consensus_out
    del disagreement_out

    gc.collect()

    print(
        f"Rows completed: "
        f"{min(yoff + ysize, HEIGHT):,}"
        f"/{HEIGHT:,}"
    )

if cursor != VALID_PIXEL_COUNT:

    raise Exception(
        "Prediction valid-pixel count mismatch."
    )

for ds in [
    rf_ds,
    xgb_ds,
    consensus_ds,
    disagreement_ds,
]:
    ds.FlushCache()

rf_ds = None
xgb_ds = None
consensus_ds = None
disagreement_ds = None

print(
    "Full-domain prediction: PASS"
)


# ============================================================
# 13. COMMON 5-CLASS BREAKS
#
# Pooled RF + XGB quintiles.
# Same thresholds are applied to RF, XGB, and consensus maps.
# ============================================================

print(
    "\n========== COMMON 5-CLASS BREAKS =========="
)

POOLED = np.concatenate([
    RF_VALID,
    XGB_VALID,
])

BREAKS = np.quantile(
    POOLED,
    [
        0.20,
        0.40,
        0.60,
        0.80,
    ],
).astype(
    np.float64
)

del POOLED
gc.collect()

print(
    "Q20:",
    f"{BREAKS[0]:.6f}"
)

print(
    "Q40:",
    f"{BREAKS[1]:.6f}"
)

print(
    "Q60:",
    f"{BREAKS[2]:.6f}"
)

print(
    "Q80:",
    f"{BREAKS[3]:.6f}"
)

with open(
    BREAKS_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "Break",
        "Score",
        "Class_Below",
        "Class_Above",
    ])

    writer.writerows([

        [
            "Q20",
            BREAKS[0],
            "1_Very_Low",
            "2_Low",
        ],

        [
            "Q40",
            BREAKS[1],
            "2_Low",
            "3_Moderate",
        ],

        [
            "Q60",
            BREAKS[2],
            "3_Moderate",
            "4_High",
        ],

        [
            "Q80",
            BREAKS[3],
            "4_High",
            "5_Very_High",
        ],

    ])


# ============================================================
# 14. MAP STATISTICS
# ============================================================

CONSENSUS_VALID = (
    (
        RF_VALID
        +
        XGB_VALID
    )
    /
    2.0
).astype(
    np.float32
)

DISAGREEMENT_VALID = np.abs(
    RF_VALID
    -
    XGB_VALID
).astype(
    np.float32
)

def summarize(
    name,
    values,
):

    return [

        name,

        float(
            np.min(
                values
            )
        ),

        float(
            np.quantile(
                values,
                0.05
            )
        ),

        float(
            np.mean(
                values
            )
        ),

        float(
            np.median(
                values
            )
        ),

        float(
            np.quantile(
                values,
                0.95
            )
        ),

        float(
            np.max(
                values
            )
        ),

        float(
            np.std(
                values
            )
        ),

    ]

stats_rows = [

    summarize(
        "Random_Forest",
        RF_VALID,
    ),

    summarize(
        "XGBoost",
        XGB_VALID,
    ),

    summarize(
        "Consensus_Mean",
        CONSENSUS_VALID,
    ),

    summarize(
        "Absolute_Disagreement",
        DISAGREEMENT_VALID,
    ),

]

with open(
    STATS_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([

        "Map",

        "Minimum",

        "P05",

        "Mean",

        "Median",

        "P95",

        "Maximum",

        "Standard_Deviation",

    ])

    writer.writerows(
        stats_rows
    )


# ============================================================
# 15. CONTINUOUS RF-XGB AGREEMENT
# ============================================================

CORRELATION = float(
    np.corrcoef(
        RF_VALID,
        XGB_VALID,
    )[0, 1]
)

MAE = float(
    np.mean(
        np.abs(
            RF_VALID
            -
            XGB_VALID
        )
    )
)

RMSE = float(
    np.sqrt(
        np.mean(
            (
                RF_VALID
                -
                XGB_VALID
            )
            ** 2
        )
    )
)

print(
    "\n========== RF-XGB CONTINUOUS AGREEMENT =========="
)

print(
    "Pearson r :",
    f"{CORRELATION:.6f}"
)

print(
    "MAE       :",
    f"{MAE:.6f}"
)

print(
    "RMSE      :",
    f"{RMSE:.6f}"
)

print(
    "Mean |diff|:",
    f"{np.mean(DISAGREEMENT_VALID):.6f}"
)

print(
    "P95 |diff| :",
    f"{np.quantile(DISAGREEMENT_VALID, 0.95):.6f}"
)


# ============================================================
# 16. CLASSIFY VALID-SCORE ARRAYS
# ============================================================

def classify_values(
    values,
    breaks,
):

    # np.digitize returns 0..4.
    # Add 1 => classes 1..5.
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

RF_CLASS_VALID = classify_values(
    RF_VALID,
    BREAKS,
)

XGB_CLASS_VALID = classify_values(
    XGB_VALID,
    BREAKS,
)

CONSENSUS_CLASS_VALID = classify_values(
    CONSENSUS_VALID,
    BREAKS,
)

EXACT_AGREEMENT = float(
    np.mean(
        RF_CLASS_VALID
        ==
        XGB_CLASS_VALID
    )
)

WITHIN_ONE = float(
    np.mean(
        np.abs(
            RF_CLASS_VALID.astype(
                np.int16
            )
            -
            XGB_CLASS_VALID.astype(
                np.int16
            )
        )
        <=
        1
    )
)

KAPPA = float(
    cohen_kappa_score(
        RF_CLASS_VALID,
        XGB_CLASS_VALID,
        labels=[
            1,
            2,
            3,
            4,
            5,
        ],
    )
)

print(
    "\n========== RF-XGB 5-CLASS AGREEMENT =========="
)

print(
    "Exact agreement :",
    f"{100 * EXACT_AGREEMENT:.2f}%"
)

print(
    "Within +/-1     :",
    f"{100 * WITHIN_ONE:.2f}%"
)

print(
    "Cohen kappa     :",
    f"{KAPPA:.6f}"
)

with open(
    AGREEMENT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "Metric",
        "Value",
    ])

    writer.writerows([

        [
            "Valid_Pixel_N",
            VALID_PIXEL_COUNT,
        ],

        [
            "RF_XGB_Pearson_r",
            CORRELATION,
        ],

        [
            "RF_XGB_MAE",
            MAE,
        ],

        [
            "RF_XGB_RMSE",
            RMSE,
        ],

        [
            "Mean_Absolute_Disagreement",
            float(
                np.mean(
                    DISAGREEMENT_VALID
                )
            ),
        ],

        [
            "P95_Absolute_Disagreement",
            float(
                np.quantile(
                    DISAGREEMENT_VALID,
                    0.95
                )
            ),
        ],

        [
            "Common5Class_Exact_Agreement",
            EXACT_AGREEMENT,
        ],

        [
            "Common5Class_WithinOneClass",
            WITHIN_ONE,
        ],

        [
            "Common5Class_Cohen_Kappa",
            KAPPA,
        ],

    ])


# ============================================================
# 17. CLASS AREA SUMMARY
# ============================================================

PIXEL_AREA_KM2 = (
    30.0
    *
    30.0
    /
    1e6
)

CLASS_NAMES = {
    1:
        "Very Low",
    2:
        "Low",
    3:
        "Moderate",
    4:
        "High",
    5:
        "Very High",
}

class_area_rows = []

for map_name, classes in [

    (
        "Random_Forest",
        RF_CLASS_VALID,
    ),

    (
        "XGBoost",
        XGB_CLASS_VALID,
    ),

    (
        "Consensus",
        CONSENSUS_CLASS_VALID,
    ),

]:

    for class_id in range(
        1,
        6,
    ):

        count = int(
            np.count_nonzero(
                classes
                ==
                class_id
            )
        )

        class_area_rows.append([

            map_name,

            class_id,

            CLASS_NAMES[
                class_id
            ],

            count,

            count
            *
            PIXEL_AREA_KM2,

            100
            *
            count
            /
            VALID_PIXEL_COUNT,

        ])

with open(
    CLASS_AREA_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([

        "Map",

        "Class_ID",

        "Class_Name",

        "Pixel_Count",

        "Area_km2",

        "Percent_of_Valid_Domain",

    ])

    writer.writerows(
        class_area_rows
    )


# ============================================================
# 18. SECOND PASS — WRITE COMMON 5-CLASS RASTERS
# ============================================================

rf_score_ds = gdal.Open(
    RF_SCORE_FILE,
    gdal.GA_ReadOnly,
)

xgb_score_ds = gdal.Open(
    XGB_SCORE_FILE,
    gdal.GA_ReadOnly,
)

cons_score_ds = gdal.Open(
    CONSENSUS_SCORE_FILE,
    gdal.GA_ReadOnly,
)

rf_score_band = rf_score_ds.GetRasterBand(1)
xgb_score_band = xgb_score_ds.GetRasterBand(1)
cons_score_band = cons_score_ds.GetRasterBand(1)

rf_class_ds = create_byte_raster(
    RF_CLASS_FILE,
    mask_ds,
    "RF common 5-class susceptibility",
)

xgb_class_ds = create_byte_raster(
    XGB_CLASS_FILE,
    mask_ds,
    "XGBoost common 5-class susceptibility",
)

cons_class_ds = create_byte_raster(
    CONSENSUS_CLASS_FILE,
    mask_ds,
    "Consensus common 5-class susceptibility",
)

rf_class_band = rf_class_ds.GetRasterBand(1)
xgb_class_band = xgb_class_ds.GetRasterBand(1)
cons_class_band = cons_class_ds.GetRasterBand(1)

print(
    "\n========== WRITING 5-CLASS RASTERS =========="
)

for yoff in range(
    0,
    HEIGHT,
    CHUNK_ROWS,
):

    ysize = min(
        CHUNK_ROWS,
        HEIGHT - yoff,
    )

    mask_block = mask_band.ReadAsArray(
        0,
        yoff,
        WIDTH,
        ysize,
    )

    valid = (
        mask_block == 1
    )

    rf_score_block = rf_score_band.ReadAsArray(
        0,
        yoff,
        WIDTH,
        ysize,
    )

    xgb_score_block = xgb_score_band.ReadAsArray(
        0,
        yoff,
        WIDTH,
        ysize,
    )

    cons_score_block = cons_score_band.ReadAsArray(
        0,
        yoff,
        WIDTH,
        ysize,
    )

    rf_class_out = np.zeros(
        (ysize, WIDTH),
        dtype=np.uint8,
    )

    xgb_class_out = np.zeros(
        (ysize, WIDTH),
        dtype=np.uint8,
    )

    cons_class_out = np.zeros(
        (ysize, WIDTH),
        dtype=np.uint8,
    )

    rf_class_out[
        valid
    ] = classify_values(
        rf_score_block[
            valid
        ],
        BREAKS,
    )

    xgb_class_out[
        valid
    ] = classify_values(
        xgb_score_block[
            valid
        ],
        BREAKS,
    )

    cons_class_out[
        valid
    ] = classify_values(
        cons_score_block[
            valid
        ],
        BREAKS,
    )

    rf_class_band.WriteArray(
        rf_class_out,
        0,
        yoff,
    )

    xgb_class_band.WriteArray(
        xgb_class_out,
        0,
        yoff,
    )

    cons_class_band.WriteArray(
        cons_class_out,
        0,
        yoff,
    )

    print(
        f"Class rows completed: "
        f"{min(yoff + ysize, HEIGHT):,}"
        f"/{HEIGHT:,}"
    )

for ds in [
    rf_class_ds,
    xgb_class_ds,
    cons_class_ds,
]:
    ds.FlushCache()

rf_class_ds = None
xgb_class_ds = None
cons_class_ds = None

rf_score_ds = None
xgb_score_ds = None
cons_score_ds = None


# ============================================================
# 19. MODEL METADATA
# ============================================================

with open(
    MODEL_METADATA_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "Item",
        "Value",
    ])

    writer.writerows([

        [
            "Final_Label_Version",
            "B03U urban-aware Sentinel-1",
        ],

        [
            "Training_Dataset",
            "B04U all-four-event-valid composite",
        ],

        [
            "Training_N",
            len(Y_TRAIN),
        ],

        [
            "Positive_N",
            POS,
        ],

        [
            "Negative_N",
            NEG,
        ],

        [
            "Final_RF",
            "500 trees; sqrt max_features; min_samples_leaf=2; balanced_subsample",
        ],

        [
            "Final_XGBoost",
            "Exact locked B08U full-data explanatory model",
        ],

        [
            "Score_Interpretation",
            "0-1 susceptibility score; not claimed as calibrated probability",
        ],

        [
            "Consensus",
            "Arithmetic mean of RF and XGBoost scores",
        ],

        [
            "Disagreement",
            "Absolute RF-XGBoost score difference",
        ],

        [
            "Class_Method",
            "Common quintile thresholds from pooled RF+XGBoost valid-domain scores",
        ],

        [
            "Valid_30m_Pixels",
            VALID_PIXEL_COUNT,
        ],

        [
            "Pixel_Size_m",
            30,
        ],

    ])


# ============================================================
# 20. CLEANUP + FINAL
# ============================================================

mask_ds = None

for factor in FACTOR_ORDER:
    factor_band[factor] = None
    factor_ds[factor] = None

print(
    "\n=============================================="
)

print(
    "BUET B09A FINAL 30 m MAPPING COMPLETE"
)

print(
    "=============================================="
)

print(
    "\nOutputs:"
)

for path in [

    RF_SCORE_FILE,

    XGB_SCORE_FILE,

    CONSENSUS_SCORE_FILE,

    DISAGREEMENT_FILE,

    RF_CLASS_FILE,

    XGB_CLASS_FILE,

    CONSENSUS_CLASS_FILE,

    BREAKS_CSV,

    STATS_CSV,

    AGREEMENT_CSV,

    CLASS_AREA_CSV,

    MODEL_METADATA_CSV,

    RF_MODEL_FILE,

]:
    print(path)

print(
    "\nFINAL STATUS: BUET B09A PASS"
)

print(
    "=============================================="
)
