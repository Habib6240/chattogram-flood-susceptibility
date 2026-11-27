import os
import csv
import numpy as np
from osgeo import gdal, osr

# ============================================================
# B04U — URBAN-AWARE MULTI-EVENT ML MASTER DATASET
#
# Same modelling design as B04.
# ONLY change: event labels now come from B03U urban-aware masks.
#
# One spatial sample = one row.
# Static predictors = 13 factors (15 ML columns after aspect encoding).
# Event labels = 2019, 2021, 2023, 2026.
# Sampling = 150 m systematic grid (every 5th 30 m pixel).
# ============================================================

BASE = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
)

OUTPUT_DIR = os.path.join(
    BASE, "13_BUET_ML", "02B_ML_Dataset_UrbanAware"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MASTER_NPZ = os.path.join(
    OUTPUT_DIR, "BUET_B04U_ML_Master_150m.npz"
)
MASTER_CSV = os.path.join(
    OUTPUT_DIR, "BUET_B04U_ML_Master_150m.csv"
)
QC_CSV = os.path.join(
    OUTPUT_DIR, "BUET_B04U_Dataset_QC.csv"
)
EVENT_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR, "BUET_B04U_Event_Label_Summary.csv"
)
FEATURE_METADATA_CSV = os.path.join(
    OUTPUT_DIR, "BUET_B04U_Feature_Definitions.csv"
)
EVENT_METADATA_CSV = os.path.join(
    OUTPUT_DIR, "BUET_B04U_Event_Metadata.csv"
)

FINAL_MASK_FILE = os.path.join(
    BASE,
    "11_FINAL_ANALYSIS_INPUTS",
    "08_Final_Common_Analysis_Mask_30m.tif"
)

S1_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "01B_Sentinel1_Event_Masks_UrbanAware"
)

EVENT_FILES = {
    "Flood_2019": os.path.join(
        S1_DIR, "BUET_B03U_E03_2019_Jul_FloodMask30m.tif"
    ),
    "Flood_2021": os.path.join(
        S1_DIR, "BUET_B03U_E04_2021_Jul_FloodMask30m.tif"
    ),
    "Flood_2023": os.path.join(
        S1_DIR, "BUET_B03U_E05_2023_Aug_FloodMask30m.tif"
    ),
    "Flood_2026": os.path.join(
        S1_DIR, "BUET_B03U_E07_2026_Jul_FloodMask30m.tif"
    ),
}

FACTOR_FILENAMES = {
    "Elevation": "DEM_Clipped_UTM46N.tif",
    "Slope": "Slope_deg_30m.tif",
    "Aspect": "Aspect_deg_30_FLATFIX.tif",
    "Profile_Curvature": "Profile_Curvature_30m.tif",
    "Tangential_Curvature": "Tangential_Curvature_30m.tif",
    "TWI": "TWI_Chittagong_30m_STACK.tif",
    "Drainage_Density": "Drainage_Density_Chittagong_30m_STACK.tif",
    "Rainfall": "01_Rainfall_CHIRPSv3_MAP_1991_2020_Chattogram_30m_STACK.tif",
    "LULC": "02_LULC_ESA_WorldCover_2021_Chattogram_30m_STACK.tif",
    "NDVI": "03_NDVI_Sentinel2_Median_2021_Chattogram_30m_STACK.tif",
    "HSG": "04_Soil_HYSOGs250m_Chattogram_30m_STACK.tif",
    "Distance_River": "05_Distance_to_River_HydroRIVERS_UPLAND25_Chattogram_30m_STACK.tif",
    "Distance_Road": "06_Distance_to_Road_OSM2021_Chattogram_30m_STACK.tif",
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

print("\n========== B04U FILE DISCOVERY ==========")

# Build exact-filename TIFF index under BASE.
tif_index = {}
for root, dirs, files in os.walk(BASE):
    for filename in files:
        if filename.lower().endswith((".tif", ".tiff")):
            tif_index.setdefault(filename.lower(), []).append(
                os.path.join(root, filename)
            )

def find_unique_raster(filename):
    matches = tif_index.get(filename.lower(), [])
    if len(matches) == 0:
        raise Exception("Raster not found:\n" + filename)
    if len(matches) > 1:
        print("\nDUPLICATE FILE FOUND:", filename)
        for p in matches:
            print(" ", p)
        raise Exception(
            "Duplicate exact raster filename. Use one canonical copy."
        )
    return matches[0]

FACTOR_PATHS = {
    factor: find_unique_raster(FACTOR_FILENAMES[factor])
    for factor in FACTOR_ORDER
}

for factor in FACTOR_ORDER:
    print(f"{factor:24s} -> {FACTOR_PATHS[factor]}")

for path in [FINAL_MASK_FILE] + list(EVENT_FILES.values()):
    if not os.path.exists(path):
        raise Exception("Required file missing:\n" + path)

print("\nDirect-path file QC : PASS")

def read_raster(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise Exception("Cannot open:\n" + path)

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()

    numeric = arr.astype(np.float64)
    valid = np.isfinite(numeric)

    if nodata is not None:
        if np.isnan(nodata):
            valid &= ~np.isnan(numeric)
        else:
            valid &= (arr != nodata)

    return ds, arr, valid, nodata

def same_projection(wkt1, wkt2):
    s1 = osr.SpatialReference()
    s2 = osr.SpatialReference()
    s1.ImportFromWkt(wkt1)
    s2.ImportFromWkt(wkt2)
    return bool(s1.IsSame(s2))

def check_grid(ds, ref_ds, name):
    if (
        ds.RasterXSize != ref_ds.RasterXSize
        or ds.RasterYSize != ref_ds.RasterYSize
    ):
        raise Exception(name + ": dimension mismatch.")

    if not np.allclose(
        ds.GetGeoTransform(),
        ref_ds.GetGeoTransform(),
        atol=0.01,
    ):
        raise Exception(name + ": geotransform mismatch.")

    if not same_projection(ds.GetProjection(), ref_ds.GetProjection()):
        raise Exception(name + ": CRS mismatch.")

# ============================================================
# Analysis mask
# ============================================================

mask_ds, mask_arr, mask_valid, mask_nodata = read_raster(
    FINAL_MASK_FILE
)
analysis_mask = (mask_arr == 1)
analysis_pixels = int(np.count_nonzero(analysis_mask))

print("\n========== ANALYSIS MASK ==========")
print("Analysis pixels :", f"{analysis_pixels:,}")

if analysis_pixels != 4722216:
    print("WARNING: expected 4,722,216 final-analysis pixels.")

# ============================================================
# 150 m systematic sampling
# ============================================================

SAMPLE_STEP = 5
ROW_OFFSET = 2
COL_OFFSET = 2

systematic_grid = np.zeros(
    analysis_mask.shape,
    dtype=bool,
)

systematic_grid[
    ROW_OFFSET::SAMPLE_STEP,
    COL_OFFSET::SAMPLE_STEP,
] = True

sample_mask = analysis_mask & systematic_grid
sample_rows, sample_cols = np.where(sample_mask)
N = len(sample_rows)

print("\n========== 150 m SYSTEMATIC SAMPLE ==========")
print("Sample N :", f"{N:,}")

gt = mask_ds.GetGeoTransform()

X_COORD = (
    gt[0]
    + (sample_cols + 0.5) * gt[1]
    + (sample_rows + 0.5) * gt[2]
)

Y_COORD = (
    gt[3]
    + (sample_cols + 0.5) * gt[4]
    + (sample_rows + 0.5) * gt[5]
)

# 3 km and 5 km block IDs
xmin = gt[0]
ytop = gt[3]

block3_col = np.floor((X_COORD - xmin) / 3000.0).astype(np.int32)
block3_row = np.floor((ytop - Y_COORD) / 3000.0).astype(np.int32)
BLOCK_3KM = (
    block3_row.astype(np.int64) * 10000
    + block3_col.astype(np.int64)
)

block5_col = np.floor((X_COORD - xmin) / 5000.0).astype(np.int32)
block5_row = np.floor((ytop - Y_COORD) / 5000.0).astype(np.int32)
BLOCK_5KM = (
    block5_row.astype(np.int64) * 10000
    + block5_col.astype(np.int64)
)

print("Unique 3 km blocks:", len(np.unique(BLOCK_3KM)))
print("Unique 5 km blocks:", len(np.unique(BLOCK_5KM)))

# ============================================================
# Static factors
# ============================================================

DATA = {}

print("\n========== EXTRACTING STATIC FACTORS ==========")

for factor in FACTOR_ORDER:
    ds, arr, valid, nodata = read_raster(FACTOR_PATHS[factor])
    check_grid(ds, mask_ds, factor)

    invalid = int(np.count_nonzero(~valid[sample_mask]))
    if invalid > 0:
        raise Exception(
            f"{factor}: {invalid} invalid values inside B04U sample."
        )

    values = arr[sample_mask]

    if factor in ["LULC", "HSG"]:
        DATA[factor] = values.astype(np.int16)
    else:
        DATA[factor] = values.astype(np.float32)

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
# Aspect encoding
# ============================================================

aspect = DATA["Aspect"].astype(np.float64)

ASPECT_FLAT = (aspect == -1).astype(np.uint8)
ASPECT_SIN = np.zeros(N, dtype=np.float32)
ASPECT_COS = np.zeros(N, dtype=np.float32)

nonflat = (aspect >= 0)
theta = np.deg2rad(aspect[nonflat])

ASPECT_SIN[nonflat] = np.sin(theta).astype(np.float32)
ASPECT_COS[nonflat] = np.cos(theta).astype(np.float32)

print("\n========== ASPECT ENCODING ==========")
print("Flat samples    :", f"{np.count_nonzero(ASPECT_FLAT == 1):,}")
print("Non-flat samples:", f"{np.count_nonzero(ASPECT_FLAT == 0):,}")

# ============================================================
# Urban-aware event labels
# ============================================================

EVENT_LABELS = {}
event_summary_rows = []

print("\n========== EXTRACTING B03U EVENT LABELS ==========")

for event_name, event_path in EVENT_FILES.items():
    ds, arr, valid_raw, nodata = read_raster(event_path)
    check_grid(ds, mask_ds, event_name)

    valid_label = (
        valid_raw
        & ((arr == 0) | (arr == 1))
    )

    labels = np.full(N, -1, dtype=np.int8)

    sample_values = arr[sample_mask]
    sample_valid = valid_label[sample_mask]

    labels[sample_valid] = sample_values[
        sample_valid
    ].astype(np.int8)

    EVENT_LABELS[event_name] = labels

    valid_n = int(np.count_nonzero(labels >= 0))
    flood_n = int(np.count_nonzero(labels == 1))
    nonflood_n = int(np.count_nonzero(labels == 0))
    invalid_n = int(np.count_nonzero(labels == -1))
    flood_pct = (
        100 * flood_n / valid_n
        if valid_n > 0
        else np.nan
    )

    event_summary_rows.append([
        event_name,
        valid_n,
        flood_n,
        nonflood_n,
        invalid_n,
        flood_pct,
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
# Composite label
# ============================================================

EVENT_MATRIX = np.column_stack([
    EVENT_LABELS["Flood_2019"],
    EVENT_LABELS["Flood_2021"],
    EVENT_LABELS["Flood_2023"],
    EVENT_LABELS["Flood_2026"],
]).astype(np.int8)

VALID_EVENT_COUNT = np.sum(
    EVENT_MATRIX >= 0,
    axis=1,
).astype(np.uint8)

FLOOD_EVENT_COUNT = np.sum(
    EVENT_MATRIX == 1,
    axis=1,
).astype(np.uint8)

ALL4_VALID = (VALID_EVENT_COUNT == 4)

COMPOSITE_ANY4 = np.full(
    N,
    -1,
    dtype=np.int8,
)

COMPOSITE_ANY4[ALL4_VALID] = (
    FLOOD_EVENT_COUNT[ALL4_VALID] > 0
).astype(np.int8)

all4_n = int(np.count_nonzero(COMPOSITE_ANY4 >= 0))
any_flood_n = int(np.count_nonzero(COMPOSITE_ANY4 == 1))
stable_nonflood_n = int(np.count_nonzero(COMPOSITE_ANY4 == 0))

print("\n========== B03U FOUR-EVENT COMPOSITE ==========")
print("All-4-event valid :", f"{all4_n:,}")
print("Any flood >=1    :", f"{any_flood_n:,}")
print("Stable non-flood :", f"{stable_nonflood_n:,}")
print(
    "Composite flood %:",
    f"{100 * any_flood_n / all4_n:.3f}%"
)

# ============================================================
# Categorical QC
# ============================================================

print("\n========== CATEGORICAL QC ==========")
print("LULC codes:", np.unique(DATA["LULC"]).tolist())
print("HSG codes :", np.unique(DATA["HSG"]).tolist())

# ============================================================
# Save summaries
# ============================================================

with open(
    EVENT_SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "Event",
        "Valid_Sample_N",
        "Flood_N",
        "NonFlood_N",
        "Invalid_N",
        "Flood_Percent_of_Valid",
    ])
    writer.writerows(event_summary_rows)

with open(
    EVENT_METADATA_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "Event",
        "Observation_Type",
        "Pre_Date",
        "Observed_Date",
        "Sensor_Track",
        "Label_Version",
    ])
    writer.writerows([
        ["2019", "DURING_EVENT", "2019-07-02", "2019-07-14", "S1A_ASC_R41", "B03U urban-aware"],
        ["2021", "DURING_EVENT", "2021-07-17", "2021-07-29", "S1A_DESC_R77", "B03U urban-aware"],
        ["2023", "POST_PEAK_PERSISTENT_INUNDATION", "2023-07-31", "2023-08-12", "S1A_DESC_R77", "B03U urban-aware"],
        ["2026", "DURING_EVENT", "2026-06-28", "2026-07-10", "S1D_DESC_R77", "B03U urban-aware"],
    ])

feature_rows = [
    ["Elevation", "Continuous", "Raw"],
    ["Slope", "Continuous", "Raw"],
    ["Aspect_sin", "Circular", "Derived"],
    ["Aspect_cos", "Circular", "Derived"],
    ["Aspect_flat", "Binary", "Derived"],
    ["Profile_Curvature", "Continuous", "Raw"],
    ["Tangential_Curvature", "Continuous", "Raw"],
    ["TWI", "Continuous", "Raw"],
    ["Drainage_Density", "Continuous", "Raw"],
    ["Rainfall", "Continuous", "Raw"],
    ["LULC", "Categorical", "One-hot in modelling"],
    ["NDVI", "Continuous", "Raw"],
    ["HSG", "Categorical", "One-hot in modelling"],
    ["Distance_River", "Continuous", "Raw"],
    ["Distance_Road", "Continuous", "Raw"],
]

with open(
    FEATURE_METADATA_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "ML_Feature",
        "Variable_Type",
        "Encoding",
    ])
    writer.writerows(feature_rows)

# ============================================================
# Save NPZ
# ============================================================

print("\n========== SAVING B04U NPZ ==========")

np.savez_compressed(
    MASTER_NPZ,
    Raster_Row=sample_rows.astype(np.int32),
    Raster_Column=sample_cols.astype(np.int32),
    X=X_COORD.astype(np.float64),
    Y=Y_COORD.astype(np.float64),
    Block_3km=BLOCK_3KM,
    Block_5km=BLOCK_5KM,
    Elevation=DATA["Elevation"],
    Slope=DATA["Slope"],
    Aspect_sin=ASPECT_SIN,
    Aspect_cos=ASPECT_COS,
    Aspect_flat=ASPECT_FLAT,
    Profile_Curvature=DATA["Profile_Curvature"],
    Tangential_Curvature=DATA["Tangential_Curvature"],
    TWI=DATA["TWI"],
    Drainage_Density=DATA["Drainage_Density"],
    Rainfall=DATA["Rainfall"],
    LULC=DATA["LULC"],
    NDVI=DATA["NDVI"],
    HSG=DATA["HSG"],
    Distance_River=DATA["Distance_River"],
    Distance_Road=DATA["Distance_Road"],
    Flood_2019=EVENT_LABELS["Flood_2019"],
    Flood_2021=EVENT_LABELS["Flood_2021"],
    Flood_2023=EVENT_LABELS["Flood_2023"],
    Flood_2026=EVENT_LABELS["Flood_2026"],
    Valid_Event_Count=VALID_EVENT_COUNT,
    Flood_Event_Count=FLOOD_EVENT_COUNT,
    Composite_AnyFlood_All4Valid=COMPOSITE_ANY4,
)

# ============================================================
# Save human-readable CSV
# ============================================================

print("\n========== SAVING B04U CSV ==========")

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
    "Composite_AnyFlood_All4Valid",
]

with open(
    MASTER_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow(headers)

    for i in range(N):
        writer.writerow([
            int(sample_rows[i]),
            int(sample_cols[i]),
            float(X_COORD[i]),
            float(Y_COORD[i]),
            int(BLOCK_3KM[i]),
            int(BLOCK_5KM[i]),
            float(DATA["Elevation"][i]),
            float(DATA["Slope"][i]),
            float(ASPECT_SIN[i]),
            float(ASPECT_COS[i]),
            int(ASPECT_FLAT[i]),
            float(DATA["Profile_Curvature"][i]),
            float(DATA["Tangential_Curvature"][i]),
            float(DATA["TWI"][i]),
            float(DATA["Drainage_Density"][i]),
            float(DATA["Rainfall"][i]),
            int(DATA["LULC"][i]),
            float(DATA["NDVI"][i]),
            int(DATA["HSG"][i]),
            float(DATA["Distance_River"][i]),
            float(DATA["Distance_Road"][i]),
            int(EVENT_LABELS["Flood_2019"][i]),
            int(EVENT_LABELS["Flood_2021"][i]),
            int(EVENT_LABELS["Flood_2023"][i]),
            int(EVENT_LABELS["Flood_2026"][i]),
            int(VALID_EVENT_COUNT[i]),
            int(FLOOD_EVENT_COUNT[i]),
            int(COMPOSITE_ANY4[i]),
        ])

        if (i + 1) % 50000 == 0:
            print("CSV rows written:", f"{i+1:,}")

print("CSV rows written:", f"{N:,}")

# ============================================================
# QC summary
# ============================================================

with open(
    QC_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow(["QC_Metric", "Value"])
    writer.writerows([
        ["Reference_CRS", "EPSG:32646"],
        ["Reference_Resolution_m", 30],
        ["Original_Common_Analysis_Pixels", analysis_pixels],
        ["Systematic_Sampling_Spacing_m", 150],
        ["B04U_Master_Rows", N],
        ["Unique_3km_Blocks", len(np.unique(BLOCK_3KM))],
        ["Unique_5km_Blocks", len(np.unique(BLOCK_5KM))],
        ["All4Event_Valid_Rows", all4_n],
        ["AnyFlood_All4Valid_Rows", any_flood_n],
        ["StableNonFlood_All4Valid_Rows", stable_nonflood_n],
        ["Composite_Flood_Percent", 100 * any_flood_n / all4_n],
        ["Label_Version", "B03U urban-aware"],
        ["Dataset_Status", "PASS"],
    ])

print("\n==============================================")
print("BUET B04U URBAN-AWARE ML DATASET COMPLETE")
print("==============================================")
print("Master rows       :", f"{N:,}")
print("All-4-event valid :", f"{all4_n:,}")
print("Any flood >=1     :", f"{any_flood_n:,}")
print("Stable non-flood  :", f"{stable_nonflood_n:,}")
print(
    "Composite flood % :",
    f"{100 * any_flood_n / all4_n:.3f}%"
)
print("\nFINAL STATUS: BUET B04U PASS")
