from qgis.core import QgsProject
from osgeo import gdal
import numpy as np
import os
import csv

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "12_DU_MCDA",
    "00_Input_Standardization_QC"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

MASK_NAME = "08_Final_Common_Analysis_Mask_30m"

CONTINUOUS_FACTORS = [
    "DEM_Clipped_UTM46N",
    "Slope_deg_30m",
    "Profile_Curvature_30m",
    "Tangential_Curvature_30m",
    "TWI_Chittagong_30m_STACK",
    "Drainage_Density_Chittagong_30m_STACK",
    "01_Rainfall_CHIRPSv3_MAP_1991_2020_Chattogram_30m_STACK",
    "03_NDVI_Sentinel2_Median_2021_Chattogram_30m_STACK",
    "05_Distance_to_River_HydroRIVERS_UPLAND25_Chattogram_30m_STACK",
    "06_Distance_to_Road_OSM2021_Chattogram_30m_STACK"
]

ASPECT_NAME = "Aspect_deg_30_FLATFIX"

LULC_NAME = (
    "02_LULC_ESA_WorldCover_2021_"
    "Chattogram_30m_STACK"
)

HSG_NAME = (
    "04_Soil_HYSOGs250m_"
    "Chattogram_30m_STACK"
)


# ============================================================
# OUTPUTS
# ============================================================

CONT_STATS = os.path.join(
    OUTPUT_DIR,
    "08A_Continuous_Factor_Statistics.csv"
)

ASPECT_FREQ = os.path.join(
    OUTPUT_DIR,
    "08B_Aspect_Class_Frequency.csv"
)

LULC_FREQ = os.path.join(
    OUTPUT_DIR,
    "08C_LULC_Class_Frequency.csv"
)

HSG_FREQ = os.path.join(
    OUTPUT_DIR,
    "08D_HSG_Class_Frequency.csv"
)


# ============================================================
# HELPERS
# ============================================================

project = QgsProject.instance()


def get_layer(name):

    layers = project.mapLayersByName(name)

    if not layers:
        raise Exception(
            "Layer not found: " + name
        )

    return layers[0]


def read_raster(name):

    lyr = get_layer(name)

    ds = gdal.Open(
        lyr.source()
    )

    if ds is None:
        raise Exception(
            "Could not open: " + name
        )

    band = ds.GetRasterBand(1)

    arr = band.ReadAsArray()

    nodata = band.GetNoDataValue()

    valid = np.isfinite(arr)

    if nodata is not None:

        if np.isnan(nodata):

            valid &= ~np.isnan(arr)

        else:

            valid &= (
                arr != nodata
            )

    return arr, valid


# ============================================================
# FINAL ANALYSIS MASK
# ============================================================

mask_arr, mask_valid = read_raster(
    MASK_NAME
)

analysis_mask = (
    mask_valid
    &
    (mask_arr == 1)
)

n_analysis = np.count_nonzero(
    analysis_mask
)

print(
    "\n========== FINAL ANALYSIS DOMAIN =========="
)

print(
    "Final analysis pixels :",
    f"{n_analysis:,}"
)

print(
    "Expected              :",
    "4,722,216"
)

if n_analysis != 4722216:

    print(
        "WARNING: Pixel count differs "
        "from final-mask audit."
    )


# ============================================================
# CONTINUOUS FACTOR STATISTICS
# ============================================================

stats_rows = []

print(
    "\n========== CONTINUOUS FACTOR STATISTICS =========="
)

for name in CONTINUOUS_FACTORS:

    arr, valid = read_raster(
        name
    )

    use = (
        analysis_mask
        &
        valid
    )

    values = arr[use].astype(
        np.float64
    )

    if len(values) != n_analysis:
        raise Exception(
            name
            + " is not valid across the complete final mask."
        )

    percentiles = np.percentile(
        values,
        [
            1,
            5,
            10,
            25,
            50,
            75,
            90,
            95,
            99
        ]
    )

    row = [
        name,
        len(values),
        np.min(values),
        np.mean(values),
        np.std(
            values,
            ddof=1
        ),
        percentiles[0],
        percentiles[1],
        percentiles[2],
        percentiles[3],
        percentiles[4],
        percentiles[5],
        percentiles[6],
        percentiles[7],
        percentiles[8],
        np.max(values)
    ]

    stats_rows.append(
        row
    )

    print(
        f"{name:68s}"
        f" Min={np.min(values):10.4f}"
        f" P50={np.median(values):10.4f}"
        f" Max={np.max(values):10.4f}"
    )


with open(
    CONT_STATS,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Factor",
        "N",
        "Minimum",
        "Mean",
        "SD",
        "P01",
        "P05",
        "P10",
        "P25",
        "P50_Median",
        "P75",
        "P90",
        "P95",
        "P99",
        "Maximum"
    ])

    writer.writerows(
        stats_rows
    )


# ============================================================
# ASPECT FREQUENCY
# Flat + 8 directions
# ============================================================

aspect, aspect_valid = read_raster(
    ASPECT_NAME
)

use = (
    analysis_mask
    &
    aspect_valid
)

aspect_vals = aspect[use].astype(
    np.float64
)

aspect_labels = np.full(
    len(aspect_vals),
    "",
    dtype=object
)

# -1 = explicit flat
flat = np.isclose(
    aspect_vals,
    -1.0,
    atol=1e-6
)

aspect_labels[flat] = "Flat"

directional = ~flat

deg = (
    aspect_vals[directional]
    % 360.0
)

direction_labels = np.empty(
    len(deg),
    dtype=object
)

# N: 337.5–360 and 0–22.5
direction_labels[
    (deg >= 337.5)
    |
    (deg < 22.5)
] = "N"

direction_labels[
    (deg >= 22.5)
    &
    (deg < 67.5)
] = "NE"

direction_labels[
    (deg >= 67.5)
    &
    (deg < 112.5)
] = "E"

direction_labels[
    (deg >= 112.5)
    &
    (deg < 157.5)
] = "SE"

direction_labels[
    (deg >= 157.5)
    &
    (deg < 202.5)
] = "S"

direction_labels[
    (deg >= 202.5)
    &
    (deg < 247.5)
] = "SW"

direction_labels[
    (deg >= 247.5)
    &
    (deg < 292.5)
] = "W"

direction_labels[
    (deg >= 292.5)
    &
    (deg < 337.5)
] = "NW"

aspect_labels[
    directional
] = direction_labels

aspect_order = [
    "Flat",
    "N",
    "NE",
    "E",
    "SE",
    "S",
    "SW",
    "W",
    "NW"
]


with open(
    ASPECT_FREQ,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Aspect_Class",
        "Count",
        "Percent"
    ])

    for cls in aspect_order:

        count = np.count_nonzero(
            aspect_labels == cls
        )

        pct = (
            100.0
            * count
            / n_analysis
        )

        writer.writerow([
            cls,
            count,
            pct
        ])


# ============================================================
# GENERIC CATEGORICAL FREQUENCY
# ============================================================

def categorical_frequency(
    raster_name,
    class_labels,
    output_file
):

    arr, valid = read_raster(
        raster_name
    )

    use = (
        analysis_mask
        &
        valid
    )

    vals = arr[use]

    classes, counts = np.unique(
        vals,
        return_counts=True
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Class_Code",
            "Class_Name",
            "Count",
            "Percent"
        ])

        for cls, count in zip(
            classes,
            counts
        ):

            code = int(cls)

            writer.writerow([
                code,
                class_labels.get(
                    code,
                    "Unknown"
                ),
                int(count),
                100.0
                * count
                / n_analysis
            ])


# ============================================================
# LULC
# ============================================================

LULC_LABELS = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen"
}

categorical_frequency(
    LULC_NAME,
    LULC_LABELS,
    LULC_FREQ
)


# ============================================================
# HSG
# ============================================================

HSG_LABELS = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    11: "A/D",
    12: "B/D",
    13: "C/D",
    14: "D/D"
}

categorical_frequency(
    HSG_NAME,
    HSG_LABELS,
    HSG_FREQ
)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print(
    "\n========== ASPECT FREQUENCY =========="
)

for cls in aspect_order:

    count = np.count_nonzero(
        aspect_labels == cls
    )

    print(
        f"{cls:5s}: "
        f"{count:9,d} "
        f"({100*count/n_analysis:6.2f}%)"
    )


print(
    "\n========== OUTPUT FILES =========="
)

print(CONT_STATS)
print(ASPECT_FREQ)
print(LULC_FREQ)
print(HSG_FREQ)

print(
    "\nFINAL STATUS: "
    "DU FINAL-FACTOR STATISTICS COMPLETE"
)