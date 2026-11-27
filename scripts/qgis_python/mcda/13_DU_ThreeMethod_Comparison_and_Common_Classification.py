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

OUTPUT_DIR = os.path.join(
    BASE,
    "12_DU_MCDA",
    "05_Three_Method_Comparison"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


NODATA_FLOAT = -9999.0
NODATA_CLASS = 0


# ============================================================
# 02. OUTPUT FILES
# ============================================================

BREAKS_CSV = os.path.join(
    OUTPUT_DIR,
    "DU13_Common_Classification_Breaks.csv"
)

METRICS_CSV = os.path.join(
    OUTPUT_DIR,
    "DU13_Continuous_Method_Comparison.csv"
)

AREA_CSV = os.path.join(
    OUTPUT_DIR,
    "DU13_Susceptibility_Class_Area.csv"
)

AGREEMENT_CSV = os.path.join(
    OUTPUT_DIR,
    "DU13_Class_Agreement.csv"
)


CLASS_FILES = {

    "AHP":
        os.path.join(
            OUTPUT_DIR,
            "DU13_AHP_CommonClass_1to5.tif"
        ),

    "FAHP":
        os.path.join(
            OUTPUT_DIR,
            "DU13_FAHP_CommonClass_1to5.tif"
        ),

    "ANP":
        os.path.join(
            OUTPUT_DIR,
            "DU13_ANP_CommonClass_1to5.tif"
        )
}


DIFFERENCE_FILES = {

    "AHP_minus_FAHP":
        os.path.join(
            OUTPUT_DIR,
            "DU13_AHP_minus_FAHP.tif"
        ),

    "AHP_minus_ANP":
        os.path.join(
            OUTPUT_DIR,
            "DU13_AHP_minus_ANP.tif"
        ),

    "FAHP_minus_ANP":
        os.path.join(
            OUTPUT_DIR,
            "DU13_FAHP_minus_ANP.tif"
        )
}


# ============================================================
# 03. READ RASTER
# ============================================================

def read_raster(path):

    if not os.path.exists(path):

        raise Exception(
            "Raster not found:\n"
            + path
        )

    ds = gdal.Open(
        path
    )

    if ds is None:

        raise Exception(
            "Could not open:\n"
            + path
        )

    band = ds.GetRasterBand(1)

    arr = band.ReadAsArray().astype(
        np.float32
    )

    nodata = band.GetNoDataValue()

    valid = np.isfinite(
        arr
    )

    if nodata is not None:

        if np.isnan(nodata):

            valid &= ~np.isnan(
                arr
            )

        else:

            valid &= (
                arr != nodata
            )

    return ds, arr, valid


# ============================================================
# 04. LOAD THREE MCDA MAPS
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


# ============================================================
# 05. GRID QC
# ============================================================

reference_x = ahp_ds.RasterXSize
reference_y = ahp_ds.RasterYSize

reference_gt = ahp_ds.GetGeoTransform()
reference_projection = ahp_ds.GetProjection()


for name, ds in [

    ("FAHP", fahp_ds),
    ("ANP", anp_ds)

]:

    if (
        ds.RasterXSize != reference_x
        or
        ds.RasterYSize != reference_y
    ):

        raise Exception(
            name +
            " raster dimension mismatch."
        )


    if not np.allclose(
        ds.GetGeoTransform(),
        reference_gt,
        atol=0.01
    ):

        raise Exception(
            name +
            " geotransform mismatch."
        )


print(
    "\n========== DU STEP 13 GRID QC =========="
)

print(
    "AHP / FAHP / ANP grid : PASS"
)


# ============================================================
# 06. COMMON VALID DOMAIN
# ============================================================

common_valid = (
    ahp_valid
    &
    fahp_valid
    &
    anp_valid
)


n_valid = int(
    np.count_nonzero(
        common_valid
    )
)


print(
    "Common valid pixels   :",
    f"{n_valid:,}"
)


if n_valid != 4722216:

    raise Exception(
        "Expected 4,722,216 common valid pixels."
    )


# ============================================================
# 07. EXTRACT VALID VALUES
# ============================================================

A = ahp[
    common_valid
].astype(
    np.float32
)

F = fahp[
    common_valid
].astype(
    np.float32
)

N = anp[
    common_valid
].astype(
    np.float32
)


# ============================================================
# 08. COMMON POOLED CLASSIFICATION BREAKS
#
# Three maps are already on the same 0–1 criterion scale.
#
# We pool their continuous susceptibility values and derive
# common quintile thresholds.
#
# These identical breaks will then be used for ALL methods.
# ============================================================

pooled = np.concatenate(
    [
        A,
        F,
        N
    ]
)


q20, q40, q60, q80 = np.quantile(
    pooled,
    [
        0.20,
        0.40,
        0.60,
        0.80
    ]
)


del pooled


BREAKS = [
    float(q20),
    float(q40),
    float(q60),
    float(q80)
]


print(
    "\n========== COMMON CLASS BREAKS =========="
)

print(
    f"Very Low / Low       : {q20:.6f}"
)

print(
    f"Low / Moderate       : {q40:.6f}"
)

print(
    f"Moderate / High      : {q60:.6f}"
)

print(
    f"High / Very High     : {q80:.6f}"
)


# ============================================================
# 09. SAVE CLASS BREAKS
# ============================================================

with open(
    BREAKS_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Class_Code",
        "Class_Name",
        "Lower_Bound",
        "Upper_Bound"
    ])

    writer.writerow([
        1,
        "Very Low",
        0,
        q20
    ])

    writer.writerow([
        2,
        "Low",
        q20,
        q40
    ])

    writer.writerow([
        3,
        "Moderate",
        q40,
        q60
    ])

    writer.writerow([
        4,
        "High",
        q60,
        q80
    ])

    writer.writerow([
        5,
        "Very High",
        q80,
        1
    ])


# ============================================================
# 10. CLASSIFICATION FUNCTION
#
# 1 = Very Low
# 2 = Low
# 3 = Moderate
# 4 = High
# 5 = Very High
# 0 = NoData
# ============================================================

def classify(
    raster,
    valid_mask
):

    output = np.zeros(
        raster.shape,
        dtype=np.uint8
    )


    values = raster[
        valid_mask
    ]


    classes = np.digitize(
        values,
        bins=np.array(
            BREAKS,
            dtype=np.float32
        ),
        right=False
    ) + 1


    output[
        valid_mask
    ] = classes.astype(
        np.uint8
    )


    return output


ahp_class = classify(
    ahp,
    common_valid
)

fahp_class = classify(
    fahp,
    common_valid
)

anp_class = classify(
    anp,
    common_valid
)


# ============================================================
# 11. WRITE CLASS RASTER
# ============================================================

def write_byte_raster(
    path,
    arr
):

    driver = gdal.GetDriverByName(
        "GTiff"
    )

    out = driver.Create(
        path,
        reference_x,
        reference_y,
        1,
        gdal.GDT_Byte,
        options=[
            "COMPRESS=LZW"
        ]
    )

    out.SetGeoTransform(
        reference_gt
    )

    out.SetProjection(
        reference_projection
    )

    band = out.GetRasterBand(1)

    band.WriteArray(
        arr
    )

    band.SetNoDataValue(
        NODATA_CLASS
    )

    band.FlushCache()

    out = None


write_byte_raster(
    CLASS_FILES["AHP"],
    ahp_class
)

write_byte_raster(
    CLASS_FILES["FAHP"],
    fahp_class
)

write_byte_raster(
    CLASS_FILES["ANP"],
    anp_class
)


print(
    "\nThree common-class rasters : DONE"
)


# ============================================================
# 12. CONTINUOUS COMPARISON METRICS
# ============================================================

def continuous_metrics(
    x,
    y
):

    difference = x - y

    pearson = np.corrcoef(
        x,
        y
    )[0, 1]


    mae = np.mean(
        np.abs(
            difference
        )
    )


    rmse = np.sqrt(
        np.mean(
            difference ** 2
        )
    )


    mean_difference = np.mean(
        difference
    )


    max_abs_difference = np.max(
        np.abs(
            difference
        )
    )


    return {

        "Pearson_r":
            float(pearson),

        "MAE":
            float(mae),

        "RMSE":
            float(rmse),

        "Mean_Difference_First_minus_Second":
            float(mean_difference),

        "Maximum_Absolute_Difference":
            float(max_abs_difference)

    }


PAIR_DATA = [

    (
        "AHP",
        "FAHP",
        A,
        F
    ),

    (
        "AHP",
        "ANP",
        A,
        N
    ),

    (
        "FAHP",
        "ANP",
        F,
        N
    )

]


metric_rows = []


print(
    "\n========== CONTINUOUS MAP COMPARISON =========="
)


for name1, name2, x, y in PAIR_DATA:

    m = continuous_metrics(
        x,
        y
    )


    metric_rows.append([

        name1,

        name2,

        m["Pearson_r"],

        m["MAE"],

        m["RMSE"],

        m[
            "Mean_Difference_First_minus_Second"
        ],

        m[
            "Maximum_Absolute_Difference"
        ]

    ])


    print(
        f"{name1:4s} vs {name2:4s}"
        f"  r={m['Pearson_r']:.6f}"
        f"  MAE={m['MAE']:.6f}"
        f"  RMSE={m['RMSE']:.6f}"
    )


with open(
    METRICS_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([

        "Method_1",

        "Method_2",

        "Pearson_r",

        "MAE",

        "RMSE",

        "Mean_Difference_Method1_minus_Method2",

        "Maximum_Absolute_Difference"

    ])

    writer.writerows(
        metric_rows
    )


# ============================================================
# 13. CLASS AREA STATISTICS
# ============================================================

pixel_area_m2 = abs(
    reference_gt[1]
    *
    reference_gt[5]
)

pixel_area_km2 = (
    pixel_area_m2
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
        "Very High"

}


METHOD_CLASSES = {

    "AHP":
        ahp_class,

    "FAHP":
        fahp_class,

    "ANP":
        anp_class

}


area_rows = []


print(
    "\n========== CLASS AREA STATISTICS =========="
)


for method, class_raster in METHOD_CLASSES.items():

    print(
        "\n" + method
    )

    for class_code in range(
        1,
        6
    ):

        count = int(
            np.count_nonzero(
                class_raster[
                    common_valid
                ]
                ==
                class_code
            )
        )


        area_km2 = (
            count
            *
            pixel_area_km2
        )


        percent = (
            count
            /
            n_valid
            *
            100
        )


        area_rows.append([

            method,

            class_code,

            CLASS_NAMES[
                class_code
            ],

            count,

            area_km2,

            percent

        ])


        print(
            f"{CLASS_NAMES[class_code]:10s}"
            f" pixels={count:9,d}"
            f" area={area_km2:9.2f} km2"
            f" ({percent:6.2f}%)"
        )


with open(
    AREA_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([

        "Method",

        "Class_Code",

        "Class_Name",

        "Pixel_Count",

        "Area_km2",

        "Percent_of_Final_Analysis_Area"

    ])

    writer.writerows(
        area_rows
    )


# ============================================================
# 14. CLASS AGREEMENT + COHEN KAPPA
# ============================================================

def class_agreement_metrics(
    class1,
    class2,
    valid
):

    x = class1[
        valid
    ]

    y = class2[
        valid
    ]


    observed = np.mean(
        x == y
    )


    confusion = np.zeros(
        (
            5,
            5
        ),
        dtype=np.int64
    )


    for i in range(
        1,
        6
    ):

        for j in range(
            1,
            6
        ):

            confusion[
                i - 1,
                j - 1
            ] = np.count_nonzero(
                (x == i)
                &
                (y == j)
            )


    row_totals = confusion.sum(
        axis=1
    )

    col_totals = confusion.sum(
        axis=0
    )

    total = confusion.sum()


    expected = np.sum(
        row_totals
        *
        col_totals
    ) / (
        total ** 2
    )


    if np.isclose(
        1 - expected,
        0
    ):

        kappa = np.nan

    else:

        kappa = (
            observed - expected
        ) / (
            1 - expected
        )


    within_one_class = np.mean(
        np.abs(
            x.astype(np.int16)
            -
            y.astype(np.int16)
        )
        <= 1
    )


    return (
        observed,
        within_one_class,
        kappa
    )


agreement_rows = []


CLASS_PAIR_DATA = [

    (
        "AHP",
        "FAHP",
        ahp_class,
        fahp_class
    ),

    (
        "AHP",
        "ANP",
        ahp_class,
        anp_class
    ),

    (
        "FAHP",
        "ANP",
        fahp_class,
        anp_class
    )

]


print(
    "\n========== CLASS AGREEMENT =========="
)


for name1, name2, c1, c2 in CLASS_PAIR_DATA:

    exact, within1, kappa = (
        class_agreement_metrics(
            c1,
            c2,
            common_valid
        )
    )


    agreement_rows.append([

        name1,

        name2,

        exact * 100,

        within1 * 100,

        kappa

    ])


    print(
        f"{name1:4s} vs {name2:4s}"
        f" exact={exact*100:.2f}%"
        f" within±1={within1*100:.2f}%"
        f" kappa={kappa:.6f}"
    )


with open(
    AGREEMENT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([

        "Method_1",

        "Method_2",

        "Exact_Class_Agreement_Percent",

        "Agreement_Within_One_Class_Percent",

        "Cohen_Kappa"

    ])

    writer.writerows(
        agreement_rows
    )


# ============================================================
# 15. WRITE DIFFERENCE RASTER
# ============================================================

def write_difference(
    path,
    raster1,
    raster2
):

    output = np.full(
        raster1.shape,
        NODATA_FLOAT,
        dtype=np.float32
    )


    output[
        common_valid
    ] = (
        raster1[
            common_valid
        ]
        -
        raster2[
            common_valid
        ]
    )


    driver = gdal.GetDriverByName(
        "GTiff"
    )


    out = driver.Create(
        path,
        reference_x,
        reference_y,
        1,
        gdal.GDT_Float32,
        options=[
            "COMPRESS=LZW"
        ]
    )


    out.SetGeoTransform(
        reference_gt
    )

    out.SetProjection(
        reference_projection
    )


    band = out.GetRasterBand(1)

    band.WriteArray(
        output
    )

    band.SetNoDataValue(
        NODATA_FLOAT
    )

    band.FlushCache()

    out = None


write_difference(
    DIFFERENCE_FILES[
        "AHP_minus_FAHP"
    ],
    ahp,
    fahp
)


write_difference(
    DIFFERENCE_FILES[
        "AHP_minus_ANP"
    ],
    ahp,
    anp
)


write_difference(
    DIFFERENCE_FILES[
        "FAHP_minus_ANP"
    ],
    fahp,
    anp
)


print(
    "\nDifference rasters : DONE"
)


# ============================================================
# 16. FINAL REPORT
# ============================================================

print(
    "\n=============================================="
)

print(
    "DU STEP 13 OUTPUT DIRECTORY:"
)

print(
    OUTPUT_DIR
)

print(
    "\nCreated:"
)

print(
    "3 common-class susceptibility rasters"
)

print(
    "3 continuous difference rasters"
)

print(
    "1 common-class-break CSV"
)

print(
    "1 continuous comparison CSV"
)

print(
    "1 class-area CSV"
)

print(
    "1 class-agreement CSV"
)

print(
    "\nFINAL STATUS: DU STEP 13 PASS"
)

print(
    "=============================================="
)