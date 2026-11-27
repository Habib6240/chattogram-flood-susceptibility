import csv
import os
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

INPUT_CSV = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
    r"\Factor_Screening_Results\00_Samples"
    r"\05_Factor_Screening_Sample_All14.csv"
)
OUTPUT_DIR = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
    r"\Factor_Screening_Results"
    r"\02_Categorical_Circular_Screening"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


CONTINUOUS = [
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


# ============================================================
# READ DATA
# ============================================================

rows = []

with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for rec in reader:
        rows.append(rec)


n = len(rows)

aspect = np.array(
    [float(r["Aspect"]) for r in rows],
    dtype=float
)

lulc = np.array(
    [int(float(r["LULC"])) for r in rows],
    dtype=int
)

hsg = np.array(
    [int(float(r["HSG"])) for r in rows],
    dtype=int
)

X = np.column_stack([
    np.array(
        [float(r[name]) for r in rows],
        dtype=float
    )
    for name in CONTINUOUS
])


print("\n================ DATA QC ================")
print("Rows             :", n)
print("Continuous       :", len(CONTINUOUS))
print("Aspect min/max   :", np.min(aspect), np.max(aspect))
print("LULC classes     :", sorted(np.unique(lulc)))
print("HSG classes      :", sorted(np.unique(hsg)))


# ============================================================
# CLASS FREQUENCIES
# ============================================================

def save_frequency(values, variable, filename):

    classes, counts = np.unique(
        values,
        return_counts=True
    )

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Variable",
            "Class",
            "Count",
            "Percent"
        ])

        for cls, count in zip(classes, counts):

            writer.writerow([
                variable,
                int(cls),
                int(count),
                100.0 * count / len(values)
            ])


LULC_FREQ = os.path.join(
    OUTPUT_DIR,
    "06A_LULC_Class_Frequency.csv"
)

HSG_FREQ = os.path.join(
    OUTPUT_DIR,
    "06B_HSG_Class_Frequency.csv"
)

save_frequency(
    lulc,
    "LULC",
    LULC_FREQ
)

save_frequency(
    hsg,
    "HSG",
    HSG_FREQ
)


# ============================================================
# CORRELATION RATIO ETA-SQUARED
# categorical -> continuous association
# ============================================================

def eta_squared(categories, values):

    grand_mean = np.mean(values)

    ss_total = np.sum(
        (values - grand_mean) ** 2
    )

    if ss_total == 0:
        return 0.0

    ss_between = 0.0

    for cls in np.unique(categories):

        vals = values[
            categories == cls
        ]

        ss_between += (
            len(vals)
            * (np.mean(vals) - grand_mean) ** 2
        )

    return ss_between / ss_total


eta_rows = []

for i, name in enumerate(CONTINUOUS):

    eta_lulc = eta_squared(
        lulc,
        X[:, i]
    )

    eta_hsg = eta_squared(
        hsg,
        X[:, i]
    )

    eta_rows.append([
        name,
        eta_lulc,
        eta_hsg
    ])


ETA_FILE = os.path.join(
    OUTPUT_DIR,
    "06C_Categorical_Continuous_EtaSquared.csv"
)

with open(
    ETA_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Continuous_Factor",
        "EtaSquared_LULC",
        "EtaSquared_HSG"
    ])

    writer.writerows(eta_rows)


# ============================================================
# BIAS-CORRECTED CRAMER'S V
# LULC vs HSG
# ============================================================

def corrected_cramers_v(x, y):

    x_classes = np.unique(x)
    y_classes = np.unique(y)

    table = np.zeros(
        (
            len(x_classes),
            len(y_classes)
        ),
        dtype=float
    )

    for i, xi in enumerate(x_classes):

        for j, yj in enumerate(y_classes):

            table[i, j] = np.sum(
                (x == xi) &
                (y == yj)
            )

    n = np.sum(table)

    row_sum = np.sum(
        table,
        axis=1,
        keepdims=True
    )

    col_sum = np.sum(
        table,
        axis=0,
        keepdims=True
    )

    expected = (
        row_sum @ col_sum
    ) / n

    valid = expected > 0

    chi2 = np.sum(
        ((table[valid] - expected[valid]) ** 2)
        / expected[valid]
    )

    phi2 = chi2 / n

    r, k = table.shape

    phi2_corr = max(
        0,
        phi2 -
        ((k - 1) * (r - 1))
        / (n - 1)
    )

    r_corr = (
        r -
        ((r - 1) ** 2)
        / (n - 1)
    )

    k_corr = (
        k -
        ((k - 1) ** 2)
        / (n - 1)
    )

    denom = min(
        k_corr - 1,
        r_corr - 1
    )

    if denom <= 0:
        return 0.0, chi2

    v = np.sqrt(
        phi2_corr / denom
    )

    return v, chi2


cramers_v, chi2 = corrected_cramers_v(
    lulc,
    hsg
)


CRAMER_FILE = os.path.join(
    OUTPUT_DIR,
    "06D_LULC_HSG_CramersV.csv"
)

with open(
    CRAMER_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Variable_1",
        "Variable_2",
        "Corrected_Cramers_V",
        "Chi_Square",
        "N"
    ])

    writer.writerow([
        "LULC",
        "HSG",
        cramers_v,
        chi2,
        n
    ])


# ============================================================
# CIRCULAR ASPECT TRANSFORMATION
# ============================================================

aspect_rad = np.deg2rad(
    aspect
)

aspect_sin = np.sin(
    aspect_rad
)

aspect_cos = np.cos(
    aspect_rad
)


# ============================================================
# RANK FUNCTION FOR SPEARMAN
# ============================================================

def average_rank(a):

    a = np.asarray(
        a,
        dtype=float
    )

    order = np.argsort(
        a,
        kind="mergesort"
    )

    sorted_a = a[order]

    ranks_sorted = np.empty(
        len(a),
        dtype=float
    )

    i = 0

    while i < len(a):

        j = i + 1

        while (
            j < len(a)
            and sorted_a[j] == sorted_a[i]
        ):
            j += 1

        ranks_sorted[i:j] = (
            ((i + 1) + j) / 2.0
        )

        i = j

    ranks = np.empty(
        len(a),
        dtype=float
    )

    ranks[order] = ranks_sorted

    return ranks


# ============================================================
# ASPECT vs CONTINUOUS ASSOCIATIONS
# ============================================================

aspect_rows = []

for i, name in enumerate(CONTINUOUS):

    vals = X[:, i]

    p_sin = np.corrcoef(
        aspect_sin,
        vals
    )[0, 1]

    p_cos = np.corrcoef(
        aspect_cos,
        vals
    )[0, 1]

    s_sin = np.corrcoef(
        average_rank(aspect_sin),
        average_rank(vals)
    )[0, 1]

    s_cos = np.corrcoef(
        average_rank(aspect_cos),
        average_rank(vals)
    )[0, 1]

    aspect_rows.append([
        name,
        p_sin,
        p_cos,
        s_sin,
        s_cos
    ])


ASPECT_ASSOC_FILE = os.path.join(
    OUTPUT_DIR,
    "06E_AspectSinCos_Associations.csv"
)

with open(
    ASPECT_ASSOC_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Factor",
        "Pearson_AspectSin",
        "Pearson_AspectCos",
        "Spearman_AspectSin",
        "Spearman_AspectCos"
    ])

    writer.writerows(
        aspect_rows
    )


# ============================================================
# VIF: 10 CONTINUOUS + ASPECT SIN/COS
# ============================================================

X_aug = np.column_stack([
    X,
    aspect_sin,
    aspect_cos
])

names_aug = (
    CONTINUOUS
    + ["Aspect_sin", "Aspect_cos"]
)


def calculate_vif(data, names):

    mean = np.mean(
        data,
        axis=0
    )

    std = np.std(
        data,
        axis=0
    )

    Z = (
        (data - mean)
        / std
    )

    results = []

    for j in range(
        Z.shape[1]
    ):

        y = Z[:, j]

        other = np.delete(
            Z,
            j,
            axis=1
        )

        A = np.column_stack([
            np.ones(len(y)),
            other
        ])

        beta, _, _, _ = np.linalg.lstsq(
            A,
            y,
            rcond=None
        )

        pred = A @ beta

        ss_res = np.sum(
            (y - pred) ** 2
        )

        ss_tot = np.sum(
            (y - np.mean(y)) ** 2
        )

        r2 = 1.0 - ss_res / ss_tot

        vif = (
            np.inf
            if r2 >= 1
            else 1.0 / (1.0 - r2)
        )

        tolerance = (
            0.0
            if not np.isfinite(vif)
            else 1.0 / vif
        )

        results.append([
            names[j],
            r2,
            vif,
            tolerance
        ])

    results.sort(
        key=lambda x: x[2],
        reverse=True
    )

    return results


vif_results = calculate_vif(
    X_aug,
    names_aug
)


ASPECT_VIF_FILE = os.path.join(
    OUTPUT_DIR,
    "06F_VIF_ContinuousPlusAspectSinCos.csv"
)

with open(
    ASPECT_VIF_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Factor",
        "R_squared",
        "VIF",
        "Tolerance"
    ])

    writer.writerows(
        vif_results
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n========== LULC FREQUENCIES ==========")

for cls in sorted(
    np.unique(lulc)
):

    count = np.sum(
        lulc == cls
    )

    print(
        f"Class {cls:3d}: "
        f"{count:5d} "
        f"({100*count/n:6.2f}%)"
    )


print("\n========== HSG FREQUENCIES ==========")

for cls in sorted(
    np.unique(hsg)
):

    count = np.sum(
        hsg == cls
    )

    print(
        f"Class {cls:3d}: "
        f"{count:5d} "
        f"({100*count/n:6.2f}%)"
    )


print("\n========== LULC-HSG ASSOCIATION ==========")
print(
    "Corrected Cramer's V =",
    round(cramers_v, 4)
)


print("\n========== ETA-SQUARED ==========")

for row in sorted(
    eta_rows,
    key=lambda x:
    max(x[1], x[2]),
    reverse=True
):

    print(
        f"{row[0]:65s} "
        f"LULC η²={row[1]:.4f} "
        f"HSG η²={row[2]:.4f}"
    )


print("\n========== VIF WITH ASPECT SIN/COS ==========")

for row in vif_results:

    print(
        f"{row[0]:65s} "
        f"VIF={row[2]:7.4f}"
    )


print("\n========== OUTPUTS ==========")
print(LULC_FREQ)
print(HSG_FREQ)
print(ETA_FILE)
print(CRAMER_FILE)
print(ASPECT_ASSOC_FILE)
print(ASPECT_VIF_FILE)

print(
    "\nFINAL: categorical/circular screening complete."
)