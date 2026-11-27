import csv
import os
import numpy as np

# ============================================================
# USER SETTINGS
# ============================================================

INPUT_CSV = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
    r"\Factor_Screening_Results\00_Samples"
    r"\03_Factor_Screening_Sample_1km.csv"
)

OUTPUT_DIR = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
    r"\Factor_Screening_Results"
    r"\01_Continuous_Screening"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONTINUOUS FACTORS
# ============================================================

FACTORS = [
    "DEM_Clipped_UTM46N",
    "Slope_deg_30m",
    "TRI_Riley_30m",
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
# READ CSV
# ============================================================

data_rows = []

with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    missing_columns = [
        name for name in FACTORS
        if name not in reader.fieldnames
    ]

    if missing_columns:
        raise Exception(
            "Missing factor columns: "
            + ", ".join(missing_columns)
        )

    for rec in reader:

        values = []

        for name in FACTORS:

            value = rec[name]

            if value is None or value.strip() == "":
                values.append(np.nan)
            else:
                values.append(float(value))

        data_rows.append(values)


X = np.asarray(data_rows, dtype=np.float64)

n_samples, n_factors = X.shape


# ============================================================
# BASIC QC
# ============================================================

missing_total = np.count_nonzero(~np.isfinite(X))

print("\n================ DATA QC ================")
print("Samples       :", n_samples)
print("Factors       :", n_factors)
print("Missing cells :", missing_total)

if missing_total > 0:
    raise Exception(
        "Missing/non-finite values detected. "
        "Screening stopped."
    )

if n_samples != 4320:
    print(
        "WARNING: Expected 4320 samples, found",
        n_samples
    )


# ============================================================
# AVERAGE-RANK FUNCTION FOR SPEARMAN
# Handles tied values correctly
# ============================================================

def average_rank(a):

    a = np.asarray(a, dtype=np.float64)
    n = len(a)

    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]

    ranks_sorted = np.empty(n, dtype=np.float64)

    i = 0

    while i < n:

        j = i + 1

        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1

        # ranks are 1-based
        average = ((i + 1) + j) / 2.0

        ranks_sorted[i:j] = average

        i = j

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted

    return ranks


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

desc_rows = []

for i, name in enumerate(FACTORS):

    a = X[:, i]

    desc_rows.append([
        name,
        len(a),
        np.mean(a),
        np.std(a, ddof=1),
        np.min(a),
        np.percentile(a, 25),
        np.median(a),
        np.percentile(a, 75),
        np.max(a)
    ])


DESC_FILE = os.path.join(
    OUTPUT_DIR,
    "04A_Descriptive_Statistics.csv"
)

with open(
    DESC_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Factor",
        "N",
        "Mean",
        "SD",
        "Minimum",
        "Q1_25pct",
        "Median",
        "Q3_75pct",
        "Maximum"
    ])

    writer.writerows(desc_rows)


# ============================================================
# PEARSON CORRELATION
# ============================================================

pearson = np.corrcoef(
    X,
    rowvar=False
)


# ============================================================
# SPEARMAN CORRELATION
# Pearson correlation of average ranks
# ============================================================

ranked_X = np.column_stack([
    average_rank(X[:, i])
    for i in range(n_factors)
])

spearman = np.corrcoef(
    ranked_X,
    rowvar=False
)


# ============================================================
# SAVE CORRELATION MATRICES
# ============================================================

def save_matrix(path, matrix):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["Factor"] + FACTORS
        )

        for i, name in enumerate(FACTORS):

            writer.writerow(
                [name]
                + list(matrix[i, :])
            )


PEARSON_FILE = os.path.join(
    OUTPUT_DIR,
    "04B_Pearson_Correlation_Matrix.csv"
)

SPEARMAN_FILE = os.path.join(
    OUTPUT_DIR,
    "04C_Spearman_Correlation_Matrix.csv"
)

save_matrix(
    PEARSON_FILE,
    pearson
)

save_matrix(
    SPEARMAN_FILE,
    spearman
)


# ============================================================
# CORRELATION PAIRS
# ============================================================

pair_rows = []

for i in range(n_factors):

    for j in range(i + 1, n_factors):

        pair_rows.append([
            FACTORS[i],
            FACTORS[j],
            spearman[i, j],
            pearson[i, j],
            abs(spearman[i, j]),
            abs(pearson[i, j])
        ])


# Primary sorting = absolute Spearman
pair_rows.sort(
    key=lambda x: x[4],
    reverse=True
)


PAIR_FILE = os.path.join(
    OUTPUT_DIR,
    "04D_Strongest_Correlation_Pairs.csv"
)

with open(
    PAIR_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Factor_1",
        "Factor_2",
        "Spearman_rho",
        "Pearson_r",
        "Abs_Spearman",
        "Abs_Pearson"
    ])

    writer.writerows(pair_rows)


# ============================================================
# VIF FUNCTION
# VIF = 1 / (1 - R^2)
# ============================================================

def calculate_vif(data, names):

    # Standardization improves numerical stability
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0, ddof=0)

    if np.any(stds == 0):
        raise Exception(
            "Constant predictor detected."
        )

    Z = (
        (data - means)
        / stds
    )

    results = []

    p = Z.shape[1]

    for j in range(p):

        y = Z[:, j]

        X_other = np.delete(
            Z,
            j,
            axis=1
        )

        # Add intercept
        A = np.column_stack([
            np.ones(len(y)),
            X_other
        ])

        beta, _, _, _ = np.linalg.lstsq(
            A,
            y,
            rcond=None
        )

        y_pred = A @ beta

        ss_res = np.sum(
            (y - y_pred) ** 2
        )

        ss_tot = np.sum(
            (y - np.mean(y)) ** 2
        )

        r2 = 1.0 - (
            ss_res / ss_tot
        )

        if r2 >= 1.0:
            vif = np.inf
            tolerance = 0.0
        else:
            vif = 1.0 / (1.0 - r2)
            tolerance = 1.0 / vif

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


# ============================================================
# INITIAL VIF — ALL 11 FACTORS
# ============================================================

vif_all = calculate_vif(
    X,
    FACTORS
)


VIF_ALL_FILE = os.path.join(
    OUTPUT_DIR,
    "04E_VIF_All11_Continuous_Factors.csv"
)

with open(
    VIF_ALL_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Factor",
        "R_squared_against_others",
        "VIF",
        "Tolerance"
    ])

    writer.writerows(vif_all)


# ============================================================
# SECOND VIF — REMOVE TRI
# ============================================================

TRI_NAME = "TRI_Riley_30m"

keep_indices = [
    i for i, name in enumerate(FACTORS)
    if name != TRI_NAME
]

factors_without_tri = [
    FACTORS[i]
    for i in keep_indices
]

X_without_tri = X[:, keep_indices]

vif_without_tri = calculate_vif(
    X_without_tri,
    factors_without_tri
)


VIF_NO_TRI_FILE = os.path.join(
    OUTPUT_DIR,
    "04F_VIF_After_TRI_Removal.csv"
)

with open(
    VIF_NO_TRI_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Factor",
        "R_squared_against_others",
        "VIF",
        "Tolerance"
    ])

    writer.writerows(
        vif_without_tri
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n========== DESCRIPTIVE QC ==========")
print("N =", n_samples)
print("Missing =", missing_total)


print("\n========== TOP 15 CORRELATION PAIRS ==========")

for row in pair_rows[:15]:

    print(
        f"{row[0]}  <->  {row[1]}"
    )

    print(
        f"    Spearman = {row[2]:.6f} | "
        f"Pearson = {row[3]:.6f}"
    )


print("\n========== VIF: ALL 11 ==========")

for row in vif_all:

    print(
        f"{row[0]:65s} "
        f"VIF={row[2]:8.4f}  "
        f"Tolerance={row[3]:8.4f}"
    )


print("\n========== VIF: TRI REMOVED ==========")

for row in vif_without_tri:

    print(
        f"{row[0]:65s} "
        f"VIF={row[2]:8.4f}  "
        f"Tolerance={row[3]:8.4f}"
    )


print("\n========== OUTPUT FILES ==========")
print(DESC_FILE)
print(PEARSON_FILE)
print(SPEARMAN_FILE)
print(PAIR_FILE)
print(VIF_ALL_FILE)
print(VIF_NO_TRI_FILE)

print("\nFINAL: Continuous-factor screening calculations complete.")
print("==============================================")