import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# B08V — FINAL URBAN-AWARE SHAP DEPENDENCE
#
# Reads B08U explanation sample.
# Provides:
# - 20-quantile dependence for leading continuous factors
# - low-20% vs high-20% SHAP contrast
# - LULC class-wise SHAP
#
# No model fitting.
# ============================================================

BASE = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
)

INPUT_CSV = os.path.join(
    BASE,
    "13_BUET_ML",
    "06B_Explainability_UrbanAware",
    "BUET_B08U_XGB_SHAP_Explanation_Sample.csv",
)

OUTPUT_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "06B_Explainability_UrbanAware",
    "B08V_Dependence",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTINUOUS_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B08V_Continuous_SHAP_Dependence_Bins.csv",
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B08V_Dependence_Summary.csv",
)

LULC_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B08V_LULC_SHAP_By_Class.csv",
)

if not os.path.exists(INPUT_CSV):
    raise Exception(
        "B08U explanation sample not found:\n" + INPUT_CSV
    )

df = pd.read_csv(INPUT_CSV)

print("\n========== B08V DATA ==========")
print("Rows:", f"{len(df):,}")

# Leading continuous factors after B08U.
FACTORS = [
    "NDVI",
    "Elevation",
    "Rainfall",
    "Slope",
    "TWI",
    "Distance_River",
    "Distance_Road",
]

continuous_rows = []
summary_rows = []

print("\n========== CONTINUOUS SHAP DEPENDENCE ==========")

for factor in FACTORS:

    value_col = factor + "_Value"
    shap_col = factor + "_SHAP"

    temp = df[
        [
            value_col,
            shap_col,
            "Observed_Composite_Label",
        ]
    ].copy()

    temp = temp.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    temp["Bin"] = pd.qcut(
        temp[value_col],
        q=20,
        duplicates="drop",
    )

    factor_bin_rows = []

    for bin_number, (interval, group) in enumerate(
        temp.groupby("Bin", observed=True),
        start=1,
    ):
        median_value = float(group[value_col].median())
        mean_shap = float(group[shap_col].mean())
        median_shap = float(group[shap_col].median())
        flood_fraction = float(
            group["Observed_Composite_Label"].mean()
        )
        n = len(group)

        row = [
            factor,
            bin_number,
            float(interval.left),
            float(interval.right),
            median_value,
            mean_shap,
            median_shap,
            n,
            flood_fraction,
        ]

        continuous_rows.append(row)
        factor_bin_rows.append(row)

    q20 = float(np.quantile(temp[value_col], 0.20))
    q80 = float(np.quantile(temp[value_col], 0.80))

    low = temp[temp[value_col] <= q20]
    high = temp[temp[value_col] >= q80]

    low_shap = float(low[shap_col].mean())
    high_shap = float(high[shap_col].mean())
    difference = high_shap - low_shap

    if difference > 0:
        direction = "Higher values increase model susceptibility relative to lower values"
    elif difference < 0:
        direction = "Higher values decrease model susceptibility relative to lower values"
    else:
        direction = "No net quintile contrast"

    summary_rows.append([
        factor,
        q20,
        q80,
        low_shap,
        high_shap,
        difference,
        direction,
    ])

    print(
        f"{factor:18s}"
        f" low20 mean SHAP={low_shap:+.4f}"
        f" high20 mean SHAP={high_shap:+.4f}"
        f" delta={difference:+.4f}"
    )

    x = [row[4] for row in factor_bin_rows]
    y = [row[5] for row in factor_bin_rows]

    plt.figure(figsize=(7, 4.8))
    plt.plot(x, y, marker="o")
    plt.axhline(0, linewidth=1)
    plt.xlabel(factor)
    plt.ylabel("Mean grouped SHAP value")
    plt.title(f"{factor} - urban-aware SHAP dependence")
    plt.tight_layout()

    figure_path = os.path.join(
        OUTPUT_DIR,
        "BUET_B08V_" + factor + "_SHAP_Dependence.png",
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

with open(
    CONTINUOUS_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "Factor",
        "Quantile_Bin",
        "Bin_Lower",
        "Bin_Upper",
        "Median_Factor_Value",
        "Mean_SHAP",
        "Median_SHAP",
        "N",
        "Observed_Flood_Fraction",
    ])
    writer.writerows(continuous_rows)

with open(
    SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "Factor",
        "P20",
        "P80",
        "Low20_Mean_SHAP",
        "High20_Mean_SHAP",
        "High_minus_Low_SHAP",
        "Broad_Direction",
    ])
    writer.writerows(summary_rows)

# ------------------------------------------------------------
# LULC class-wise dependence
# ------------------------------------------------------------

LULC_LABELS = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    90: "Herbaceous wetland",
    95: "Mangroves",
}

lulc_rows = []

print("\n========== LULC SHAP BY CLASS ==========")

for code in sorted(df["LULC_Value"].unique()):

    code_int = int(code)

    subset = df[
        df["LULC_Value"] == code
    ]

    n = len(subset)
    mean_shap = float(subset["LULC_SHAP"].mean())
    median_shap = float(subset["LULC_SHAP"].median())
    mean_abs_shap = float(
        np.mean(np.abs(subset["LULC_SHAP"]))
    )
    flood_fraction = float(
        subset["Observed_Composite_Label"].mean()
    )

    label = LULC_LABELS.get(
        code_int,
        "Unknown",
    )

    lulc_rows.append([
        code_int,
        label,
        n,
        mean_shap,
        median_shap,
        mean_abs_shap,
        flood_fraction,
    ])

    print(
        f"{code_int:3d}"
        f" {label:25s}"
        f" N={n:6,d}"
        f" mean SHAP={mean_shap:+.4f}"
        f" flood%={100 * flood_fraction:6.2f}"
    )

with open(
    LULC_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "LULC_Code",
        "LULC_Class",
        "N",
        "Mean_LULC_SHAP",
        "Median_LULC_SHAP",
        "Mean_Absolute_LULC_SHAP",
        "Observed_Flood_Fraction",
    ])
    writer.writerows(lulc_rows)

plot_rows = sorted(
    lulc_rows,
    key=lambda row: row[3],
)

labels = [row[1] for row in plot_rows]
values = [row[3] for row in plot_rows]

plt.figure(figsize=(8, 5.5))
plt.barh(labels, values)
plt.axvline(0, linewidth=1)
plt.xlabel("Mean grouped LULC SHAP value")
plt.ylabel("ESA WorldCover class")
plt.title("Urban-aware LULC effect on XGBoost susceptibility")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "BUET_B08V_LULC_SHAP_By_Class.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\n==============================================")
print("BUET B08V URBAN-AWARE SHAP DEPENDENCE COMPLETE")
print("==============================================")
print("\nFINAL STATUS: BUET B08V PASS")
