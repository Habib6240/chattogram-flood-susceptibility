import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# B10A — PUBLICATION FIGURES + COMPACT RESULT TABLES
#
# Uses only locked outputs from B08U, B09A, and B09B.
# No model fitting. No metric tuning.
# ============================================================

BASE = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
)

B08U_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "06B_Explainability_UrbanAware",
)

B09A_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "07_Final_30m_Maps_UrbanAware",
)

B09B_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "08_ML_vs_MCDA_Comparison",
)

OUTPUT_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "09_Publication_Figures_Tables",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

EVENT_PERF = os.path.join(
    B09B_DIR,
    "BUET_B09B_Strict_Event_Performance_All_Methods.csv",
)

MACRO = os.path.join(
    B09B_DIR,
    "BUET_B09B_Strict_Macro_Performance_All_Methods.csv",
)

TOP20 = os.path.join(
    B09B_DIR,
    "BUET_B09B_Top20_Susceptibility_Overlap.csv",
)

SHAP = os.path.join(
    B08U_DIR,
    "BUET_B08U_XGB_SHAP_Global_13Factor_Importance.csv",
)

CLASS_AREA = os.path.join(
    B09A_DIR,
    "BUET_B09A_Class_Area_Summary.csv",
)

for p in [EVENT_PERF, MACRO, TOP20, SHAP, CLASS_AREA]:
    if not os.path.exists(p):
        raise Exception("Required locked output missing:\n" + p)

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

event_df = pd.read_csv(EVENT_PERF)
macro_df = pd.read_csv(MACRO)
top20_df = pd.read_csv(TOP20)
shap_df = pd.read_csv(SHAP)
class_df = pd.read_csv(CLASS_AREA)

print("\n========== B10A INPUT QC ==========")
print("Event-performance rows:", len(event_df))
print("Macro rows            :", len(macro_df))
print("SHAP factors          :", len(shap_df))
print("Class-area rows       :", len(class_df))

# ============================================================
# FIGURE 1 — STRICT EVENT AUC BY METHOD
# ============================================================

method_order = [
    "Random_Forest",
    "XGBoost",
    "ML_Consensus",
    "AHP",
    "FAHP",
    "ANP",
]

display_names = {
    "Random_Forest": "RF",
    "XGBoost": "XGBoost",
    "ML_Consensus": "ML Consensus",
    "AHP": "AHP",
    "FAHP": "FAHP",
    "ANP": "ANP",
}

events = ["2019", "2021", "2023", "2026"]

x = np.arange(len(events), dtype=float)
width = 0.12

plt.figure(figsize=(9.0, 5.3))

for i, method in enumerate(method_order):
    vals = []
    for event in events:
        row = event_df[
            (event_df["Held_Event"].astype(str) == event)
            & (event_df["Method"] == method)
        ]
        if len(row) != 1:
            raise Exception(f"Missing/duplicate event result: {event}, {method}")
        vals.append(float(row.iloc[0]["ROC_AUC"]))

    offset = (i - (len(method_order) - 1) / 2.0) * width
    plt.bar(
        x + offset,
        vals,
        width=width,
        label=display_names[method],
    )

plt.xticks(x, events)
plt.ylim(0.65, 0.95)
plt.xlabel("Held-out flood event")
plt.ylabel("ROC-AUC")
plt.title("Strict unseen-event + unseen-geography performance")
plt.legend(ncol=3, frameon=False)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure_B10A_01_Strict_Event_AUC.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ============================================================
# FIGURE 2 — MACRO AUC
# ============================================================

macro_lookup = macro_df.set_index("Method")

macro_auc = [
    float(macro_lookup.loc[m, "Macro_Event_AUC"])
    for m in method_order
]

plt.figure(figsize=(8.2, 4.8))
plt.bar(
    [display_names[m] for m in method_order],
    macro_auc,
)
plt.ylim(0.70, 0.90)
plt.ylabel("Macro event ROC-AUC")
plt.title("Macro-event predictive performance")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure_B10A_02_Macro_AUC.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ============================================================
# FIGURE 3 — MACRO AP
# ============================================================

macro_ap = [
    float(macro_lookup.loc[m, "Macro_Event_AP"])
    for m in method_order
]

plt.figure(figsize=(8.2, 4.8))
plt.bar(
    [display_names[m] for m in method_order],
    macro_ap,
)
plt.ylabel("Macro event average precision")
plt.title("Macro-event precision-recall performance")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure_B10A_03_Macro_AP.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ============================================================
# FIGURE 4 — FINAL SHAP IMPORTANCE
# ============================================================

shap_plot = shap_df.sort_values(
    "Relative_Importance_Percent",
    ascending=True,
)

plt.figure(figsize=(8.5, 6.4))
plt.barh(
    shap_plot["Factor"],
    shap_plot["Relative_Importance_Percent"],
)
plt.xlabel("Relative grouped SHAP importance (%)")
plt.ylabel("Conditioning factor")
plt.title("Final urban-aware XGBoost factor importance")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure_B10A_04_Final_SHAP_Importance.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ============================================================
# FIGURE 5 — ML CONSENSUS vs MCDA TOP-20% OVERLAP
# ============================================================

mcda_methods = ["AHP", "FAHP", "ANP"]
overlap_values = []

for mcda in mcda_methods:
    row = top20_df[
        ((top20_df["Method_1"] == "ML_Consensus") & (top20_df["Method_2"] == mcda))
        |
        ((top20_df["Method_2"] == "ML_Consensus") & (top20_df["Method_1"] == mcda))
    ]

    if len(row) != 1:
        raise Exception("Missing ML consensus top-20 comparison for " + mcda)

    overlap_values.append(
        100.0 * float(row.iloc[0]["Shared_Percent_of_Smaller_Top20"])
        if float(row.iloc[0]["Shared_Percent_of_Smaller_Top20"]) <= 1.0
        else float(row.iloc[0]["Shared_Percent_of_Smaller_Top20"])
    )

plt.figure(figsize=(6.8, 4.7))
plt.bar(mcda_methods, overlap_values)
plt.ylim(0, 100)
plt.ylabel("Shared top-20% susceptibility area (%)")
plt.title("High-susceptibility overlap: ML consensus vs MCDA")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure_B10A_05_ML_MCDA_Top20_Overlap.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ============================================================
# FIGURE 6 — CONSENSUS SUSCEPTIBILITY CLASS AREA
# ============================================================

consensus_class = class_df[
    class_df["Map"] == "Consensus"
].copy()

if len(consensus_class) != 5:
    raise Exception("Expected five consensus class-area rows.")

consensus_class = consensus_class.sort_values("Class_ID")

plt.figure(figsize=(7.5, 4.8))
plt.bar(
    consensus_class["Class_Name"],
    consensus_class["Percent_of_Valid_Domain"],
)
plt.ylabel("Valid-domain area (%)")
plt.xlabel("Susceptibility class")
plt.title("Final ML consensus susceptibility-class distribution")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure_B10A_06_Consensus_Class_Area.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ============================================================
# FINAL COMPACT TABLE — use sample SD across 4 held events
# ============================================================

summary_rows = []

for method in method_order:
    sub = event_df[event_df["Method"] == method].copy()

    aucs = sub["ROC_AUC"].to_numpy(dtype=float)
    aps = sub["Average_Precision"].to_numpy(dtype=float)

    if len(aucs) != 4:
        raise Exception("Expected four held events for " + method)

    summary_rows.append([
        display_names[method],
        float(np.mean(aucs)),
        float(np.std(aucs, ddof=1)),
        float(np.min(aucs)),
        float(np.max(aucs)),
        float(np.mean(aps)),
        float(np.std(aps, ddof=1)),
    ])

summary_csv = os.path.join(
    OUTPUT_DIR,
    "Table_B10A_Final_Strict_Performance_SampleSD.csv",
)

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Method",
        "Macro_AUC",
        "Sample_SD_AUC",
        "Minimum_Event_AUC",
        "Maximum_Event_AUC",
        "Macro_AP",
        "Sample_SD_AP",
    ])
    writer.writerows(summary_rows)

# ============================================================
# HEADLINE RESULT TABLE
# ============================================================

headline_csv = os.path.join(
    OUTPUT_DIR,
    "Table_B10A_Headline_Results.csv",
)

ml_cons_auc = next(r[1] for r in summary_rows if r[0] == "ML Consensus")
ml_cons_ap = next(r[5] for r in summary_rows if r[0] == "ML Consensus")

mcda_auc_candidates = [
    r for r in summary_rows
    if r[0] in ["AHP", "FAHP", "ANP"]
]

best_mcda_auc_row = max(mcda_auc_candidates, key=lambda r: r[1])
best_mcda_ap_row = max(mcda_auc_candidates, key=lambda r: r[5])

with open(headline_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Result", "Value"])
    writer.writerows([
        ["ML consensus macro AUC", ml_cons_auc],
        ["Best MCDA macro AUC method", best_mcda_auc_row[0]],
        ["Best MCDA macro AUC", best_mcda_auc_row[1]],
        ["ML minus best-MCDA AUC", ml_cons_auc - best_mcda_auc_row[1]],
        ["ML consensus macro AP", ml_cons_ap],
        ["Best MCDA macro AP method", best_mcda_ap_row[0]],
        ["Best MCDA macro AP", best_mcda_ap_row[5]],
        ["ML minus best-MCDA AP", ml_cons_ap - best_mcda_ap_row[5]],
    ])

print("\n==============================================")
print("BUET B10A PUBLICATION FIGURES + TABLES COMPLETE")
print("==============================================")
print("Output folder:")
print(OUTPUT_DIR)
print("\nFINAL STATUS: BUET B10A PASS")
