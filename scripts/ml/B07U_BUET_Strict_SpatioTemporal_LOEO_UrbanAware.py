import os
import csv
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

# ============================================================
# B07U — STRICT SPATIO-TEMPORAL LOEO
# Urban-aware Sentinel-1 label version
#
# DESIGN
# ------------------------------------------------------------
# 4 held-out events x 5 reused B06 spatial folds = 20 scenarios
# per model (40 model evaluations total).
#
# For held event e:
#   training events = the other 3 events
#   training label  = any-flood across those 3, but ONLY where
#                     all three labels are valid
#   test label      = held event's observed 0/1 label
#
# For spatial fold f:
#   train = valid 3-event training labels AND Spatial_Fold != f
#   test  = held-event-valid labels AND Spatial_Fold == f
#
# Thus every scenario tests simultaneously:
#   - an unseen flood event
#   - unseen 5-km geographic blocks
#
# Same RF/XGB hyperparameters as B05U/B06U.
# No held-event tuning.
# Primary metrics: ROC-AUC, AP, Brier score.
# ============================================================

BASE = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
)

INPUT_NPZ = os.path.join(
    BASE,
    "13_BUET_ML",
    "02B_ML_Dataset_UrbanAware",
    "BUET_B04U_ML_Master_150m.npz",
)

B06_FOLD_FILE = os.path.join(
    BASE,
    "13_BUET_ML",
    "04_Spatial_CV",
    "BUET_B06_5km_Block_Spatial_Folds.csv",
)

OUTPUT_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "05B_Strict_LOEO_UrbanAware",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENARIO_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B07U_LOEO_Scenario_Performance.csv",
)

EVENT_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B07U_LOEO_Event_Summary.csv",
)

OVERALL_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B07U_LOEO_Overall_Summary.csv",
)

PAIRED_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B07U_RF_vs_XGB_Scenario_Comparison.csv",
)

OOF_NPZ = os.path.join(
    OUTPUT_DIR,
    "BUET_B07U_LOEO_OutOfFold_Predictions.npz",
)

if not os.path.exists(INPUT_NPZ):
    raise Exception("B04U dataset not found:\n" + INPUT_NPZ)

if not os.path.exists(B06_FOLD_FILE):
    raise Exception("Original B06 fold map not found:\n" + B06_FOLD_FILE)

D = np.load(INPUT_NPZ, allow_pickle=False)

# ============================================================
# 01. FEATURES
# ============================================================

X = np.column_stack([
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
    D["Distance_Road"],
]).astype(np.float32)

BLOCKS = D["Block_5km"].astype(np.int64)
X_COORD = D["X"].astype(np.float64)
Y_COORD = D["Y"].astype(np.float64)
ROWS = D["Raster_Row"].astype(np.int32)
COLS = D["Raster_Column"].astype(np.int32)

EVENT_LABELS = {
    "2019": D["Flood_2019"].astype(np.int8),
    "2021": D["Flood_2021"].astype(np.int8),
    "2023": D["Flood_2023"].astype(np.int8),
    "2026": D["Flood_2026"].astype(np.int8),
}

EVENTS = ["2019", "2021", "2023", "2026"]

N = len(BLOCKS)

print("\n========== B07U MASTER DATASET ==========")
print("Rows:", f"{N:,}")
print("5-km blocks:", len(np.unique(BLOCKS)))

# ============================================================
# 02. REUSE ORIGINAL B06 FOLD MAP
# ============================================================

block_to_fold = {}

with open(B06_FOLD_FILE, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        block_to_fold[int(row["Block_5km"])] = int(row["Spatial_Fold"])

missing_blocks = sorted(set(np.unique(BLOCKS).tolist()) - set(block_to_fold.keys()))
if missing_blocks:
    raise Exception(
        "Some B04U blocks are missing from original B06 fold map: "
        + str(missing_blocks[:10])
    )

SPATIAL_FOLD = np.array(
    [block_to_fold[int(b)] for b in BLOCKS],
    dtype=np.uint8,
)

print("\n========== B06 FOLD REUSE QC ==========")
print("Mapped rows:", f"{len(SPATIAL_FOLD):,}")
print("Fold IDs:", sorted(np.unique(SPATIAL_FOLD).tolist()))
print("Status: PASS")

# ============================================================
# 03. FEATURE PREPROCESSING DESIGN
# ============================================================

NUMERIC_INDICES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14
]
CATEGORICAL_INDICES = [10, 12]


def make_preprocessor():
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float32,
        )

    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_INDICES),
            ("categorical", encoder, CATEGORICAL_INDICES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_models(y_train):
    pos = int(np.count_nonzero(y_train == 1))
    neg = int(np.count_nonzero(y_train == 0))

    if pos == 0 or neg == 0:
        raise Exception("Training scenario has only one class.")

    scale_pos_weight = neg / pos

    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )

    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=2,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    return rf, xgb, scale_pos_weight


def metrics(y_true, prob):
    if len(np.unique(y_true)) < 2:
        raise Exception("Test scenario has only one class; AUC undefined.")

    auc = float(roc_auc_score(y_true, prob))
    ap = float(average_precision_score(y_true, prob))
    brier = float(brier_score_loss(y_true, prob))

    flood_prob = float(np.mean(prob[y_true == 1]))
    nonflood_prob = float(np.mean(prob[y_true == 0]))
    separation = flood_prob - nonflood_prob

    return {
        "AUC": auc,
        "AP": ap,
        "Brier": brier,
        "Mean_Prob_Flood": flood_prob,
        "Mean_Prob_NonFlood": nonflood_prob,
        "Probability_Separation": separation,
    }

# ============================================================
# 04. STRICT LOEO SCENARIOS
# ============================================================

scenario_rows = []

# OOF storage per held event; only held-event-valid rows receive predictions.
oof = {}
for event in EVENTS:
    n = N
    oof[event] = {
        "Observed": np.full(n, -1, dtype=np.int8),
        "Fold": np.zeros(n, dtype=np.uint8),
        "RF": np.full(n, np.nan, dtype=np.float32),
        "XGB": np.full(n, np.nan, dtype=np.float32),
    }

for held_event in EVENTS:

    training_events = [e for e in EVENTS if e != held_event]

    train_matrix = np.column_stack(
        [EVENT_LABELS[e] for e in training_events]
    )

    valid_train_label = np.all(train_matrix >= 0, axis=1)

    train_target = np.full(N, -1, dtype=np.int8)
    train_target[valid_train_label] = np.any(
        train_matrix[valid_train_label] == 1,
        axis=1,
    ).astype(np.int8)

    held_labels = EVENT_LABELS[held_event]
    held_valid = held_labels >= 0

    print("\n##################################################")
    print("HELD-OUT EVENT:", held_event)
    print("##################################################")
    print("Training events:", ", ".join(training_events))
    print(
        "Valid 3-event training-label rows:",
        f"{np.count_nonzero(valid_train_label):,}",
    )
    print(
        "Valid held-event label rows:",
        f"{np.count_nonzero(held_valid):,}",
    )

    for fold in range(1, 6):

        train_mask = valid_train_label & (SPATIAL_FOLD != fold)
        test_mask = held_valid & (SPATIAL_FOLD == fold)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        y_train = train_target[train_idx]
        y_test = held_labels[test_idx]

        train_blocks = np.unique(BLOCKS[train_idx])
        test_blocks = np.unique(BLOCKS[test_idx])
        overlap = np.intersect1d(train_blocks, test_blocks)

        if len(overlap) != 0:
            raise Exception(
                f"Spatial leakage in held event {held_event}, fold {fold}."
            )

        print("\n----------------------------------------------")
        print(f"Event {held_event} | Spatial Fold {fold}/5")
        print("----------------------------------------------")
        print(
            f"Train N: {len(train_idx):,} | flood: "
            f"{np.count_nonzero(y_train == 1):,} | flood%: "
            f"{100*np.mean(y_train == 1):.2f}%"
        )
        print(
            f"Test N : {len(test_idx):,} | flood: "
            f"{np.count_nonzero(y_test == 1):,} | flood%: "
            f"{100*np.mean(y_test == 1):.2f}%"
        )
        print("Block overlap:", len(overlap))

        preprocessor = make_preprocessor()
        X_train = preprocessor.fit_transform(X[train_idx]).astype(np.float32)
        X_test = preprocessor.transform(X[test_idx]).astype(np.float32)

        if not np.all(np.isfinite(X_train)):
            raise Exception("Non-finite values in training matrix.")
        if not np.all(np.isfinite(X_test)):
            raise Exception("Non-finite values in test matrix.")

        rf, xgb, scale_pos_weight = build_models(y_train)

        print("RF training...")
        rf.fit(X_train, y_train)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        rf_m = metrics(y_test, rf_prob)
        print(
            f"RF  AUC={rf_m['AUC']:.6f} "
            f"AP={rf_m['AP']:.6f} "
            f"Brier={rf_m['Brier']:.6f}"
        )

        print("XGB training...")
        xgb.fit(X_train, y_train)
        xgb_prob = xgb.predict_proba(X_test)[:, 1]
        xgb_m = metrics(y_test, xgb_prob)
        print(
            f"XGB AUC={xgb_m['AUC']:.6f} "
            f"AP={xgb_m['AP']:.6f} "
            f"Brier={xgb_m['Brier']:.6f}"
        )

        # Save OOF predictions for held event.
        oof[held_event]["Observed"][test_idx] = y_test
        oof[held_event]["Fold"][test_idx] = fold
        oof[held_event]["RF"][test_idx] = rf_prob.astype(np.float32)
        oof[held_event]["XGB"][test_idx] = xgb_prob.astype(np.float32)

        for model_name, m in [
            ("Random_Forest", rf_m),
            ("XGBoost", xgb_m),
        ]:
            scenario_rows.append({
                "Held_Event": held_event,
                "Training_Events": ";".join(training_events),
                "Spatial_Fold": fold,
                "Model": model_name,
                "Train_N": len(train_idx),
                "Train_Flood_N": int(np.count_nonzero(y_train == 1)),
                "Train_Flood_Percent": float(100*np.mean(y_train == 1)),
                "Test_N": len(test_idx),
                "Test_Flood_N": int(np.count_nonzero(y_test == 1)),
                "Test_Flood_Percent": float(100*np.mean(y_test == 1)),
                "Train_Block_N": len(train_blocks),
                "Test_Block_N": len(test_blocks),
                "Block_Overlap_N": len(overlap),
                "Encoded_Feature_N": X_train.shape[1],
                "XGB_Scale_Pos_Weight": scale_pos_weight if model_name == "XGBoost" else "",
                "ROC_AUC": m["AUC"],
                "Average_Precision": m["AP"],
                "Brier_Score": m["Brier"],
                "Mean_Probability_Flood": m["Mean_Prob_Flood"],
                "Mean_Probability_NonFlood": m["Mean_Prob_NonFlood"],
                "Probability_Separation": m["Probability_Separation"],
            })

# ============================================================
# 05. SAVE SCENARIO TABLE
# ============================================================

scenario_headers = [
    "Held_Event",
    "Training_Events",
    "Spatial_Fold",
    "Model",
    "Train_N",
    "Train_Flood_N",
    "Train_Flood_Percent",
    "Test_N",
    "Test_Flood_N",
    "Test_Flood_Percent",
    "Train_Block_N",
    "Test_Block_N",
    "Block_Overlap_N",
    "Encoded_Feature_N",
    "XGB_Scale_Pos_Weight",
    "ROC_AUC",
    "Average_Precision",
    "Brier_Score",
    "Mean_Probability_Flood",
    "Mean_Probability_NonFlood",
    "Probability_Separation",
]

with open(SCENARIO_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=scenario_headers)
    writer.writeheader()
    writer.writerows(scenario_rows)

# ============================================================
# 06. EVENT-LEVEL POOLED SUMMARY
# ============================================================

event_summary_rows = []

print("\n==================================================")
print("EVENT-LEVEL STRICT LOEO SUMMARY")
print("==================================================")

for held_event in EVENTS:
    valid = oof[held_event]["Observed"] >= 0
    y = oof[held_event]["Observed"][valid]

    for model_name, key in [
        ("Random_Forest", "RF"),
        ("XGBoost", "XGB"),
    ]:
        p = oof[held_event][key][valid]

        if np.any(~np.isfinite(p)):
            raise Exception(
                f"Missing OOF predictions for {held_event} {model_name}."
            )

        pooled = metrics(y, p)

        fold_rows = [
            r for r in scenario_rows
            if r["Held_Event"] == held_event and r["Model"] == model_name
        ]

        fold_auc = np.array([r["ROC_AUC"] for r in fold_rows], dtype=float)
        fold_ap = np.array([r["Average_Precision"] for r in fold_rows], dtype=float)

        row = {
            "Held_Event": held_event,
            "Model": model_name,
            "Fold_Mean_AUC": float(np.mean(fold_auc)),
            "Fold_SD_AUC": float(np.std(fold_auc, ddof=1)),
            "Fold_Min_AUC": float(np.min(fold_auc)),
            "Fold_Max_AUC": float(np.max(fold_auc)),
            "Fold_Mean_AP": float(np.mean(fold_ap)),
            "Fold_SD_AP": float(np.std(fold_ap, ddof=1)),
            "Pooled_N": len(y),
            "Pooled_Flood_N": int(np.count_nonzero(y == 1)),
            "Pooled_Flood_Percent": float(100*np.mean(y == 1)),
            "Pooled_AUC": pooled["AUC"],
            "Pooled_AP": pooled["AP"],
            "Pooled_Brier": pooled["Brier"],
            "Pooled_Mean_Prob_Flood": pooled["Mean_Prob_Flood"],
            "Pooled_Mean_Prob_NonFlood": pooled["Mean_Prob_NonFlood"],
            "Pooled_Probability_Separation": pooled["Probability_Separation"],
        }

        event_summary_rows.append(row)

        print(
            f"{held_event} {model_name:15s} "
            f"mean-fold AUC={row['Fold_Mean_AUC']:.6f} "
            f"SD={row['Fold_SD_AUC']:.6f} "
            f"pooled AUC={row['Pooled_AUC']:.6f} "
            f"pooled AP={row['Pooled_AP']:.6f}"
        )

EVENT_HEADERS = list(event_summary_rows[0].keys())
with open(EVENT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=EVENT_HEADERS)
    writer.writeheader()
    writer.writerows(event_summary_rows)

# ============================================================
# 07. OVERALL MACRO SUMMARY
# ============================================================

overall_rows = []

print("\n==================================================")
print("OVERALL STRICT LOEO SUMMARY")
print("==================================================")

for model_name in ["Random_Forest", "XGBoost"]:
    e_rows = [r for r in event_summary_rows if r["Model"] == model_name]
    s_rows = [r for r in scenario_rows if r["Model"] == model_name]

    event_auc = np.array([r["Pooled_AUC"] for r in e_rows], dtype=float)
    event_ap = np.array([r["Pooled_AP"] for r in e_rows], dtype=float)
    event_brier = np.array([r["Pooled_Brier"] for r in e_rows], dtype=float)

    scenario_auc = np.array([r["ROC_AUC"] for r in s_rows], dtype=float)
    scenario_ap = np.array([r["Average_Precision"] for r in s_rows], dtype=float)

    row = {
        "Model": model_name,
        "Macro_Event_AUC": float(np.mean(event_auc)),
        "Event_AUC_SD": float(np.std(event_auc, ddof=1)),
        "Event_AUC_Min": float(np.min(event_auc)),
        "Event_AUC_Max": float(np.max(event_auc)),
        "Macro_Event_AP": float(np.mean(event_ap)),
        "Macro_Event_Brier": float(np.mean(event_brier)),
        "Scenario_Mean_AUC": float(np.mean(scenario_auc)),
        "Scenario_SD_AUC": float(np.std(scenario_auc, ddof=1)),
        "Scenario_Min_AUC": float(np.min(scenario_auc)),
        "Scenario_Max_AUC": float(np.max(scenario_auc)),
        "Scenario_Mean_AP": float(np.mean(scenario_ap)),
        "Scenario_SD_AP": float(np.std(scenario_ap, ddof=1)),
    }

    overall_rows.append(row)

    print(
        f"{model_name:15s} "
        f"macro event AUC={row['Macro_Event_AUC']:.6f} "
        f"SD={row['Event_AUC_SD']:.6f} "
        f"scenario mean AUC={row['Scenario_Mean_AUC']:.6f} "
        f"scenario SD={row['Scenario_SD_AUC']:.6f}"
    )

OVERALL_HEADERS = list(overall_rows[0].keys())
with open(OVERALL_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=OVERALL_HEADERS)
    writer.writeheader()
    writer.writerows(overall_rows)

# ============================================================
# 08. PAIRED RF VS XGB — 20 SCENARIOS
# ============================================================

paired_rows = []

for held_event in EVENTS:
    for fold in range(1, 6):
        rf_row = next(
            r for r in scenario_rows
            if r["Held_Event"] == held_event
            and r["Spatial_Fold"] == fold
            and r["Model"] == "Random_Forest"
        )
        xgb_row = next(
            r for r in scenario_rows
            if r["Held_Event"] == held_event
            and r["Spatial_Fold"] == fold
            and r["Model"] == "XGBoost"
        )

        paired_rows.append({
            "Held_Event": held_event,
            "Spatial_Fold": fold,
            "RF_AUC": rf_row["ROC_AUC"],
            "XGB_AUC": xgb_row["ROC_AUC"],
            "XGB_minus_RF_AUC": xgb_row["ROC_AUC"] - rf_row["ROC_AUC"],
            "RF_AP": rf_row["Average_Precision"],
            "XGB_AP": xgb_row["Average_Precision"],
            "XGB_minus_RF_AP": xgb_row["Average_Precision"] - rf_row["Average_Precision"],
        })

with open(PAIRED_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
    writer.writeheader()
    writer.writerows(paired_rows)

auc_diff = np.array([r["XGB_minus_RF_AUC"] for r in paired_rows], dtype=float)
ap_diff = np.array([r["XGB_minus_RF_AP"] for r in paired_rows], dtype=float)

print("\n========== RF vs XGB: 20 PAIRED SCENARIOS ==========")
print("Mean XGB-RF AUC difference:", f"{np.mean(auc_diff):.6f}")
print("XGB AUC wins:", f"{np.count_nonzero(auc_diff > 0)}/20")
print("RF AUC wins :", f"{np.count_nonzero(auc_diff < 0)}/20")
print("Mean XGB-RF AP difference :", f"{np.mean(ap_diff):.6f}")

# ============================================================
# 09. SAVE OOF ARRAYS
# ============================================================

save_dict = {
    "X": X_COORD,
    "Y": Y_COORD,
    "Raster_Row": ROWS,
    "Raster_Column": COLS,
    "Block_5km": BLOCKS,
    "Spatial_Fold": SPATIAL_FOLD,
}

for event in EVENTS:
    save_dict[f"Observed_{event}"] = oof[event]["Observed"]
    save_dict[f"RF_{event}"] = oof[event]["RF"]
    save_dict[f"XGB_{event}"] = oof[event]["XGB"]

np.savez_compressed(OOF_NPZ, **save_dict)

# ============================================================
# 10. FINAL
# ============================================================

print("\n==============================================")
print("BUET B07U STRICT SPATIO-TEMPORAL LOEO COMPLETE")
print("==============================================")
print("FINAL STATUS: BUET B07U PASS")
