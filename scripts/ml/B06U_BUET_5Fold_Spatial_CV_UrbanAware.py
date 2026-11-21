import os
import csv
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    balanced_accuracy_score,
    f1_score,
    brier_score_loss,
)
from xgboost import XGBClassifier

# ============================================================
# B06U — 5-FOLD GEOGRAPHIC SPATIAL CV, URBAN-AWARE LABELS
#
# Design is intentionally unchanged from B06.
# - Same B04U predictors
# - Same 5 km blocks
# - EXACT reuse of original B06 block -> spatial-fold mapping
# - Same RF/XGB hyperparameters
# - Preprocessing fitted within each training fold
# - XGB class weight calculated within each training fold
# - Primary metrics: ROC-AUC and AP
# - Threshold metrics are supplementary; threshold is chosen from
#   TRAINING predictions only, never test labels.
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

# Reuse exact original B06 fold geometry for paired before/after comparison.
ORIGINAL_FOLD_FILE = os.path.join(
    BASE,
    "13_BUET_ML",
    "04_Spatial_CV",
    "BUET_B06_5km_Block_Spatial_Folds.csv",
)

OUTPUT_DIR = os.path.join(
    BASE,
    "13_BUET_ML",
    "04B_Spatial_CV_UrbanAware",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FOLD_PERFORMANCE_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B06U_Fold_Performance.csv",
)
SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B06U_Spatial_CV_Summary.csv",
)
OOF_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B06U_OutOfFold_Predictions.csv",
)
PAIRED_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B06U_RF_vs_XGB_Paired_Fold_Comparison.csv",
)
FOLD_COPY_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B06U_5km_Block_Spatial_Folds.csv",
)

for p in [INPUT_NPZ, ORIGINAL_FOLD_FILE]:
    if not os.path.exists(p):
        raise Exception("Required input missing:\n" + p)

D = np.load(INPUT_NPZ, allow_pickle=False)

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
    D["Distance_Road"],
]).astype(np.float32)

y_all = D["Composite_AnyFlood_All4Valid"].astype(np.int8)
block_all = D["Block_5km"].astype(np.int64)
x_all = D["X"].astype(np.float64)
ycoord_all = D["Y"].astype(np.float64)
row_all = D["Raster_Row"].astype(np.int32)
col_all = D["Raster_Column"].astype(np.int32)

valid = (y_all >= 0)
X = X_ALL[valid]
y = y_all[valid]
blocks = block_all[valid]
xcoord = x_all[valid]
ycoord = ycoord_all[valid]
rows = row_all[valid]
cols = col_all[valid]

print("\n========== B06U DATASET ==========")
print("Rows             :", f"{len(y):,}")
print("Flood            :", f"{np.count_nonzero(y == 1):,}")
print("Non-flood        :", f"{np.count_nonzero(y == 0):,}")
print("Flood %          :", f"{100*np.mean(y==1):.3f}%")
print("Unique 5km blocks:", len(np.unique(blocks)))

# ============================================================
# Load exact B06 spatial fold assignment
# ============================================================

block_to_fold = {}
fold_rows_raw = []

with open(ORIGINAL_FOLD_FILE, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames or []
    if "Block_5km" not in fields or "Spatial_Fold" not in fields:
        raise Exception(
            "Original B06 fold CSV must contain Block_5km and Spatial_Fold. "
            "Fields found: " + str(fields)
        )
    for r in reader:
        b = int(r["Block_5km"])
        fold = int(r["Spatial_Fold"])
        block_to_fold[b] = fold
        fold_rows_raw.append(r)

unique_blocks = np.unique(blocks)
missing = [int(b) for b in unique_blocks if int(b) not in block_to_fold]
if missing:
    raise Exception(
        "B06U blocks missing from original B06 fold mapping: " + str(missing[:20])
    )

spatial_fold = np.array(
    [block_to_fold[int(b)] for b in blocks],
    dtype=np.uint8,
)

print("\n========== B06 FOLD REUSE QC ==========")
print("Mapped rows:", f"{len(spatial_fold):,}")
print("Fold IDs   :", sorted(np.unique(spatial_fold).tolist()))
print("Status     : PASS")

# Save a local copy of the fold mapping.
with open(FOLD_COPY_CSV, "w", newline="", encoding="utf-8") as f:
    if fold_rows_raw:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows_raw[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows_raw)

NUMERIC_INDICES = [0,1,2,3,4,5,6,7,8,9,11,13,14]
CATEGORICAL_INDICES = [10,12]


def make_preprocessor():
    try:
        enc = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
    except TypeError:
        enc = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float32,
        )

    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_INDICES),
            ("categorical", enc, CATEGORICAL_INDICES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def training_youden_threshold(y_train, p_train):
    fpr, tpr, thresholds = roc_curve(y_train, p_train)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx])


def metric_row(model_name, fold, y_test, p_test, threshold,
               train_n, test_n, train_pos, test_pos,
               train_blocks_n, test_blocks_n):
    pred = (p_test >= threshold).astype(np.uint8)
    return {
        "Spatial_Fold": fold,
        "Model": model_name,
        "Train_N": train_n,
        "Test_N": test_n,
        "Train_Flood_N": train_pos,
        "Test_Flood_N": test_pos,
        "Train_Flood_Percent": 100*train_pos/train_n,
        "Test_Flood_Percent": 100*test_pos/test_n,
        "Train_Block_N": train_blocks_n,
        "Test_Block_N": test_blocks_n,
        "Threshold_From_Training": threshold,
        "ROC_AUC": roc_auc_score(y_test, p_test),
        "Average_Precision": average_precision_score(y_test, p_test),
        "Balanced_Accuracy": balanced_accuracy_score(y_test, pred),
        "F1": f1_score(y_test, pred, zero_division=0),
        "Brier": brier_score_loss(y_test, p_test),
    }

# OOF arrays
rf_oof = np.full(len(y), np.nan, dtype=np.float32)
xgb_oof = np.full(len(y), np.nan, dtype=np.float32)

results = []

for fold in range(1, 6):
    print("\n" + "="*55)
    print(f"B06U SPATIAL FOLD {fold}/5")
    print("="*55)

    test_idx = np.where(spatial_fold == fold)[0]
    train_idx = np.where(spatial_fold != fold)[0]

    train_blocks = np.unique(blocks[train_idx])
    test_blocks = np.unique(blocks[test_idx])
    overlap = np.intersect1d(train_blocks, test_blocks)
    if len(overlap) != 0:
        raise Exception(f"Fold {fold}: block leakage detected.")

    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    train_pos = int(np.count_nonzero(y_train == 1))
    train_neg = int(np.count_nonzero(y_train == 0))
    test_pos = int(np.count_nonzero(y_test == 1))

    print("Train N       :", f"{len(train_idx):,}")
    print("Test N        :", f"{len(test_idx):,}")
    print("Train blocks  :", len(train_blocks))
    print("Test blocks   :", len(test_blocks))
    print("Block overlap :", len(overlap))
    print("Train flood % :", f"{100*train_pos/len(train_idx):.3f}%")
    print("Test flood %  :", f"{100*test_pos/len(test_idx):.3f}%")

    pre = make_preprocessor()
    X_train = pre.fit_transform(X_train_raw).astype(np.float32)
    X_test = pre.transform(X_test_raw).astype(np.float32)

    # ---------------- RF ----------------
    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )

    print("RF training...")
    rf.fit(X_train, y_train)
    rf_train_prob = rf.predict_proba(X_train)[:,1]
    rf_test_prob = rf.predict_proba(X_test)[:,1]
    rf_thr = training_youden_threshold(y_train, rf_train_prob)
    rf_result = metric_row(
        "Random_Forest", fold, y_test, rf_test_prob, rf_thr,
        len(train_idx), len(test_idx), train_pos, test_pos,
        len(train_blocks), len(test_blocks)
    )
    results.append(rf_result)
    rf_oof[test_idx] = rf_test_prob.astype(np.float32)
    print(
        f"RF  AUC={rf_result['ROC_AUC']:.6f} "
        f"AP={rf_result['Average_Precision']:.6f} "
        f"BA={rf_result['Balanced_Accuracy']:.6f} "
        f"F1={rf_result['F1']:.6f}"
    )

    # ---------------- XGB ----------------
    scale_pos_weight = train_neg / train_pos
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

    print("XGB training...")
    xgb.fit(X_train, y_train)
    xgb_train_prob = xgb.predict_proba(X_train)[:,1]
    xgb_test_prob = xgb.predict_proba(X_test)[:,1]
    xgb_thr = training_youden_threshold(y_train, xgb_train_prob)
    xgb_result = metric_row(
        "XGBoost", fold, y_test, xgb_test_prob, xgb_thr,
        len(train_idx), len(test_idx), train_pos, test_pos,
        len(train_blocks), len(test_blocks)
    )
    results.append(xgb_result)
    xgb_oof[test_idx] = xgb_test_prob.astype(np.float32)
    print(
        f"XGB AUC={xgb_result['ROC_AUC']:.6f} "
        f"AP={xgb_result['Average_Precision']:.6f} "
        f"BA={xgb_result['Balanced_Accuracy']:.6f} "
        f"F1={xgb_result['F1']:.6f}"
    )

if np.any(~np.isfinite(rf_oof)) or np.any(~np.isfinite(xgb_oof)):
    raise Exception("OOF prediction coverage incomplete.")

# ============================================================
# Save fold performance
# ============================================================

headers = [
    "Spatial_Fold","Model","Train_N","Test_N","Train_Flood_N","Test_Flood_N",
    "Train_Flood_Percent","Test_Flood_Percent","Train_Block_N","Test_Block_N",
    "Threshold_From_Training","ROC_AUC","Average_Precision","Balanced_Accuracy",
    "F1","Brier"
]

with open(FOLD_PERFORMANCE_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

# ============================================================
# OOF predictions
# ============================================================

with open(OOF_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Raster_Row","Raster_Column","X","Y","Block_5km","Spatial_Fold",
        "Observed_Composite_Label","RF_OOF_Probability","XGB_OOF_Probability"
    ])
    for i in range(len(y)):
        writer.writerow([
            int(rows[i]), int(cols[i]), float(xcoord[i]), float(ycoord[i]),
            int(blocks[i]), int(spatial_fold[i]), int(y[i]),
            float(rf_oof[i]), float(xgb_oof[i])
        ])

# ============================================================
# Summary
# ============================================================

summary_rows = []
model_summaries = {}

print("\n========== B06U SPATIAL CV SUMMARY ==========")

for model_name, oof in [
    ("Random_Forest", rf_oof),
    ("XGBoost", xgb_oof),
]:
    mr = [r for r in results if r["Model"] == model_name]
    aucs = np.array([r["ROC_AUC"] for r in mr], dtype=float)
    aps = np.array([r["Average_Precision"] for r in mr], dtype=float)
    bas = np.array([r["Balanced_Accuracy"] for r in mr], dtype=float)
    f1s = np.array([r["F1"] for r in mr], dtype=float)

    pooled_auc = roc_auc_score(y, oof)
    pooled_ap = average_precision_score(y, oof)
    pooled_brier = brier_score_loss(y, oof)

    row = {
        "Model": model_name,
        "Mean_Fold_AUC": float(np.mean(aucs)),
        "SD_Fold_AUC": float(np.std(aucs, ddof=1)),
        "Min_Fold_AUC": float(np.min(aucs)),
        "Max_Fold_AUC": float(np.max(aucs)),
        "Mean_Fold_AP": float(np.mean(aps)),
        "SD_Fold_AP": float(np.std(aps, ddof=1)),
        "Mean_Fold_BA": float(np.mean(bas)),
        "Mean_Fold_F1": float(np.mean(f1s)),
        "Pooled_OOF_AUC": float(pooled_auc),
        "Pooled_OOF_AP": float(pooled_ap),
        "Pooled_OOF_Brier": float(pooled_brier),
    }
    summary_rows.append(row)
    model_summaries[model_name] = row

    print(
        f"{model_name:15s} "
        f"mean AUC={row['Mean_Fold_AUC']:.6f} "
        f"SD={row['SD_Fold_AUC']:.6f} "
        f"min={row['Min_Fold_AUC']:.6f} "
        f"max={row['Max_Fold_AUC']:.6f} "
        f"pooled AUC={row['Pooled_OOF_AUC']:.6f} "
        f"pooled AP={row['Pooled_OOF_AP']:.6f}"
    )

summary_headers = [
    "Model","Mean_Fold_AUC","SD_Fold_AUC","Min_Fold_AUC","Max_Fold_AUC",
    "Mean_Fold_AP","SD_Fold_AP","Mean_Fold_BA","Mean_Fold_F1",
    "Pooled_OOF_AUC","Pooled_OOF_AP","Pooled_OOF_Brier"
]

with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_headers)
    writer.writeheader()
    writer.writerows(summary_rows)

# ============================================================
# Paired RF-XGB fold comparison
# ============================================================

rf_rows = {r["Spatial_Fold"]: r for r in results if r["Model"] == "Random_Forest"}
xgb_rows = {r["Spatial_Fold"]: r for r in results if r["Model"] == "XGBoost"}

paired = []
for fold in range(1,6):
    ra = rf_rows[fold]
    xa = xgb_rows[fold]
    paired.append({
        "Spatial_Fold": fold,
        "RF_AUC": ra["ROC_AUC"],
        "XGB_AUC": xa["ROC_AUC"],
        "XGB_minus_RF_AUC": xa["ROC_AUC"] - ra["ROC_AUC"],
        "RF_AP": ra["Average_Precision"],
        "XGB_AP": xa["Average_Precision"],
        "XGB_minus_RF_AP": xa["Average_Precision"] - ra["Average_Precision"],
    })

with open(PAIRED_CSV, "w", newline="", encoding="utf-8") as f:
    headers2 = list(paired[0].keys())
    writer = csv.DictWriter(f, fieldnames=headers2)
    writer.writeheader()
    writer.writerows(paired)

auc_diffs = np.array([p["XGB_minus_RF_AUC"] for p in paired])
ap_diffs = np.array([p["XGB_minus_RF_AP"] for p in paired])

print("\n========== RF vs XGB: 5 PAIRED FOLDS ==========")
print("Mean XGB-RF AUC difference:", f"{np.mean(auc_diffs):+.6f}")
print("XGB AUC wins:", f"{np.count_nonzero(auc_diffs > 0)}/5")
print("RF AUC wins :", f"{np.count_nonzero(auc_diffs < 0)}/5")
print("Mean XGB-RF AP difference :", f"{np.mean(ap_diffs):+.6f}")

print("\n==============================================")
print("BUET B06U 5-FOLD SPATIAL CV COMPLETE")
print("==============================================")
print("FINAL STATUS: BUET B06U PASS")
