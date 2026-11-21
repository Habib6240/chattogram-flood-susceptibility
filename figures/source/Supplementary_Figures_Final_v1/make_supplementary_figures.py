#!/usr/bin/env python3
"""Generates Supplementary Figures S2 (MCDA sensitivity) and S3 (fold-level performance).
Usage:
  python make_supplementary_figures.py --prepare SOURCE_DIR   # (optional) rebuild plotted_data/*.csv from the audited source CSVs
  python make_supplementary_figures.py                        # plot from plotted_data/*.csv -> PNG and SVG
Source CSVs expected in SOURCE_DIR: DU15_Weight_Perturbation_PlusMinus20pct.csv, DU15_Leave_One_Factor_Out.csv,
DU15_ANP_Link_Removal_Sensitivity.csv, BUET_B06U_Fold_Performance.csv, BUET_B07U_LOEO_Scenario_Performance.csv.
No values are estimated; every plotted value is read from these files."""
import sys, os, argparse
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.join(HERE, "plotted_data")
ORDER = ["Rainfall", "Elevation", "Slope", "Drainage_Density", "Distance_River", "TWI", "LULC", "HSG", "NDVI", "Distance_Road", "Profile_Curvature", "Tangential_Curvature", "Aspect"]
LAB = {"Rainfall": "Rainfall", "Elevation": "Elevation", "Slope": "Slope", "Drainage_Density": "Drainage density", "Distance_River": "Distance to river", "TWI": "TWI", "LULC": "LULC", "HSG": "HSG",
       "NDVI": "NDVI", "Distance_Road": "Distance to road", "Profile_Curvature": "Profile curvature", "Tangential_Curvature": "Tangential curvature", "Aspect": "Aspect"}
# Okabe-Ito colour-blind-safe palette
C = {"AHP": "#0072B2", "FAHP": "#E69F00", "ANP": "#009E73", "RF": "#0072B2", "XGBoost": "#D55E00",
     2019: "#0072B2", 2021: "#E69F00", 2023: "#CC79A7", 2026: "#009E73"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.linewidth": 0.6, "svg.fonttype": "path", "figure.dpi": 150})

def prepare(src):
    os.makedirs(PD, exist_ok=True)
    pw = pd.read_csv(os.path.join(src, "DU15_Weight_Perturbation_PlusMinus20pct.csv"))
    pw[["Method", "Perturbed_Criterion", "Perturbation", "Baseline_AUC", "Scenario_AUC", "Delta_AUC"]].to_csv(os.path.join(PD, "figS2a_weight_perturbation.csv"), index=False)
    lo = pd.read_csv(os.path.join(src, "DU15_Leave_One_Factor_Out.csv"))
    lo[["Method", "Removed_Criterion", "Baseline_AUC", "LeaveOneOut_AUC", "Delta_AUC"]].to_csv(os.path.join(PD, "figS2b_leave_one_factor_out.csv"), index=False)
    an = pd.read_csv(os.path.join(src, "DU15_ANP_Link_Removal_Sensitivity.csv"))
    an[["Source_Criterion", "Removed_Link_Target", "Scenario_AUC", "Delta_AUC", "Status"]].to_csv(os.path.join(PD, "figS2c_anp_link_removal.csv"), index=False)
    b6 = pd.read_csv(os.path.join(src, "BUET_B06U_Fold_Performance.csv"))
    b6[["Spatial_Fold", "Model", "ROC_AUC", "Average_Precision"]].to_csv(os.path.join(PD, "figS3a_spatial_cv_folds.csv"), index=False)
    b7 = pd.read_csv(os.path.join(src, "BUET_B07U_LOEO_Scenario_Performance.csv"))
    b7[["Held_Event", "Spatial_Fold", "Model", "ROC_AUC", "Average_Precision", "Test_N", "Test_Flood_N"]].to_csv(os.path.join(PD, "figS3b_strict_scenarios.csv"), index=False)

def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")

def fig_s2():
    pw = pd.read_csv(os.path.join(PD, "figS2a_weight_perturbation.csv")); lo = pd.read_csv(os.path.join(PD, "figS2b_leave_one_factor_out.csv")); an = pd.read_csv(os.path.join(PD, "figS2c_anp_link_removal.csv"))
    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.95], hspace=0.5, wspace=0.12)
    ya = np.arange(len(ORDER))[::-1]
    off = {"AHP": 0.22, "FAHP": 0.0, "ANP": -0.22}
    # (a)
    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a")
    for m in ["AHP", "FAHP", "ANP"]:
        for pert, filled in [("-20%", True), ("+20%", False)]:
            d = pw[(pw.Method == m) & (pw.Perturbation == pert)].set_index("Perturbed_Criterion")
            y = [ya[i] + off[m] for i, f in enumerate(ORDER)]
            ax.scatter([d.loc[f, "Delta_AUC"] for f in ORDER], y, s=16, marker="o", facecolor=C[m] if filled else "white", edgecolor=C[m], linewidth=0.9, zorder=3)
    ax.axvline(0, color="#444", lw=0.6, zorder=1)
    ax.set_yticks(ya); ax.set_yticklabels([LAB[f] for f in ORDER]); ax.set_xlabel("Change in ROC-AUC")
    ax.set_title("Weight perturbation (\u00b120%)", loc="left"); ax.grid(axis="x", color="#ddd", lw=0.4, zorder=0)
    # (b)
    bx = fig.add_subplot(gs[0, 1], sharey=ax); panel_label(bx, "b")
    for m in ["AHP", "FAHP", "ANP"]:
        d = lo[lo.Method == m].set_index("Removed_Criterion")
        bx.scatter([d.loc[f, "Delta_AUC"] for f in ORDER], [ya[i] + off[m] for i, f in enumerate(ORDER)], s=16, marker="s", color=C[m], zorder=3)
    bx.axvline(0, color="#444", lw=0.6, zorder=1); plt.setp(bx.get_yticklabels(), visible=False)
    bx.set_xlabel("Change in ROC-AUC"); bx.set_title("Leave-one-factor-out", loc="left"); bx.grid(axis="x", color="#ddd", lw=0.4, zorder=0)
    h = [Line2D([], [], marker="o", ls="", color=C[m], label=m) for m in ["AHP", "FAHP", "ANP"]] + [Line2D([], [], marker="o", ls="", mfc=c, mec="#333", color="#333", label=l) for c, l in [("#333", "\u221220% (a)"), ("white", "+20% (a)")]] + [Line2D([], [], marker="s", ls="", color="#333", label="Factor removed (b)")]
    fig.legend(handles=h, loc="center", bbox_to_anchor=(0.5, 0.463), frameon=False, ncol=6, handletextpad=0.3, columnspacing=1.0)
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    # (c)
    cx = fig.add_subplot(gs[1, :]); panel_label(cx, "c")
    d = an.sort_values("Delta_AUC").reset_index(drop=True)
    cols = ["#D55E00" if v < 0 else "#009E73" for v in d.Delta_AUC]
    cx.bar(np.arange(len(d)), d.Delta_AUC, color=cols, width=0.8, zorder=3)
    cx.axhline(0, color="#444", lw=0.6)
    cx.set_xlim(-1, len(d)); cx.set_xlabel("ANP link-removal scenarios (57), sorted by change in ROC-AUC"); cx.set_ylabel("Change in ROC-AUC"); cx.set_xticks([])
    cx.grid(axis="y", color="#ddd", lw=0.4, zorder=0); cx.set_title("ANP link removal", loc="left")
    def lab(i, dy):
        r = d.iloc[i]; t = f"{LAB[r.Source_Criterion]} \u2192 {LAB[r.Removed_Link_Target]}"
        cx.annotate(t, (i, r.Delta_AUC), xytext=(0, dy), textcoords="offset points", ha="left" if i < len(d) / 2 else "right", va="bottom" if dy > 0 else "top", fontsize=6.5)
    lab(0, -3); lab(1, -3); lab(len(d) - 1, 3); lab(len(d) - 2, 3)
    cx.set_ylim(d.Delta_AUC.min() * 1.5, d.Delta_AUC.max() * 1.35)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(HERE, f"Figure_S2_MCDA_sensitivity.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def fig_s3():
    a = pd.read_csv(os.path.join(PD, "figS3a_spatial_cv_folds.csv")); b = pd.read_csv(os.path.join(PD, "figS3b_strict_scenarios.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"wspace": 0.28})
    ax = axs[0]; panel_label(ax, "a")
    for m, mk in [("Random_Forest", "RF"), ("XGBoost", "XGBoost")]:
        d = a[a.Model == m].sort_values("Spatial_Fold"); xo = -0.08 if mk == "RF" else 0.08
        ax.plot(d.Spatial_Fold + xo, d.ROC_AUC, "-o", color=C[mk], ms=4.5, lw=0.8, zorder=3)
        ax.plot(d.Spatial_Fold + xo, d.Average_Precision, "--D", color=C[mk], ms=4, lw=0.8, mfc="white", zorder=3)
    ax.set_xticks(range(1, 6)); ax.set_xlabel("Spatial fold"); ax.set_ylabel("Metric value (composite target)"); ax.set_ylim(0.5, 1.0)
    ax.set_title("5-fold spatial block cross-validation", loc="left"); ax.grid(axis="y", color="#ddd", lw=0.4, zorder=0)
    ax.legend(handles=[Line2D([], [], color=C["RF"], marker="o", label="RF"), Line2D([], [], color=C["XGBoost"], marker="o", label="XGBoost"), Line2D([], [], color="#333", marker="o", label="ROC-AUC"), Line2D([], [], color="#333", ls="--", marker="D", mfc="white", label="Average precision")],
              loc="lower left", frameon=True, framealpha=0.9, edgecolor="#bbb", ncol=2, handletextpad=0.3)
    bx = axs[1]; panel_label(bx, "b")
    for e in [2019, 2021, 2023, 2026]:
        for m, mk in [("Random_Forest", "o"), ("XGBoost", "s")]:
            d = b[(b.Held_Event == e) & (b.Model == m)].sort_values("Spatial_Fold")
            xo = {2019: -0.18, 2021: -0.06, 2023: 0.06, 2026: 0.18}[e] + (-0.02 if m == "Random_Forest" else 0.02)
            bx.scatter(d.Spatial_Fold + xo, d.ROC_AUC, marker=mk, s=18, color=C[e], zorder=3, edgecolor="white", linewidth=0.3)
    bx.set_xticks(range(1, 6)); bx.set_xlabel("Held-out spatial fold"); bx.set_ylabel("ROC-AUC (held-event label)"); bx.set_ylim(0.65, 0.95)
    bx.set_title("Strict held-event and held-geography evaluation", loc="left"); bx.grid(axis="y", color="#ddd", lw=0.4, zorder=0)
    h = [Line2D([], [], marker="o", ls="", color=C[e], label=str(e)) for e in [2019, 2021, 2023, 2026]] + [Line2D([], [], marker="o", ls="", color="#333", label="RF"), Line2D([], [], marker="s", ls="", color="#333", label="XGBoost")]
    bx.legend(handles=h, loc="lower left", ncol=3, frameon=True, framealpha=0.9, edgecolor="#bbb", handletextpad=0.2, columnspacing=0.8)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(HERE, f"Figure_S3_fold_level_performance.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--prepare", default=None); args = ap.parse_args()
    if args.prepare: prepare(args.prepare)
    fig_s2(); fig_s3(); print("figures written")
