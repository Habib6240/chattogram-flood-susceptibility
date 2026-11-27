import pandas as pd
import os

# ============================================================
# SETTINGS
# ============================================================

BASE = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
    r"\13_BUET_ML\00_Event_Inventory"
)

INPUT_CSV = os.path.join(
    BASE,
    "03_B02A_Selected_Pair_Coverage_Audit",
    "BUET_B02A_Selected_Pair_Coverage_Audit.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "04_B02B_Final_Event_Selection"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "BUET_B02B_Event_Selection_Audit.csv"
)


# ============================================================
# READ
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)


# ============================================================
# QC
# ============================================================

required = [
    "pair_id",
    "event_id",
    "pre_date",
    "flood_date",
    "gap_days",
    "platform",
    "orbit_pass",
    "relative_orbit",
    "pre_scene_count",
    "flood_scene_count",
    "common_coverage_pct"
]

missing = [
    x for x in required
    if x not in df.columns
]

if missing:
    raise Exception(
        "Missing columns: "
        + ", ".join(missing)
    )


# ============================================================
# OBJECTIVE COVERAGE CLASS
# ============================================================

def coverage_class(x):

    if x >= 95:
        return "FULL_CORE"

    elif x >= 50:
        return "PARTIAL_CANDIDATE"

    else:
        return "REJECT_LOW_COVERAGE"


df["coverage_class"] = (
    df["common_coverage_pct"]
    .apply(coverage_class)
)


# ============================================================
# EVENT-SPECIFIC DECISION
#
# These decisions combine:
# - same-track geometry
# - 12-day temporal interval
# - common coverage
# - duplicate alternatives within the same flood event
#
# 2023 is NOT in B02A yet; it will be audited separately.
# ============================================================

decision = {

    "E02_2017_R114":
        (
            "HOLD_PARTIAL",
            "Valid 12-day same-track pair, but only ~64% "
            "common AOI coverage."
        ),

    "E03_2019_R41":
        (
            "CORE_KEEP",
            "Near-full AOI coverage and clean 12-day "
            "same-track PRE-flood pair."
        ),

    "E03_2019_R114_ALT":
        (
            "REJECT_ALTERNATIVE",
            "Same event has substantially better R41 pair."
        ),

    "E04_2021_R41":
        (
            "KEEP_AS_ALTERNATIVE",
            "Near-full coverage, but R77 provides 100% "
            "common coverage for the same event."
        ),

    "E04_2021_R77_ALT":
        (
            "CORE_KEEP",
            "100% common AOI coverage and clean 12-day "
            "same-track PRE-flood pair."
        ),

    "E06_2024_R114":
        (
            "HOLD_PARTIAL",
            "Flood-date pair is temporally useful, but only "
            "~56.5% common AOI coverage."
        ),

    "E07_2026_R77":
        (
            "CORE_KEEP",
            "100% common AOI coverage and clean 12-day "
            "same-track PRE-flood pair."
        )
}


statuses = []
reasons = []

for pair_id in df["pair_id"]:

    status, reason = decision.get(
        pair_id,
        (
            "REVIEW",
            "No predefined event decision."
        )
    )

    statuses.append(status)
    reasons.append(reason)


df["selection_status"] = statuses
df["selection_reason"] = reasons


# ============================================================
# FINAL SORT
# ============================================================

priority = {
    "CORE_KEEP": 1,
    "KEEP_AS_ALTERNATIVE": 2,
    "HOLD_PARTIAL": 3,
    "REJECT_ALTERNATIVE": 4,
    "REVIEW": 5
}

df["priority"] = (
    df["selection_status"]
    .map(priority)
    .fillna(99)
)

df = df.sort_values(
    [
        "priority",
        "event_id",
        "common_coverage_pct"
    ],
    ascending=[
        True,
        True,
        False
    ]
)


# ============================================================
# EXPORT
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# CONSOLE
# ============================================================

print(
    "\n========== BUET B02B EVENT SELECTION =========="
)

for _, r in df.iterrows():

    print(
        f"{r['pair_id']:20s} "
        f"Coverage={r['common_coverage_pct']:6.2f}%  "
        f"{r['selection_status']}"
    )

print(
    "\nCORE EVENTS:",
    list(
        df.loc[
            df["selection_status"] == "CORE_KEEP",
            "event_id"
        ]
    )
)

print(
    "\nOutput:",
    OUTPUT_CSV
)

print(
    "\nFINAL STATUS: B02B SELECTION AUDIT COMPLETE"
)