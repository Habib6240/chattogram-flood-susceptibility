#!/usr/bin/env python3
"""Step 3. Verify regenerated Distance_River against (a) the author's original sidecars if available, else (b) the published
summary statistics (n = 188,895; min 3.5687; max 14,031.58; mean 2,375.575 m).

Tolerances are EMPIRICAL: they were set from the agreement observed between an independent exact-geometry recomputation and the
original values (max |diff| 32.4 m, median 7.0 m, Pearson 0.99999 on all 188,895 cells). They are not a pre-specified standard.
The original tool and its settings are not recorded, so bit-identical values are not expected (differences of up to about one
30 m grid cell).
"""
import argparse, csv, gzip, sys, numpy as np
TOL = dict(max_abs_m=35.0, median_abs_m=10.0, mean_abs_m=12.0, pearson=0.9999, mean_stat_m=5.0, max_stat_m=35.0)
ap = argparse.ArgumentParser(); ap.add_argument("--regen", required=True, help="REGEN_Distance_River_Lattice.csv.gz")
ap.add_argument("--original", default=None, help="original Distance_River values (master CSV from the restricted record or a lattice sidecar), for a cell-by-cell check")
a = ap.parse_args()
def rd(p):
    with gzip.open(p, "rt", newline="") as f: return {(r["Raster_Row"], r["Raster_Column"]): float(r["Distance_River"]) for r in csv.DictReader(f)}
R = rd(a.regen); ok = True
v = np.array(list(R.values()))
print("regenerated: n=%d min=%.4f max=%.2f mean=%.3f" % (len(v), v.min(), v.max(), v.mean()))
def chk(name, cond): 
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)
chk("n == 188895", len(v) == 188895)
if a.original:
    O = rd(a.original); keys = list(O); chk("same lattice keys", set(keys) == set(R))
    o = np.array([O[k] for k in keys]); r = np.array([R[k] for k in keys]); e = np.abs(r - o)
    print("max|d|=%.2f median|d|=%.2f mean|d|=%.2f pearson=%.7f" % (e.max(), np.median(e), e.mean(), np.corrcoef(r, o)[0, 1]))
    chk("max |diff| <= %g m" % TOL["max_abs_m"], e.max() <= TOL["max_abs_m"]); chk("median |diff| <= %g m" % TOL["median_abs_m"], np.median(e) <= TOL["median_abs_m"])
    chk("mean |diff| <= %g m" % TOL["mean_abs_m"], e.mean() <= TOL["mean_abs_m"]); chk("Pearson >= %g" % TOL["pearson"], np.corrcoef(r, o)[0, 1] >= TOL["pearson"])
else:
    chk("mean within %g m of 2375.575" % TOL["mean_stat_m"], abs(v.mean() - 2375.575) <= TOL["mean_stat_m"])
    chk("max within %g m of 14031.58" % TOL["max_stat_m"], abs(v.max() - 14031.58) <= TOL["max_stat_m"])
    chk("min within %g m of 3.5687" % TOL["max_stat_m"], abs(v.min() - 3.5687) <= TOL["max_stat_m"])
    print("(statistics-only check: weaker than the cell-by-cell check)")
sys.exit(0 if ok else 1)
