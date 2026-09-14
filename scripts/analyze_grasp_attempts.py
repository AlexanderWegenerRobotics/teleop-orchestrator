"""Why didn't it close? Per-attempt breakdown of a live autonomous run.

The gripper failing to close has several distinguishable causes, and the
top-level plot can't separate them:

  1. the policy never asked            -> predicted width never dips
  2. it asked but too weakly           -> dips that miss the close threshold
  3. it asked and the wire dropped it  -> dips below threshold, flag never set
  4. it asked, closed, and missed      -> flag set, fingers moved, no grasp
  5. it asked in the wrong place       -> dips, but nowhere near a parcel

This walks every downward excursion in the policy's predicted gripper width
and reports which of those it was, including how far the end-effector was from
the nearest candidate at the moment of peak commitment. That last column is the
one that separates "the model can't see the object" from "the model can see it
but won't commit".

Reads the frames/candidate_features group (CANDIDATE_FEATURE_NAMES =
px_u, px_v, dist_left, dist_right) when present; runs without it, minus the
proximity column.

Usage:
    python scripts/analyze_grasp_attempts.py logs/run_20260808_155302.hdf5
    python scripts/analyze_grasp_attempts.py logs/run.hdf5 --threshold 0.04 --min-dip 0.005
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

ARMS = ["arm_left", "arm_right"]
OPEN_WIDTH_M = 0.08
# Column indices into candidate_features, per contracts/frame.py's
# CANDIDATE_FEATURE_NAMES. Looked up by name there; mirrored here so a
# reordering upstream shows up as a wrong number rather than silently.
FEAT_PX_U, FEAT_PX_V, FEAT_DIST_LEFT, FEAT_DIST_RIGHT = 0, 1, 2, 3
CAND_TYPE_OBJECT = 0  # bins are type 1; only objects are graspable


def find_dips(width: np.ndarray, baseline: float, min_dip: float):
    """Returns [(start, trough, end)] index triples for each downward excursion
    of at least min_dip below baseline. A 'dip' is a contiguous run below
    (baseline - min_dip); the trough is its argmin."""
    below = width < (baseline - min_dip)
    if not below.any():
        return []
    edges = np.diff(below.astype(int))
    starts = list(np.where(edges > 0)[0] + 1)
    ends = list(np.where(edges < 0)[0] + 1)
    if below[0]:
        starts.insert(0, 0)
    if below[-1]:
        ends.append(len(width))
    return [(s, s + int(np.argmin(width[s:e])), e) for s, e in zip(starts, ends) if e > s]


def nearest_object_dist(f, arm: str, tick: int):
    """Distance from this arm's EE to the nearest graspable candidate at tick,
    or None if the run predates candidate_features logging."""
    if "frames/candidate_features" not in f:
        return None
    feats = f["frames/candidate_features"][tick]
    mask = f["frames/candidate_mask"][tick]
    types = f["frames/candidate_types"][tick]
    col = FEAT_DIST_LEFT if arm == "arm_left" else FEAT_DIST_RIGHT
    sel = mask & (types == CAND_TYPE_OBJECT)
    if not sel.any():
        return None
    return float(np.min(feats[sel, col]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--threshold", type=float, default=0.04,
                    help="close threshold; keep in sync with run.py's _GRIPPER_CLOSE_THRESHOLD_M")
    ap.add_argument("--min-dip", type=float, default=0.005,
                    help="minimum drop below the open baseline to count as an attempt (m)")
    args = ap.parse_args()

    with h5py.File(args.run, "r") as f:
        ts = f["frames/timestamp_ns"][:]
        t = (ts - ts[0]) / 1e9
        has_cmd = "commanded" in f
        has_cand = "frames/candidate_features" in f

        print(f"{args.run}  |  {len(t)} ticks, {t[-1]:.1f} s, {len(t) / t[-1]:.1f} Hz")
        if not has_cand:
            print("  (no frames/candidate_features -- run predates it; proximity column unavailable)")
        print()

        for arm in ARMS:
            width = f[f"modules/policy/{arm}/gripper"][:]
            baseline = float(np.percentile(width, 90))  # the 'resting open' level
            dips = find_dips(width, baseline, args.min_dip)

            flag = f[f"commanded/{arm}/gripper_close_flag"][:] if has_cmd else None
            meas = f[f"commanded/{arm}/meas_gripper_width"][:] if has_cmd else None
            grasp = f[f"commanded/{arm}/meas_grasp_state"][:] if has_cmd else None

            print(f"=== {arm}   baseline(open) {baseline:.4f} m, threshold {args.threshold:.4f} m")
            if not dips:
                print("    no downward excursions at all -- the policy never attempted a grasp.\n")
                continue

            hdr = f"    {'#':>2} {'t(s)':>7} {'trough':>7} {'miss(mm)':>9} {'flag':>5} {'fingers':>8} {'grasp':>6}"
            if has_cand:
                hdr += f" {'nearest obj':>12}"
            print(hdr)
            for i, (s, tr, e) in enumerate(dips, 1):
                w = width[tr]
                miss = (w - args.threshold) * 1000.0
                fired = bool(flag[s:e].max() > 0.5) if flag is not None else None
                moved = (f"{(np.nanmax(meas[s:e]) - np.nanmin(meas[s:e])) * 1000:.1f}mm"
                         if meas is not None else "n/a")
                conf = ("yes" if grasp is not None and np.nanmax(grasp[s:e]) > 1 else "no")
                row = (f"    {i:>2} {t[tr]:>7.1f} {w:>7.4f} {miss:>+9.1f} "
                       f"{('YES' if fired else 'no'):>5} {moved:>8} {conf:>6}")
                if has_cand:
                    d = nearest_object_dist(f, arm, tr)
                    row += f" {(f'{d * 1000:.0f} mm' if d is not None else 'n/a'):>12}"
                print(row)

            missed = [width[tr] for _, tr, _ in dips if width[tr] >= args.threshold]
            if missed:
                worst = min(missed)
                print(f"    -> {len(missed)}/{len(dips)} attempts missed the threshold; "
                      f"closest got within {(worst - args.threshold) * 1000:.1f} mm of firing.")
                print(f"       a threshold of {worst + 0.001:.3f} would have fired "
                      f"{sum(1 for _, tr, _ in dips if width[tr] < worst + 0.001)}/{len(dips)}.")
            print()


if __name__ == "__main__":
    main()
