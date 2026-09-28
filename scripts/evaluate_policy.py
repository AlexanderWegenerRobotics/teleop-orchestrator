"""Score an evaluation run (run.py --trials N --timeout S) from the sim logs.

usage:
  python scripts/evaluate_policy.py logs/eval/act_sorting_relpos_v4_20260928_101500 \
      --sim-logs ../teleop-simulator/logs [--copy-sim]
  python scripts/evaluate_policy.py logs/eval/<run_a> logs/eval/<run_b> --sim-logs ...   (compare)

Each trial's sim episode folder is found by its episode_end label
("eval:<run_id>:<trial>", written by the orchestrator's episode_restart), with a
fallback on time overlap. Everything is measured inside the trial window
[start_ns, end_ns] from manifest.json, so the homing afterwards never counts.

Per trial:
  parcels         parcels in the scene whose colour has a bin in color_bin_mapping
  correct/wrong   parcels resting in their own / another bin at the end
                  (xy within BIN_R of the bin centre, below Z_REST, not moving)
  on_table        parcels still on the table at the end
  success         every parcel in its correct bin
  t_first / t_all time the first / last parcel settled in its correct bin
  closes, lifted  gripper closes and how many lifted a parcel >= 3 cm within 2 s
  faults          arm FAULT transitions, with the peak |F_ext| before them

Writes results.csv (one row per trial) and summary.json into the eval folder.
With --copy-sim the matched sim folders are copied to <eval>/sim/trial_XX, so
the evaluation is self-contained.
"""
import argparse
import json
import math
import os
import shutil
import sys

import numpy as np
import pandas as pd

BIN_R = 0.08          # parcel centre within this xy distance of a bin centre = in that bin
Z_REST = 0.80         # ...and below this height (not being carried over it)
V_REST = 0.03         # ...and slower than this (m/s)
SETTLE_S = 1.0        # a placement counts once it has held this long
ON_TABLE_Z, ON_TABLE_X = 0.75, 0.80
LIFT_M, LIFT_WINDOW_S, NEAR_XY = 0.03, 2.0, 0.06
RATE = 10.0           # scoring grid, Hz
FAULT = 6


# ── sim folder lookup ────────────────────────────────────────────────────────
def read_meta(folder):
    """(episode_end label, seed, color_bin_mapping dict) from arm_left_meta.csv."""
    path = os.path.join(folder, "arm_left_meta.csv")
    label, seed, mapping = None, None, {}
    if not os.path.exists(path):
        return label, seed, mapping
    with open(path) as f:
        rows = [ln.rstrip("\n").split(";") for ln in f if ln.strip()]
    if not rows:
        return label, seed, mapping
    hdr = rows[0]
    for r in rows[1:]:
        d = dict(zip(hdr, r))
        if d.get("event") == "episode_config":
            seed = d.get("seed")
            try:
                mapping = json.loads(d.get("color_bin_mapping") or "{}")
            except ValueError:
                mapping = {}
        elif d.get("event") == "episode_end":
            label = d.get("color_bin_mapping")   # the end reason lands in this column
    return label, seed, mapping


def index_sim_folders(roots):
    """label -> folder, plus (folder, t_first, t_last) for the time fallback."""
    by_label, spans = {}, []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            folder = os.path.join(root, name)
            if not os.path.isfile(os.path.join(folder, "scene.csv")):
                continue
            label, _, _ = read_meta(folder)
            if label:
                by_label.setdefault(label, folder)
            try:
                t = pd.read_csv(os.path.join(folder, "scene.csv"), sep=";", usecols=["wall_clock_ns"])
                spans.append((folder, int(t.wall_clock_ns.iloc[0]), int(t.wall_clock_ns.iloc[-1])))
            except Exception:
                pass
    return by_label, spans


def find_folder(trial, by_label, spans):
    if trial["label"] in by_label:
        return by_label[trial["label"]], "label"
    mid = (trial["start_ns"] + trial["end_ns"]) // 2
    for folder, t0, t1 in spans:
        if t0 <= mid <= t1:
            return folder, "time"
    return None, None


# ── scoring ──────────────────────────────────────────────────────────────────
def _grid(df, grid):
    idx = np.clip(np.searchsorted(df.wall_clock_ns.values, grid), 0, len(df) - 1)
    return df.iloc[idx].reset_index(drop=True)


def _held(mask, n):
    """True where mask has been True for the last n samples."""
    out = np.zeros_like(mask)
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        out[i] = run >= n
    return out


def score_trial(folder, t0, t1):
    label, seed, mapping = read_meta(folder)
    sp = os.path.join(folder, "scene.csv")
    hdr = pd.read_csv(sp, sep=";", nrows=0).columns
    use = [c for c in hdr if c == "wall_clock_ns" or c == "n_objects"
           or (c.startswith("obj") and c.split("_", 1)[1] in ("color", "x", "y", "z"))
           or (c.startswith("bin") and c.split("_", 1)[1] in ("name", "x", "y"))]
    scene = pd.read_csv(sp, sep=";", usecols=use)
    grid = np.arange(t0, t1, int(1e9 / RATE), dtype=np.int64)
    S = _grid(scene, grid)
    n_obj = int(S["n_objects"].iloc[0]) if "n_objects" in S else 4

    bins = {}
    for b in range(8):
        if f"bin{b}_name" in S and np.isfinite(S[f"bin{b}_x"].iloc[0]):
            bins[str(S[f"bin{b}_name"].iloc[0])] = (float(S[f"bin{b}_x"].iloc[0]), float(S[f"bin{b}_y"].iloc[0]))

    parcels = [k for k in range(n_obj)
               if f"obj{k}_color" in S and str(S[f"obj{k}_color"].iloc[0]) in mapping]
    hold = max(1, int(SETTLE_S * RATE))
    row = {"seed": seed, "parcels": len(parcels)}
    correct_since, final = [], {"correct": 0, "wrong": 0, "on_table": 0, "other": 0}
    for k in parcels:
        xyz = S[[f"obj{k}_x", f"obj{k}_y", f"obj{k}_z"]].values
        v = np.r_[0.0, np.linalg.norm(np.diff(xyz, axis=0), axis=1) * RATE]
        v = pd.Series(v).rolling(int(0.5 * RATE), min_periods=1).mean().values
        rest = (xyz[:, 2] < Z_REST) & (v < V_REST)
        target = mapping[str(S[f"obj{k}_color"].iloc[0])]
        in_bin = {name: rest & (np.hypot(xyz[:, 0] - bx, xyz[:, 1] - by) < BIN_R)
                  for name, (bx, by) in bins.items()}
        ok = _held(in_bin.get(target, np.zeros(len(grid), bool)), hold)
        wrong = _held(np.any([m for n, m in in_bin.items() if n != target] or [np.zeros(len(grid), bool)], axis=0), hold)
        correct_since.append(np.argmax(ok) / RATE if ok.any() else np.nan)
        if ok[-1]:
            final["correct"] += 1
        elif wrong[-1]:
            final["wrong"] += 1
        elif xyz[-1, 2] < ON_TABLE_Z and xyz[-1, 0] < ON_TABLE_X:
            final["on_table"] += 1
        else:
            final["other"] += 1
    row.update(final)
    row["success"] = bool(parcels) and final["correct"] == len(parcels)
    cs = np.array(correct_since, float)
    row["t_first_s"] = float(np.nanmin(cs)) if np.isfinite(cs).any() else np.nan
    row["t_all_s"] = float(np.nanmax(cs)) if row["success"] else np.nan

    closes = lifted = faults = 0
    fmax = 0.0
    obj_xyz = np.stack([S[[f"obj{k}_x", f"obj{k}_y", f"obj{k}_z"]].values for k in parcels], 1) if parcels else None
    for arm in ("arm_left", "arm_right"):
        p = os.path.join(folder, f"{arm}.csv")
        if not os.path.exists(p):
            continue
        cols = ["wall_clock_ns", "state", "gripper_cmd", "O_T_EE_world_12", "O_T_EE_world_13",
                "F_ext_0", "F_ext_1", "F_ext_2"]
        h = pd.read_csv(p, sep=";", nrows=0).columns
        d = pd.read_csv(p, sep=";", usecols=[c for c in cols if c in h])
        d = d[d.state.notna() & (d.wall_clock_ns >= t0) & (d.wall_clock_ns <= t1)].reset_index(drop=True)
        if d.empty:
            continue
        st = d.state.values
        for j in np.where((st[1:] == FAULT) & (st[:-1] != FAULT))[0] + 1:
            faults += 1
            w = d.iloc[max(0, j - 250):j + 1]
            fmax = max(fmax, float(np.linalg.norm(w[["F_ext_0", "F_ext_1", "F_ext_2"]].values, axis=1).max()))
        D = _grid(d, grid)
        closed = D.gripper_cmd.values < 0.04
        eng = D.state.values == 4
        for c in np.where(closed[1:] & ~closed[:-1] & eng[1:])[0] + 1:
            closes += 1
            if obj_xyz is None:
                continue
            ee = D[["O_T_EE_world_12", "O_T_EE_world_13"]].values[c]
            dxy = np.linalg.norm(obj_xyz[c, :, :2] - ee, axis=1)
            k = int(np.argmin(dxy))
            if dxy[k] > NEAR_XY:
                continue
            end = min(c + int(LIFT_WINDOW_S * RATE), len(grid) - 1)
            if obj_xyz[c:end + 1, k, 2].max() - obj_xyz[c, k, 2] > LIFT_M:
                lifted += 1
    row.update({"closes": closes, "lifted": lifted, "faults": faults, "fault_max_N": round(fmax)})
    return row


# ── statistics ───────────────────────────────────────────────────────────────
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_p(a, b, c, d):
    """Two-sided Fisher exact test on [[a, b], [c, d]]."""
    n1, n2, m = a + b, c + d, a + c
    n = n1 + n2

    def pmf(x):
        return math.comb(n1, x) * math.comb(n2, m - x) / math.comb(n, m)
    p0 = pmf(a)
    lo, hi = max(0, m - n2), min(m, n1)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p0 * (1 + 1e-9)))


def summarize(df, run_id):
    n = len(df)
    k = int(df.success.sum())
    lo, hi = wilson(k, n)
    parcels = int(df.parcels.sum())
    s = {
        "run_id": run_id, "trials": n, "success": k, "success_rate": k / n if n else np.nan,
        "success_ci95": [lo, hi],
        "parcels": parcels, "correct": int(df.correct.sum()), "wrong_bin": int(df.wrong.sum()),
        "sorted_fraction": df.correct.sum() / parcels if parcels else np.nan,
        "closes": int(df.closes.sum()), "lifted": int(df.lifted.sum()),
        "grasp_rate": df.lifted.sum() / df.closes.sum() if df.closes.sum() else np.nan,
        "t_first_median_s": float(df.t_first_s.median()),
        "t_all_median_s": float(df.t_all_s.median()),
        "faults": int(df.faults.sum()),
        "seeds": [str(x) for x in df.seed],
    }
    return s


def print_summary(s):
    lo, hi = s["success_ci95"]
    print(f"\n== {s['run_id']} ==")
    print(f"  success        {s['success']}/{s['trials']} ({s['success_rate']:.0%}, 95% CI {lo:.0%}-{hi:.0%})")
    print(f"  parcels sorted {s['correct']}/{s['parcels']} ({s['sorted_fraction']:.0%}), wrong bin {s['wrong_bin']}")
    print(f"  grasps         {s['lifted']}/{s['closes']} lifted ({s['grasp_rate']:.0%})")
    print(f"  time           first parcel {s['t_first_median_s']:.1f} s, all sorted {s['t_all_median_s']:.1f} s (medians)")
    print(f"  faults         {s['faults']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("eval_dirs", nargs="+")
    ap.add_argument("--sim-logs", default="../teleop-simulator/logs",
                    help="the sim's log folder (where the trial episode folders are)")
    ap.add_argument("--copy-sim", action="store_true",
                    help="copy each trial's sim folder into <eval>/sim/trial_XX")
    args = ap.parse_args()

    summaries = []
    for ed in args.eval_dirs:
        with open(os.path.join(ed, "manifest.json")) as f:
            man = json.load(f)
        by_label, spans = index_sim_folders([os.path.join(ed, "sim"), args.sim_logs])
        rows = []
        for t in man["trials"]:
            if t.get("end_reason") != "timeout":
                print(f"  [skip] trial {t['trial']}: ended by {t.get('end_reason')}")
                continue
            folder, how = find_folder(t, by_label, spans)
            if folder is None:
                print(f"  [skip] trial {t['trial']}: no sim folder with label {t['label']} "
                      f"or covering its time window")
                continue
            if args.copy_sim:
                dst = os.path.join(ed, "sim", f"trial_{t['trial']:02d}")
                if os.path.abspath(folder) != os.path.abspath(dst) and not os.path.exists(dst):
                    shutil.copytree(folder, dst)
                    folder = dst
            r = score_trial(folder, int(t["start_ns"]), int(t["end_ns"]))
            r.update({"trial": t["trial"], "sim_folder": folder, "matched_by": how,
                      "duration_s": t.get("duration_s"), "reset_ok": t.get("reset_ok")})
            rows.append(r)
            print(f"  trial {t['trial']:2d} [{os.path.basename(folder)}] seed {r['seed']}: "
                  f"{r['correct']}/{r['parcels']} correct, {r['wrong']} wrong, {r['on_table']} on table | "
                  f"grasps {r['lifted']}/{r['closes']} | faults {r['faults']}"
                  + (f" | all sorted at {r['t_all_s']:.1f} s" if r["success"] else ""))
        if not rows:
            print(f"{ed}: nothing to score")
            continue
        df = pd.DataFrame(rows)
        cols = ["trial", "seed", "parcels", "correct", "wrong", "on_table", "other", "success",
                "t_first_s", "t_all_s", "closes", "lifted", "faults", "fault_max_N",
                "duration_s", "reset_ok", "matched_by", "sim_folder"]
        df[cols].to_csv(os.path.join(ed, "results.csv"), index=False)
        s = summarize(df, man.get("run_id", os.path.basename(ed)))
        s["checkpoint"] = man.get("checkpoint")
        s["timeout_s"] = man.get("timeout_s")
        with open(os.path.join(ed, "summary.json"), "w") as f:
            json.dump(s, f, indent=2, default=float)
        print_summary(s)
        summaries.append(s)

    if len(summaries) >= 2:
        a = summaries[0]
        print("\n== comparison (Fisher exact, two-sided, against the first run) ==")
        for b in summaries[1:]:
            p_succ = fisher_p(a["success"], a["trials"] - a["success"], b["success"], b["trials"] - b["success"])
            p_grasp = fisher_p(a["lifted"], a["closes"] - a["lifted"], b["lifted"], b["closes"] - b["lifted"])
            p_sort = fisher_p(a["correct"], a["parcels"] - a["correct"], b["correct"], b["parcels"] - b["correct"])
            print(f"  {b['run_id']} vs {a['run_id']}: success p={p_succ:.3f}, parcels sorted p={p_sort:.3f}, "
                  f"grasp p={p_grasp:.3f}")
            if a["seeds"] != b["seeds"]:
                print("    note: the two runs did not see the same layouts (seed lists differ)")


if __name__ == "__main__":
    sys.exit(main())
