#!/usr/bin/env python3
"""
TC Classification — ensemble ML classifier.

Applies 6-model ensemble (RF/XGB/LGBM x ERA5/JRA3Q) to tracker output.
Ensemble threshold=3, min_hours=24 (hardcoded from verification).

TC output uses anal (raw tracker) wind/psl, not env parameters.
  - maxwind: max 850hPa wind within 350km of center
  - minpsl: min sea-level pressure within 500km of center

Usage:
    # Explicit file paths (preferred, used by run_detect.py)
    python run_classify.py --envi-file DIR/DBTRACK_X_envi.txt \\
                           --anal-file DIR/DBTRACK_X.txt \\
                           --output-file DIR/TC_X.txt

    # Legacy: glob-based discovery (fallback)
    python run_classify.py --model ACCESS-CM2 --scenario hist --input-dir DIR
"""
import argparse
import sys
import numpy as np
import joblib
from pathlib import Path

# Hardcoded from verification
ENSEMBLE_THRESHOLD = 3
MIN_HOURS = 24
MIN_STEPS = MIN_HOURS // 6
MODELS = ["rf", "xgb", "lgbm"]
DATASETS = ["ERA5", "JRA3Q"]
FEATURES = ["lat", "lon", "wind", "vws", "vor", "pres", "wcore250", "wcore250500"]
COL_IDX = {"lat": 5, "lon": 4, "wind": 6, "vws": 7,
           "vor": 8, "pres": 9, "wcore250": 10, "wcore250500": 11}
FEAT_INDICES = [COL_IDX[f] for f in FEATURES]


def load_envi(envi_path):
    """Load envi file. Returns (data_rows, track_boundaries).
    data_rows: list of [13 floats] per timestep
    track_boundaries: list of (header_line, start_idx, nstep)
    """
    all_rows = []
    track_bounds = []

    with open(envi_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 3:
            header = lines[i].rstrip()
            nstep = int(parts[1])
            start = len(all_rows)
            for j in range(1, nstep + 1):
                if i + j < len(lines):
                    p = lines[i + j].split()
                    if len(p) == 13:
                        all_rows.append([float(x) for x in p])
            actual = len(all_rows) - start
            if actual > 0:
                track_bounds.append((header, start, actual))
            i += nstep + 1
        else:
            i += 1

    data = np.array(all_rows) if all_rows else np.empty((0, 13))
    return data, track_bounds


def load_anal(anal_path):
    """Load anal (raw tracker) file. Returns list of track dicts.
    Each track: {header, lines: [(yyyy mm dd hh, lon, lat, wind, psl), ...]}
    """
    tracks = []
    with open(anal_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 3:
            header = lines[i].rstrip()
            nstep = int(parts[1])
            track_lines = []
            for j in range(1, nstep + 1):
                if i + j < len(lines):
                    track_lines.append(lines[i + j].rstrip())
            tracks.append({"header": header, "lines": track_lines})
            i += nstep + 1
        else:
            i += 1
    return tracks


def classify_and_write(envi_path, anal_path, output_path, classifiers):
    """Run ensemble classification on one envi+anal pair, write TC output."""
    data, track_bounds = load_envi(str(envi_path))
    if len(data) == 0:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        open(output_path, "w").close()
        print(f"  No data -> {output_path}")
        return 0

    anal_by_tid = {}
    anal_dup = 0
    for t in load_anal(str(anal_path)):
        tid = int(t["header"].split()[2])
        if tid in anal_by_tid:
            anal_dup += 1          # duplicate track id in DBTRACK -> malformed
        anal_by_tid[tid] = t

    # Preflight (#7): envi must match anal (DBTRACK) per track as a SET of points,
    # key = (y, m, d, h, lon, lat). Labels are bound below BY KEY (not by row
    # position), so the two need not be in the same ORDER — a pure order difference
    # is logged (for detect-side investigation), not failed. What MUST hold:
    #  - identical track-id sets (a track absent from envi is caught, #1),
    #  - identical point set per track (stale/partial envi is caught),
    #  - no duplicate (time,lon,lat) within a track (else key->label is ambiguous).
    # Coords are (lon, lat) tuples in real column order — both anal (detect.py:383)
    # and envi write lon then lat, so a genuine lon/lat swap must NOT pass.
    envi_tids = [int(h.split()[2]) for h, _s, _n in track_bounds]
    envi_dup = len(envi_tids) - len(set(envi_tids))
    if envi_dup or anal_dup:
        raise SystemExit(
            f"ERROR: duplicate track ids (envi dup={envi_dup}, anal dup={anal_dup}) "
            f"for {envi_path}; malformed track files — re-run detect/env stage")
    if set(envi_tids) != set(anal_by_tid):
        raise SystemExit(
            f"ERROR: envi/anal track-id sets differ for {envi_path} "
            f"(envi {len(set(envi_tids))} vs anal {len(anal_by_tid)} unique tids); "
            f"re-run env stage")

    def _seq_rows(rows):
        return [(int(r[0]), int(r[1]), int(r[2]), int(r[3]),
                 round(float(r[4]), 2), round(float(r[5]), 2)) for r in rows]

    def _seq_lines(lines):
        out = []
        for ln in lines:
            q = ln.split()
            out.append((int(q[0]), int(q[1]), int(q[2]), int(q[3]),
                        round(float(q[4]), 2), round(float(q[5]), 2)))
        return out

    mism = 0
    order_diff = []
    for header, start, nstep in track_bounds:
        tid = int(header.split()[2])
        t = anal_by_tid.get(tid)
        if t is None:
            mism += 1
            continue
        eseq = _seq_rows(data[start:start + nstep])
        aseq = _seq_lines(t["lines"])
        if len(set(eseq)) != len(eseq) or len(set(aseq)) != len(aseq):
            raise SystemExit(
                f"ERROR: duplicate (time,lon,lat) within track {tid} in {envi_path}; "
                f"ambiguous label binding — fix detect/env output")
        if set(eseq) != set(aseq):
            mism += 1
        elif eseq != aseq:
            order_diff.append(tid)
    if mism:
        raise SystemExit(
            f"ERROR: envi/anal point-SET mismatch for {mism} track(s) in "
            f"{envi_path} (stale/partial envi?); re-run env stage")
    if order_diff:
        # Same points, different DBTRACK-vs-envi order. Labels bind by key so output
        # is still correct; surface the ids for detect-side ordering investigation.
        print(f"  INFO: {len(order_diff)} track(s) identical point set but different "
              f"DBTRACK/envi order (labels bound by key); tids[:50]={order_diff[:50]}",
              file=sys.stderr)

    X = data[:, FEAT_INDICES]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    preds = np.zeros((len(classifiers), len(X)), dtype=np.int32)
    for idx, (key, model) in enumerate(classifiers.items()):
        preds[idx] = model.predict(X)

    votes = np.sum(preds > 0, axis=0)
    y_ens = (votes >= ENSEMBLE_THRESHOLD).astype(np.int32)

    for header, start, nstep in track_bounds:
        tc_count = np.sum(y_ens[start:start + nstep] == 1)
        if tc_count < MIN_STEPS:
            y_ens[start:start + nstep] = 0

    # Bind labels BY KEY, not by row position (#2 fix). tid is globally unique and
    # keys are unique within a track (checked in preflight), so (tid, y, m, d, h,
    # lon, lat) is an unambiguous global key. Built AFTER the min-steps zeroing so it
    # reflects final labels.
    pred_by_key = {}
    for header, start, nstep in track_bounds:
        tid = int(header.split()[2])
        for r in range(start, start + nstep):
            row = data[r]
            key = (tid, int(row[0]), int(row[1]), int(row[2]), int(row[3]),
                   round(float(row[4]), 2), round(float(row[5]), 2))
            pred_by_key[key] = int(y_ens[r])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n_tc_tracks = 0
    with open(output_path, "w") as f:
        for header, start, nstep in track_bounds:
            if not np.any(y_ens[start:start + nstep] == 1):
                continue
            tid = int(header.split()[2])
            if tid not in anal_by_tid:
                continue
            anal_lines = anal_by_tid[tid]["lines"]
            # Output in original DBTRACK order; look up each timestep's label by key.
            f.write(f"{header.split()[0]}    {len(anal_lines):6d}    {tid:6d}\n")
            for ln in anal_lines:
                p = ln.split()
                key = (tid, int(p[0]), int(p[1]), int(p[2]), int(p[3]),
                       round(float(p[4]), 2), round(float(p[5]), 2))
                # Direct index, not .get(key, 0): preflight guarantees the key
                # exists, so a miss means an internal inconsistency — fail loudly
                # (KeyError) rather than silently emit a 0 label.
                f.write(f"{p[0]} {p[1]} {p[2]} {p[3]}  "
                        f"{float(p[4]):8.2f} {float(p[5]):8.2f}  "
                        f"{float(p[6]):7.2f} {float(p[7]):9.2f}  "
                        f"{pred_by_key[key]:d}\n")
            n_tc_tracks += 1

    print(f"  {n_tc_tracks} TC tracks -> {output_path}")
    return n_tc_tracks


def main():
    parser = argparse.ArgumentParser(description="TC Classification")
    # Explicit file mode (preferred)
    parser.add_argument("--envi-file", default=None, help="Path to envi file")
    parser.add_argument("--anal-file", default=None, help="Path to anal (tracker) file")
    parser.add_argument("--output-file", default=None, help="Path to TC output file")
    # Legacy glob mode (fallback)
    parser.add_argument("--model", default=None, help="Model name (legacy)")
    parser.add_argument("--scenario", default=None, help="Scenario (legacy)")
    parser.add_argument("--input-dir", default=None, help="Directory with anal+envi files (legacy)")
    parser.add_argument("--output-dir", default=None, help="Output directory (legacy)")
    args = parser.parse_args()

    # Load classifiers
    lib_dir = Path(__file__).parent / "lib"
    classifiers = {}
    for mn in MODELS:
        for dn in DATASETS:
            pkl_path = lib_dir / f"{mn}_{dn}.pkl"
            classifiers[f"{mn}_{dn}"] = joblib.load(str(pkl_path))
    print(f"Loaded {len(classifiers)} classifiers")

    # Mode 1: Explicit file paths
    if args.envi_file:
        envi_path = Path(args.envi_file)
        anal_path = Path(args.anal_file) if args.anal_file else Path(
            str(envi_path).replace("_envi.txt", ".txt"))
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            # Default: TC_ prefix in same directory
            name = envi_path.name.replace("DBTRACK_", "TC_").replace("_envi.txt", ".txt")
            output_path = envi_path.parent / name
        classify_and_write(envi_path, anal_path, output_path, classifiers)
        return 0

    # Mode 2: Legacy glob-based discovery
    if not args.input_dir:
        print("Error: provide --envi-file or --input-dir", file=sys.stderr)
        return 1

    project_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else project_dir

    envi_files = sorted(project_dir.glob("DBTRACK_*_envi.txt"))
    if not envi_files:
        print(f"Error: No envi files in {project_dir}")
        return 1

    for envi_path in envi_files:
        print(f"\nProcessing {envi_path.name}")
        anal_path = Path(str(envi_path).replace("_envi.txt", ".txt"))
        # TC_ output naming
        tc_name = envi_path.name.replace("DBTRACK_", "TC_").replace("_envi.txt", ".txt")
        output_path = output_dir / tc_name
        classify_and_write(envi_path, anal_path, output_path, classifiers)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
