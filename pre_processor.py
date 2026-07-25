"""
Procmon CSV -> Sliding-window feature sequences for anomaly detection.

Pipeline:
  1. Load raw Procmon CSV export
  2. Parse timestamps, per-PID sort
  3. Parse the free-text 'Detail' column into numeric/categorical sub-features
  4. Engineer per-event features (operation type, path features, timing deltas)
  5. Build sliding windows of N consecutive events per PID
  6. Aggregate each window into a fixed-size feature vector
  7. Output: windowed_features.csv (ready for Isolation Forest / Markov / LSTM)

Usage:
    python preprocess_procmon.py --input Logfile.csv --output windowed_features.csv --window 20 --stride 5
"""

import argparse
import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Loading & basic cleaning
# ---------------------------------------------------------------------------

def load_procmon_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Parse "Time of Day" (Procmon default format: HH:MM:SS.ffffff, no date)
    # We only have time-of-day, so treat as a monotonically increasing float
    # (seconds since first event) -- fine as long as the capture doesn't cross midnight.
    def to_seconds(t):
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    df["t_sec"] = df["Time of Day"].apply(to_seconds)
    df = df.sort_values(["PID", "t_sec"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Detail column parsing
# ---------------------------------------------------------------------------
# The Detail field is a semi-structured "Key: Value, Key: Value" string that
# varies by Operation type. We extract the handful of fields that matter most
# for injection detection rather than trying to parse everything generically.

DETAIL_PATTERNS = {
    "offset": re.compile(r"Offset:\s*(\d+)"),
    "length": re.compile(r"Length:\s*(\d+)"),
    "page_execute": re.compile(r"PageProtection:\s*PAGE_EXECUTE"),
    "allocation_size": re.compile(r"AllocationSize:\s*(\d+)"),
    "end_of_file": re.compile(r"EndOfFile:\s*(\d+)"),
    "desired_access_execute": re.compile(r"Desired Access:.*Execute"),
    "desired_access_write": re.compile(r"Desired Access:.*Write"),
}


def parse_detail(detail: str) -> dict:
    if not isinstance(detail, str):
        detail = ""
    out = {}
    for key, pat in DETAIL_PATTERNS.items():
        m = pat.search(detail)
        if key in ("page_execute", "desired_access_execute", "desired_access_write"):
            out[key] = bool(m)
        else:
            out[key] = float(m.group(1)) if m else np.nan
    return out


# ---------------------------------------------------------------------------
# 3. Path features (avoid raw string -> use structural signals instead)
# ---------------------------------------------------------------------------

SUSPICIOUS_DIRS = ("downloads", "temp", "appdata\\local\\temp", "public")
EXEC_EXTENSIONS = (".dll", ".exe", ".sys", ".scr")


def path_features(path: str) -> dict:
    if not isinstance(path, str) or path == "":
        return {
            "path_depth": 0,
            "is_exec_ext": False,
            "in_suspicious_dir": False,
            "path_len": 0,
        }
    p = path.lower()
    return {
        "path_depth": p.count("\\"),
        "is_exec_ext": p.endswith(EXEC_EXTENSIONS),
        "in_suspicious_dir": any(d in p for d in SUSPICIOUS_DIRS),
        "path_len": len(p),
    }


# ---------------------------------------------------------------------------
# 4. Per-event feature construction
# ---------------------------------------------------------------------------

def build_event_features(df: pd.DataFrame) -> pd.DataFrame:
    detail_feats = df["Detail"].apply(parse_detail).apply(pd.Series)
    path_feats = df["Path"].apply(path_features).apply(pd.Series)

    events = pd.concat([df[["PID", "t_sec", "Operation", "Result"]], detail_feats, path_feats], axis=1)

    # time delta between consecutive events, per PID
    events["dt"] = events.groupby("PID")["t_sec"].diff().fillna(0)

    # one-hot the Operation (this becomes your "token" for sequence models too)
    events["op"] = events["Operation"].astype("category")

    events["is_success"] = (events["Result"] == "SUCCESS").astype(int)

    return events


# ---------------------------------------------------------------------------
# 5. Sliding window aggregation
# ---------------------------------------------------------------------------

# The operations most relevant to process injection detection (MITRE T1055).
# Anything outside this list gets bucketed as "other" to keep the vector small.
KEY_OPS = [
    "CreateFile", "CreateFileMapping", "ReadFile", "WriteFile",
    "Thread Create", "Process Create", "LoadImage", "VirtualAlloc",
    "WriteProcessMemory", "CreateRemoteThread", "QueryBasicInformationFile",
    "QueryStandardInformationFile", "CloseFile", "FileSystemControl",
]


def op_bucket(op: str) -> str:
    return op if op in KEY_OPS else "Other"


def windowize(events: pd.DataFrame, window: int, stride: int) -> pd.DataFrame:
    events["op_bucket"] = events["Operation"].apply(op_bucket)
    rows = []

    for pid, group in events.groupby("PID"):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < window:
            windows = [(0, n)]  # short capture -> single window
        else:
            windows = [(i, i + window) for i in range(0, n - window + 1, stride)]

        for start, end in windows:
            w = group.iloc[start:end]
            if w.empty:
                continue

            feat = {
                "pid": pid,
                "window_start_t": w["t_sec"].iloc[0],
                "window_end_t": w["t_sec"].iloc[-1],
                "n_events": len(w),
                "mean_dt": w["dt"].mean(),
                "std_dt": (w["dt"].std() if len(w) > 1 else 0.0) or 0.0,
                "max_dt": w["dt"].max(),
                "frac_success": w["is_success"].mean(),
                "any_page_execute": int(w["page_execute"].any()),
                "any_exec_ext": int(w["is_exec_ext"].any()),
                "any_suspicious_dir": int(w["in_suspicious_dir"].any()),
                "any_desired_access_execute": int(w["desired_access_execute"].any()),
                "mean_offset": w["offset"].mean(),
                "offset_monotonic": int(w["offset"].dropna().is_monotonic_increasing) if w["offset"].notna().sum() > 1 else 1,
            }

            # operation frequency vector -- the core "sequence shape" signal
            op_counts = w["op_bucket"].value_counts(normalize=True)
            for op in KEY_OPS + ["Other"]:
                feat[f"op_frac_{op}"] = op_counts.get(op, 0.0)

            rows.append(feat)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="windowed_features.csv")
    ap.add_argument("--window", type=int, default=20, help="events per window")
    ap.add_argument("--stride", type=int, default=5, help="step size between windows")
    args = ap.parse_args()

    df = load_procmon_csv("svchost_first_run.csv")
    events = build_event_features(df)
    windows = windowize(events, args.window, args.stride)
    windows.to_csv(args.output, index=False)
    print(f"Wrote {len(windows)} windows ({args.window} events, stride {args.stride}) to {args.output}")


if __name__ == "__main__":
    main()