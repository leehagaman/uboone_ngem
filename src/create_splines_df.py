
import uproot
import numpy as np
import polars as pl
import os
import time
import sys
import argparse
from pathlib import Path

parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.file_locations import intermediate_files_location, data_files_location

SPLINE_ROOT_FILE = "/nevis/riverside/data/leehagaman/ngem/data_files/run4b_nuoverlay_retuple_splines.root"
SPLINE_TREE = "spline_weights"
FILENAME = "checkout_MCC9.10_Run4b_v10_04_07_20_BNB_nu_overlay_retuple_retuple_hist.root"
FILETYPE = "nu_overlay"
DETAILED_RUN_PERIOD = "4b"

OUTPUT_PARQUET = f"{intermediate_files_location}/spline_weights_df.parquet"


def load_spline_weights_df(root_file_path, frac_events=1.0):
    """Load the spline_weights tree from the ROOT file into a polars DataFrame."""
    print(f"Opening {root_file_path}...")
    print(f"  Reading tree: {SPLINE_TREE}")
    f = uproot.open(root_file_path)
    t = f[SPLINE_TREE]

    total_entries = t.num_entries
    n_events = total_entries if frac_events >= 1.0 else max(1, int(total_entries * frac_events))

    # Identify weight columns (everything except run/subrun/event/entry)
    id_cols = {"run", "subrun", "event", "entry"}
    weight_cols = [c for c in t.keys() if c not in id_cols]

    print(f"  {total_entries} total entries, loading {n_events} ({frac_events:.0%})")
    print(f"  Weight columns ({len(weight_cols)}): {weight_cols}")

    slice_kwargs = {"entry_stop": n_events}

    print("  Loading id columns...")
    id_data = t.arrays(["run", "subrun", "event"], library="np", **slice_kwargs)

    print("  Loading weight columns...")
    weight_data = t.arrays(weight_cols, library="np", **slice_kwargs)

    f.close()

    data = {col: id_data[col] for col in ["run", "subrun", "event"]}
    for col in weight_cols:
        arr = weight_data[col]
        data[col] = [row.tolist() for row in arr]

    df = pl.DataFrame(data)
    df = df.with_columns([
        pl.col("run").cast(pl.Int32),
        pl.col("subrun").cast(pl.Int32),
        pl.col("event").cast(pl.Int32),
        pl.lit(DETAILED_RUN_PERIOD).alias("detailed_run_period"),
        pl.lit(FILENAME).alias("filename"),
        pl.lit(FILETYPE).alias("filetype"),
    ])

    print(f"  df shape: {df.shape}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Save spline_weights tree as a standalone weights dataframe")
    parser.add_argument("-f", "--frac_events", type=float, default=1.0,
                        help="Fraction of events to load from the spline ROOT file, in (0, 1]. Default: 1.0")
    args = parser.parse_args()

    if not (0.0 < args.frac_events <= 1.0):
        raise ValueError("--frac_events must be in the interval (0, 1].")

    start_time = time.time()

    spline_df = load_spline_weights_df(SPLINE_ROOT_FILE, frac_events=args.frac_events)

    print(f"Writing to {OUTPUT_PARQUET}...")
    spline_df.write_parquet(OUTPUT_PARQUET)
    file_size_gb = os.path.getsize(OUTPUT_PARQUET) / 1024**3
    elapsed = time.time() - start_time
    print(f"Done. {file_size_gb:.2f} GB, {elapsed:.1f} s")


if __name__ == "__main__":
    main()
