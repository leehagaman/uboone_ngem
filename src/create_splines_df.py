
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

from src.file_locations import intermediate_files_location

SPLINE_ROOT_FILES = [
    "/nevis/riverside/data/leehagaman/PROfit_files/spline_weight_files/checkout_run4b_nu_overlay_retuple_splines.root",
    "/nevis/riverside/data/leehagaman/PROfit_files/spline_weight_files/checkout_run4b_nue_overlay_splines.root",
    "/nevis/riverside/data/leehagaman/PROfit_files/spline_weight_files/checkout_run_4b_dirt_overlay_splines.root",
    "/nevis/riverside/data/leehagaman/PROfit_files/spline_weight_files/checkout_run4b_ncpi0_overlay_splines.root",
    "/nevis/riverside/data/leehagaman/PROfit_files/spline_weight_files/checkout_run4b_numuccpi0_overlay_splines.root",
]

OUTPUT_PARQUET = f"{intermediate_files_location}/spline_weights_df.parquet"


def _filetype_from_filename(filename):
    fn = filename.lower()
    if "nu_overlay" in fn or "nuoverlay" in fn:
        return "nu_overlay"
    if "nue_overlay" in fn:
        return "nue_overlay"
    if "dirt" in fn:
        return "dirt_overlay"
    if "nc_pi0" in fn or "ncpi0" in fn or "nc_pio" in fn or "ncpio" in fn:
        return "nc_pi0_overlay"
    if "ccpi0" in fn:
        return "numucc_pi0_overlay"
    raise ValueError(f"Unknown filetype for spline file: {filename}")


def _detailed_run_period_from_filename(filename):
    fn = filename.lower()
    if "4a.root" in filename:
        return "4a"
    elif "4b.root" in filename:
        return "4b"
    elif "4c.root" in filename:
        return "4c"
    elif "4d.root" in filename:
        return "4d"
    elif "4bcd.root" in filename:
        return "4bcd"
    elif "5.root" in filename:
        return "5"
    elif "run4a" in fn or "_4a_" in fn:
        return "4a"
    elif "run4b" in fn or "_4b_" in fn or fn.endswith("4b_splines.root"):
        return "4b"
    elif "run4c" in fn or "_4c_" in fn:
        return "4c"
    elif "run4d" in fn or "_4d_" in fn:
        return "4d"
    elif "run4bcd" in fn:
        return "4bcd"
    elif "run5" in fn:
        return "5"
    raise ValueError(f"Cannot determine detailed_run_period from spline filename: {filename}")


def load_spline_weights_df(root_file_path, frac_events=1.0, chunk_size=10_000):
    """Load the spline_weights tree (and weightsReint from nuselection) into a polars DataFrame."""
    filename = os.path.basename(root_file_path)
    filetype = _filetype_from_filename(filename)
    detailed_run_period = _detailed_run_period_from_filename(filename)

    print(f"Opening {root_file_path}...")
    print(f"  filetype={filetype}, detailed_run_period={detailed_run_period}")
    f = uproot.open(root_file_path)
    t = f["spline_weights"]
    t_nusel = f['nuselection/NeutrinoSelectionFilter']

    total_entries = t.num_entries
    n_events = total_entries if frac_events >= 1.0 else max(1, int(total_entries * frac_events))

    id_cols = {"run", "subrun", "event", "entry", "samdef"}
    weight_cols = [c for c in t.keys() if c not in id_cols]

    n_chunks = (n_events + chunk_size - 1) // chunk_size
    print(f"  {total_entries} total entries, loading {n_events} ({frac_events:.0%}) in {n_chunks} chunks of {chunk_size}")
    print(f"  Weight columns ({len(weight_cols)}): {weight_cols}")

    chunk_dfs = []
    for chunk_idx, chunk_start in enumerate(range(0, n_events, chunk_size)):
        chunk_stop = min(chunk_start + chunk_size, n_events)
        print(f"  chunk {chunk_idx + 1}/{n_chunks}: events {chunk_start}-{chunk_stop}...")
        slice_kwargs = {"entry_start": chunk_start, "entry_stop": chunk_stop}

        id_data = t.arrays(["run", "subrun", "event"], library="np", **slice_kwargs)
        nusel_data = t_nusel.arrays(['weightsReint'], library="np", **slice_kwargs)
        weight_data = t.arrays(weight_cols, library="np", **slice_kwargs)

        data = {col: id_data[col] for col in ["run", "subrun", "event"]}
        for col in weight_cols:
            data[col] = [row.tolist() for row in weight_data[col]]
        data['weightsReint'] = [(row.astype(np.float64) / 1000.0).tolist() for row in nusel_data['weightsReint']]

        chunk_df = pl.DataFrame(data)
        chunk_df = chunk_df.with_columns([
            pl.col("run").cast(pl.Int32),
            pl.col("subrun").cast(pl.Int32),
            pl.col("event").cast(pl.Int32),
            pl.lit(detailed_run_period).alias("detailed_run_period"),
            pl.lit(filename).alias("filename"),
            pl.lit(filetype).alias("filetype"),
        ])
        chunk_dfs.append(chunk_df)

    f.close()

    df = pl.concat(chunk_dfs, how="vertical") if len(chunk_dfs) > 1 else chunk_dfs[0]
    print(f"  df shape: {df.shape}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Save spline_weights tree as a standalone weights dataframe")
    parser.add_argument("-f", "--frac_events", type=float, default=1.0,
                        help="Fraction of events to load from each spline ROOT file, in (0, 1]. Default: 1.0")
    args = parser.parse_args()

    if not (0.0 < args.frac_events <= 1.0):
        raise ValueError("--frac_events must be in the interval (0, 1].")

    start_time = time.time()

    dfs = []
    for root_file_path in SPLINE_ROOT_FILES:
        dfs.append(load_spline_weights_df(root_file_path, frac_events=args.frac_events))

    spline_df = pl.concat(dfs, how="diagonal_relaxed") if len(dfs) > 1 else dfs[0]

    print(f"Writing to {OUTPUT_PARQUET}...")
    spline_df.write_parquet(OUTPUT_PARQUET)
    file_size_gb = os.path.getsize(OUTPUT_PARQUET) / 1024**3
    elapsed = time.time() - start_time
    print(f"Done. {file_size_gb:.2f} GB, {elapsed:.1f} s")


if __name__ == "__main__":
    main()
