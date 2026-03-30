
import uproot
import numpy as np
import pandas as pd
import polars as pl
import sys
import os
import time
import argparse
from pathlib import Path

# Add parent directory to path to allow imports with src. prefix
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.file_locations import data_files_location, intermediate_files_location

from src.memory_monitoring import start_memory_logger

from src.pyroot_loading import get_rw_sys_weights_dic

def _get_file_metadata(filename, frac_events=1):
    """Collect per-file metadata without reading any weight data.

    Returns a dict with keys:
        filetype, detailed_run_period, n_events, root_file_size_gb
    """
    if "beam_off" in filename.lower() or "beamoff" in filename.lower() or "ext" in filename.lower():
        filetype = "ext"
    elif "nu_overlay" in filename.lower():
        filetype = "nu_overlay"
    elif "nue_overlay" in filename.lower():
        filetype = "nue_overlay"
    elif "dirt" in filename.lower():
        filetype = "dirt_overlay"
    elif "nc_pi0" in filename.lower() or "ncpi0" in filename.lower() or "nc_pio" in filename.lower() or "ncpio" in filename.lower():
        filetype = "nc_pi0_overlay"
    elif "ccpi0_overlay" in filename.lower():
        filetype = "numucc_pi0_overlay"
    elif "delete_one_gamma" in filename.lower():
        filetype = "delete_one_gamma_overlay"
    elif "isotropic_one_gamma" in filename.lower():
        filetype = "isotropic_one_gamma_overlay"
    elif "beam_on" in filename.lower():
        filetype = "data"
    else:
        raise ValueError("Unknown filetype!", filename)

    if filetype == "data" or filetype == "ext" or filetype == "isotropic_one_gamma_overlay" or filetype == "delete_one_gamma_overlay":
        raise ValueError("Data, EXT, and 1g overlay files don't have systematics variables!")

    root_file_size_gb = os.path.getsize(f"{data_files_location}/{filename}") / 1024**3

    if not (0.0 < frac_events <= 1.0):
        raise ValueError("--frac_events/-f must be in the interval (0, 1].")

    f = uproot.open(f"{data_files_location}/{filename}")
    total_entries = f["wcpselection"]["T_eval"].num_entries
    n_events = total_entries if frac_events >= 1.0 else max(1, int(total_entries * frac_events))
    f.close()

    print(f"{total_entries=}, {frac_events=}, {n_events=}")

    detailed_run_period = "?"
    if "4a.root" in filename:
        detailed_run_period = "4a"
    elif "4b.root" in filename:
        detailed_run_period = "4b"
    elif "4c.root" in filename:
        detailed_run_period = "4c"
    elif "4d.root" in filename:
        detailed_run_period = "4d"
    elif "4bcd.root" in filename:
        detailed_run_period = "4bcd"
    elif "5.root" in filename:
        detailed_run_period = "5"
    elif "4a" in filename.lower(): # if the filename doesn't end with the run period, look for run strings in the file names
        detailed_run_period = "4a"
    elif "run4b" in filename.lower():
        detailed_run_period = "4b"
    elif "run4c" in filename.lower():
        detailed_run_period = "4c"
    elif "run4d" in filename.lower():
        detailed_run_period = "4d"
    elif "run4bcd" in filename.lower():
        detailed_run_period = "4bcd"
    elif "run5" in filename.lower():
        detailed_run_period = "5"
    else:
        raise ValueError("Invalid detailed run period!", filename)

    return {
        "filetype": filetype,
        "detailed_run_period": detailed_run_period,
        "n_events": n_events,
        "root_file_size_gb": root_file_size_gb,
    }


def _load_chunk(filename, filetype, detailed_run_period, entry_start, entry_stop, **_):
    """Load events [entry_start, entry_stop) and return a polars DataFrame.

    **_ absorbs unused metadata keys (n_events, root_file_size_gb) so callers can
    pass the full _get_file_metadata dict via **meta.
    """
    chunk_size = entry_stop - entry_start
    slice_kwargs = {"entry_start": entry_start, "entry_stop": entry_stop}

    f = uproot.open(f"{data_files_location}/{filename}")

    print("  loading run, subrun, event, and CV weights using uproot...")
    dic = f["nuselection"]["NeutrinoSelectionFilter"].arrays(
        ["run", "sub", "evt", "weightSpline", "weightTune", "weightSplineTimesTune"],
        library="np", **slice_kwargs)
    curr_weights_df = pl.DataFrame({col: dic[col] for col in dic})
    curr_weights_df = curr_weights_df.rename({"sub": "subrun", "evt": "event"})

    print("  loading wc_kine_reco_Enu for preselection using uproot...")
    dic = f["wcpselection"]["T_KINEvars"].arrays(["kine_reco_Enu"], library="np", **slice_kwargs)
    curr_weights_df = curr_weights_df.with_columns(pl.Series(name="wc_kine_reco_Enu", values=dic["kine_reco_Enu"]))
    del f
    del dic

    print("  loading systematic weights using PyROOT...")
    all_event_weights = get_rw_sys_weights_dic(
        f"{data_files_location}/{filename}",
        max_entries=chunk_size,
        start_entry=entry_start,
    )
    print("  adding systematic weights to dataframe...")

    if all_event_weights and all_event_weights[0]:
        systematic_keys = list(all_event_weights[0].keys())
        for k in systematic_keys:
            weight_lists = [event_dict[k] for event_dict in all_event_weights]
            curr_weights_df = curr_weights_df.with_columns(pl.Series(name=k, values=weight_lists, dtype=pl.List(pl.Float32)))

    previous_num_events = curr_weights_df.height
    curr_weights_df = curr_weights_df.filter(pl.col("wc_kine_reco_Enu") > 0)
    print(f"  kept {curr_weights_df.height}/{previous_num_events} events after preselection wc_kine_reco_Enu > 0")

    curr_weights_df = curr_weights_df.with_columns([
        pl.lit(detailed_run_period).alias("detailed_run_period"),
        pl.lit(filename).alias("filename"),
        pl.lit(filetype).alias("filetype"),
    ])

    return curr_weights_df


def process_rw_sys_root_file(filename, frac_events=1):
    """Load an entire ROOT file as a single DataFrame. Thin wrapper around _get_file_metadata + _load_chunk."""
    start_time = time.time()
    print(f"loading {filename}...")
    meta = _get_file_metadata(filename, frac_events)
    curr_weights_df = _load_chunk(filename, entry_start=0, entry_stop=meta["n_events"], **meta)
    end_time = time.time()
    progress_str = (
        f"\nloaded {meta['filetype']:<30}   Run {meta['detailed_run_period']:<4} "
        f"{curr_weights_df.shape[0]:>10,d} wc_generic_sel events "
        f"{meta['root_file_size_gb']:>6.2f} GB {end_time - start_time:>6.2f} s"
    )
    if frac_events < 1.0:
        progress_str += f" (f={frac_events})"
    print(progress_str)
    return meta["filetype"], curr_weights_df


if __name__ == "__main__":
    main_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Create merged dataframe from SURPRISE 4b ROOT files")
    parser.add_argument("-f", "--frac_events", type=float, default=1.0,
                        help="Fraction of events (and POT) to load from each file, in (0,1]. Default: 1.0")
    parser.add_argument("-m", "--memory_logger", action="store_true", default=False,
                        help="Start a memory logger thread")
    parser.add_argument("-w", "--weight_types", type=str, default="genie,flux,reint",
                        help="Comma-separated list of weight types to load. Default: genie,flux,reint")
    parser.add_argument("--just_one_file", action="store_true", default=False,
                        help="Only process one file for debugging purposes")
    parser.add_argument("--chunk_size", type=int, default=100_000,
                        help="Number of events per chunk when reading ROOT files. Default: 100000")
    args = parser.parse_args()

    if args.memory_logger:
        start_memory_logger(10)

    if args.frac_events < 1.0:
        print(f"Loading {args.frac_events} fraction of events from each file")

    for file in os.listdir(intermediate_files_location):
        if (file.startswith("presel_weights_df") and file.endswith(".parquet")) or \
           (file.startswith("chunk_weights_") and file.endswith(".parquet")):
            os.remove(f"{intermediate_files_location}/{file}")
    print("Deleted intermediate presel_weights_df*.parquet files")

    print(f"Starting loop over root files (chunk_size={args.chunk_size:,})...")

    filenames = os.listdir(data_files_location)
    filenames.sort()

    for file_num, filename in enumerate(filenames):

        if args.just_one_file and "checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_reco2_hist_4c.root" not in filename:
            continue
        if "UNUSED" in filename or "older_downloads" in filename:
            continue

        if "nfs000000150070ba7d00000751" in filename: # TEMPORARY: weird file in directory now
            continue

        if "beam_on" in filename.lower() or "beamon" in filename.lower():
            continue
        if "beam_off" in filename.lower() or "beamoff" in filename.lower() or "ext" in filename.lower():
            continue
        if "one_gamma" in filename.lower():
            continue

        if "detvar" in filename.lower():
            continue

        print(f"loading {filename}...")
        file_start_time = time.time()

        meta = _get_file_metadata(filename, frac_events=args.frac_events)
        filetype = meta["filetype"]
        detailed_run_period = meta["detailed_run_period"]
        n_events = meta["n_events"]

        n_chunks = (n_events + args.chunk_size - 1) // args.chunk_size
        chunk_parquet_paths = []

        for chunk_idx, chunk_start in enumerate(range(0, n_events, args.chunk_size)):
            chunk_stop = min(chunk_start + args.chunk_size, n_events)
            print(f"  chunk {chunk_idx + 1}/{n_chunks}: events {chunk_start}-{chunk_stop}...")

            curr_chunk_df = _load_chunk(filename, entry_start=chunk_start, entry_stop=chunk_stop, **meta)

            chunk_path = f"{intermediate_files_location}/chunk_weights_{file_num}_{chunk_idx}.parquet"
            curr_chunk_df.write_parquet(chunk_path)
            chunk_parquet_paths.append(chunk_path)
            del curr_chunk_df

        # combine chunks into the per-file parquet
        parquet_path = f"{intermediate_files_location}/presel_weights_df_{file_num}.parquet"
        if len(chunk_parquet_paths) == 1:
            os.rename(chunk_parquet_paths[0], parquet_path)
            print("single chunk, renamed to final parquet")
        else:
            print(f"  combining {len(chunk_parquet_paths)} chunks into {parquet_path}...")
            pl.concat(
                [pl.read_parquet(p) for p in chunk_parquet_paths],
                how="diagonal_relaxed",
            ).write_parquet(parquet_path)
            for p in chunk_parquet_paths:
                os.remove(p)

        file_end_time = time.time()
        print(f"  saved {os.path.getsize(parquet_path) / 1e9:.2f} GB (on disk)")
        progress_str = (
            f"\nloaded {filetype:<30}   Run {detailed_run_period:<4} "
            f"{n_events:>10,d} events "
            f"{meta['root_file_size_gb']:>6.2f} GB {file_end_time - file_start_time:>6.2f} s"
        )
        if args.frac_events < 1.0:
            progress_str += f" (f={args.frac_events})"
        print(progress_str)

        if args.just_one_file:
            break

    print("loading polars dataframes from parquet files...")

    presel_weights_dfs = []
    for file in os.listdir(intermediate_files_location):
        if file.startswith("presel_weights_df_") and file.endswith(".parquet"):
            presel_weights_dfs.append(pl.read_parquet(f"{intermediate_files_location}/{file}"))
    presel_weights_df = pl.concat(presel_weights_dfs, how="vertical")
    del presel_weights_dfs

    for file in os.listdir(intermediate_files_location):
        if file.startswith("presel_weights_df_") and file.endswith(".parquet"):
            os.remove(f"{intermediate_files_location}/{file}")
    print("Deleted intermediate presel_weights_df*.parquet files")

    if presel_weights_df.is_empty():
        raise ValueError("No events in the dataframe!")
    
    print(f"presel_weights_df.height={presel_weights_df.height}")

    # Extend presel_weights_df with derived event types that have no GENIE systematics.
    # numuCC_rad_corrected comes from delete_one_gamma_overlay files, and
    # NC_coherent_1g_reweighted comes from isotropic_one_gamma_overlay files.
    # Neither source file has GENIE weight trees, so we assign unit CV weights and
    # unit systematic weights (all-ones lists matching the shape of existing columns).
    print("Adding derived event types (rad_corrected, coherent_1g) with unit systematic weights...")
    presel_df_path = f"{intermediate_files_location}/presel_df_train_vars.parquet"
    if os.path.exists(presel_df_path):
        derived_events = pl.scan_parquet(presel_df_path).filter(
            pl.col("filetype").is_in(["numuCC_rad_corrected", "NC_coherent_1g_reweighted"])
        ).select(["run", "subrun", "event", "filetype", "detailed_run_period", "filename", "wc_kine_reco_Enu"]).collect()

        print(f"  found {derived_events.height} derived events in presel_df_train_vars.parquet")

        if derived_events.height > 0:
            # Build unit CV weights
            derived_events = derived_events.with_columns([
                pl.lit(1.0).cast(pl.Float32).alias("weightSpline"),
                pl.lit(1.0).cast(pl.Float32).alias("weightTune"),
                pl.lit(1.0).cast(pl.Float32).alias("weightSplineTimesTune"),
            ])

            # Build unit systematic list columns matching shape of existing presel_weights_df
            sys_list_cols = [c for c, t in presel_weights_df.schema.items() if isinstance(t, pl.List)]
            n = derived_events.height
            for col in sys_list_cols:
                list_len = len(presel_weights_df[col][0])
                derived_events = derived_events.with_columns(
                    pl.Series(col, [[1.0] * list_len] * n, dtype=pl.List(pl.Float32))
                )

            presel_weights_df = pl.concat([presel_weights_df, derived_events], how="diagonal_relaxed")
            print(f"  presel_weights_df now has {presel_weights_df.height} rows after adding derived events")
        else:
            print("  WARNING: no derived events found; skipping extension")
    else:
        print(f"  WARNING: {presel_df_path} not found; skipping derived event extension")

    print(f"saving {intermediate_files_location}/presel_weights_df.parquet...", end="", flush=True)
    start_time = time.time()
    presel_weights_df.write_parquet(f"{intermediate_files_location}/presel_weights_df.parquet")
    end_time = time.time()
    file_size_gb = os.path.getsize(f"{intermediate_files_location}/presel_weights_df.parquet") / 1024**3
    print(f"done, {file_size_gb:.2f} GB, {end_time - start_time:.2f} seconds")
    main_end_time = time.time()
    print(f"Total time to create weights dataframe: {main_end_time - main_start_time:.2f} seconds")
    