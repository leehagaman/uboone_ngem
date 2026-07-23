
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
from src.zexp_reweighting import ZEXP_MINERVA_FA_BRANCH, ZEXP_PCA_BRANCHES, compute_minerva_zexp_weights

def _get_file_metadata(filename, frac_events=1):
    """Collect per-file metadata without reading any weight data.

    Returns a dict with keys:
        filetype, detailed_run_period, n_events, root_file_size_gb
    """
    if "fullosc" in filename.lower():
        filetype = "fullosc_overlay"
    elif "beam_off" in filename.lower() or "beamoff" in filename.lower() or "ext" in filename.lower():
        filetype = "ext"
    elif "nuwro" in filename.lower():
        filetype = "nuwro_fake_data"
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

    if filetype in ("data", "ext", "nuwro_fake_data", "isotropic_one_gamma_overlay", "delete_one_gamma_overlay"):
        raise ValueError(f"{filetype} files are treated like data and have no GENIE systematics variables!", filename)

    root_file_size_gb = os.path.getsize(f"{data_files_location}/{filename}") / 1024**3

    if not (0.0 < frac_events <= 1.0):
        raise ValueError("--frac_events/-f must be in the interval (0, 1].")

    f = uproot.open(f"{data_files_location}/{filename}")
    total_entries = f["wcpselection"]["T_eval"].num_entries
    n_events = total_entries if frac_events >= 1.0 else max(1, int(total_entries * frac_events))
    f.close()

    print(f"{total_entries=}, {frac_events=}, {n_events=}")

    detailed_run_period = "?"
    if "1.root" in filename:
        detailed_run_period = "1"
    elif "2.root" in filename:
        detailed_run_period = "2"
    elif "3.root" in filename:
        detailed_run_period = "3"
    elif "4a.root" in filename:
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
    """Load events [entry_start, entry_stop) and return (syst_df, spline_df).

    syst_df : run/subrun/event + CV weights + the GENIE/flux/reint multisim systematic
              weights (the `weights` map), preselected to wc_kine_reco_Enu > 0.
    spline_df : run/subrun/event + the per-knob spline weights (the file's
              spline_weights tree) + weightsReint, preselected identically.

    The spline loading used to live in the standalone create_splines_df.py and read a
    handful of special run-4b spline files; it is folded in here now that every overlay
    file carries a spline_weights tree, so each ROOT file is opened once and splines are
    produced for all run periods.  spline_df is written to the separate
    spline_weights_df.parquet (consumed by save_PROfit_rootfiles).

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

    if filetype == "fullosc_overlay":
        # matching flag, only present in the fullosc file: fullosc==1 marks events
        # where the numu<->nue matching succeeded (same cut as in create_df.py)
        fo_dic = f["nuselection"]["NeutrinoSelectionFilter"].arrays(["fullosc"], library="np", **slice_kwargs)
        curr_weights_df = curr_weights_df.with_columns(pl.Series(name="fullosc", values=fo_dic["fullosc"]))

    # Int32 ids shared by both output dfs (the spline_weights tree is entry-aligned with
    # nuselection, so reusing these ids keeps the two parquets' join keys consistent).
    ids_df = curr_weights_df.select([
        pl.col("run").cast(pl.Int32),
        pl.col("subrun").cast(pl.Int32),
        pl.col("event").cast(pl.Int32),
    ])

    print("  loading systematic (multisim) weights using PyROOT...")
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

    print("  loading per-knob spline weights (spline_weights tree) + weightsReint...")
    spline_tree = f["spline_weights"]
    # the fullosc_* branches (matching bookkeeping + CV weight, fullosc file only)
    # are per-event scalars, not spline knobs, so they are excluded here
    spline_id_cols = {"run", "subrun", "event", "entry", "samdef",
                      "fullosc", "fullosc_numu_entry", "fullosc_numu_run",
                      "fullosc_numu_subrun", "fullosc_numu_event", "fullosc_cv_weight",
                      "numu_nue_xs_ratio_weight", "numu_nue_xs_ratio_weight_bar"}
    spline_knob_cols = [c for c in spline_tree.keys() if c not in spline_id_cols]
    spline_data = spline_tree.arrays(spline_knob_cols, library="np", **slice_kwargs)
    reint_data = f["nuselection"]["NeutrinoSelectionFilter"].arrays(["weightsReint"], library="np", **slice_kwargs)
    q2_data = f["singlephotonana"]["eventweight_tree"].arrays(["GTruth_gQ2"], library="np", **slice_kwargs)
    spline_dict = {
        "run": ids_df["run"].to_numpy(),
        "subrun": ids_df["subrun"].to_numpy(),
        "event": ids_df["event"].to_numpy(),
    }
    for col in spline_knob_cols:
        spline_dict[col] = [row.tolist() for row in spline_data[col]]
    spline_dict["weightsReint"] = [(row.astype(np.float64) / 1000.0).tolist() for row in reint_data["weightsReint"]]

    print("  computing MINERvA z-expansion axial form-factor weights...")
    zexp_weights = compute_minerva_zexp_weights(q2_data["GTruth_gQ2"], spline_data["MaCCQE_UBGenie"])
    spline_dict[ZEXP_MINERVA_FA_BRANCH] = zexp_weights[ZEXP_MINERVA_FA_BRANCH]
    for col in ZEXP_PCA_BRANCHES:
        spline_dict[col] = [row.tolist() for row in zexp_weights[col]]
    spline_df = pl.DataFrame(spline_dict)

    del f, dic, all_event_weights, spline_data, reint_data, q2_data, zexp_weights

    # identical preselection on both dfs (same entry slice -> same row order, so the
    # boolean mask from one applies to the other).
    keep_expr = pl.col("wc_kine_reco_Enu") > 0
    if filetype == "fullosc_overlay":
        keep_expr = keep_expr & (pl.col("fullosc") == 1)
    keep_mask = curr_weights_df.select(keep_expr).to_series()
    previous_num_events = curr_weights_df.height
    curr_weights_df = curr_weights_df.filter(keep_mask)
    spline_df = spline_df.filter(keep_mask)
    if filetype == "fullosc_overlay":
        curr_weights_df = curr_weights_df.drop("fullosc")
    print(f"  kept {curr_weights_df.height}/{previous_num_events} events after preselection"
          + (" wc_kine_reco_Enu > 0 & fullosc == 1" if filetype == "fullosc_overlay" else " wc_kine_reco_Enu > 0"))

    meta_cols = [
        pl.lit(detailed_run_period).alias("detailed_run_period"),
        pl.lit(filename).alias("filename"),
        pl.lit(filetype).alias("filetype"),
    ]
    curr_weights_df = curr_weights_df.with_columns(meta_cols)
    spline_df = spline_df.with_columns(meta_cols)

    return curr_weights_df, spline_df


def process_rw_sys_root_file(filename, frac_events=1):
    """Load an entire ROOT file as (syst_df, spline_df). Thin wrapper around
    _get_file_metadata + _load_chunk."""
    start_time = time.time()
    print(f"loading {filename}...")
    meta = _get_file_metadata(filename, frac_events)
    curr_weights_df, curr_spline_df = _load_chunk(filename, entry_start=0, entry_stop=meta["n_events"], **meta)
    end_time = time.time()
    progress_str = (
        f"\nloaded {meta['filetype']:<30}   Run {meta['detailed_run_period']:<4} "
        f"{curr_weights_df.shape[0]:>10,d} wc_generic_sel events "
        f"{meta['root_file_size_gb']:>6.2f} GB {end_time - start_time:>6.2f} s"
    )
    if frac_events < 1.0:
        progress_str += f" (f={frac_events})"
    print(progress_str)
    return meta["filetype"], curr_weights_df, curr_spline_df


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
    parser.add_argument("--just_one_file_target", type=str,
                        default="checkout_MCC9.10_Run123_v10_04_07_20_BNB_nu_overlay_surprise_reco2_hist_1.root",
                        help="Filename substring to use with --just_one_file. Default: a BNB nu_overlay file with CCQE-capable MaCCQE splines")
    parser.add_argument("--chunk_size", type=int, default=100_000,
                        help="Number of events per chunk when reading ROOT files. Default: 100000")
    args = parser.parse_args()

    if args.memory_logger:
        start_memory_logger(10)

    if args.frac_events < 1.0:
        print(f"Loading {args.frac_events} fraction of events from each file")

    for file in os.listdir(intermediate_files_location):
        if file.endswith(".parquet") and (
            file.startswith("presel_weights_df") or file.startswith("chunk_weights_")
            or file.startswith("spline_weights_df") or file.startswith("chunk_splines_")
            or file == "_derived_weights.parquet"
        ):
            os.remove(f"{intermediate_files_location}/{file}")
    print("Deleted intermediate weight/spline parquet files")

    print(f"Starting loop over root files (chunk_size={args.chunk_size:,})...")

    def _is_systematics_root_file(fn):
        """Overlay ROOT files that carry GENIE systematics.  Skips non-.root inputs (the
        directory also holds e.g. evtgen_pi0_dalitz.csv), auxiliary/unused files, and the
        samples treated like data -- real data, EXT, NuWro fake data, and the 1g overlays
        -- as well as detvar files.  NuWro is a different generator (no GENIE multisim
        `weights` map and no spline_weights tree), so it is the "data" role in the nuwro
        config and gets no systematic bands."""
        if not fn.endswith(".root"):
            return False
        if "UNUSED" in fn or "older_downloads" in fn or "nfs000000150070ba7d00000751" in fn:
            return False
        low = fn.lower()
        return not (
            "beam_on" in low or "beamon" in low
            or "beam_off" in low or "beamoff" in low or "ext" in low
            or "one_gamma" in low or "nuwro" in low or "detvar" in low
        )

    filenames = [f for f in sorted(os.listdir(data_files_location)) if _is_systematics_root_file(f)]
    if args.just_one_file:
        target = args.just_one_file_target
        filenames = [f for f in filenames if target in f]
        if not filenames:
            raise ValueError(f"--just_one_file_target matched no systematics ROOT files: {target}")
        filenames = filenames[:1]
        print(f"  --just_one_file target: {target}")
    print(f"Processing {len(filenames)} systematics ROOT files...")

    for file_num, filename in enumerate(filenames):

        print(f"\n=== file {file_num + 1}/{len(filenames)}: {filename} ===")
        file_start_time = time.time()

        meta = _get_file_metadata(filename, frac_events=args.frac_events)
        filetype = meta["filetype"]
        detailed_run_period = meta["detailed_run_period"]
        n_events = meta["n_events"]

        n_chunks = (n_events + args.chunk_size - 1) // args.chunk_size
        chunk_parquet_paths = []
        spline_chunk_parquet_paths = []

        for chunk_idx, chunk_start in enumerate(range(0, n_events, args.chunk_size)):
            chunk_stop = min(chunk_start + args.chunk_size, n_events)
            print(f"  chunk {chunk_idx + 1}/{n_chunks}: events {chunk_start}-{chunk_stop}...")

            curr_chunk_df, curr_spline_chunk_df = _load_chunk(
                filename, entry_start=chunk_start, entry_stop=chunk_stop, **meta)

            chunk_path = f"{intermediate_files_location}/chunk_weights_{file_num}_{chunk_idx}.parquet"
            spline_chunk_path = f"{intermediate_files_location}/chunk_splines_{file_num}_{chunk_idx}.parquet"
            curr_chunk_df.write_parquet(chunk_path)
            curr_spline_chunk_df.write_parquet(spline_chunk_path)
            chunk_parquet_paths.append(chunk_path)
            spline_chunk_parquet_paths.append(spline_chunk_path)
            del curr_chunk_df, curr_spline_chunk_df

        # combine each chunk set into its per-file parquet (stream via sink so the chunks
        # are never all held in memory at once)
        parquet_path = f"{intermediate_files_location}/presel_weights_df_{file_num}.parquet"
        spline_parquet_path = f"{intermediate_files_location}/spline_weights_df_{file_num}.parquet"
        for _paths, _out in ((chunk_parquet_paths, parquet_path),
                             (spline_chunk_parquet_paths, spline_parquet_path)):
            if len(_paths) == 1:
                os.rename(_paths[0], _out)
            else:
                print(f"  combining {len(_paths)} chunks into {_out}...")
                pl.concat([pl.scan_parquet(p) for p in _paths], how="diagonal_relaxed").sink_parquet(_out)
                for p in _paths:
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

    print("merging per-file parquets into the final dataframes...")

    weight_parts = sorted([
        f"{intermediate_files_location}/{file}"
        for file in os.listdir(intermediate_files_location)
        if file.startswith("presel_weights_df_") and file.endswith(".parquet")
    ])
    spline_parts = sorted([
        f"{intermediate_files_location}/{file}"
        for file in os.listdir(intermediate_files_location)
        if file.startswith("spline_weights_df_") and file.endswith(".parquet")
    ])
    if not weight_parts:
        raise ValueError("No events in the dataframe!")
    print(f"  {len(weight_parts)} weight parts, {len(spline_parts)} spline parts")

    # ── spline_weights_df.parquet: stream-concat the per-file spline parts ──
    spline_out = f"{intermediate_files_location}/spline_weights_df.parquet"
    print(f"saving {spline_out}...", end="", flush=True)
    pl.concat([pl.scan_parquet(p) for p in spline_parts], how="diagonal_relaxed").sink_parquet(spline_out)
    for p in spline_parts:
        os.remove(p)
    print(f"done, {os.path.getsize(spline_out) / 1024**3:.2f} GB")

    # ── presel_weights_df.parquet: derived rad/coherent rows + stream-concat ──
    # numuCC_rad_corrected (from delete_one_gamma_overlay) and NC_coherent_1g_reweighted
    # (from isotropic_one_gamma_overlay) have no GENIE weight trees, so they get unit CV
    # weights and unit systematic-weight lists (matching the existing list columns'
    # shapes).  Written to their own small parquet and streamed in alongside the rest, so
    # the full weights dataframe is never materialized in memory.
    derived_part = None
    presel_df_path = f"{intermediate_files_location}/presel_df_train_vars.parquet"
    if os.path.exists(presel_df_path):
        derived_events = pl.scan_parquet(presel_df_path).filter(
            pl.col("filetype").is_in(["numuCC_rad_corrected", "NC_coherent_1g_reweighted"])
        ).select(["run", "subrun", "event", "filetype", "detailed_run_period", "filename", "wc_kine_reco_Enu"]).collect()
        print(f"Adding {derived_events.height} derived events (rad_corrected, coherent_1g) with unit systematic weights...")

        if derived_events.height > 0:
            # Union of list (systematic) columns across all weight parts, with one example
            # list length each (parts can in principle differ in which knobs they carry).
            list_col_len = {}
            for wp in weight_parts:
                sch = pl.scan_parquet(wp).collect_schema()
                need = [c for c, t in sch.items() if isinstance(t, pl.List) and c not in list_col_len]
                if need:
                    first = pl.scan_parquet(wp).select(need).head(1).collect()
                    for c in need:
                        list_col_len[c] = len(first[c][0])

            derived_events = derived_events.with_columns([
                pl.lit(1.0).cast(pl.Float32).alias("weightSpline"),
                pl.lit(1.0).cast(pl.Float32).alias("weightTune"),
                pl.lit(1.0).cast(pl.Float32).alias("weightSplineTimesTune"),
            ])
            n = derived_events.height
            for col, list_len in list_col_len.items():
                derived_events = derived_events.with_columns(
                    pl.Series(col, [[1.0] * list_len] * n, dtype=pl.List(pl.Float32))
                )

            derived_part = f"{intermediate_files_location}/_derived_weights.parquet"
            derived_events.write_parquet(derived_part)
        else:
            print("  WARNING: no derived events found; skipping extension")
    else:
        print(f"  WARNING: {presel_df_path} not found; skipping derived event extension")

    presel_out = f"{intermediate_files_location}/presel_weights_df.parquet"
    all_weight_parts = weight_parts + ([derived_part] if derived_part else [])
    print(f"saving {presel_out}...", end="", flush=True)
    start_time = time.time()
    pl.concat([pl.scan_parquet(p) for p in all_weight_parts], how="diagonal_relaxed").sink_parquet(presel_out)
    for p in all_weight_parts:
        os.remove(p)
    end_time = time.time()
    file_size_gb = os.path.getsize(presel_out) / 1024**3
    print(f"done, {file_size_gb:.2f} GB, {end_time - start_time:.2f} seconds")
    main_end_time = time.time()
    print(f"Total time to create weights dataframe: {main_end_time - main_start_time:.2f} seconds")
    