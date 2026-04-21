
import gc
import ctypes
import uproot
import numpy as np
import pandas as pd
import polars as pl
import os
import time
import argparse


from ntuple_variables.variables import wc_T_BDT_including_training_vars, wc_T_KINEvars_including_training_vars, wc_training_only_vars
from ntuple_variables.variables import wc_T_spacepoints_vars, wc_T_eval_vars, wc_T_pf_vars, wc_T_pf_data_vars, wc_T_eval_data_vars
from ntuple_variables.variables import blip_vars, pandora_vars, glee_vars, glee_eventweight_vars, lantern_vars, vector_columns
from postprocessing import do_orthogonalization_and_POT_weighting, add_extra_true_photon_variables, do_spacepoint_postprocessing, add_signal_categories
from postprocessing import do_wc_postprocessing, do_pandora_postprocessing, do_lantern_postprocessing, do_combined_postprocessing, do_glee_postprocessing
from blip_postprocessing import do_blip_postprocessing
from postprocessing import remove_vector_variables, change_dtypes
from postprocessing import add_1g1mu_rad_corr_events, add_nc_coh_1g_reweighted_events

from file_locations import data_files_location, intermediate_files_location

from df_helpers import align_columns_for_concat
from memory_monitoring import start_memory_logger

def _filetype_from_filename(filename):
    fn = filename.lower()
    if "beam_off" in fn or "beamoff" in fn or "ext" in fn:
        return "ext"
    if "nu_overlay" in fn:
        return "nu_overlay"
    if "nue_overlay" in fn:
        return "nue_overlay"
    if "dirt" in fn:
        return "dirt_overlay"
    if "nc_pi0" in fn or "ncpi0" in fn or "nc_pio" in fn or "ncpio" in fn:
        return "nc_pi0_overlay"
    if "ccpi0" in fn:
        return "numucc_pi0_overlay"
    if "delete_one_gamma" in fn:
        return "delete_one_gamma_overlay"
    if "isotropic_one_gamma" in fn:
        return "isotropic_one_gamma_overlay"
    if "beam_on" in fn:
        return "data"
    raise ValueError("Unknown filetype!", filename)


def _get_file_metadata(filename, frac_events=1):
    """Open a ROOT file briefly to collect per-file metadata without reading branch data.

    Returns a dict with keys:
        filetype, detailed_run_period, file_POT, n_events,
        root_file_size_gb, curr_wc_T_BDT_including_training_vars, curr_wc_T_pf_vars
    """
    filetype = _filetype_from_filename(filename)

    root_file_size_gb = os.path.getsize(f"{data_files_location}/{filename}") / 1024**3

    if not (0.0 < frac_events <= 1.0):
        raise ValueError("--frac_events/-f must be in the interval (0, 1].")

    f = uproot.open(f"{data_files_location}/{filename}")
    total_entries = f["wcpselection"]["T_eval"].num_entries
    n_events = total_entries if frac_events >= 1.0 else max(1, int(total_entries * frac_events))

    print(f"{total_entries=}, {frac_events=}, {n_events=}")

    curr_wc_T_pf_vars = wc_T_pf_vars
    curr_wc_T_BDT_including_training_vars = wc_T_BDT_including_training_vars
    if (("v10_04_07_09" in filename) or (filename == "checkout_MCC9.10_Run4b_v10_04_07_20_BNB_beam_off_metapatch_retuple_retuple_hist.root")
                 or (filename == "checkout_MCC9.10_Run4b_v10_04_07_20_BNB_nu_overlay_retuple_retuple_hist.root")):
        print(f"    TEMPORARY: NOT LOADING WCPMTInfo VARIABLES FOR {filetype}")
        curr_wc_T_BDT_including_training_vars = [var for var in wc_T_BDT_including_training_vars if "WCPMTInfo" not in var]

    # data and EXT POT and trigger numbers from https://docs.google.com/spreadsheets/d/1RUiX2M6zoob9R0YWPLummHzmX5UeLLEtS-7ZU-x2gA4
    # also from Karan's processing, https://docs.google.com/document/d/1SWZtfo9MIGpODVopGwWTM2LNEN-d7GmkWhYOmChK4kk/edit?tab=t.0

    # using these numbers for now
    # eventually can replace these with the full data statistics, to get a more accurate POT/trigger ratio in each run period
    run4a_open_data_POT = 2.098e19
    run4a_open_data_num_triggers = 4836758
    run4a_pot_per_trigger = run4a_open_data_POT / run4a_open_data_num_triggers

    run4b_open_data_POT = 4.038e19
    run4b_open_data_num_triggers = 9218529
    run4b_pot_per_trigger = run4b_open_data_POT / run4b_open_data_num_triggers

    file_POT_total = np.sum(f["wcpselection"]["T_pot"].arrays("pot_tor875good", library="np")["pot_tor875good"])
    f.close()

    if filetype == "ext":
        if filename == "checkout_MCC9.10_Run4a_BNB_beam_off_data_surprise_reco2_hist.root": # run 4a
            run4a_ext_num_triggers = 27940007
            file_POT_total = run4a_ext_num_triggers * run4a_pot_per_trigger
        elif filename == "checkout_MCC9.10_Run4b_v10_04_07_20_BNB_beam_off_metapatch_retuple_retuple_hist.root": # run 4b
            run4b_ext_num_triggers = 89010180
            file_POT_total = run4b_ext_num_triggers * run4b_pot_per_trigger
        elif filename == "checkout_MCC9.10_Run4acd5_v10_04_07_14_BNB_beam_off_surprise_reco2_hist_4c.root": # run 4c
            run4b_ext_num_triggers = 53659787
            file_POT_total = run4b_ext_num_triggers * run4b_pot_per_trigger
        elif filename == "checkout_MCC9.10_Run4acd5_v10_04_07_14_BNB_beam_off_surprise_reco2_hist_4d.root": # run 4d
            run4b_ext_num_triggers = 76563108
            file_POT_total = run4b_ext_num_triggers * run4b_pot_per_trigger
        elif filename == "checkout_MCC9.10_Run4acd5_v10_04_07_14_BNB_beam_off_surprise_reco2_hist_5.root": # run 5
            run4b_ext_num_triggers = 111457148
            file_POT_total = run4b_ext_num_triggers * run4b_pot_per_trigger
        else:
            raise ValueError("Invalid EXT file, num triggers not found!")
    elif filetype == "data":
        if filename == "checkout_MCC9.10_Run4a_BNB_beam_on_data_surprise_reco2_hist_opendata_19550.root": # run 4a open data
            file_POT_total = run4a_open_data_POT
        elif filename == "checkout_MCC9.10_Run4b_v10_04_07_20_BNB_beam_on_metapatch_retuple_retuple_hist_opendata_20700.root": # run 4b open data
            file_POT_total = run4b_open_data_POT
        else:
            raise ValueError("Invalid data file!")

    file_POT = file_POT_total * frac_events

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
        "file_POT": file_POT,
        "n_events": n_events,
        "root_file_size_gb": root_file_size_gb,
        "curr_wc_T_BDT_including_training_vars": curr_wc_T_BDT_including_training_vars,
        "curr_wc_T_pf_vars": curr_wc_T_pf_vars,
    }


def _load_chunk(filename, filetype, detailed_run_period, file_POT,
                curr_wc_T_BDT_including_training_vars, curr_wc_T_pf_vars,
                entry_start, entry_stop, **_):
    """Load events [entry_start, entry_stop) from a ROOT file and return a DataFrame.

    **_ absorbs unused metadata keys (n_events, root_file_size_gb) so callers can
    pass the full _get_file_metadata dict via **meta.
    """
    f = uproot.open(f"{data_files_location}/{filename}")
    slice_kwargs = {"entry_start": entry_start, "entry_stop": entry_stop}

    # loading Wire-Cell variables
    dic = {}
    dic.update(f["wcpselection"]["T_BDTvars"].arrays(curr_wc_T_BDT_including_training_vars, library="np", **slice_kwargs))
    dic.update(f["wcpselection"]["T_KINEvars"].arrays(wc_T_KINEvars_including_training_vars, library="np", **slice_kwargs))
    dic.update(f["wcpselection"]["T_spacepoints"].arrays(wc_T_spacepoints_vars, library="np", **slice_kwargs))
    if filetype == "ext" or filetype == "data":
        dic.update(f["wcpselection"]["T_PFeval"].arrays(wc_T_pf_data_vars, library="np", **slice_kwargs))
        dic.update(f["wcpselection"]["T_eval"].arrays(wc_T_eval_data_vars, library="np", **slice_kwargs))
    else:
        dic.update(f["wcpselection"]["T_PFeval"].arrays(curr_wc_T_pf_vars, library="np", **slice_kwargs))
        dic.update(f["wcpselection"]["T_eval"].arrays(wc_T_eval_vars, library="np", **slice_kwargs))
    all_df = pd.DataFrame({col: arr.tolist() if arr.ndim != 1 else arr for col, arr in dic.items()}).add_prefix("wc_")
    del dic
    all_df["wc_file_POT"] = file_POT

    # loading blip variables (blip variables already have the "blip_" prefix)
    dic = {}
    dic.update(f["nuselection"]["NeutrinoSelectionFilter"].arrays(blip_vars, library="np", **slice_kwargs))
    blip_df = pd.DataFrame({col: arr.tolist() if arr.ndim != 1 else arr for col, arr in dic.items()})
    del dic
    all_df = pd.concat([all_df, blip_df], axis=1)
    del blip_df

    # loading pandora variables
    dic = {}
    dic.update(f["nuselection"]["NeutrinoSelectionFilter"].arrays(pandora_vars, library="np", **slice_kwargs))
    pandora_df = pd.DataFrame({col: arr.tolist() if arr.ndim != 1 else arr for col, arr in dic.items()}).add_prefix("pandora_")
    del dic
    all_df = pd.concat([all_df, pandora_df], axis=1)
    del pandora_df

    # loading gLEE variables
    dic = {}
    dic.update(f["singlephotonana"]["vertex_tree"].arrays(glee_vars, library="np", **slice_kwargs))
    if filetype != "ext" and filetype != "data":
        dic.update(f["singlephotonana"]["eventweight_tree"].arrays(glee_eventweight_vars, library="np", **slice_kwargs))
    glee_df = pd.DataFrame({col: arr.tolist() if arr.ndim != 1 else arr for col, arr in dic.items()}).add_prefix("glee_")
    del dic
    all_df = pd.concat([all_df, glee_df], axis=1)
    del glee_df

    # loading LANTERN variables
    dic = {}
    dic.update(f["lantern"]["EventTree"].arrays(lantern_vars, library="np", **slice_kwargs))
    lantern_df = pd.DataFrame({col: arr.tolist() if arr.ndim != 1 else arr for col, arr in dic.items()}).add_prefix("lantern_")
    del dic
    all_df = pd.concat([all_df, lantern_df], axis=1)
    del lantern_df

    del f

    # remove some of these prefixes, for things that should be universal
    all_df.rename(columns={"wc_run": "run", "wc_subrun": "subrun", "wc_event": "event"}, inplace=True)

    all_df["detailed_run_period"] = detailed_run_period
    all_df["filename"] = filename
    all_df["filetype"] = filetype

    return all_df


def process_root_file(filename, frac_events=1):
    """Load an entire ROOT file as a single DataFrame. Thin wrapper around _get_file_metadata + _load_chunk."""
    start_time = time.time()
    print(f"loading {filename}...")
    meta = _get_file_metadata(filename, frac_events)
    all_df = _load_chunk(filename, entry_start=0, entry_stop=meta["n_events"], **meta)
    end_time = time.time()
    events_per_POT = all_df.shape[0] / (meta["file_POT"] / 1e19)
    progress_str = (
        f"\nloaded {meta['filetype']:<30}   Run {meta['detailed_run_period']:<4} "
        f"{all_df.shape[0]:>10,d} events {meta['file_POT']:>10.2e} POT "
        f"{events_per_POT:>6.2f} events / 1e19 POT "
        f"{meta['root_file_size_gb']:>6.2f} GB {end_time - start_time:>6.2f} s"
    )
    if frac_events < 1.0:
        progress_str += f" (f={frac_events})"
    print(progress_str)
    return meta["filetype"], meta["detailed_run_period"], all_df, meta["file_POT"]


if __name__ == "__main__":
    main_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Create merged dataframe from SURPRISE 4b ROOT files")
    parser.add_argument("-f", "--frac_events", type=float, default=1.0,
                        help="Fraction of events (and POT) to load from each file, in (0,1]. Default: 1.0")
    parser.add_argument("-m", "--memory_logger", action="store_true", default=False,
                        help="Start a memory logger thread")
    parser.add_argument("--just_one_file", action="store_true", default=False,
                        help="Only process one file for debugging purposes")
    parser.add_argument("--create_file_dfs", action="store_true", default=False,
                        help="Create file-level dataframes for each file")
    parser.add_argument("--merge_file_dfs", action="store_true", default=False,
                        help="Merge file-level dataframes into a single dataframe")
    parser.add_argument("--chunk_size", type=int, default=100_000,
                        help="Number of events per chunk when reading ROOT files. Default: 100000")
    args = parser.parse_args()

    if args.memory_logger:
        start_memory_logger(1)

    if args.create_file_dfs:
        print("Creating file-level dataframes for each file...")

        if args.frac_events < 1.0:
            print(f"Loading {args.frac_events} fraction of events from each file")

        for file in os.listdir(intermediate_files_location):
            if (file.startswith("curr_df_pl_") and file.endswith(".parquet")) or \
               (file.startswith("chunk_") and file.endswith(".parquet")):
                os.remove(f"{intermediate_files_location}/{file}")
        print("Deleted intermediate df parquet files")

        print(f"Starting loop over root files (chunk_size={args.chunk_size:,})...")

        filenames_with_unused = os.listdir(data_files_location)
        filenames_with_unused.sort()
        # sorting these puts an NC Pi0 overlay first, which will have all the WCPMTInfo and truth variables present, 
        # so it can be used to add columns to future dataframes with missing values

        filenames = []
        for filename in filenames_with_unused:
            if args.just_one_file and "checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_reco2_hist_4c.root" not in filename:
                continue

            if "UNUSED" in filename or "older_downloads" in filename:
                continue

            if "nfs000000150070ba7d00000751" in filename: # TEMPORARY: weird file in directory now
                continue

            if "detvar" in filename.lower():
                continue

            filenames.append(filename)

        pot_dic = {}

        for file_num, filename in enumerate(filenames):

            print(f"Processing file {file_num} / {len(filenames)}")
            print(f"loading {filename}...")
            file_start_time = time.time()

            meta = _get_file_metadata(filename, frac_events=args.frac_events)
            filetype = meta["filetype"]
            detailed_run_period = meta["detailed_run_period"]

            file_POT = meta["file_POT"]
            n_events = meta["n_events"]

            pot_dic[(filetype, detailed_run_period)] = file_POT

            n_chunks = (n_events + args.chunk_size - 1) // args.chunk_size
            chunk_parquet_paths = []

            for chunk_idx, chunk_start in enumerate(range(0, n_events, args.chunk_size)):
                chunk_stop = min(chunk_start + args.chunk_size, n_events)
                print(f"  chunk {chunk_idx + 1}/{n_chunks}: events {chunk_start}-{chunk_stop}...")

                curr_df = _load_chunk(filename, entry_start=chunk_start, entry_stop=chunk_stop, **meta)

                print("  doing post-processing that requires vector variables...")

                curr_df = do_wc_postprocessing(curr_df)
                curr_df = add_extra_true_photon_variables(curr_df)
                curr_df = do_spacepoint_postprocessing(curr_df)
                curr_df = do_pandora_postprocessing(curr_df)
                curr_df = do_blip_postprocessing(curr_df)
                curr_df = do_lantern_postprocessing(curr_df)
                curr_df = do_glee_postprocessing(curr_df)

                curr_df = remove_vector_variables(curr_df)

                # converting to polars
                curr_df_pl = pl.from_pandas(curr_df)
                del curr_df

                # Validate filetype column after conversion to polars
                filetype_values = curr_df_pl["filetype"].unique().to_list()
                if '' in filetype_values or None in filetype_values:
                    empty_count = curr_df_pl.filter(pl.col("filetype") == '').height
                    null_count = curr_df_pl.filter(pl.col("filetype").is_null()).height
                    if empty_count > 0 or null_count > 0:
                        raise ValueError(f"filetype column has empty/null values after polars conversion for {filename} chunk {chunk_idx}: {empty_count} empty, {null_count} null")

                curr_df_pl = curr_df_pl.with_columns([pl.col(pl.Float64).cast(pl.Float32)])
                curr_df_pl = curr_df_pl.with_columns([pl.col(pl.Int32).cast(pl.Int64)])

                chunk_path = f"{intermediate_files_location}/chunk_{file_num}_{chunk_idx}.parquet"
                curr_df_pl.write_parquet(chunk_path)
                chunk_parquet_paths.append(chunk_path)
                del curr_df_pl

            # combine chunks into the per-file parquet
            parquet_path = f"{intermediate_files_location}/curr_df_pl_{file_num}.parquet"
            if len(chunk_parquet_paths) == 1:
                os.rename(chunk_parquet_paths[0], parquet_path)
                print("single chunk, renamed to final parquet")
            else:
                print(f"combining {len(chunk_parquet_paths)} chunks into {parquet_path}...")
                pl.concat(
                    [pl.read_parquet(p) for p in chunk_parquet_paths],
                    how="diagonal_relaxed",
                ).write_parquet(parquet_path)
                for p in chunk_parquet_paths:
                    os.remove(p)
            print(f"curr_df_pl size: {os.path.getsize(parquet_path) / 1e9:.2f} GB (on disk)")
            print("saved to parquet file")

            print(f"Reloading {parquet_path} to ensure on-disk integrity...")
            reloaded_df = pl.read_parquet(parquet_path)
            if "filetype" not in reloaded_df.columns:
                raise ValueError(f"{parquet_path} is missing the filetype column after writing!")
            empty_count = reloaded_df.filter(pl.col("filetype") == '').height
            null_count = reloaded_df.filter(pl.col("filetype").is_null()).height
            if empty_count > 0 or null_count > 0:
                raise ValueError(
                    f"{parquet_path} has corrupted filetype values after writing: "
                    f"{empty_count} empty strings, {null_count} nulls"
                )
            del reloaded_df

            file_end_time = time.time()
            events_per_POT = n_events / (file_POT / 1e19)
            progress_str = (
                f"\nloaded {filetype:<30}   Run {detailed_run_period:<4} "
                f"{n_events:>10,d} events {file_POT:>10.2e} POT "
                f"{events_per_POT:>6.2f} events / 1e19 POT "
                f"{meta['root_file_size_gb']:>6.2f} GB {file_end_time - file_start_time:>6.2f} s"
            )
            if args.frac_events < 1.0:
                progress_str += f" (f={args.frac_events})"
            print(progress_str)

                # TODO: When we have more files, do weighting to make each set of run fractions match the run fractions in data


        print("saving pot_dic to csv file...")
        if os.path.exists(f"{intermediate_files_location}/pot_dic.csv"):
            os.remove(f"{intermediate_files_location}/pot_dic.csv")
        with open(f"{intermediate_files_location}/pot_dic.csv", "w") as f:
            for key, value in pot_dic.items():
                f.write(f"{key[0]},{key[1]},{value}\n")

        print("done creating file-level dataframes for each file")

    if args.merge_file_dfs:
        print("Merging file-level dataframes into a single dataframe...")

        print("loading pot_dic from csv file...")
        with open(f"{intermediate_files_location}/pot_dic.csv", "r") as f:
            pot_dic = {}
            for line in f:
                filetype, detailed_run_period, value = line.strip().split(",")
                pot_dic[(filetype, detailed_run_period)] = float(value)

        for file in os.listdir(intermediate_files_location):
            if file == "presel_df_train_vars.parquet" or file == "all_df.parquet":
                os.remove(f"{intermediate_files_location}/{file}")
        print("Deleted final df parquet files")

        print("loading polars dataframes from parquet files...")

        parquet_files = sorted([
            f"{intermediate_files_location}/{file}"
            for file in os.listdir(intermediate_files_location)
            if file.startswith("curr_df_pl_") and file.endswith(".parquet")
        ])
        print(f"Found {len(parquet_files)} parquet files")
        all_df = pl.scan_parquet(parquet_files, missing_columns="insert", extra_columns="ignore").collect()
        print(f"all_df size: {all_df.estimated_size() / 1e9:.2f} GB")
        
        # Validate filetype column immediately after concatenation
        known_filetypes = {
            "ext", "data", "nu_overlay", "nue_overlay", "dirt_overlay",
            "nc_pi0_overlay", "numucc_pi0_overlay",
            "delete_one_gamma_overlay", "isotropic_one_gamma_overlay",
        }
        if "filetype" in all_df.columns:
            # Cast to String first so Categorical comparisons don't silently miss values
            filetype_str = all_df.get_column("filetype").cast(pl.String)
            null_count = filetype_str.is_null().sum()
            empty_count = (filetype_str == "").sum()
            unknown_vals = [v for v in filetype_str.unique().to_list() if v is not None and v not in known_filetypes]
            if null_count > 0 or empty_count > 0 or unknown_vals:
                print(f"ERROR after concat: filetype has {empty_count} empty strings, {null_count} nulls, unknown values: {unknown_vals}")
                bad_mask = filetype_str.is_null() | (filetype_str == "")
                for uv in unknown_vals:
                    bad_mask = bad_mask | (filetype_str == uv)
                bad_rows = all_df.filter(bad_mask)
                for row in bad_rows.head(5).iter_rows(named=True):
                    print(f"  Bad filetype row: filetype={row.get('filetype')!r}, filename={row.get('filename', 'N/A')}")
                    print(f"    Stale parquet detected — delete intermediate parquets and re-run --create_file_dfs")
                raise ValueError(f"filetype column corrupted after concatenation: {empty_count} empty, {null_count} null, unknown: {unknown_vals}")

        if all_df.is_empty():
            raise ValueError("No events in the dataframe!")
        
        print(f"all_df.height={all_df.height}")

        # Defrag immediately: scan_parquet over N files produces an N-chunk df and
        # fragments the heap from many small allocations.  Write/reload once here
        # so ALL downstream postprocessing (including add_signal_categories) runs on
        # a fresh single-chunk df with a clean heap (~35 GB baseline instead of ~80 GB).
        temp_defrag_path = f"{intermediate_files_location}/_temp_defrag_all_df.parquet"
        print(f"Early defrag: writing {len(parquet_files)}-chunk df to {temp_defrag_path}...", end="", flush=True)
        _t0 = time.time()
        all_df.lazy().sink_parquet(temp_defrag_path)
        del all_df
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        all_df = pl.read_parquet(temp_defrag_path)
        os.remove(temp_defrag_path)
        gc.collect()
        print(f" done in {time.time() - _t0:.1f}s")

        # Polars sink_parquet can corrupt low-cardinality String columns (like filetype)
        # via dictionary encoding. Detect and fix any rows where filetype became ''.
        bad_filetype_mask = pl.col("filetype").is_null() | (pl.col("filetype") == "")
        bad_count = all_df.filter(bad_filetype_mask).height
        if bad_count > 0:
            print(f"WARNING: {bad_count} rows have empty/null filetype after defrag — re-inferring from filename")
            fixed_filetype = pl.Series(
                "filetype",
                [ft if (ft is not None and ft != "") else _filetype_from_filename(fn)
                 for ft, fn in zip(all_df["filetype"].to_list(), all_df["filename"].to_list())]
            )
            all_df = all_df.with_columns(fixed_filetype)
            still_bad = all_df.filter(bad_filetype_mask).height
            if still_bad > 0:
                raise ValueError(f"{still_bad} rows still have empty/null filetype after re-inference!")
            print(f"  Fixed {bad_count} rows.")

        print("doing post-processing that doesn't require vector variables using polars...")

        all_df = do_combined_postprocessing(all_df)

        # Convert dtypes early so all subsequent postprocessing works on smaller arrays.
        # Float64→Float32 and Int64→Int32 are done in batches to avoid holding two full
        # copies of the dataframe at once.
        print("Converting dtypes to reduce memory usage (before heavy postprocessing)...")
        memory_before = all_df.estimated_size() / (1024**3)
        print(f"Estimated memory usage before conversion: {memory_before:.4f} GB")

        float64_cols = [col for col, dtype in all_df.schema.items() if dtype == pl.Float64]
        int64_cols   = [col for col, dtype in all_df.schema.items() if dtype == pl.Int64]
        print(f"Converting {len(float64_cols)} Float64 columns to Float32")
        print(f"Converting {len(int64_cols)} Int64 columns to Int32 (clipping to Int32 range)")

        if float64_cols:
            all_df = all_df.with_columns([pl.col(col).cast(pl.Float32) for col in float64_cols])
            gc.collect()

        int32_min, int32_max = -2147483648, 2147483647
        batch_size = 50
        for i in range(0, len(int64_cols), batch_size):
            batch = int64_cols[i:i + batch_size]
            all_df = all_df.with_columns([
                pl.col(col).clip(int32_min, int32_max).cast(pl.Int32) for col in batch
            ])
            gc.collect()

        memory_after = all_df.estimated_size() / (1024**3)
        print(f"Estimated memory usage after conversion: {memory_after:.4f} GB")
        print(f"Memory saved: {memory_before - memory_after:.4f} GB ({(memory_before - memory_after) / memory_before * 100:.1f}%)")
        gc.collect()

        normalizing_POT = 0
        for key, value in pot_dic.items():
            if key[0] == "data":
                normalizing_POT += value
        if normalizing_POT == 0:
            normalizing_POT = 1.11e21 # if we don't use a data file, assume we want full runs 1-5 data

        run4b_normalizing_POT = 0
        for key, value in pot_dic.items():
            if key[0] == "data" and key[1] == "4b":
                run4b_normalizing_POT += value
        if run4b_normalizing_POT == 0:
            run4b_normalizing_POT = normalizing_POT

        all_df = do_orthogonalization_and_POT_weighting(all_df, pot_dic, normalizing_POT=normalizing_POT)
        all_df = do_orthogonalization_and_POT_weighting(all_df, pot_dic, normalizing_POT=run4b_normalizing_POT, run4b_only=True)

        # do_orthogonalization_and_POT_weighting adds new Float64 weight columns; convert them now.
        new_float64_cols = [col for col, dtype in all_df.schema.items() if dtype == pl.Float64]
        if new_float64_cols:
            print(f"Converting {len(new_float64_cols)} new Float64 columns added by postprocessing: {new_float64_cols}")
            all_df = all_df.with_columns([pl.col(col).cast(pl.Float32) for col in new_float64_cols])
            gc.collect()

        all_df = add_signal_categories(all_df)
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            print("malloc_trim successful")
        except Exception:
            print("malloc_trim failed")

        train_test_score_bytes = (
            all_df.select(["filename", "run", "subrun", "event"])
            .hash_rows(seed=0)
            .to_numpy()
            & np.uint64(0xFF)
        ).astype(np.uint8)
        train_test_score = train_test_score_bytes.astype(np.float32) / 256.0
        train_mask = train_test_score < 0.5
        all_df = all_df.with_columns(
            pl.Series("train_test_score", train_test_score),
            pl.Series("will_use_for_50_50_training", train_mask),
        )

        print(f"Total number of events in all_df: {all_df.height}")
        print(f"Number of events in all_df with will_use_for_50_50_training == True: {all_df.select(pl.col('will_use_for_50_50_training').sum()).item()}")

        # Write all_df (with signal categories + train_test_score) to a temp parquet
        # so that add_1g1mu_rad_corr_events and add_nc_coh_1g_reweighted_events can
        # use scan_parquet with predicate pushdown to collect only the small filtered
        # sub-dfs cheaply, then read the full df fresh from disk.
        temp_defrag_path = f"{intermediate_files_location}/_temp_defrag_all_df.parquet"
        print(f"Writing temp parquet to {temp_defrag_path}...", end="", flush=True)
        start_time = time.time()
        all_df.lazy().sink_parquet(temp_defrag_path)
        del all_df
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            print("malloc_trim successful")
        except Exception:
            print("malloc_trim failed")
            pass
        # Use scan_parquet (lazy) to compute the two small filtered sub-dfs with
        # predicate pushdown (reads only the relevant rows, not the full 35 GB).
        # Both add_* functions, when given a LazyFrame, collect and return only
        # the *new* rows to append as a small eager DataFrame.
        # After both small dfs are collected we read the full df once and concat.
        # Temp file is kept until after read_parquet then removed.
        print(f" done in {time.time() - start_time:.1f}s")
        all_lf = pl.scan_parquet(temp_defrag_path)

        rad_corrected_df = add_1g1mu_rad_corr_events(all_lf)
        coherent_1g_df = add_nc_coh_1g_reweighted_events(all_lf)
        del all_lf

        print("Reading full df from parquet...")
        start_read = time.time()
        all_df = pl.read_parquet(temp_defrag_path)
        os.remove(temp_defrag_path)
        gc.collect()
        print(f"  read done in {time.time() - start_read:.1f}s, all_df has {all_df.height} rows")

        print("Concatenating full df with new rows...")
        all_df = pl.concat([all_df, rad_corrected_df, coherent_1g_df], how="diagonal_relaxed")
        del rad_corrected_df, coherent_1g_df
        gc.collect()
        print(f"  all_df has {all_df.height} rows after adding rad_corr and coherent events")

        dup_mask = pl.struct("filetype", "run", "subrun", "event").is_duplicated()
        n_dups = all_df.select(dup_mask.sum()).item()
        if n_dups > 0:
            dups = all_df.filter(dup_mask).select(["filename", "filetype", "run", "subrun", "event", "wc_truth_nuEnergy"])
            print(f"Found {n_dups} duplicate rows, first 10:\n{dups.head(10)}")
            raise ValueError("Duplicate filename/run/subrun/event!")

        # Use sink_parquet via lazy API to stream the filtered data directly to disk
        # without materializing a second full copy of all_df in memory.
        print(f"saving {intermediate_files_location}/presel_df_train_vars.parquet...", end="", flush=True)
        start_time = time.time()
        all_df.lazy().filter(pl.col("wc_kine_reco_Enu") > 0).sink_parquet(
            f"{intermediate_files_location}/presel_df_train_vars.parquet"
        )
        end_time = time.time()
        file_size_gb = os.path.getsize(f"{intermediate_files_location}/presel_df_train_vars.parquet") / 1024**3
        print(f"done, {file_size_gb:.2f} GB, {end_time - start_time:.2f} seconds")

        print(f"saving {intermediate_files_location}/all_df.parquet...", end="", flush=True)
        start_time = time.time()
        # remove the large number of WC training-only-variables for a smaller file size

        remove_columns = wc_training_only_vars

        all_df = all_df.drop(remove_columns)

        all_df.write_parquet(f"{intermediate_files_location}/all_df.parquet")
        end_time = time.time()
        file_size_gb = os.path.getsize(f"{intermediate_files_location}/all_df.parquet") / 1024**3
        print(f"done, {file_size_gb:.2f} GB, {end_time - start_time:.2f} seconds")
        main_end_time = time.time()
        print(f"Total time to create the dataframes: {main_end_time - main_start_time:.2f} seconds")

        print("done merging file-level dataframes into a single dataframe")
    
