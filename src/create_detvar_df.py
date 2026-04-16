
import uproot
import numpy as np
import pandas as pd
import polars as pl
import os
import gc
import time
import argparse


from ntuple_variables.variables import wc_T_BDT_including_training_vars, wc_T_KINEvars_including_training_vars, wc_training_only_vars
from ntuple_variables.variables import wc_T_spacepoints_vars, wc_T_eval_vars, wc_T_pf_vars, wc_T_pf_data_vars, wc_T_eval_data_vars
from ntuple_variables.variables import blip_vars, pandora_vars, glee_vars, lantern_vars, vector_columns
from postprocessing import do_orthogonalization_and_POT_weighting, add_extra_true_photon_variables, do_spacepoint_postprocessing, add_signal_categories
from postprocessing import do_wc_postprocessing, do_pandora_postprocessing, do_lantern_postprocessing, do_combined_postprocessing, do_glee_postprocessing
from blip_postprocessing import do_blip_postprocessing
from postprocessing import remove_vector_variables, add_1g1mu_rad_corr_events, add_nc_coh_1g_reweighted_events

from file_locations import data_files_location, intermediate_files_location

from df_helpers import align_columns_for_concat
from memory_monitoring import start_memory_logger

def _get_file_metadata(filename, frac_events=1):
    """Open a ROOT file briefly to collect per-file metadata without reading branch data.

    Returns a dict with keys:
        filetype, vartype, detailed_run_period, file_POT, n_events,
        root_file_size_gb, curr_wc_T_BDT_including_training_vars, curr_wc_T_pf_vars
    """
    if "nu_overlay" in filename.lower():
        filetype = "nu_overlay"
    elif "nue_overlay" in filename.lower():
        filetype = "nue_overlay"
    elif "dirt" in filename.lower():
        filetype = "dirt_overlay"
    elif "nc_pi0" in filename.lower() or "ncpi0" in filename.lower() or "nc_pio" in filename.lower() or "ncpio" in filename.lower():
        filetype = "nc_pi0_overlay"
    elif "ccpi0" in filename.lower():
        filetype = "numucc_pi0_overlay"
    elif "delete_one_gamma" in filename.lower():
        filetype = "delete_one_gamma_overlay"
    elif "isotropic_one_gamma" in filename.lower():
        filetype = "isotropic_one_gamma_overlay"
    else:
        raise ValueError("Unknown filetype!", filename)

    if not filetype or filetype == '':
        raise ValueError(f"filetype is empty or None for filename: {filename}")

    if "lya" in filename.lower():
        vartype = "LYAtt"
    elif "lyd" in filename.lower():
        vartype = "LYDown"
    elif "lyr" in filename.lower():
        vartype = "LYRayleigh"
    elif "wmx" in filename.lower():
        vartype = "WireModX"
    elif "wmyz" in filename.lower():
        vartype = "WireModYZ"
    elif "recomb2" in filename.lower():
        vartype = "Recomb2"
    elif "sce" in filename.lower():
        vartype = "SCE"
    elif "cv" in filename.lower():
        vartype = "CV"
    else:
        raise ValueError("Unknown vartype!", filename)

    root_file_size_gb = os.path.getsize(f"{data_files_location}/{filename}") / 1024**3

    if not (0.0 < frac_events <= 1.0):
        raise ValueError("--frac_events/-f must be in the interval (0, 1].")

    f = uproot.open(f"{data_files_location}/{filename}")
    total_entries = f["wcpselection"]["T_eval"].num_entries
    n_events = total_entries if frac_events >= 1.0 else max(1, int(total_entries * frac_events))

    print(f"{total_entries=}, {frac_events=}, {n_events=}")

    # this nanosecond timing variable only exists in the CV and merged files
    curr_wc_T_pf_vars = [var for var in wc_T_pf_vars if var != "evtTimeNS_cor"]

    curr_wc_T_BDT_including_training_vars = wc_T_BDT_including_training_vars
    if "v10_04_07_09" in filename:
        print(f"    TEMPORARY: NOT LOADING WCPMTInfo VARIABLES FOR {filetype}")
        curr_wc_T_BDT_including_training_vars = [var for var in wc_T_BDT_including_training_vars if "WCPMTInfo" not in var]

    file_POT_total = np.sum(f["wcpselection"]["T_pot"].arrays("pot_tor875good", library="np")["pot_tor875good"])
    f.close()

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

    file_POT = file_POT_total * frac_events

    return {
        "filetype": filetype,
        "vartype": vartype,
        "detailed_run_period": detailed_run_period,
        "file_POT": file_POT,
        "n_events": n_events,
        "root_file_size_gb": root_file_size_gb,
        "curr_wc_T_BDT_including_training_vars": curr_wc_T_BDT_including_training_vars,
        "curr_wc_T_pf_vars": curr_wc_T_pf_vars,
    }


def _load_chunk(filename, filetype, vartype, detailed_run_period, file_POT,
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

    previous_num_events = all_df.shape[0]
    all_df = all_df.query("wc_kine_reco_Enu > 0").reset_index(drop=True)
    print(f"    kept {all_df.shape[0]}/{previous_num_events} events after preselection wc_kine_reco_Enu > 0")

    all_df["detailed_run_period"] = detailed_run_period
    all_df["filename"] = filename
    all_df["filetype"] = filetype
    all_df["vartype"] = vartype

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
        f"\nloaded {meta['filetype']:<30}   Vartype {meta['vartype']:<12} Run {meta['detailed_run_period']:<4} "
        f"{all_df.shape[0]:>10,d} events {meta['file_POT']:>10.2e} POT "
        f"{events_per_POT:>6.2f} events / 1e19 POT "
        f"{meta['root_file_size_gb']:>6.2f} GB {end_time - start_time:>6.2f} s"
    )
    if frac_events < 1.0:
        progress_str += f" (f={frac_events})"
    print(progress_str)
    return meta["filetype"], meta["detailed_run_period"], meta["vartype"], all_df, meta["file_POT"]


if __name__ == "__main__":
    main_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Create merged dataframe from SURPRISE 4b ROOT files")
    parser.add_argument("-f", "--frac_events", type=float, default=1.0,
                        help="Fraction of events (and POT) to load from each file, in (0,1]. Default: 1.0")
    parser.add_argument("-m", "--memory_logger", action="store_true", default=False,
                        help="Start a memory logger thread")
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
        if ((file.startswith("curr_detvar_df_pl_") and file.endswith(".parquet")) or
                (file.startswith("chunk_detvar_") and file.endswith(".parquet")) or
                file == "detvar_presel_df_train_vars.parquet"):
            os.remove(f"{intermediate_files_location}/{file}")
    print("Deleted intermediate parquet files")

    print(f"Starting loop over root files (chunk_size={args.chunk_size:,})...")

    filenames = os.listdir(data_files_location)
    filenames.sort()
    # sorting these puts an NC Pi0 overlay first, which will have all the WCPMTInfo and truth variables present,
    # so it can be used to add columns to future dataframes with missing values

    pot_dic = {}

    for file_num, filename in enumerate(filenames):

        if args.just_one_file and "checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_reco2_hist_4c.root" not in filename:
            continue

        if "UNUSED" in filename or "older_downloads" in filename:
            continue

        if "nfs000000150070ba7d00000751" in filename: # TEMPORARY: weird file in directory now
            continue

        if not "detvar" in filename.lower():
            continue

        print(f"loading {filename}...")
        file_start_time = time.time()

        meta = _get_file_metadata(filename, frac_events=args.frac_events)
        filetype = meta["filetype"]
        vartype = meta["vartype"]
        detailed_run_period = meta["detailed_run_period"]
        file_POT = meta["file_POT"]
        n_events = meta["n_events"]

        if vartype == "CV":
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

            chunk_path = f"{intermediate_files_location}/chunk_detvar_{file_num}_{chunk_idx}.parquet"
            curr_df_pl.write_parquet(chunk_path)
            chunk_parquet_paths.append(chunk_path)
            del curr_df_pl

        # combine chunks into the per-file parquet
        parquet_path = f"{intermediate_files_location}/curr_detvar_df_pl_{file_num}.parquet"
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
        print(f"curr_detvar_df_pl size: {os.path.getsize(parquet_path) / 1e9:.2f} GB (on disk)")
        print("saved to parquet file")

        file_end_time = time.time()
        events_per_POT = n_events / (file_POT / 1e19)
        progress_str = (
            f"\nloaded {filetype:<30}   Vartype {vartype:<12} Run {detailed_run_period:<4} "
            f"{n_events:>10,d} events {file_POT:>10.2e} POT "
            f"{events_per_POT:>6.2f} events / 1e19 POT "
            f"{meta['root_file_size_gb']:>6.2f} GB {file_end_time - file_start_time:>6.2f} s"
        )
        if args.frac_events < 1.0:
            progress_str += f" (f={args.frac_events})"
        print(progress_str)

    print("loading polars dataframes from parquet files...")

    pl_dfs = []
    for file in os.listdir(intermediate_files_location):

        if file.startswith("curr_detvar_df_pl_") and file.endswith(".parquet"):
            print(f"Reading {file}")
            curr_df = pl.read_parquet(f"{intermediate_files_location}/{file}")
            # Validate filetype column after reading from parquet - should never be empty
            if 'filetype' in curr_df.columns:
                empty_count = curr_df.filter(pl.col("filetype") == '').height
                if empty_count > 0:
                    raise ValueError(f"filetype column has {empty_count} empty string values after reading parquet file {file}. This should never happen!")
            pl_dfs.append(curr_df)
            print(f"Read {file}, estimated size: {pl_dfs[-1].estimated_size() / 1e9:.2f} GB")
    all_df = pl.concat(align_columns_for_concat(pl_dfs), how="vertical")
    del pl_dfs
    print(f"all_df size: {all_df.estimated_size() / 1e9:.2f} GB")
    
    # Validate filetype column immediately after concatenation
    if "filetype" in all_df.columns:
        empty_count = all_df.filter(pl.col("filetype") == '').height
        null_count = all_df.filter(pl.col("filetype").is_null()).height
        if empty_count > 0 or null_count > 0:
            print(f"ERROR after concat: filetype has {empty_count} empty strings and {null_count} nulls")
            if empty_count > 0:
                problem_row = all_df.filter(pl.col("filetype") == '').head(1).row(0, named=True)
                print(f"Example empty filetype row: filename={problem_row.get('filename', 'N/A')}")
            raise ValueError(f"filetype column corrupted after concatenation: {empty_count} empty, {null_count} null")

    for file in os.listdir(intermediate_files_location):
        if file.startswith("curr_detvar_df_pl_") and file.endswith(".parquet"):
            os.remove(f"{intermediate_files_location}/{file}")
    print("Deleted intermediate curr_detvar_df_pl*.parquet files")

    if all_df.is_empty():
        raise ValueError("No events in the dataframe!")
    
    print(f"all_df.height={all_df.height}")

    print("doing post-processing that doesn't require vector variables using polars...")

    all_df = do_combined_postprocessing(all_df)
    #normalizing_POT = 1.11e21
    normalizing_POT = 2.098e19 + 4.038e19 # for run 4a and run 4b open data
    
    all_df = do_orthogonalization_and_POT_weighting(all_df, pot_dic, normalizing_POT=normalizing_POT)

    all_df = add_signal_categories(all_df)

    temp_defrag_path = f"{intermediate_files_location}/_temp_defrag_detvar_df.parquet"
    print(f"Writing temp parquet to {temp_defrag_path}...", end="", flush=True)
    all_df.lazy().sink_parquet(temp_defrag_path)
    print("done")
    gc.collect()

    all_lf = pl.scan_parquet(temp_defrag_path)
    rad_corrected_df = add_1g1mu_rad_corr_events(all_lf)
    coherent_1g_df = add_nc_coh_1g_reweighted_events(all_lf)
    del all_lf

    all_df = pl.read_parquet(temp_defrag_path)
    os.remove(temp_defrag_path)
    gc.collect()

    all_df = pl.concat([all_df, rad_corrected_df, coherent_1g_df], how="diagonal_relaxed")
    del rad_corrected_df, coherent_1g_df
    gc.collect()
    print(f"  all_df has {all_df.height} rows after adding rad_corr and coherent events")

    file_RSEs = []
    for filetype, vartype, run, subrun, event in zip(all_df["filetype"].to_numpy(), all_df["vartype"].to_numpy(), all_df["run"].to_numpy(), all_df["subrun"].to_numpy(), all_df["event"].to_numpy()):
        file_RSE = f"{filetype}_{vartype}_{run:06d}_{subrun:06d}_{event:06d}"
        file_RSEs.append(file_RSE)
    assert len(file_RSEs) == len(set(file_RSEs)), "Duplicate filetype/vartype/run/subrun/event!"

    print(f"saving {intermediate_files_location}/detvar_presel_df_train_vars.parquet...", end="", flush=True)
    start_time = time.time()
    presel_df = all_df.filter(pl.col("wc_kine_reco_Enu") > 0)
    presel_df.write_parquet(f"{intermediate_files_location}/detvar_presel_df_train_vars.parquet")
    end_time = time.time()
    file_size_gb = os.path.getsize(f"{intermediate_files_location}/detvar_presel_df_train_vars.parquet") / 1024**3
    print(f"done, {file_size_gb:.2f} GB, {end_time - start_time:.2f} seconds")

    main_end_time = time.time()
    print(f"Total time to create the dataframes: {main_end_time - main_start_time:.2f} seconds")
    