
import uproot
import numpy as np
import pandas as pd
import polars as pl
import sys
import os
import time
import argparse

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from postprocessing import do_orthogonalization_and_POT_weighting, add_extra_true_photon_variables, do_spacepoint_postprocessing, add_signal_categories
from postprocessing import do_wc_postprocessing, do_pandora_postprocessing, do_blip_postprocessing, do_lantern_postprocessing, do_combined_postprocessing, do_glee_postprocessing
from postprocessing import remove_vector_variables

from file_locations import data_files_location, intermediate_files_location

from df_helpers import align_columns_for_concat, compress_df
from memory_monitoring import start_memory_logger

from systematics import get_rw_sys_weights_dic

def process_root_file(filename, frac_events = 1):

    if "beam_off" in filename.lower() or "beamoff" in filename.lower() or "ext" in filename.lower(): # EXT file
        filetype = "ext"
    elif "nu_overlay" in filename.lower():
        filetype = "nu_overlay"
    elif "nue_overlay" in filename.lower():
        filetype = "nue_overlay"
    elif "dirt" in filename.lower():
        filetype = "dirt_overlay"
    elif "nc_pi0" in filename.lower() or "ncpi0" in filename.lower() or "nc_pio" in filename.lower() or "ncpio" in filename.lower():
        filetype = "nc_pi0_overlay"
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

    start_time = time.time()

    print(f"loading {filename}...")

    f = uproot.open(f"{data_files_location}/{filename}")

    # determine how many events to read based on requested fraction
    if not (0.0 < frac_events <= 1.0):
        raise ValueError("--frac_events/-f must be in the interval (0, 1].")
    total_entries = f["wcpselection"]["T_eval"].num_entries
    n_events = total_entries if frac_events >= 1.0 else max(1, int(total_entries * frac_events))

    print("loading run, subrun, event, and CV weights using uproot...")
    slice_kwargs = {} if n_events >= total_entries else {"entry_stop": n_events}
    dic = f["nuselection"]["NeutrinoSelectionFilter"].arrays(["run", "sub", "evt", "weightSpline", "weightTune", "weightSplineTimesTune"], library="np", **slice_kwargs)
    curr_weights_df = pl.DataFrame({col: dic[col].tolist() for col in dic})
    curr_weights_df = curr_weights_df.rename({"sub": "subrun", "evt": "event"})

    print("loading wc_kine_reco_Enu for preselection using uproot...")
    dic = f["wcpselection"]["T_KINEvars"].arrays(["kine_reco_Enu"], library="np", **slice_kwargs)
    curr_weights_df = curr_weights_df.with_columns(pl.Series(name="wc_kine_reco_Enu", values=dic["kine_reco_Enu"].tolist()))
    del f
    del dic

    print("loading systematic weights using PyROOT...")
    all_event_weights = get_rw_sys_weights_dic(
        f"{data_files_location}/{filename}",
        max_entries=n_events,
    )
    print("adding systematic weights to dataframe...")

    systematic_keys = list(all_event_weights[0].keys())
    for k in systematic_keys:
        weight_lists = [event_dict[k] for event_dict in all_event_weights]
        curr_weights_df = curr_weights_df.with_columns(pl.Series(name=k, values=weight_lists, dtype=pl.List(pl.Float32)))

    previous_num_events = curr_weights_df.height
    curr_weights_df = curr_weights_df.filter(pl.col("wc_kine_reco_Enu") > 0)
    print(f"kept {curr_weights_df.height}/{previous_num_events} events with after preselection using wc_kine_reco_Enu > 0")
    
    detailed_run_period = "?"
    if "4a.root" in filename:
        detailed_run_period = "4a"
    elif "4b.root" in filename:
        detailed_run_period = "4b"
    elif "4c.root" in filename:
        detailed_run_period = "4c"
    elif "4d.root" in filename:
        detailed_run_period = "4d"
    elif "5.root" in filename:
        detailed_run_period = "5"
    elif "run4b" in filename.lower(): # if the filename doesn't end with the run period, look for run strings in the file names
        detailed_run_period = "4b"
    elif "run4c" in filename.lower():
        detailed_run_period = "4c"
    elif "run4d" in filename.lower():
        detailed_run_period = "4d"
    elif "run5" in filename.lower():
        detailed_run_period = "5"
    elif "run45" in filename.lower():
        detailed_run_period = "45"
    else:
        raise ValueError("Invalid detailed run period!", filename)

    curr_weights_df = curr_weights_df.with_columns(pl.Series(name="detailed_run_period", values=[detailed_run_period] * curr_weights_df.shape[0]))
    curr_weights_df = curr_weights_df.with_columns(pl.Series(name="filename", values=[filename] * curr_weights_df.shape[0]))
    curr_weights_df = curr_weights_df.with_columns(pl.Series(name="filetype", values=[filetype] * curr_weights_df.shape[0]))

    end_time = time.time()

    progress_str = f"\nloaded {filetype:<30}   Run {detailed_run_period:<4} {curr_weights_df.shape[0]:>10,d} events {root_file_size_gb:>6.2f} GB {end_time - start_time:>6.2f} s"
    if frac_events < 1.0:
        progress_str += f" (f={frac_events})"
    print(progress_str)

    return filetype, curr_weights_df


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
    args = parser.parse_args()

    if args.memory_logger:
        start_memory_logger(10)

    if args.frac_events < 1.0:
        print(f"Loading {args.frac_events} fraction of events from each file")

    for file in os.listdir(intermediate_files_location):
        if file.startswith("presel_weights_df") and file.endswith(".parquet"):
            os.remove(f"{intermediate_files_location}/{file}")
    print("Deleted intermediate presel_weights_df*.parquet files")

    print("Starting loop over root files...")
    weights_dfs_pl = pl.DataFrame()

    filenames = os.listdir(data_files_location)
    filenames.sort()
    # sorting these puts an NC Pi0 overlay first, which will have all the WCPMTInfo and truth variables present, 
    # so it can be used to add columns to future dataframes with missing values
    
    for file_num, filename in enumerate(filenames):

        if args.just_one_file and "checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_reco2_hist_4c.root" not in filename:
            continue

        if "UNUSED" in filename or "older_downloads" in filename:
            continue

        if "beam_on" in filename.lower() or "beamon" in filename.lower():
            continue
        if "beam_off" in filename.lower() or "beamoff" in filename.lower() or "ext" in filename.lower():
            continue
        if "one_gamma" in filename.lower():
            continue

        filetype, curr_presel_weights_df = process_root_file(filename, frac_events=args.frac_events)

        print(f"curr_presel_weights_df size: {curr_presel_weights_df.estimated_size() / 1e9:.2f} GB")
        curr_presel_weights_df.write_parquet(f"{intermediate_files_location}/presel_weights_df_{file_num}.parquet")
        print("saved to parquet file")
        del curr_presel_weights_df

        if args.just_one_file:
            break

    print("loading polars dataframes from parquet files...")

    presel_weights_dfs = []
    for file in os.listdir(intermediate_files_location):
        if file.startswith("presel_weights_df") and file.endswith(".parquet"):
            presel_weights_dfs.append(pl.read_parquet(f"{intermediate_files_location}/{file}"))
    presel_weights_df = pl.concat(presel_weights_dfs, how="vertical")
    del presel_weights_dfs

    if presel_weights_df.is_empty():
        raise ValueError("No events in the dataframe!")
    
    print(f"finished looping over root files, presel_weights_df.shape={presel_weights_df.height}")

    print(f"saving {intermediate_files_location}/presel_weights_df.parquet...", end="", flush=True)
    start_time = time.time()
    presel_weights_df.write_parquet(f"{intermediate_files_location}/presel_weights_df.parquet")
    end_time = time.time()
    file_size_gb = os.path.getsize(f"{intermediate_files_location}/presel_weights_df.parquet") / 1024**3
    print(f"done, {file_size_gb:.2f} GB, {end_time - start_time:.2f} seconds")
    main_end_time = time.time()
    print(f"Total time to create weights dataframe: {main_end_time - main_start_time:.2f} seconds")
    