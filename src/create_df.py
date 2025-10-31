
import uproot
import numpy as np
import pandas as pd
import polars as pl
import os
import time
import argparse


from ntuple_variables.variables import wc_T_BDT_including_training_vars, wc_T_KINEvars_including_training_vars, wc_training_only_vars
from ntuple_variables.variables import wc_T_spacepoints_vars, wc_T_eval_vars, wc_T_pf_vars, wc_T_pf_data_vars, wc_T_eval_data_vars
from ntuple_variables.variables import blip_vars, pandora_vars, glee_vars, lantern_vars, vector_columns
from postprocessing import do_orthogonalization_and_POT_weighting, add_extra_true_photon_variables, do_spacepoint_postprocessing, add_signal_categories
from postprocessing import do_wc_postprocessing, do_pandora_postprocessing, do_blip_postprocessing, do_lantern_postprocessing, do_combined_postprocessing, do_glee_postprocessing
from postprocessing import remove_vector_variables

from file_locations import data_files_location, intermediate_files_location

from df_helpers import align_columns_for_concat, compress_df
from memory_monitoring import start_memory_logger

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
    
    root_file_size_gb = os.path.getsize(f"{data_files_location}/{filename}") / 1024**3

    start_time = time.time()

    print(f"loading {filename}...")

    f = uproot.open(f"{data_files_location}/{filename}")

    if filetype == "data": # quartering the data to use as open data
        curr_frac_events = frac_events / 4
    else:
        curr_frac_events = frac_events

    # determine how many events to read based on requested fraction
    if not (0.0 < frac_events <= 1.0):
        raise ValueError("--frac_events/-f must be in the interval (0, 1].")
    total_entries = f["wcpselection"]["T_eval"].num_entries
    n_events = total_entries if curr_frac_events >= 1.0 else max(1, int(total_entries * curr_frac_events))
    slice_kwargs = {} if n_events >= total_entries else {"entry_stop": n_events}

    curr_wc_T_pf_vars = wc_T_pf_vars

    curr_wc_T_BDT_including_training_vars = wc_T_BDT_including_training_vars
    if "v10_04_07_09" in filename:
        print(f"    TEMPORARY: NOT LOADING WCPMTInfo VARIABLES FOR {filetype}")
        curr_wc_T_BDT_including_training_vars = [var for var in wc_T_BDT_including_training_vars if "WCPMTInfo" not in var]
            

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
    file_POT_total = np.sum(f["wcpselection"]["T_pot"].arrays("pot_tor875good", library="np")["pot_tor875good"])
    for col in dic:
        dic[col] = dic[col].tolist()
    wc_df = pd.DataFrame(dic)

    if filetype == "ext":
        # TODO: When we have more files, do weighting to make each set of run fractions match the run fractions in data

        # trigger numbers from https://docs.google.com/spreadsheets/d/1RUiX2M6zoob9R0YWPLummHzmX5UeLLEtS-7ZU-x2gA4
        if filename == "MCC9.10_Run4b_v10_04_07_09_Run4b_BNB_beam_off_surprise_reco2_hist.root":
            ext_num_triggers = 88445969
            corresponding_data_num_triggers = 31582916
            corresponding_data_POT = 1.332e20
            file_POT_total = corresponding_data_POT * ext_num_triggers / corresponding_data_num_triggers
        else:
            raise ValueError("Invalid EXT file, num triggers not found!")
    elif filetype == "data":
        if filename == "MCC9.10_Run4b_v10_04_07_11_BNB_beam_on_surprise_reco2_hist.root":
            file_POT_total = 1.332e20
        else:
            raise ValueError("Invalid data file!")
    
    file_POT = file_POT_total * curr_frac_events
    wc_df["wc_file_POT"] = file_POT
    
    # loading blip variables
    dic = {}
    dic.update(f["nuselection"]["NeutrinoSelectionFilter"].arrays(blip_vars, library="np", **slice_kwargs))
    for col in dic:
        dic[col] = dic[col].tolist()
    blip_df = pd.DataFrame(dic)

    # loading pandora variables
    dic = {}
    dic.update(f["nuselection"]["NeutrinoSelectionFilter"].arrays(pandora_vars, library="np", **slice_kwargs))
    for col in dic:
        dic[col] = dic[col].tolist()
    pandora_df = pd.DataFrame(dic)

    # loading gLEE variables
    dic = {}
    dic.update(f["singlephotonana"]["vertex_tree"].arrays(glee_vars, library="np", **slice_kwargs))
    for col in dic:
        dic[col] = dic[col].tolist()
    glee_df = pd.DataFrame(dic)

    # loading LANTERN variables
    dic = {}
    dic.update(f["lantern"]["EventTree"].arrays(lantern_vars, library="np", **slice_kwargs))
    for col in dic:
        dic[col] = dic[col].tolist()
    lantern_df = pd.DataFrame(dic)

    del f
    del dic

    wc_df = wc_df.add_prefix("wc_")
    # blip_df = blip_df.add_prefix("blip_") # blip variables already have the "blip_" prefix
    pandora_df = pandora_df.add_prefix("pandora_")
    glee_df = glee_df.add_prefix("glee_")
    lantern_df = lantern_df.add_prefix("lantern_")

    all_df = pd.concat([wc_df, blip_df, pandora_df, glee_df, lantern_df], axis=1)
    del wc_df
    del blip_df
    del pandora_df
    del glee_df
    del lantern_df

    # remove some of these prefixes, for things that should be universal
    all_df.rename(columns={"wc_run": "run", "wc_subrun": "subrun", "wc_event": "event"}, inplace=True)

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

    all_df["detailed_run_period"] = detailed_run_period
    all_df["filename"] = filename
    all_df["filetype"] = filetype

    end_time = time.time()

    events_per_POT = all_df.shape[0] / (file_POT / 1e19)

    progress_str = f"\nloaded {filetype:<30}   Run {detailed_run_period:<4} {all_df.shape[0]:>10,d} events {file_POT:>10.2e} POT {events_per_POT:>6.2f} events / 1e19 POT {root_file_size_gb:>6.2f} GB {end_time - start_time:>6.2f} s"
    if frac_events < 1.0:
        progress_str += f" (f={frac_events})"
    print(progress_str)

    return filetype, all_df, file_POT


if __name__ == "__main__":
    main_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Create merged dataframe from SURPRISE 4b ROOT files")
    parser.add_argument("-f", "--frac_events", type=float, default=1.0,
                        help="Fraction of events (and POT) to load from each file, in (0,1]. Default: 1.0")
    parser.add_argument("-m", "--memory_logger", action="store_true", default=False,
                        help="Start a memory logger thread")
    parser.add_argument("--just_one_file", action="store_true", default=False,
                        help="Only process one file for debugging purposes")
    args = parser.parse_args()

    if args.memory_logger:
        start_memory_logger(10)

    if args.frac_events < 1.0:
        print(f"Loading {args.frac_events} fraction of events from each file")

    for file in os.listdir(intermediate_files_location):
        if ((file.startswith("curr_df_pl_") and file.endswith(".parquet")) or file == "presel_df_train_vars.parquet" or file == "all_df.parquet"):
            os.remove(f"{intermediate_files_location}/{file}")
    print("Deleted intermediate parquet files")

    print("Starting loop over root files...")
    all_df_pl = pl.DataFrame()
    all_ncpi0_POTs = []
    all_nu_POTs = []
    all_nue_POTs = []
    all_dirt_POTs = []
    all_ext_POTs = []
    all_delete_one_gamma_POTs = []
    all_isotropic_one_gamma_POTs = []
    all_data_POTs = []

    filenames = os.listdir(data_files_location)
    filenames.sort()
    # sorting these puts an NC Pi0 overlay first, which will have all the WCPMTInfo and truth variables present, 
    # so it can be used to add columns to future dataframes with missing values
    
    for file_num, filename in enumerate(filenames):

        if args.just_one_file and "checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_reco2_hist_4c.root" not in filename:
            continue

        if "UNUSED" in filename or "older_downloads" in filename:
            continue

        filetype, curr_df, curr_POT = process_root_file(filename, frac_events=args.frac_events)
        if filetype == "nc_pi0_overlay":
            all_ncpi0_POTs.append(curr_POT)
        elif filetype == "nu_overlay":
            all_nu_POTs.append(curr_POT)
        elif filetype == "nue_overlay":
            all_nue_POTs.append(curr_POT)
        elif filetype == "dirt_overlay":
            all_dirt_POTs.append(curr_POT)
        elif filetype == "ext":
            all_ext_POTs.append(curr_POT)
        elif filetype == "delete_one_gamma_overlay":
            all_delete_one_gamma_POTs.append(curr_POT)
        elif filetype == "isotropic_one_gamma_overlay":
            all_isotropic_one_gamma_POTs.append(curr_POT)
        elif filetype == "data":
            all_data_POTs.append(curr_POT)
        else:
            raise ValueError("Unknown filetype!", filetype)


        print("doing post-processing that requires vector variables...")

        curr_df = do_wc_postprocessing(curr_df)
        curr_df = add_extra_true_photon_variables(curr_df)
        curr_df = do_spacepoint_postprocessing(curr_df)
        curr_df = do_pandora_postprocessing(curr_df)
        curr_df = do_blip_postprocessing(curr_df)
        curr_df = do_lantern_postprocessing(curr_df)
        curr_df = do_glee_postprocessing(curr_df)

        curr_df = remove_vector_variables(curr_df)
        #curr_df = compress_df(curr_df) # not needed, because we're converting to polars which will be more compressed

        # converting to polars
        curr_df_pl = pl.from_pandas(curr_df)
        del curr_df

        curr_df_pl = curr_df_pl.with_columns([pl.col(pl.Float64).cast(pl.Float32)])
        curr_df_pl = curr_df_pl.with_columns([pl.col(pl.Int32).cast(pl.Int64)])

        print(f"curr_df_pl size: {curr_df_pl.estimated_size() / 1e9:.2f} GB")
        curr_df_pl.write_parquet(f"{intermediate_files_location}/curr_df_pl_{file_num}.parquet")
        print("saved to parquet file")
        del curr_df_pl

    print("loading polars dataframes from parquet files...")

    pl_dfs = []
    for file in os.listdir(intermediate_files_location):
        if file.startswith("curr_df_pl_") and file.endswith(".parquet"):
            print(f"Reading {file}")
            pl_dfs.append(pl.read_parquet(f"{intermediate_files_location}/{file}"))
            print(f"Read {file}, estimated size: {pl_dfs[-1].estimated_size() / 1e9:.2f} GB")
    all_df = pl.concat(align_columns_for_concat(pl_dfs), how="vertical")
    del pl_dfs
    print(f"all_df size: {all_df.estimated_size() / 1e9:.2f} GB")

    for file in os.listdir(intermediate_files_location):
        if file.startswith("curr_df_pl_") and file.endswith(".parquet"):
            os.remove(f"{intermediate_files_location}/{file}")
    print("Deleted intermediate curr_df_pl*.parquet files")

    if all_df.is_empty():
        raise ValueError("No events in the dataframe!")
    
    print(f"all_df.height={all_df.height}")

    # TODO: When we have more files, do weighting to make each set of run fractions match the run fractions in data

    pot_dic = {
        "nc_pi0_overlay": sum(all_ncpi0_POTs),
        "nu_overlay": sum(all_nu_POTs),
        "nue_overlay": sum(all_nue_POTs),
        "dirt_overlay": sum(all_dirt_POTs),
        "ext": sum(all_ext_POTs),
        "delete_one_gamma_overlay": sum(all_delete_one_gamma_POTs),
        "isotropic_one_gamma_overlay": sum(all_isotropic_one_gamma_POTs),
        "data": sum(all_data_POTs),
    }

    print("doing post-processing that doesn't require vector variables using polars...")

    all_df = do_combined_postprocessing(all_df)
    if pot_dic["data"] > 0:
        normalizing_POT = pot_dic["data"]
    else:
        normalizing_POT = 1.11e21
    
    all_df = do_orthogonalization_and_POT_weighting(all_df, pot_dic, normalizing_POT=normalizing_POT)
    all_df = add_signal_categories(all_df)

    file_RSEs = []
    for filetype, run, subrun, event in zip(all_df["filetype"].to_numpy(), all_df["run"].to_numpy(), all_df["subrun"].to_numpy(), all_df["event"].to_numpy()):
        file_RSE = f"{filetype}_{run:06d}_{subrun:06d}_{event:06d}"
        file_RSEs.append(file_RSE)
    assert len(file_RSEs) == len(set(file_RSEs)), "Duplicate filetype/run/subrun/event!"

    print(f"saving {intermediate_files_location}/presel_df_train_vars.parquet...", end="", flush=True)
    start_time = time.time()
    presel_df = all_df.filter(pl.col("wc_kine_reco_Enu") > 0)
    presel_df.write_parquet(f"{intermediate_files_location}/presel_df_train_vars.parquet")
    end_time = time.time()
    file_size_gb = os.path.getsize(f"{intermediate_files_location}/presel_df_train_vars.parquet") / 1024**3
    print(f"done, {file_size_gb:.2f} GB, {end_time - start_time:.2f} seconds")

    print(f"saving {intermediate_files_location}/all_df.parquet...", end="", flush=True)
    start_time = time.time()
    # remove the large number of WC training-only-variables for a smaller file size
    remove_columns = wc_training_only_vars
    final_save_columns = [col for col in all_df.columns if col not in remove_columns]

    all_df_no_training_columns = all_df[final_save_columns]
    all_df_no_training_columns.write_parquet(f"{intermediate_files_location}/all_df.parquet")
    end_time = time.time()
    file_size_gb = os.path.getsize(f"{intermediate_files_location}/all_df.parquet") / 1024**3
    print(f"done, {file_size_gb:.2f} GB, {end_time - start_time:.2f} seconds")
    main_end_time = time.time()
    print(f"Total time to create the dataframes: {main_end_time - main_start_time:.2f} seconds")
    