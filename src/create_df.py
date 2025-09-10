
import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
import time
import argparse

from variables import wc_T_BDT_including_training_vars, wc_T_KINEvars_including_training_vars, wc_training_only_vars
from variables import wc_T_bdt_vars, wc_T_kine_vars, wc_T_eval_vars, wc_T_pf_vars, wc_T_pf_data_vars, wc_T_eval_data_vars
from variables import blip_vars, pelee_vars, glee_vars, lantern_vars, vector_columns
from postprocessing import do_orthogonalization_and_POT_weighting, add_extra_true_photon_variables, add_signal_categories
from postprocessing import do_wc_postprocessing, do_blip_postprocessing, do_lantern_postprocessing, do_glee_postprocessing, do_combined_postprocessing

from file_locations import data_files_location, intermediate_files_location

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
    else:
        raise ValueError("Unknown filetype!", filename)
    
    root_file_size_gb = os.path.getsize(f"{data_files_location}/{filename}") / 1024**3

    start_time = time.time()

    f = uproot.open(f"{data_files_location}/{filename}")

    # determine how many events to read based on requested fraction
    if not (0.0 < frac_events <= 1.0):
        raise ValueError("--frac_events/-f must be in the interval (0, 1].")
    total_entries = f["wcpselection"]["T_eval"].num_entries
    n_events = total_entries if frac_events >= 1.0 else max(1, int(total_entries * frac_events))
    slice_kwargs = {} if n_events >= total_entries else {"entry_stop": n_events}

    # loading Wire-Cell variables
    dic = {}
    dic.update(f["wcpselection"]["T_BDTvars"].arrays(wc_T_BDT_including_training_vars, library="np", **slice_kwargs))
    dic.update(f["wcpselection"]["T_KINEvars"].arrays(wc_T_KINEvars_including_training_vars, library="np", **slice_kwargs))
    if filetype == "ext" or filetype == "data":
        dic.update(f["wcpselection"]["T_PFeval"].arrays(wc_T_pf_data_vars, library="np", **slice_kwargs))
        dic.update(f["wcpselection"]["T_eval"].arrays(wc_T_eval_data_vars, library="np", **slice_kwargs))
    else:
        dic.update(f["wcpselection"]["T_PFeval"].arrays(wc_T_pf_vars, library="np", **slice_kwargs))
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
    
    file_POT = file_POT_total * frac_events
    wc_df["file_POT"] = file_POT
    
    # loading blip variables
    dic = {}
    dic.update(f["nuselection"]["NeutrinoSelectionFilter"].arrays(blip_vars, library="np", **slice_kwargs))
    for col in dic:
        dic[col] = dic[col].tolist()
    blip_df = pd.DataFrame(dic)

    # loading PeLEE variables
    dic = {}
    dic.update(f["nuselection"]["NeutrinoSelectionFilter"].arrays(pelee_vars, library="np", **slice_kwargs))
    for col in dic:
        dic[col] = dic[col].tolist()
    pelee_df = pd.DataFrame(dic)

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

    wc_df = wc_df.add_prefix("wc_")
    # blip_df = blip_df.add_prefix("blip_") # blip variables already have the "blip_" prefix
    pelee_df = pelee_df.add_prefix("pelee_")
    glee_df = glee_df.add_prefix("glee_")
    lantern_df = lantern_df.add_prefix("lantern_")

    all_df = pd.concat([wc_df, blip_df, pelee_df, glee_df, lantern_df], axis=1)

    # remove some of these prefixes, for things that should be universal
    all_df.rename(columns={"wc_run": "run", "wc_subrun": "subrun", "wc_event": "event"}, inplace=True)

    all_df["filename"] = filename
    all_df["filetype"] = filetype

    end_time = time.time()

    progress_str = f"loaded {filetype:<20} {all_df.shape[0]:>10,d} events {file_POT:>10.2e} POT {root_file_size_gb:>6.2f} GB {end_time - start_time:>6.2f} s"
    if frac_events < 1.0:
        progress_str += f" (f={frac_events})"
    print(progress_str)

    return filetype, all_df, file_POT


if __name__ == "__main__":
    main_start_time = time.time()

    parser = argparse.ArgumentParser(description="Create merged dataframe from SURPRISE 4b ROOT files")
    parser.add_argument("-f", "--frac_events", type=float, default=1.0,
                        help="Fraction of events (and POT) to load from each file, in (0,1]. Default: 1.0")
    args = parser.parse_args()

    if args.frac_events < 1.0:
        print(f"Loading {args.frac_events} fraction of events from each file")

    print("Starting loop over root files...")
    all_df = pd.DataFrame()
    all_ncpi0_POTs = []
    all_nu_POTs = []
    all_nue_POTs = []
    all_dirt_POTs = []
    all_ext_POTs = []
    for filename in os.listdir(data_files_location):
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
        else:
            raise ValueError("Unknown filetype!", filetype)

        all_df = pd.concat([all_df, curr_df])
        all_df.reset_index(drop=True, inplace=True)

    del curr_df # should save a bit of memory

    # TODO: When we have more files, do weighting to make each set of run fractions match the run fractions in data

    pot_dic = {
        "nc_pi0_overlay": sum(all_ncpi0_POTs),
        "nu_overlay": sum(all_nu_POTs),
        "nue_overlay": sum(all_nue_POTs),
        "dirt_overlay": sum(all_dirt_POTs),
        "ext": sum(all_ext_POTs),
    }

    print("doing post-processing...")
    all_df = do_orthogonalization_and_POT_weighting(all_df, pot_dic)
    all_df = do_wc_postprocessing(all_df)
    all_df = add_extra_true_photon_variables(all_df)
    all_df = add_signal_categories(all_df)

    all_df = do_blip_postprocessing(all_df)
    all_df = do_lantern_postprocessing(all_df)
    all_df = do_glee_postprocessing(all_df)
    all_df = do_combined_postprocessing(all_df)

    file_RSEs = []
    for filetype, run, subrun, event in zip(all_df["filetype"].to_numpy(), all_df["run"].to_numpy(), all_df["subrun"].to_numpy(), all_df["event"].to_numpy()):
        file_RSE = f"{filetype}_{run:06d}_{subrun:06d}_{event:06d}"
        file_RSEs.append(file_RSE)
    assert len(file_RSEs) == len(set(file_RSEs)), "Duplicate filetype/run/subrun/event!"

    print("removing vector variables (they were only used for pre-processing)...")
    save_columns = [col for col in all_df.columns if col not in vector_columns]
    all_df = all_df[save_columns]


    print(f"saving {intermediate_files_location}/presel_df_train_vars.pkl...", end="", flush=True)
    start_time = time.time()
    presel_df = all_df.query("wc_kine_reco_Enu > 0 and wc_shw_sp_n_20mev_showers > 0") # the generic selection wc_kine_reco_Enu > 0 was already applied
    presel_df.reset_index(drop=True, inplace=True)
    presel_df.to_pickle(f"{intermediate_files_location}/presel_df_train_vars.pkl")
    end_time = time.time()
    file_size_gb = os.path.getsize(f"{intermediate_files_location}/presel_df_train_vars.pkl") / 1024**3
    print(f"done, {file_size_gb:.2f} GB, {end_time - start_time:.2f} seconds")

    print(f"saving {intermediate_files_location}/intermediate_files/all_df.pkl...", end="", flush=True)
    start_time = time.time()
    # remove the large number of WC training-only-variables for a smaller file size
    remove_columns = wc_training_only_vars
    final_save_columns = [col for col in all_df.columns if col not in remove_columns]

    all_df_no_training_columns = all_df[final_save_columns]
    all_df_no_training_columns.to_pickle(f"{intermediate_files_location}/all_df.pkl")
    end_time = time.time()
    file_size_gb = os.path.getsize(f"{intermediate_files_location}/all_df.pkl") / 1024**3
    print(f"done, {file_size_gb:.2f} GB, {end_time - start_time:.2f} seconds")
    main_end_time = time.time()
    print(f"Total time to create the dataframes: {main_end_time - main_start_time:.2f} seconds")
    