
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
from variables import blip_vars, pelee_vars, glee_vars, lantern_vars
from postprocessing import do_orthogonalization_and_POT_weighting, add_extra_true_photon_variables, add_signal_categories
from postprocessing import do_wc_postprocessing, do_blip_postprocessing, do_lantern_postprocessing, do_glee_postprocessing, do_combined_postprocessing

def process_root_file(file_category, frac_events: float = 1.0):

    # loading the root file
    if file_category == "SURPRISE_4b_NC_pi0_overlay":
        filename = "SURPRISE_Test_Samples_v10_04_07_05_Run4b_hyper_unified_reco2_BNB_nu_NC_pi0_overlay_may8_reco2_hist_62280465_snapshot.root"
        filetype = "nc_pi0_overlay"
    elif file_category == "SURPRISE_4b_nu_overlay":
        filename = "SURPRISE_Test_Samples_v10_04_07_05_Run4b_hyper_unified_reco2_BNB_nu_overlay_may8_reco2_hist_62280499_snapshot.root"
        filetype = "nu_overlay"
    elif file_category == "SURPRISE_4b_dirt_overlay":
        filename = "SURPRISE_Test_Samples_v10_04_07_05_Run4b_hyper_unified_reco2_BNB_dirt_may8_reco2_hist_62280564_snapshot.root"
        filetype = "dirt_overlay"
    elif file_category == "SURPRISE_4b_ext":
        filename = "SURPRISE_Test_Samples_v10_04_07_05_Run4b_hyper_unified_reco2_BNB_beam_off_may8_reco2_hist_goodruns_62280841_snapshot.root"
        filetype = "ext"
        # from https://docs.google.com/spreadsheets/d/1AVrUfAffE6mQw5t-gQnlXxxcyhwVxtPGGUp_7c1MT2w/edit?gid=1068115471#gid=1068115471
        # SURPRISE Samples for Collaboration Meeting
        # Hyper unified (uboonecode v10_04_07_05)
        # Accessed 2025_07_09
        wc_processed_ext_num_spills = 34369859
        wc_processed_data_num_spills = 28396891
        wc_processed_data_POT = 1.197e20
    elif file_category == "SURPRISE_4b_data":
        raise ValueError("Not looking at data yet!")
    else:
        raise ValueError("Invalid root file type!")
    
    root_file_size_gb = os.path.getsize(f"data_files/{filename}") / 1024**3

    start_time = time.time()

    f = uproot.open(f"data_files/{filename}")

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
        file_POT_total = wc_processed_data_POT * wc_processed_ext_num_spills / wc_processed_data_num_spills
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

    all_df["filetype"] = filetype

    end_time = time.time()

    progress_str = f"loaded {filetype:<20} {all_df.shape[0]:>10,d} events {file_POT:>10.2e} POT {root_file_size_gb:>6.2f} GB {end_time - start_time:>6.2f} s"
    if frac_events < 1.0:
        progress_str += f" (f={frac_events})"
    print(progress_str)

    return all_df, file_POT


if __name__ == "__main__":
    main_start_time = time.time()

    parser = argparse.ArgumentParser(description="Create merged dataframe from SURPRISE 4b ROOT files")
    parser.add_argument("-f", "--frac_events", type=float, default=1.0,
                        help="Fraction of events (and POT) to load from each file, in (0,1]. Default: 1.0")
    args = parser.parse_args()

    nc_pi0_df, nc_pi0_POT = process_root_file("SURPRISE_4b_NC_pi0_overlay", frac_events=args.frac_events)
    nu_df, nu_POT = process_root_file("SURPRISE_4b_nu_overlay", frac_events=args.frac_events)
    dirt_df, dirt_POT = process_root_file("SURPRISE_4b_dirt_overlay", frac_events=args.frac_events)
    ext_df, ext_POT = process_root_file("SURPRISE_4b_ext", frac_events=args.frac_events)

    if args.frac_events < 1.0:
        print(f"Loading {args.frac_events} fraction of events from each file")

    all_df = pd.concat([nc_pi0_df, nu_df, dirt_df, ext_df])

    pot_dic = {
        "nc_pi0_overlay": nc_pi0_POT,
        "nu_overlay": nu_POT,
        "dirt_overlay": dirt_POT,
        "ext": ext_POT,
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


    print("saving intermediate_files/generic_df_train_vars.pkl...", end="", flush=True)
    start_time = time.time()
    generic_df = all_df.query("wc_kine_reco_Enu > 0").reset_index(drop=True)    
    generic_df.to_pickle("intermediate_files/generic_df_train_vars.pkl")
    end_time = time.time()
    file_size_gb = os.path.getsize('intermediate_files/generic_df_train_vars.pkl') / 1024**3
    print(f"done, {file_size_gb:.2f} GB, {end_time - start_time:.2f} seconds")

    print("saving intermediate_files/all_df.pkl...", end="", flush=True)
    start_time = time.time()
    # restrict to fewer columns for a smaller file size
    vector_columns = [
        "wc_kine_energy_particle",
        "wc_kine_particle_type",
        "wc_truth_id",
        "wc_truth_pdg",
        "wc_truth_mother",
        "wc_truth_startMomentum",
        "wc_truth_startXYZT",
        "wc_truth_endXYZT",
        "wc_reco_id",
        "wc_reco_pdg",
        "wc_reco_mother",
        "wc_reco_startMomentum",
        "wc_reco_startXYZT",
        "wc_reco_endXYZT",

        "lantern_showerPhScore",

        "blip_x",
        "blip_y",
        "blip_z",
        "blip_dx",
        "blip_dw",
        "blip_energy",
        "blip_true_g4id",
        "blip_true_pdg",
        "blip_true_energy",

        "glee_sss_candidate_veto_score",
        "glee_sss3d_shower_score",

        "lantern_showerIsSecondary",
        "lantern_showerPID",
        "lantern_showerPhScore",
        "lantern_showerElScore",
        "lantern_showerMuScore",
        "lantern_showerPiScore",
        "lantern_showerPrScore",
        "lantern_showerCharge",
        "lantern_showerPurity",
        "lantern_showerComp",
        "lantern_showerPrimaryScore",
        "lantern_showerFromNeutralScore",
        "lantern_showerFromChargedScore",
        "lantern_showerCosTheta",
        "lantern_showerCosThetaY",
        "lantern_showerDistToVtx",
        "lantern_showerStartDirX",
        "lantern_showerStartDirY",
        "lantern_showerStartDirZ",

        "lantern_trackIsSecondary",
        "lantern_trackClassified",
        "lantern_trackCharge",
        "lantern_trackComp",
        "lantern_trackPurity",
        "lantern_trackPrimaryScore",
        "lantern_trackFromNeutralScore",
        "lantern_trackFromChargedScore",
        "lantern_trackCosTheta",
        "lantern_trackCosThetaY",
        "lantern_trackDistToVtx",
        "lantern_trackStartDirX",
        "lantern_trackStartDirY",
        "lantern_trackStartDirZ",
        "lantern_trackElScore",
        "lantern_trackPhScore",
        "lantern_trackMuScore",
        "lantern_trackPiScore",
        "lantern_trackPrScore",
        "lantern_trackPID",
    ]
    remove_columns = wc_training_only_vars + vector_columns
    final_save_columns = [col for col in all_df.columns if col not in remove_columns]

    all_df_no_training_columns = all_df[final_save_columns]
    all_df_no_training_columns.to_pickle("intermediate_files/all_df.pkl")
    end_time = time.time()
    file_size_gb = os.path.getsize('intermediate_files/all_df.pkl') / 1024**3
    print(f"done, {file_size_gb:.2f} GB, {end_time - start_time:.2f} seconds")
    main_end_time = time.time()
    print(f"Total time to create the dataframes: {main_end_time - main_start_time:.2f} seconds")
    