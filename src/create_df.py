
import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os

from variables import wc_T_BDT_including_training_vars, wc_T_KINEvars_including_training_vars
from variables import wc_T_bdt_vars, wc_T_kine_vars, wc_T_eval_vars, wc_T_pf_vars, wc_T_pf_data_vars, wc_T_eval_data_vars
from variables import blip_vars, pelee_vars
from postprocessing import do_orthogonalization_and_POT_weighting, do_wc_postprocessing, do_blip_postprocessing
from postprocessing import add_extra_true_photon_variables, add_signal_categories

def process_root_file(file_category):

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
    f = uproot.open(f"data_files/{filename}")

    # loading Wire-Cell variables
    dic = {}
    dic.update(f["wcpselection"]["T_BDTvars"].arrays(wc_T_BDT_including_training_vars, library="np"))
    dic.update(f["wcpselection"]["T_KINEvars"].arrays(wc_T_KINEvars_including_training_vars, library="np"))
    if filetype == "ext" or filetype == "data":
        dic.update(f["wcpselection"]["T_PFeval"].arrays(wc_T_pf_data_vars, library="np"))
        dic.update(f["wcpselection"]["T_eval"].arrays(wc_T_eval_data_vars, library="np"))
    else:
        dic.update(f["wcpselection"]["T_PFeval"].arrays(wc_T_pf_vars, library="np"))
        dic.update(f["wcpselection"]["T_eval"].arrays(wc_T_eval_vars, library="np"))
    file_POT = np.sum(f["wcpselection"]["T_pot"].arrays("pot_tor875good", library="np")["pot_tor875good"])
    for col in dic:
        dic[col] = dic[col].tolist()
    wc_df = pd.DataFrame(dic)
    if filetype == "ext":
        file_POT = wc_processed_data_POT * wc_processed_ext_num_spills / wc_processed_data_num_spills
    wc_df["file_POT"] = file_POT
    
    # loading blip variables
    dic = {}
    dic.update(f["nuselection"]["NeutrinoSelectionFilter"].arrays(blip_vars, library="np"))
    for col in dic:
        dic[col] = dic[col].tolist()
    blip_df = pd.DataFrame(dic)

    # loading PeLEE variables
    dic = {}
    dic.update(f["nuselection"]["NeutrinoSelectionFilter"].arrays(pelee_vars, library="np"))
    for col in dic:
        dic[col] = dic[col].tolist()
    pelee_df = pd.DataFrame(dic)

    wc_df = wc_df.add_prefix("wc_")
    # blip_df = blip_df.add_prefix("blip_") # blip variables already have the "blip_" prefix
    pelee_df = pelee_df.add_prefix("pelee_")

    all_df = pd.concat([wc_df, blip_df, pelee_df], axis=1)

    # remove some of these prefixes, for things that should be universal
    all_df.rename(columns={"wc_run": "run", "wc_subrun": "subrun", "wc_event": "event"}, inplace=True)

    all_df["filetype"] = filetype

    print(f"loaded {filetype}, {all_df.shape[0]} events, {file_POT:.2e} POT")

    return all_df, file_POT


if __name__ == "__main__":

    nc_pi0_df, nc_pi0_POT = process_root_file("SURPRISE_4b_NC_pi0_overlay")
    nu_df, nu_POT = process_root_file("SURPRISE_4b_nu_overlay")
    dirt_df, dirt_POT = process_root_file("SURPRISE_4b_dirt_overlay")
    ext_df, ext_POT = process_root_file("SURPRISE_4b_ext")

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
    all_df = do_blip_postprocessing(all_df)
    all_df = add_extra_true_photon_variables(all_df)
    all_df = add_signal_categories(all_df)

    RSEs = []
    for filetype, run, subrun, event in zip(all_df["filetype"].to_numpy(), all_df["run"].to_numpy(), all_df["subrun"].to_numpy(), all_df["event"].to_numpy()):
        RSE = f"{filetype}_{run:06d}_{subrun:06d}_{event:06d}"
        RSEs.append(RSE)
    assert len(RSEs) == len(set(RSEs)), "Duplicate RSEs!"

    print("saving to pickle...")

    # restrict to generic selected events for a smaller file size
    generic_df = all_df.query("wc_kine_reco_Enu > 0").reset_index(drop=True)
    with open("intermediate_files/generic_df_train_vars.pkl", "wb") as f:
        pickle.dump(generic_df, f)
    print(f"saved intermediate_files/generic_df_train_vars.pkl, {os.path.getsize('intermediate_files/generic_df_train_vars.pkl') / 1024**3:.2f} GB")

    # restrict to fewer columns for a smaller file size
    non_training_columns = ["run", "subrun", "event", "filetype", "wc_net_weight", "topological_signal_category", "physics_signal_category"]
    non_training_columns += ["wc_" + var for var in wc_T_bdt_vars + wc_T_kine_vars + wc_T_eval_vars + wc_T_pf_vars if var not in ["run", "subrun", "event"]]
    non_training_columns += [var for var in blip_vars]
    non_training_columns += ["pelee_" + var for var in pelee_vars]
    all_df_no_training_columns = all_df[non_training_columns]
    with open("intermediate_files/all_df.pkl", "wb") as f:
        pickle.dump(all_df_no_training_columns, f)
    print(f"saved intermediate_files/all_df.pkl, {os.path.getsize('intermediate_files/all_df.pkl') / 1024**3:.2f} GB")
    