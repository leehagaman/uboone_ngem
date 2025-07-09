
import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os

from variables import wc_T_BDT_including_training_vars, wc_T_KINEvars_including_training_vars
from variables import wc_T_bdt_vars, wc_T_kine_vars, wc_T_eval_vars, wc_T_pf_vars
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
    elif file_category == "SURPRISE_4b_data":
        raise ValueError("Not looking at data yet!")
    else:
        raise ValueError("Invalid root file type!")
    f = uproot.open(f"data_files/{filename}")

    # loading Wire-Cell variables
    dic = {}
    dic.update(f["wcpselection"]["T_BDTvars"].arrays(wc_T_BDT_including_training_vars, library="np"))
    dic.update(f["wcpselection"]["T_KINEvars"].arrays(wc_T_KINEvars_including_training_vars, library="np"))
    dic.update(f["wcpselection"]["T_eval"].arrays(wc_T_eval_vars, library="np"))
    dic.update(f["wcpselection"]["T_PFeval"].arrays(wc_T_pf_vars, library="np"))
    file_POT = np.sum(f["wcpselection"]["T_pot"].arrays("pot_tor875good", library="np")["pot_tor875good"])
    for col in dic:
        dic[col] = dic[col].tolist()
    wc_df = pd.DataFrame(dic)
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

    all_df = pd.concat([nc_pi0_df, nu_df, dirt_df])


    pot_dic = {
        "nc_pi0_overlay": nc_pi0_POT,
        "nu_overlay": nu_POT,
        "dirt_overlay": dirt_POT,
    }

    print("doing post-processing...")
    all_df = do_orthogonalization_and_POT_weighting(all_df, pot_dic)
    all_df = do_wc_postprocessing(all_df)
    all_df = do_blip_postprocessing(all_df)
    all_df = add_extra_true_photon_variables(all_df)
    all_df = add_signal_categories(all_df)

    print("saving to pickle...")
    with open("intermediate_files/all_df.pkl", "wb") as f:
        pickle.dump(all_df, f)

    # restrict to fewer columns
    # Add prefixes to match the actual column names in the DataFrame
    non_training_columns = ["run", "subrun", "event", "filetype", "wc_net_weight", "reconstructable_signal_category", "physics_signal_category"]
    non_training_columns += ["wc_" + var for var in wc_T_bdt_vars + wc_T_kine_vars + wc_T_eval_vars + wc_T_pf_vars if var not in ["run", "subrun", "event"]]
    non_training_columns += [var for var in blip_vars]
    non_training_columns += ["pelee_" + var for var in pelee_vars]
    
    all_df_no_training_vars = all_df[non_training_columns]

    with open("intermediate_files/all_df_no_training_vars.pkl", "wb") as f:
        pickle.dump(all_df_no_training_vars, f)

    print(f"saved intermediate_files/all_df.pkl, {os.path.getsize('intermediate_files/all_df.pkl') / 1024**3:.2f} GB")
    print(f"saved intermediate_files/all_df_no_training_vars.pkl, {os.path.getsize('intermediate_files/all_df_no_training_vars.pkl') / 1024**3:.2f} GB")
    