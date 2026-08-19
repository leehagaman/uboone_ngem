import uproot

data_files_location = "/nevis/riverside/data/leehagaman/ngem/data_files"

# All overlay files with GENIE systematics (spline_weights tree), from download_input_files.sh
files = [
    # nu overlay files
    "checkout_MCC9.10_Run123_v10_04_07_20_BNB_nu_overlay_surprise_reco2_hist_1.root",
    "checkout_MCC9.10_Run123_v10_04_07_20_BNB_nu_overlay_surprise_reco2_hist_2.root",
    "checkout_MCC9.10_Run123_v10_04_07_20_BNB_nu_overlay_surprise_reco2_hist_3.root",
    "checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_nu_overlay_retuple_retuple_hist_4a.root",
    "checkout_MCC9.10_Run4b_v10_04_07_20_BNB_nu_overlay_retuple_retuple_hist.root",
    "checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_nu_overlay_retuple_retuple_hist_4c.root",
    "checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_nu_overlay_retuple_retuple_hist_4d.root",
    "checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_nu_overlay_retuple_retuple_hist_5.root",

    # nue overlay files
    "checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_intrinsic_nue_overlay_surprise_reco2_hist_4a.root",
    "checkout_MCC9.10_Run4b_v10_04_07_09_BNB_nue_overlay_surprise_reco2_hist.root",
    "checkout_MCC9.10_Run4c_v10_04_07_13_BNB_intrinsic_nue_overlay_surprise_redo_reco2_hist.root",
    "checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_intrinsic_nue_overlay_surprise_reco2_hist_4d.root",
    "checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_intrinsic_nue_overlay_surprise_reco2_hist_5.root",

    # NCpi0 overlay files
    "checkout_MCC9.10_Run4a_v10_04_07_16_BNB_NCpi0_overlay_surprise_reco2_hist.root",
    "checkout_MCC9.10_Run4b_v10_04_07_09_BNB_NC_pi0_overlay_surprise_reco2_hist.root",
    "checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_reco2_hist_4c.root",
    "checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_reco2_hist_4d.root",
    "checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_reco2_hist_5.root",

    # numuCCpi0 overlay files
    "checkout_MCC9.10_Run4a_v10_04_07_16_BNB_CCpi0_overlay_surprise_reco2_hist.root",
    "checkout_MCC9.10_Run4b4c4d5_v10_04_07_15_BNB_CCpi0_overlay_surprise_reco2_hist_4b.root",
    "checkout_MCC9.10_Run4b4c4d5_v10_04_07_15_BNB_CCpi0_overlay_surprise_reco2_hist_4c.root",
    "checkout_MCC9.10_Run4b4c4d5_v10_04_07_15_BNB_CCpi0_overlay_surprise_reco2_hist_5.root",

    # dirt overlay files
    "checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_dirt_overlay_surprise_reco2_hist_4a.root",
    "checkout_MCC9.10_Run4b_v10_04_07_09_BNB_dirt_surpise_reco2_hist.root",
    "checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_dirt_overlay_surprise_reco2_hist_4c.root",
    "checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_dirt_overlay_surprise_reco2_hist_4d.root",
    "checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_dirt_overlay_surprise_reco2_hist_5.root",
]

branches = [
    "kminus_PrimaryHadronNormalization",
    "kplus_PrimaryHadronFeynmanScaling",
    "kzero_PrimaryHadronSanfordWang",
]

for filename in files:
    path = f"{data_files_location}/{filename}"
    print(path)
    try:
        with uproot.open(path) as f:
            arrays = f["spline_weights"].arrays(branches, entry_stop=1, library="np")
            for branch in branches:
                print(f"    {branch}: {len(arrays[branch][0])} universes")
    except Exception as e:
        print(f"    ERROR: {e}")
