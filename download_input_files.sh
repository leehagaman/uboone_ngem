#!/bin/bash

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <username> [local_dest]"
    exit 1
fi


USERNAME="$1"

# Destination directory (optional positional arg, defaults to current directory)
if [ -n "$2" ]; then
    LOCAL_DEST="$2"
else
    LOCAL_DEST="."
fi

# Ensure destination directory exists
mkdir -p "$LOCAL_DEST"

REMOTE_HOST="uboonegpvm02.fnal.gov"

# From https://cdcvs.fnal.gov/redmine/projects/uboone-physics-analysis/wiki/MCC910_Samples
# or from https://docs.google.com/spreadsheets/d/1RUiX2M6zoob9R0YWPLummHzmX5UeLLEtS-7ZU-x2gA4/edit?gid=450838812#gid=450838812
# Using BNB WC processed files with all trees
# Updated 2026_09_02
FILES_TO_COPY=(

    # nu overlay files
    "/exp/uboone/data/users/uboonepro/SURPRISE/run1_full_samples/BNB/checkout_MCC9.10_Run123_v10_04_07_20_BNB_nu_overlay_surprise_reco2_hist_1.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run2_full_samples/BNB/checkout_MCC9.10_Run123_v10_04_07_20_BNB_nu_overlay_surprise_reco2_hist_2_v3.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run3_full_samples/BNB/checkout_MCC9.10_Run123_v10_04_07_20_BNB_nu_overlay_surprise_reco2_hist_3.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4a_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_nu_overlay_retuple_retuple_hist_4a.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4b_full_samples/BNB/checkout_MCC9.10_Run4b_v10_04_07_20_BNB_nu_overlay_retuple_retuple_hist.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4c_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_nu_overlay_retuple_retuple_hist_4c.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4d_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_nu_overlay_retuple_retuple_hist_4d.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run5_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_nu_overlay_retuple_retuple_hist_5.root"

    # nue overlay files
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4a_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_intrinsic_nue_overlay_retuple_retuple_hist_4a.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4b_full_samples/BNB/checkout_MCC9.10_Run4b_v10_04_07_20_BNB_intrinsic_nue_overlay_retuple_retuple_hist.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4c_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_intrinsic_nue_overlay_retuple_retuple_hist_4c.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4d_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_intrinsic_nue_overlay_retuple_retuple_hist_4d.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run5_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_intrinsic_nue_overlay_retuple_retuple_hist_5.root"

    # NCpi0 overlay files
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4a_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_24_BNB_NCpi0_overlay_retuple_retuple_hist_4a.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4b_full_samples/BNB/checkout_MCC9.10_Run4b_v10_04_07_24_BNB_NCpi0_overlay_retuple_retuple_hist.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4c_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_24_BNB_NCpi0_overlay_retuple_retuple_hist_4c.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4d_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_24_BNB_NCpi0_overlay_retuple_retuple_hist_4d.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run5_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_24_BNB_NCpi0_overlay_retuple_retuple_hist_5.root"

    # numuCCpi0 overlay files
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4a_full_samples/BNB/checkout_MCC9.10_Run4abcd5_v10_04_07_24_BNB_CCpi0_overlay_retuple_retuple_hist_4ab.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4b_full_samples/BNB/checkout_MCC9.10_Run4abcd5_v10_04_07_24_BNB_CCpi0_overlay_retuple_retuple_hist_4b.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4c_full_samples/BNB/checkout_MCC9.10_Run4abcd5_v10_04_07_24_BNB_CCpi0_overlay_retuple_retuple_hist_4c.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run5_full_samples/BNB/checkout_MCC9.10_Run4abcd5_v10_04_07_24_BNB_CCpi0_overlay_retuple_retuple_hist_5.root"

    # dirt overlay files
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4a_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_dirt_overlay_retuple_retuple_hist_4a.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4b_full_samples/BNB/checkout_MCC9.10_Run4b_v10_04_07_20_BNB_dirt_overlay_retuple_retuple_hist.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4c_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_dirt_overlay_retuple_retuple_hist_4c.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4d_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_dirt_overlay_retuple_retuple_hist_4d.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run5_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_20_BNB_dirt_overlay_retuple_retuple_hist_5.root"

    # Del1g overlay files
    "/pnfs/uboone/persistent/users/uboonepro/surprise/delete_one_gamma/4a/checkout_delete_one_gamma_run45_reco2_prod_reco2_hist_4a.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/delete_one_gamma/4bcd/checkout_delete_one_gamma_run45_reco2_prod_reco2_hist_4bcd.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/delete_one_gamma/5/checkout_delete_one_gamma_run45_reco2_prod_reco2_hist_5.root"

    # Iso1g overlay files
    "/pnfs/uboone/persistent/users/uboonepro/surprise/isotropic_one_gamma/4a/checkout_isotropic_one_gamma_run45_reco2_prod_reco2_hist_4a.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/isotropic_one_gamma/4bcd/checkout_isotropic_one_gamma_run45_reco2_prod_reco2_hist_4bcd.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/isotropic_one_gamma/5/checkout_isotropic_one_gamma_run45_reco2_prod_reco2_hist_5.root"

    # EXT files
    "/exp/uboone/data/users/uboonepro/SURPRISE/run1_full_samples/BNB/checkout_MCC9.10_Run123_v10_04_07_20_BNB_beam_off_data_surprise_reco2_hist_1.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run2_full_samples/BNB/checkout_MCC9.10_Run123_v10_04_07_20_BNB_beam_off_data_surprise_reco2_hist_2.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run3_full_samples/BNB/checkout_MCC9.10_Run123_v10_04_07_20_BNB_beam_off_data_surprise_reco2_hist_3.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4a_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_23_BNB_beam_off_retuple_retuple_hist_4a.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4b_full_samples/BNB/checkout_MCC9.10_Run4b_v10_04_07_20_BNB_beam_off_metapatch_retuple_retuple_hist.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4c_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_23_BNB_beam_off_retuple_retuple_hist_4c.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4d_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_23_BNB_beam_off_retuple_retuple_hist_4d.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run5_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_23_BNB_beam_off_retuple_retuple_hist_5.root"

    # Data files
    "/exp/uboone/data/users/uboonepro/SURPRISE/run1_full_samples/BNB/checkout_MCC9.10_Run123_v10_04_07_20_BNB_beam_on_data_surprise_reco2_hist_1_5e19opendata.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run3_full_samples/BNB/checkout_MCC9.10_Run123_v10_04_07_20_BNB_beam_on_data_surprise_reco2_hist_3_1e19opendata.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4a_full_samples/BNB/checkout_MCC9.10_Run4acd5_v10_04_07_23_BNB_beam_on_retuple_retuple_hist_4a.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4b_full_samples/BNB/checkout_MCC9.10_Run4b_v10_04_07_20_BNB_beam_on_metapatch_retuple_retuple_hist_opendata_20700.root"

    # NuWro fake data files
    "/exp/uboone/data/users/uboonepro/SURPRISE/run1_full_samples/BNB/checkout_MCC9.10_Run123_v10_04_07_23_BNB_nuwro_overlay_surprise_reco2_hist_1.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run2_full_samples/BNB/checkout_MCC9.10_Run123_v10_04_07_23_BNB_nuwro_overlay_surprise_reco2_hist_2.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run3_full_samples/BNB/checkout_MCC9.10_Run123_v10_04_07_23_BNB_nuwro_overlay_surprise_reco2_hist_3.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4a_full_samples/BNB/checkout_MCC9.10_Run45_v10_04_07_23_BNB_nuwro_overlay_surprise_reco2_hist_4a.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run4c_full_samples/BNB/checkout_MCC9.10_Run45_v10_04_07_23_BNB_nuwro_overlay_surprise_reco2_hist_4c.root"
    "/exp/uboone/data/users/uboonepro/SURPRISE/run5_full_samples/BNB/checkout_MCC9.10_Run45_v10_04_07_23_BNB_nuwro_overlay_surprise_reco2_hist_5.root"

    # DetVar files

    # nu_overlay, partially runs 1-5, partially runs 4-5
    # CV
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run1/checkout_DetVar_Run123_v10_04_07_23_BNB_nu_overlay_cv_13a_surprise_reco2_hist_1.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run3/checkout_DetVar_Run123_v10_04_07_23_BNB_nu_overlay_cv_13a_surprise_reco2_hist_3.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run3/checkout_DetVar_Run123_v10_04_07_23_BNB_nu_overlay_cv_3b_1mil_surprise_reco2_hist.root" # used for all variations except SCE and Recomb2
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run3/checkout_DetVar_Run123_v10_04_07_23_BNB_nu_overlay_cv_3b_500k_surprise_reco2_hist.root" # used for SCE and Recomb2
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run4d/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_cv_surprise_reco2_hist_4d.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run5/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_cv_surprise_reco2_hist_5.root"
    # LYA
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run1/checkout_DetVar_Run123_v10_04_07_23_BNB_nu_overlay_lya_surprise_reco2_hist_1.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run3/checkout_DetVar_Run123_v10_04_07_23_BNB_nu_overlay_lya_surprise_reco2_hist_3.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run4d/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_lya_surprise_reco2_hist_4d.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run5/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_lya_surprise_reco2_hist_5.root"
    # LYD
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run1/checkout_DetVar_Run123_v10_04_07_23_BNB_nu_overlay_lyd_surprise_reco2_hist_1.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run3/checkout_DetVar_Run123_v10_04_07_23_BNB_nu_overlay_lyd_surprise_reco2_hist_3.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run4d/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_lyd_surprise_reco2_hist_4d.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run5/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_lyd_surprise_reco2_hist_5.root"
    # LYR
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run1/checkout_DetVar_Run123_v10_04_07_23_BNB_nu_overlay_lyr_surprise_reco2_hist_1.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run3/checkout_DetVar_Run123_v10_04_07_23_BNB_nu_overlay_lyr_surprise_reco2_hist_3.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run4d/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_lyr_surprise_reco2_hist_4d.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run5/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_lyr_surprise_reco2_hist_5.root"
    # Recomb2
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run4d/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_recomb2_surprise_reco2_hist_4d.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run5/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_recomb2_surprise_reco2_hist_5.root"
    # SCE
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run4d/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_sce_surprise_reco2_hist_4d.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run5/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_sce_surprise_reco2_hist_5.root"
    # WMX
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run4d/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_WMX_surprise_reco2_hist_4d.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run5/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_WMX_surprise_reco2_hist_5.root"
    # WMYZ
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run4d/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_WMYZ_surprise_reco2_hist_4d.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run5/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_WMYZ_surprise_reco2_hist_5.root"
    # WMthetaXZ
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run4d/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_WMthetaXZ_surprise_reco2_hist_4d.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run5/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_WMthetaXZ_surprise_reco2_hist_5.root"
    # WMthetaYZ
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run4d/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_WMthetaYZ_surprise_reco2_hist_4d.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/detvar/BNB/run5/checkout_DetVar_Run45_v10_04_07_19_BNB_nu_overlay_WMthetaYZ_surprise_reco2_hist_5.root"

)

for FILE in "${FILES_TO_COPY[@]}"; do
    DEST="$LOCAL_DEST/$(basename "$FILE")"
    if [ -f "$DEST" ]; then
        echo "Skipping $DEST (already exists)"
        continue
    fi
    echo "Downloading $FILE..."
    scp "${USERNAME}@${REMOTE_HOST}:${FILE}" "${LOCAL_DEST}/"

done

echo "All files downloaded"
