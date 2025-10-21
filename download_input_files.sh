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

REMOTE_HOST="uboonegpvm01.fnal.gov"

# From https://cdcvs.fnal.gov/redmine/projects/uboone-physics-analysis/wiki/MCC910_Samples
# or from https://docs.google.com/spreadsheets/d/1RUiX2M6zoob9R0YWPLummHzmX5UeLLEtS-7ZU-x2gA4/edit?gid=450838812#gid=450838812
# Using BNB WC processed files with all trees, not including beam-on data yet
# This is the list of files currently on the spreadsheet as of 2025_09_08
FILES_TO_COPY=(

    # nu overlay files
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4a_full_samples/wc_processed/wc_only/BNB/checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_nu_overlay_surprise_reco2_hist_4a.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4b_full_samples/wc_processed/BNB/MCC9.10_Run4b_v10_04_07_09_BNB_nu_overlay_surprise_reco2_hist.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4c_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_nu_overlay_surprise_reco2_hist_4c.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4d_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_nu_overlay_surprise_reco2_hist_4d.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run5_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_nu_overlay_surprise_reco2_hist_5.root"


    # nue overlay files
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4a_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_intrinsic_nue_overlay_surprise_reco2_hist_4a.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4b_full_samples/wc_processed/BNB/MCC9.10_Run4b_v10_04_07_09_BNB_nue_overlay_surprise_reco2_hist.root"
    #"/pnfs/uboone/persistent/users/uboonepro/surprise/run4c_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_intrinsic_nue_overlay_surprise_redo_reco2_hist_4c.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4c_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4c_v10_04_07_13_BNB_intrinsic_nue_overlay_surprise_redo_reco2_hist.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4d_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_intrinsic_nue_overlay_surprise_reco2_hist_4d.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run5_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_intrinsic_nue_overlay_surprise_reco2_hist_5.root"

    # NC pi0 overlay files
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4b_full_samples/wc_processed/BNB/MCC9.10_Run4b_v10_04_07_09_BNB_NC_pi0_overlay_surprise_reco2_hist.root"
    #"/pnfs/uboone/persistent/users/uboonepro/surprise/run4c_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_redo_reco2_hist_4c.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4c_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_reco2_hist_4c.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4d_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_reco2_hist_4d.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run5_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_reco2_hist_5.root"

    # Dirt files
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4a_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_dirt_overlay_surprise_reco2_hist_4a.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4b_full_samples/wc_processed/BNB/MCC9.10_Run4b_v10_04_07_09_BNB_dirt_surpise_reco2_hist.root"
    #"/pnfs/uboone/persistent/users/uboonepro/surprise/run4c_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_dirt_overlay_surprise_redo_reco2_hist_4c.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4c_full_samples/wc_processed/BNB/MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_dirt_overlay_surprise_reco2_hist_4c.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4d_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_dirt_overlay_surprise_reco2_hist_4d.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run5_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_dirt_overlay_surprise_reco2_hist_5.root"

    # Del1g files
    "/pnfs/uboone/persistent/users/uboonepro/surprise/delete_one_gamma/4a/checkout_delete_one_gamma_run45_reco2_prod_reco2_hist_4a.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/delete_one_gamma/4bcd/checkout_delete_one_gamma_run45_reco2_prod_reco2_hist_4bcd.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/delete_one_gamma/5/checkout_delete_one_gamma_run45_reco2_prod_reco2_hist_5.root"

    # Iso1g files
    "/pnfs/uboone/persistent/users/uboonepro/surprise/isotropic_one_gamma/4a/checkout_isotropic_one_gamma_run45_reco2_prod_reco2_hist_4a.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/isotropic_one_gamma/4bcd/checkout_isotropic_one_gamma_run45_reco2_prod_reco2_hist_4bcd.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/isotropic_one_gamma/5/checkout_isotropic_one_gamma_run45_reco2_prod_reco2_hist_5.root"

    # EXT files
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4b_full_samples/wc_processed/BNB/MCC9.10_Run4b_v10_04_07_09_Run4b_BNB_beam_off_surprise_reco2_hist.root"

    # Data files
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4b_full_samples/wc_processed/BNB/MCC9.10_Run4b_v10_04_07_11_BNB_beam_on_surprise_reco2_hist.root"

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
