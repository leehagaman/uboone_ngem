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

# From https://docs.google.com/spreadsheets/d/1RUiX2M6zoob9R0YWPLummHzmX5UeLLEtS-7ZU-x2gA4/edit?gid=450838812#gid=450838812
# Using BNB WC processed files with all trees, not including beam-on data yet
# This is the list of files currently on the spreadsheet as of 2025_09_08
FILES_TO_COPY=(
    # run 4a files
    "/exp/uboone/data/uboonepro/MCC9.10/run4a_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_intrinsic_nue_overlay_surprise_reco2_hist_4a.root"

    # run 4b files
    "/exp/uboone/data/uboonepro/MCC9.10/wc_processed/BNB/MCC9.10_Run4b_v10_04_07_09_BNB_nu_overlay_surprise_reco2_hist.root"
    "/exp/uboone/data/uboonepro/MCC9.10/wc_processed/BNB/MCC9.10_Run4b_v10_04_07_09_BNB_nue_overlay_surprise_reco2_hist.root"
    "/exp/uboone/data/uboonepro/MCC9.10/wc_processed/BNB/MCC9.10_Run4b_v10_04_07_09_BNB_dirt_surpise_reco2_hist.root"
    "/exp/uboone/data/uboonepro/MCC9.10/wc_processed/BNB/MCC9.10_Run4b_v10_04_07_09_BNB_NC_pi0_overlay_surprise_reco2_hist.root"
    "/exp/uboone/data/uboonepro/MCC9.10/wc_processed/BNB/MCC9.10_Run4b_v10_04_07_09_Run4b_BNB_beam_off_surprise_reco2_hist.root"

    # run 4c files
    "/exp/uboone/data/uboonepro/MCC9.10/run4c_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_reco2_hist_4c.root"
    "/pnfs/uboone/persistent/users/uboonepro/surprise/run4c_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4c_v10_04_07_13_BNB_intrinsic_nue_overlay_surprise_redo_reco2_hist.root"

    # run 4d files
    "/exp/uboone/data/uboonepro/MCC9.10/run4d_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_intrinsic_nue_overlay_surprise_reco2_hist_4d.root"
    "/exp/uboone/data/uboonepro/MCC9.10/run4d_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_reco2_hist_4d.root"

    # run 5 files
    "/exp/uboone/data/uboonepro/MCC9.10/run5_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4a4c4d5_v10_04_07_13_BNB_intrinsic_nue_overlay_surprise_reco2_hist_5.root"
    "/exp/uboone/data/uboonepro/MCC9.10/run5_full_samples/wc_processed/BNB/checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_reco2_hist_5.root"

    # test runs 4-5 signal sample files
    "/exp/uboone/data/users/lhagaman/delete_one_gamma_run45_1k.root"
    "/exp/uboone/data/users/lhagaman/isotropic_one_gamma_run45_1k.root"
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
