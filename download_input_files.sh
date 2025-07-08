#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <username>"
    exit 1
fi


USERNAME="$1"

REMOTE_HOST="uboonegpvm01.fnal.gov"

# these file paths are documented at https://docs.google.com/spreadsheets/d/1AVrUfAffE6mQw5t-gQnlXxxcyhwVxtPGGUp_7c1MT2w/edit?gid=1068115471#gid=1068115471
REMOTE_PATH="/exp/uboone/data/users/eyandel/combined_reco/mcc910_test/processed_checkout_root_files"
FILES_TO_COPY=(
    "SURPRISE_Test_Samples_v10_04_07_05_Run4b_hyper_unified_reco2_BNB_dirt_may8_reco2_hist_62280564_snapshot.root"
    "SURPRISE_Test_Samples_v10_04_07_05_Run4b_hyper_unified_reco2_BNB_nu_NC_pi0_overlay_may8_reco2_hist_62280465_snapshot.root"
    "SURPRISE_Test_Samples_v10_04_07_05_Run4b_hyper_unified_reco2_BNB_nu_overlay_may8_reco2_hist_62280499_snapshot.root"
)
LOCAL_DEST="./data_files"

for FILE in "${FILES_TO_COPY[@]}"; do
    DEST="$LOCAL_DEST/$(basename "$FILE")"
    if [ -f "$DEST" ]; then
        echo "Skipping $DEST (already exists)"
        continue
    fi
    echo "Downloading $FILE..."
    scp "${USERNAME}@${REMOTE_HOST}:${REMOTE_PATH}/${FILE}" "${LOCAL_DEST}/"
done

echo "All files downloaded"
