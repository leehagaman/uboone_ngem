#!/bin/bash
# Runs the standard PROfit chain on one of the XMLs in profit_xmls/:
#   1. PROfit process   (reads the minimal_*_df.root files, builds the binary caches)
#   2. PROfit plot      (pre-fit error-band plots)
#   3. PROfit profile   (systematics-only profile)
#   4. PROfit profile --legacy-postfit-error   (same, output quantifier "legacypostfiterror")
# It uses the already-compiled binary <profit_location>/build/bin/PROfit (recompile it yourself after
# updating PROfit).
# After each PROfit step, src/pdf_grid.py merges the 20 per-channel pages of the error-band / profile
# histogram PDFs into single-page grids (+ jpgs). The chain stops at the first failed step.
#
# PROfit writes into its working directory, <profit_output_location>/<tag>/, which keeps the large binary
# caches, root files, text outputs, a copy of the XML used, and one log per step (PROfit --log files
# process.log, plot.log, ... and the matching stdout/stderr in process.out, plot.out, ...).
# All plots (PDFs and the merged grids) are moved to plots/profit/<tag>/ in this repo.
#
# Usage (from anywhere; profile takes a while, so usually in the background):
#   nohup bash run_profit.sh <xml_name> <tag> > profit_nohup.out 2>&1 &
# xml_name is a file in profit_xmls/ with or without .xml, e.g. bagels_full_open_data.
# profit_location and profit_output_location are set in src/file_locations.py.

NTHREAD="${NTHREAD:-40}"
SEED=404

[ $# -eq 2 ] || { echo "usage: bash run_profit.sh <xml_name> <tag>"; exit 1; }
XML_NAME="${1%.xml}"
TAG="$2"

cd "$(dirname "$0")" || exit 1
REPO_DIR="$PWD"
XML="$REPO_DIR/profit_xmls/$XML_NAME.xml"
[ -f "$XML" ] || { echo "no such XML: $XML"; exit 1; }
source ../uv_base/bin/activate || exit 1

# PROfit locations, from src/file_locations.py
PROFIT_DIR=$(cd src && python -c "from file_locations import profit_location; print(profit_location)") || exit 1
PROFIT_OUTPUT_DIR=$(cd src && python -c "from file_locations import profit_output_location; print(profit_output_location)") || exit 1

PROFIT="$PROFIT_DIR/build/bin/PROfit"
OUT_DIR="$PROFIT_OUTPUT_DIR/$TAG"
PLOT_DIR="$REPO_DIR/plots/profit/$TAG"

start_all=$SECONDS
step_start() { echo "[$(date '+%F %T')] starting: $1"; step_t0=$SECONDS; }
step_done()  { echo "[$(date '+%F %T')] finished: $1 ($(( (SECONDS - step_t0) / 60 )) min)"; }
fail()       { echo "[$(date '+%F %T')] FAILED: $1 (see $2)"; exit 1; }
# move the PDFs PROfit just wrote into the plot directory (also after a failed step, for debugging)
collect_plots() { mv -f "$OUT_DIR"/*.pdf "$PLOT_DIR"/ 2>/dev/null; }
# merge_grid <input pdf> <output pdf>, both in the plot directory
merge_grid() { python "$REPO_DIR/src/pdf_grid.py" "$PLOT_DIR/$1" "$PLOT_DIR/$2" --trim 20,0,20,0; }

echo "xml: $XML"
echo "tag: $TAG"
echo "PROfit: $PROFIT ($(git -C "$PROFIT_DIR" branch --show-current) @ $(git -C "$PROFIT_DIR" rev-parse --short HEAD))"
echo "outputs: $OUT_DIR"
echo "plots: $PLOT_DIR"

[ -x "$PROFIT" ] || { echo "no PROfit binary at $PROFIT (compile PROfit first)"; exit 1; }

mkdir -p "$OUT_DIR" "$PLOT_DIR" || exit 1
cd "$OUT_DIR" || exit 1
cp "$XML" "$OUT_DIR/" # keep the exact XML used next to its outputs

PROFILE_ARGS=(-v2 -n"$NTHREAD" --seed $SEED -p grad-deep --syst-only --plot-bounds ratmin 0 ratmax 2 --scale-by-width --progress)

step_start "process"
"$PROFIT" -x "$XML" -t "$TAG" -v3 --log process.log process > process.out 2>&1 || fail "process" "$OUT_DIR/process.log, process.out"
step_done "process"

step_start "plot"
"$PROFIT" -x "$XML" -t "$TAG" -v3 --seed $SEED --log plot.log --plot-bounds ratmin 0 ratmax 2 --scale-by-width plot > plot.out 2>&1 \
    || { collect_plots; fail "plot" "$OUT_DIR/plot.log, plot.out"; }
collect_plots
merge_grid "${TAG}_v1_PROplot_Variable_0_ErrorBand.pdf" "merged_plots_${TAG}.pdf" || fail "plot grid" "this log"
step_done "plot"

step_start "profile"
"$PROFIT" -x "$XML" -t "$TAG" --log profile.log "${PROFILE_ARGS[@]}" profile > profile.out 2>&1 \
    || { collect_plots; fail "profile" "$OUT_DIR/profile.log, profile.out"; }
collect_plots
merge_grid "${TAG}_v1_PROfile_hists.pdf" "merged_hist_plots_${TAG}.pdf" || fail "profile grid" "this log"
step_done "profile"

step_start "profile (legacy post-fit error)"
"$PROFIT" -x "$XML" -t "$TAG" -o legacypostfiterror --log profile_legacypostfiterror.log "${PROFILE_ARGS[@]}" profile --legacy-postfit-error \
    > profile_legacypostfiterror.out 2>&1 \
    || { collect_plots; fail "profile (legacy post-fit error)" "$OUT_DIR/profile_legacypostfiterror.log, profile_legacypostfiterror.out"; }
collect_plots
merge_grid "${TAG}_legacypostfiterror_PROfile_hists.pdf" "merged_hist_plots_${TAG}_legacypostfiterror.pdf" \
    || fail "legacy profile grid" "this log"
step_done "profile (legacy post-fit error)"

echo "[$(date '+%F %T')] all done ($(( (SECONDS - start_all) / 60 )) min); plots in $PLOT_DIR, other outputs in $OUT_DIR"
