"""Compares several BDT trainings: a summary table (learning rate, best boosting round, fit time, validation
physics-weighted log loss) and overlaid Iso1g and Del1g efficiency vs purity curves.

Same definitions as ipynb_notebooks/efficiencies.ipynb: the signals are the true in-FV events of the
injected iso1g / del1g overlays (scaled by the same injected normalization factors), the background is
the nominal open-data prediction, only BDT test and not_trained events are used, and the efficiency is
relative to all true signal events before any selection. Each curve scans the summed probabilities of
the signal's reco categories; the star is the nominal selection (src/reco_categories.py).

Usage:
    python src/compare_bdt_trainings.py --output_name lr_scan_2026_10_26 --trainings lr_scan_0.3_2026_10_26 lr_scan_0.1_2026_10_26
        (run_learning_rate_scan.sh prints the command for its trainings)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from file_locations import intermediate_files_location
from df_helpers import select_bdt_test_events
from signal_categories import train_category_labels, del1g_detailed_category_labels
from reco_categories import reco_category_expr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHT = "wc_net_weight_open_data"

# the nominal open-data prediction: the background for every purity
PREDICTION_FILETYPES = [
    "nu_overlay", "nue_overlay", "nc_pi0_overlay", "ncdeltarad_overlay", "numucc_pi0_overlay",
    "dirt_overlay", "ext", "numucc_rad_corr_sim", "NC_coherent_1g_reweighted",
]

SIGNALS = {
    "iso1g": dict(label=r"Iso1g $1\gamma$", sample="isotropic_one_gamma_overlay", norm_factor=0.05,
                  truth=["iso1g"], reco=["1g0p"]),
    "del1g": dict(label=r"Del1g $1\gamma$", sample="delete_one_gamma_overlay", norm_factor=0.5,
                  truth=["del1g_Np", "del1g_0p", "del1g_Np1mu", "del1g_0p1mu"],
                  reco=["1gNp", "1g0p", "1gNp1mu", "1g0p1mu"]),
}

def load_events():
    """All events needed by every training: generic-selected prediction and signal-sample events, plus the
    signal samples' events failing the generic selection (for the efficiency denominators)."""
    samples = [sig["sample"] for sig in SIGNALS.values()]
    return (
        pl.scan_parquet(f"{intermediate_files_location}/all_df.parquet")
        .filter(pl.col("filetype").is_in(PREDICTION_FILETYPES + samples))
        .filter((pl.col("wc_kine_reco_Enu") > 0) | pl.col("filetype").is_in(samples))
        .select("filetype", "run", "subrun", "event", WEIGHT, "wc_kine_reco_Enu", "del1g_detailed_signal_category")
        .with_columns(
            (pl.col(WEIGHT) * pl.col("filetype").replace_strict(
                {sig["sample"]: sig["norm_factor"] for sig in SIGNALS.values()}, default=1.0, return_dtype=pl.Float64))
            .alias(WEIGHT),
            pl.col("del1g_detailed_signal_category").replace_strict(
                dict(enumerate(del1g_detailed_category_labels)), default="uncategorized", return_dtype=pl.String
            ).alias("truth_category"),
        )
        .collect()
    )


def eff_pur_curve(relevant, total_sig, score, n_points=400):
    """Efficiency and purity of score > cut for every cut (one cumulative sum over the events sorted by
    descending score), thinned to points evenly spaced in both event index and efficiency."""
    arrs = relevant.select(score.alias("score"), "is_signal", WEIGHT).sort("score", descending=True)
    s, is_sig, w = arrs["score"].to_numpy(), arrs["is_signal"].to_numpy(), arrs[WEIGHT].to_numpy()
    sig_cum, all_cum = np.cumsum(np.where(is_sig, w, 0.0)), np.cumsum(w)
    last_of_tie = np.r_[s[1:] != s[:-1], True]
    eff, pur = sig_cum[last_of_tie] / total_sig, sig_cum[last_of_tie] / all_cum[last_of_tie]
    keep = np.unique(np.r_[np.linspace(0, len(eff) - 1, n_points).astype(int),
                           np.minimum(np.searchsorted(eff, np.linspace(0, eff[-1], n_points)), len(eff) - 1)])
    return eff[keep], pur[keep]


def training_curves(events, training):
    """For one training: {signal: (efficiencies, purities, nominal efficiency, nominal purity)}."""
    prob_cols = [f"prob_{cat}" for cat in train_category_labels]
    preds = pl.read_parquet(PROJECT_ROOT / "training_outputs" / training / "predictions.parquet",
                            columns=["filetype", "run", "subrun", "event", "bdt_role", "bdt_prediction_weight_factor"] + prob_cols)
    df = events.join(preds, on=["filetype", "run", "subrun", "event"], how="left")
    has_prediction = pl.col("bdt_prediction_weight_factor").is_not_null()
    df = pl.concat([select_bdt_test_events(df.filter(has_prediction), [WEIGHT]), df.filter(~has_prediction)])
    generic = df.filter(pl.col("wc_kine_reco_Enu") > 0).with_columns(reco_category_expr(train_category_labels).alias("reco_category"))

    curves = {}
    for name, sig in SIGNALS.items():
        is_signal = (pl.col("filetype") == sig["sample"]) & pl.col("truth_category").is_in(sig["truth"])
        total_sig = df.filter(is_signal)[WEIGHT].sum()
        relevant = generic.filter(pl.col("filetype").is_in(PREDICTION_FILETYPES) | is_signal).with_columns(is_signal.alias("is_signal"))
        eff, pur = eff_pur_curve(relevant, total_sig, pl.sum_horizontal([f"prob_{cat}" for cat in sig["reco"]]))
        nominal = relevant.filter(pl.col("reco_category").is_in(sig["reco"]))
        nominal_sig = nominal.filter("is_signal")[WEIGHT].sum()
        curves[name] = (eff, pur, nominal_sig / total_sig, nominal_sig / nominal[WEIGHT].sum())
    return curves


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trainings", nargs="+", required=True, help="training_outputs directory names")
    parser.add_argument("--output_name", required=True, help="name of this comparison, used in the plot file names")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "plots" / "bdt_comparison"))
    args = parser.parse_args()
    trainings = {name: name for name in args.trainings}

    print(f"{'training':<40s} {'learning rate':>13s} {'best round':>10s} {'fit min':>8s} {'val physics-weighted loss':>26s}")
    for training in trainings:
        with open(PROJECT_ROOT / "training_outputs" / training / "training_config.json") as f:
            config = json.load(f)
        fit_minutes = config.get("fit_minutes")
        print(f"{training:<40s} {config['xgb_params'].get('learning_rate', 0.3):>13g} {config['best_iteration']:>10d} "
              f"{'' if fit_minutes is None else f'{fit_minutes:.1f}':>8s} "
              f"{config['physics_weighted_logloss_at_best_iteration']['validation']['prior_corrected']:>26.5f}")

    print("loading events...")
    events = load_events()
    all_curves = {}
    for training in trainings:
        print(f"computing curves for {training}...")
        all_curves[training] = training_curves(events, training)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for name, sig in SIGNALS.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        for color, (training, label) in zip(colors, trainings.items()):
            eff, pur, nominal_eff, nominal_pur = all_curves[training][name]
            ax.plot(eff, pur, color=color, label=label)
            ax.scatter(nominal_eff, nominal_pur, color=color, marker="*", s=250, edgecolors="black", zorder=3)
            print(f"  {name} {label}: nominal efficiency {nominal_eff:.3f}, purity {nominal_pur:.3f}")
        ax.scatter([], [], color="white", marker="*", s=250, edgecolors="black", label=f"Nominal ({' | '.join(sig['reco'])})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.set_xlabel("Efficiency")
        ax.set_ylabel("Purity")
        ax.set_title(f"True {sig['label']} (x{sig['norm_factor']:g} on top of the open-data prediction)")
        ax.legend(loc="lower left")
        fig.tight_layout()
        savename = output_dir / f"eff_vs_pur_{name}_{args.output_name}.png"
        fig.savefig(savename, dpi=150)
        print(f"saved {savename}")


if __name__ == "__main__":
    main()
