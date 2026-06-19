"""Compute per-event NC-coherent-1g reweighting weights for isotropic-1g events.

This module replicates the weight-producing logic of
``ipynb_notebooks/coherent_1g_reweighting.ipynb`` so that
``coh_1g_reweighting.parquet`` can be regenerated as part of the
main dataframe-creation pipeline instead of by hand.

The produced parquet has columns:
    run, subrun, event, coherent_1g_weight_per_pot, coherent_1g_keep
The weight is POT-independent (events per POT); the target normalizing POT is
applied later by apply_nc_coh_1g_reweighting.  Only events with
coherent_1g_weight_per_pot > 0 are saved.
"""

import os
from datetime import datetime

import numpy as np
import polars as pl
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

from file_locations import intermediate_files_location, other_files_location

# plots/ folder lives at the repo root (parent of this src/ directory)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLOT_DIR = os.path.join(_REPO_ROOT, "plots", "coherent_1g_reweighting")

# NC coherent single-photon simulation file (see notebook cell 2)
_NC_COHERENT_ROOT = (
    "preTopo_vertex_NCSingleCoherent_Run123_TestSplit_v50.0_OrderedSample_NCDelta.uniq.root"
)

# ── POT-scaling constants (notebook cell 8) ──────────────────────────────────
# The target (normalizing) POT is intentionally NOT set here: it is applied later
# by apply_nc_coh_1g_reweighting, so the weights saved by this module are
# POT-independent (units of events per POT).  This lets a single computed
# reweighting be applied at multiple POT normalizations (e.g. different run-fractions).
# The constants below are intrinsic to the NC coherent reference simulation and do
# not depend on the analyzed run-fractions.
reference_nc_coherent_pot = 2.22242e+24        # discussion with Mark on slack 2026_02_24
reference_nc_coherent_num_events = 16139
train_test_fudge_factor = 0.5                  # to make the numbers match better

# Paper reference for the expected NC coherent 1g count in FV (arxiv 2502.06091),
# used as a sanity-check target for the number implied by our df.
coherent_eff_topological = 0.281
coherent_eff_rel_to_topological = 0.3909
coherent_num_sel = 1.1

# ── Reweighting binning (notebook cell 6) ────────────────────────────────────
bins_costheta = np.linspace(-1, 1, 101)
bins_energy   = np.concatenate([np.linspace(0, 2000, 101), [1e6]])


def _load_nc_coherent_simulation():
    """Load the NC coherent single-photon truth gamma energy/costheta (cell 2)."""
    f = uproot.open(
        os.path.join(other_files_location, _NC_COHERENT_ROOT)
    )["singlephotonana/vertex_tree"]
    daughters_pdg = f["mctruth_daughters_pdg"].array(library="np")
    daughters_E = f["mctruth_daughters_E"].array(library="np")
    daughters_pz = f["mctruth_daughters_pz"].array(library="np")

    nc_coherent_gamma_energy = []
    nc_coherent_gamma_costheta = []
    for i in range(len(daughters_pdg)):
        if daughters_pdg[i][0] != 22:
            raise Exception(
                "PROBLEM! First daughter particle isn't a photon in the nc coherent 1g simulation",
                daughters_pdg[i])
        nc_coherent_gamma_energy.append(daughters_E[i][0] * 1000)
        nc_coherent_gamma_costheta.append(daughters_pz[i][0] / daughters_E[i][0])

    return np.array(nc_coherent_gamma_costheta), np.array(nc_coherent_gamma_energy)


def compute_nc_coh_1g_reweighting(df, make_plots=True, net_weight_var="wc_net_weight_open_data"):
    """Compute and save the binned NC-coherent-1g reweighting from the dataframe.

    Replicates ``coherent_1g_reweighting.ipynb``. Accepts an eager polars
    DataFrame or a LazyFrame. Writes ``coh_1g_reweighting.parquet``
    to ``intermediate_files_location`` and (if ``make_plots``) all diagnostic
    plots to ``plots/coherent_1g_reweighting/``.

    ``net_weight_var`` is the per-config net-weight column used to weight the iso1g
    events when building the iso1g->coherent shape ratio (defaults to the open-data
    weighting).  The saved weight is per-POT (the absolute normalization is divided
    out and re-applied per config in apply_nc_coh_1g_reweighting), so this choice
    only sets the iso1g shape used to derive the bin-by-bin reweighting.
    """
    print(f"computing NC coherent 1g reweighting weights from the dataframe (net_weight_var={net_weight_var})")

    lf = df if isinstance(df, pl.LazyFrame) else df.lazy()

    os.makedirs(_PLOT_DIR, exist_ok=True)

    # ── NC coherent simulation target shape (cell 2) ──────────────────────────
    nc_coherent_gamma_costheta, nc_coherent_gamma_energy = _load_nc_coherent_simulation()
    num_coherent_simulated_events = len(nc_coherent_gamma_energy)
    print(f"  NC coherent simulated events: {num_coherent_simulated_events:,}")

    # ── Iso1g events from the dataframe (cell 5) ──────────────────────────────
    iso1g_df = (
        lf.filter(pl.col("filetype") == "isotropic_one_gamma_overlay")
        .select(
            "run", "subrun", "event",
            "wc_true_leading_shower_energy", "wc_true_leading_shower_costheta",
            net_weight_var, "wc_truth_inFV",
            "wc_truth_vtxX", "wc_truth_vtxY", "wc_truth_vtxZ",
        )
        .collect()
    )
    print(f"  isotropic 1g events: {iso1g_df.height:,}")

    iso1g_costheta = iso1g_df["wc_true_leading_shower_costheta"].to_numpy()
    iso1g_energy   = iso1g_df["wc_true_leading_shower_energy"].to_numpy()
    iso1g_w        = iso1g_df[net_weight_var].to_numpy()

    if make_plots:
        _plot_input_distributions(
            nc_coherent_gamma_costheta, nc_coherent_gamma_energy,
            iso1g_costheta, iso1g_energy, iso1g_w)

    # ── 2D reweighting iso1g -> coherent shape (cell 6) ───────────────────────
    H_coherent, _, _ = np.histogram2d(
        nc_coherent_gamma_costheta, nc_coherent_gamma_energy,
        bins=[bins_costheta, bins_energy])
    H_iso1g, _, _ = np.histogram2d(
        iso1g_costheta, iso1g_energy,
        bins=[bins_costheta, bins_energy], weights=iso1g_w)

    weight_ratios = np.where(H_iso1g > 0, H_coherent / H_iso1g, 0.0)
    i_cos = np.clip(np.digitize(iso1g_costheta, bins_costheta) - 1, 0, len(bins_costheta) - 2)
    i_E   = np.clip(np.digitize(iso1g_energy,   bins_energy)   - 1, 0, len(bins_energy)   - 2)
    event_weights = weight_ratios[i_cos, i_E]

    iso1g_df = iso1g_df.with_columns(
        pl.Series("coherent_1g_weight_nopot", event_weights)
    )

    H_iso1g_after, _, _ = np.histogram2d(
        iso1g_costheta, iso1g_energy,
        bins=[bins_costheta, bins_energy], weights=iso1g_w * event_weights)
    H_iso1g_after_unweighted, _, _ = np.histogram2d(
        iso1g_costheta, iso1g_energy,
        bins=[bins_costheta, bins_energy], weights=np.array(iso1g_w * event_weights > 0))

    print(f"  num Iso1g events with positive weight: {np.sum(event_weights > 0):,}")
    print(f"  num Iso1g events with zero weight: {np.sum(event_weights == 0):,}")

    if make_plots:
        _plot_reweighting(H_iso1g, H_coherent, H_iso1g_after,
                          H_iso1g_after_unweighted, event_weights)

    # ── Random sampling -> coherent_1g_keep (cell 7) ──────────────────────────
    target_fraction = 0.50
    total_after      = H_iso1g_after.sum()
    total_unweighted = H_iso1g_after_unweighted.sum()
    H_after_norm      = H_iso1g_after / total_after if total_after > 0 else H_iso1g_after
    H_unweighted_norm = (H_iso1g_after_unweighted / total_unweighted
                         if total_unweighted > 0 else H_iso1g_after_unweighted)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(H_unweighted_norm > 0, H_after_norm / H_unweighted_norm, 0.0)

    mask_positive = event_weights > 0
    i_cos_all = np.clip(np.digitize(iso1g_costheta, bins_costheta) - 1, 0, len(bins_costheta) - 2)
    i_E_all   = np.clip(np.digitize(iso1g_energy,   bins_energy)   - 1, 0, len(bins_energy)   - 2)
    event_prob_raw = np.where(mask_positive, ratio[i_cos_all, i_E_all], 0.0)

    n_positive  = int(mask_positive.sum())
    target_kept = target_fraction * n_positive
    min_raw     = event_prob_raw[event_prob_raw > 0].min()
    lo, hi = 0.0, 1.0 / min_raw
    for _ in range(60):
        mid = (lo + hi) / 2
        expected = np.sum(np.minimum(1.0, mid * event_prob_raw))
        if expected < target_kept:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2
    event_keep_prob = np.minimum(1.0, s * event_prob_raw)

    rng = np.random.default_rng(seed=42)
    keep_mask = rng.random(iso1g_df.height) < event_keep_prob

    iso1g_df = iso1g_df.with_columns(pl.Series("coherent_1g_keep", keep_mask))
    print(f"  Target fraction: {target_fraction:.2f}  ({target_kept:,.0f} events)")
    print(f"  Events kept:     {keep_mask.sum():,} / {n_positive:,}  ({keep_mask.sum() / n_positive:.3f})")

    if make_plots:
        _plot_sampling(iso1g_costheta, iso1g_energy, keep_mask,
                       H_iso1g_after_unweighted, H_iso1g_after, target_fraction)

    # ── Per-POT weight (cell 8) ───────────────────────────────────────────────
    # The target POT is factored out: the saved weight is per unit POT, and
    # apply_nc_coh_1g_reweighting multiplies by the desired normalizing POT
    # (coherent_1g_weight = coherent_1g_weight_per_pot * normalizing_POT).
    iso1g_df = iso1g_df.with_columns(
        pl.Series(
            "coherent_1g_weight_per_pot",
            iso1g_df["coherent_1g_weight_nopot"]
            * reference_nc_coherent_num_events / num_coherent_simulated_events
            / reference_nc_coherent_pot
            * train_test_fudge_factor)
    )

    iso1g_fv_df = iso1g_df.filter(
        (10.0 < pl.col("wc_truth_vtxX")) & (pl.col("wc_truth_vtxX") < 246.4)
        & (-101.5 < pl.col("wc_truth_vtxY")) & (pl.col("wc_truth_vtxY") < 101.5)
        & (10.0 < pl.col("wc_truth_vtxZ")) & (pl.col("wc_truth_vtxZ") < 986.8)
    )
    paper_coherent_num_fv_687e20 = coherent_num_sel / (coherent_eff_topological * coherent_eff_rel_to_topological)
    print(f"  Num NC coherent 1g events simulated in FV in 6.87e20 POT from paper: {paper_coherent_num_fv_687e20}")

    df_coherent_num_fv_687e20 = iso1g_fv_df["coherent_1g_weight_per_pot"].sum() * 6.87e20
    print(f"  Num NC coherent 1g events in FV in 6.87e20 POT from our df: {df_coherent_num_fv_687e20}")

    # ── Save weights (cell 9) ─────────────────────────────────────────────────
    weights_to_save = iso1g_df.filter(
        pl.col("coherent_1g_weight_per_pot") > 0
    ).select(["run", "subrun", "event", "coherent_1g_weight_per_pot", "coherent_1g_keep"])

    out_path = f"{intermediate_files_location}/coh_1g_reweighting.parquet"
    weights_to_save.write_parquet(out_path)
    print(f"  saved {out_path}")
    print(f"  {weights_to_save.height:,} / {iso1g_df.height:,} events have valid weights")

    # ── Write a log of this run into the plot directory ──────────────────────
    now = datetime.now()
    log_lines = [
        "NC coherent 1g reweighting log",
        "==============================",
        f"created: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.astimezone().tzname()})",
        "replicates: ipynb_notebooks/coherent_1g_reweighting.ipynb",
        "",
        "inputs:",
        f"  NC coherent simulation file: {_NC_COHERENT_ROOT}",
        f"  NC coherent simulated events: {num_coherent_simulated_events:,}",
        f"  isotropic 1g events (filetype=isotropic_one_gamma_overlay): {iso1g_df.height:,}",
        f"  iso1g events with positive coherent_1g_weight_nopot: {int((event_weights > 0).sum()):,}",
        f"  iso1g events with zero coherent_1g_weight_nopot: {int((event_weights == 0).sum()):,}",
        "",
        "binning:",
        f"  cos(theta) bins = linspace(-1,1,101)",
        f"  energy bins = linspace(0,2000,101) + overflow at 1e6",
        "",
        "random sampling (coherent_1g_keep):",
        f"  target_fraction = {target_fraction}",
        f"  rng seed = 42",
        f"  events kept = {int(keep_mask.sum()):,} / {n_positive:,} "
        f"({keep_mask.sum() / n_positive:.3f})",
        "",
        "POT scaling constants:",
        f"  reference_nc_coherent_pot = {reference_nc_coherent_pot:.6g}",
        f"  reference_nc_coherent_num_events = {reference_nc_coherent_num_events}",
        f"  train_test_fudge_factor = {train_test_fudge_factor}",
        f"  normalizing (target) POT: not applied here -- applied later by apply_nc_coh_1g_reweighting",
        f"  NC coherent 1g events in FV in 6.87e20 POT from paper (arxiv 2502.06091): {paper_coherent_num_fv_687e20:.4f}",
        f"  implied NC coherent 1g events in FV in 6.87e20 POT from our df: {df_coherent_num_fv_687e20:.4f}",
        "",
        "outputs:",
        f"  weights parquet: {out_path}",
        f"  valid-weight events (coherent_1g_weight_per_pot > 0): {weights_to_save.height:,} / {iso1g_df.height:,}",
        f"  saved columns: run, subrun, event, coherent_1g_weight_per_pot, coherent_1g_keep",
        f"  make_plots: {make_plots} (plots in this directory)",
    ]
    log_path = os.path.join(_PLOT_DIR, "reweighting_log.txt")
    with open(log_path, "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
    print(f"  wrote log {log_path}")

    return weights_to_save


def _plot_input_distributions(nc_coherent_gamma_costheta, nc_coherent_gamma_energy,
                              iso1g_costheta, iso1g_energy, iso1g_w):
    """NC coherent generation 2D (cell 2) and iso1g 2D (cell 5)."""
    bins = [np.linspace(-1, 1, 100), np.linspace(0, 1000, 100)]

    cmap = mpl.colormaps.get_cmap("viridis").copy()
    cmap.set_under("white")
    fig = plt.figure(figsize=(10, 5))
    plt.hist2d(nc_coherent_gamma_costheta, nc_coherent_gamma_energy, bins=bins, cmap=cmap, vmin=1e-6)
    plt.colorbar()
    plt.xlabel("cos(theta)")
    plt.ylabel("Energy (MeV)")
    plt.title("NC Coherent Single Photon Generation")
    plt.savefig(os.path.join(_PLOT_DIR, "nc_coherent_1g_energy_angle.pdf"))
    plt.savefig(os.path.join(_PLOT_DIR, "nc_coherent_1g_energy_angle.jpeg"), dpi=400)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    plt.hist2d(iso1g_costheta, iso1g_energy, bins=bins, cmap="viridis",
               norm=mpl.colors.LogNorm(), weights=iso1g_w)
    plt.colorbar()
    plt.xlabel("cos(theta)")
    plt.ylabel("Energy (MeV)")
    plt.title("Isotropic 1g Gamma Energy and Costheta")
    plt.savefig(os.path.join(_PLOT_DIR, "isotropic_1g_energy_angle.pdf"))
    plt.savefig(os.path.join(_PLOT_DIR, "isotropic_1g_energy_angle.jpeg"), dpi=400)
    plt.close(fig)


def _plot_reweighting(H_iso1g, H_coherent, H_iso1g_after,
                      H_iso1g_after_unweighted, event_weights):
    """3-panel reweighting comparison + unweighted + weight histogram (cell 6)."""
    vmax = max(H_coherent.max(), H_iso1g.max())
    if not (vmax > 0):  # empty/all-zero/NaN histogram (e.g. small --frac_events): avoid invalid LogNorm
        vmax = 1.0
    vmin = vmax * 1e-3

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, H, title in [
        (axes[0], H_iso1g,       "Iso 1g (wc_net_weight weighted)\n[before coherent_1g_weight]"),
        (axes[1], H_coherent,    "NC Coherent 1g simulation (target)"),
        (axes[2], H_iso1g_after, "Iso 1g after coherent_1g_weight"),
    ]:
        H_plot = np.where(H > 0, H, np.nan)
        im = ax.pcolormesh(bins_costheta, bins_energy, H_plot.T,
                           cmap='plasma', shading='flat',
                           norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
        plt.colorbar(im, ax=ax, label='Events / bin')
        ax.set_xlabel(r"cos $\theta$")
        ax.set_ylabel("Energy [MeV]")
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 2100)
        ax.set_title(title)
    fig.suptitle("2D (cos θ, energy) distribution before and after coherent reweighting")
    fig.tight_layout()
    fig.savefig(os.path.join(_PLOT_DIR, "three_panel_coh1g_reweighting.jpeg"), dpi=400)
    plt.close(fig)

    vmax = H_iso1g_after_unweighted.max()
    if not (vmax > 0):  # empty/all-zero/NaN histogram (e.g. small --frac_events): avoid invalid LogNorm
        vmax = 1.0
    vmin = vmax * 1e-3
    fig = plt.figure()
    plt.pcolormesh(bins_costheta, bins_energy, H_iso1g_after_unweighted.T, cmap='plasma',
                   shading='flat', norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar()
    plt.xlabel(r"cos $\theta$")
    plt.ylabel("Energy [MeV]")
    plt.title("Iso 1g after coherent_1g_weight (unweighted)\n(used for dedicated Coherent 1g BDT training)")
    plt.xlim(-1, 1)
    plt.ylim(0, 2100)
    plt.savefig(os.path.join(_PLOT_DIR, "coh1g_reweighting_unweighted.jpeg"), dpi=400)
    plt.close(fig)

    fig = plt.figure()
    plt.hist(event_weights, bins=100, range=(0, 10), density=True)
    plt.xlabel("Weight")
    plt.ylabel("Events / bin")
    plt.title("1D histogram of the weight of coherent_1g_weight_nopot")
    plt.yscale("log")
    plt.savefig(os.path.join(_PLOT_DIR, "coherent_1g_weight_nopot_hist.jpeg"), dpi=400)
    plt.close(fig)


def _plot_sampling(iso1g_costheta, iso1g_energy, keep_mask,
                   H_iso1g_after_unweighted, H_iso1g_after, target_fraction):
    """Random-sampling validation 3-panel (cell 7)."""
    H_sampled, _, _ = np.histogram2d(
        iso1g_costheta[keep_mask], iso1g_energy[keep_mask],
        bins=[bins_costheta, bins_energy])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, H, title in [
        (axes[0], H_iso1g_after_unweighted, "Iso 1g after coherent_1g_weight\n(unweighted, before sampling)"),
        (axes[1], H_iso1g_after,            "Iso 1g after coherent_1g_weight\n(weighted — target shape)"),
        (axes[2], H_sampled,                f"Iso 1g after random sampling\n(unweighted, {target_fraction:.0%} target)"),
    ]:
        H_plot = np.where(H > 0, H, np.nan)
        vmax = np.nanmax(H_plot) if np.isfinite(H_plot).any() else 0.0
        if not (vmax > 0):  # empty/all-zero/all-nan histogram (e.g. small --frac_events)
            vmax = 1.0
        vmin = vmax * 1e-3
        im = ax.pcolormesh(bins_costheta, bins_energy, H_plot.T,
                           cmap='plasma', shading='flat',
                           norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
        plt.colorbar(im, ax=ax, label='Events / bin')
        ax.set_xlabel(r"cos $\theta$")
        ax.set_ylabel("Energy [MeV]")
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 2100)
        ax.set_title(title)
    fig.suptitle("Random sampling to match NC coherent shape without event weights")
    fig.tight_layout()
    fig.savefig(os.path.join(_PLOT_DIR, "coh1g_random_sampling.jpeg"), dpi=400)
    plt.close(fig)
