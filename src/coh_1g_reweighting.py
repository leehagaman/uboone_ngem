"""Compute per-event NC-coherent-1g reweighting weights for isotropic-1g events.

This module (originally a port of the since-deleted coherent_1g_reweighting.ipynb)
writes ``coh_1g_reweighting.parquet`` as part of the
main dataframe-creation pipeline instead of by hand.

The produced parquet has columns:
    run, subrun, event, coherent_1g_weight_per_pot, coherent_1g_keep,
    coherent_1g_true_nuEnergy
The weight is POT-independent (events per POT); the target normalizing POT is
applied later by apply_nc_coh_1g_reweighting.  Only events with
coherent_1g_weight_per_pot > 0 are saved.

coherent_1g_true_nuEnergy (MeV) is a true neutrino energy assigned to each iso1g
event: the photon-gun events carry no neutrino, so one is drawn at random from the
reference NC coherent simulation events in the same (costheta, E_gamma) reweighting
bin.  Downstream (create_rw_syst_df.py) this is used to borrow flux multisim weights
from nu_overlay events of matching true neutrino energy.

Normalization (see ipynb_notebooks/nc_coherent_1g_file_comparison.ipynb): the
reference is the unbiased BatchB NC coherent simulation, generated uniformly in the
TPC active volume.  Its POT-equivalent comes from Mark's 2.22242e24 POT, which
belongs to the 34,011 events generated behind the (preTopo-selected) TestSplit file
from the same generation; that rate reproduces the paper's (arXiv:2502.06091)
predicted selected events to ~5%.  The iso1g->coherent bin ratio is built from the
unweighted iso1g events inside the TPC, so the in-TPC iso1g events sum to the
coherent rate in the TPC.  iso1g events outside the TPC (the gun fills the cryostat
at the same density per volume) get the same per-event weight, representing NC
coherent scattering on the argon outside the TPC.

A second parquet, coh_1g_reweighting_bins.parquet, holds the same reweighting per
(costheta, E_gamma) bin: coherent_1g_bin (see coherent_1g_bin_index) and
coherent_1g_rate_per_pot, the summed per-POT weight of the nominal iso1g events in that
bin (= the expected NC coherent 1g events per POT there).  create_detvar_df.py uses it
to reweight the iso1g DetVar samples, whose events are different photon-gun events
than the nominal sample's, so the run/subrun/event-keyed parquet cannot be used.
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

# NC coherent single-photon reference simulation: the complete, unfiltered BatchB
# sample (67,771 events, uniform in the TPC), from
# /pnfs/uboone/persistent/users/markross/Jan2022_gLEE_files/UniqDir/.
# (The previously used preTopo_..._TestSplit_... file is a reco-preselected subset
# whose photon kinematics are biased; see nc_coherent_1g_file_comparison.ipynb.)
_NC_COHERENT_ROOT = "vertex_NCSingleCoherent_Run123_BatchB_v50.0.uniq.root"

# ── Reference-simulation POT ─────────────────────────────────────────────────
# The target (normalizing) POT is intentionally NOT set here: it is applied later
# by apply_nc_coh_1g_reweighting, so the weights saved by this module are
# POT-independent (units of events per POT).
# Mark's POT (slack 2026-02-24) belongs to the events GENERATED behind the TestSplit
# file (sum of its run_subrun_tree subrun_counts), from the same generation as BatchB.
# Their ratio is the NC coherent 1g rate in the TPC active volume (1.53e-20 / POT,
# 10.5 events per 6.87e20 POT), and BatchB's POT-equivalent is
# n_BatchB_events / that rate (~4.43e24).
reference_nc_coherent_pot = 2.22242e+24
reference_nc_coherent_num_generated = 34011
nc_coherent_rate_per_pot_in_tpc = reference_nc_coherent_num_generated / reference_nc_coherent_pot

# TPC active volume [cm] (the reference generation volume; matches the BatchB true
# vertex ranges)
tpc_active_x = (-1.55, 254.8)
tpc_active_y = (-115.53, 117.47)
tpc_active_z = (0.1, 1036.9)

# Paper reference (arXiv:2502.06091), used as a sanity check: 1.1 predicted events in
# the coherent-rich selection, with a 28.1% topological efficiency relative to ALL
# simulated signal events in the TPC and 39.09% relative efficiency after that, so
# 1.1 / (0.281 * 0.3909) = 10.0 is an in-TPC count per 6.87e20 POT.
coherent_eff_topological = 0.281
coherent_eff_rel_to_topological = 0.3909
coherent_num_sel = 1.1

# ── Reweighting binning (notebook cell 6) ────────────────────────────────────
bins_costheta = np.linspace(-1, 1, 101)
bins_energy   = np.concatenate([np.linspace(0, 2000, 101), [1e6]])

def coherent_1g_bin_index(costheta, energy):
    """Flattened (costheta, E_gamma [MeV]) reweighting-bin index, as used for the
    nominal iso1g events (values outside the binning are clipped into the edge bins)."""
    i_cos = np.clip(np.digitize(costheta, bins_costheta) - 1, 0, len(bins_costheta) - 2)
    i_E   = np.clip(np.digitize(energy,   bins_energy)   - 1, 0, len(bins_energy)   - 2)
    return i_cos * (len(bins_energy) - 1) + i_E


def _load_nc_coherent_simulation():
    """Load the NC coherent single-photon truth gamma energy/costheta and
    the incoming true neutrino energy.

    The reference file's mctruth_nu_E branch is unfilled (-99), but every event's
    mctruth_daughters list is [gamma, nu_in, Ar40, nu_out] with status codes
    [1, 0, 2, 2], so the incoming neutrino is daughter index 1 (pdg 14, status 0).
    Energy conservation (E_nu_in ~= E_gamma + E_nu_out) was checked by hand.

    Returns (costheta, gamma_energy_MeV, nu_energy_MeV) as numpy arrays."""
    f = uproot.open(
        os.path.join(other_files_location, _NC_COHERENT_ROOT)
    )["singlephotonana/vertex_tree"]
    daughters_pdg = f["mctruth_daughters_pdg"].array(library="np")
    daughters_E = f["mctruth_daughters_E"].array(library="np")
    daughters_pz = f["mctruth_daughters_pz"].array(library="np")
    daughters_status = f["mctruth_daughters_status_code"].array(library="np")

    nc_coherent_gamma_energy = []
    nc_coherent_gamma_costheta = []
    nc_coherent_nu_energy = []
    for i in range(len(daughters_pdg)):
        if daughters_pdg[i][0] != 22:
            raise Exception(
                "PROBLEM! First daughter particle isn't a photon in the nc coherent 1g simulation",
                daughters_pdg[i])
        if abs(daughters_pdg[i][1]) != 14 or daughters_status[i][1] != 0:
            raise Exception(
                "PROBLEM! Second daughter particle isn't the incoming (status 0) muon neutrino in the nc coherent 1g simulation",
                daughters_pdg[i], daughters_status[i])
        nc_coherent_gamma_energy.append(daughters_E[i][0] * 1000)
        nc_coherent_gamma_costheta.append(daughters_pz[i][0] / daughters_E[i][0])
        nc_coherent_nu_energy.append(daughters_E[i][1] * 1000)

    return (np.array(nc_coherent_gamma_costheta), np.array(nc_coherent_gamma_energy),
            np.array(nc_coherent_nu_energy))


def compute_nc_coh_1g_reweighting(df, make_plots=True):
    """Compute and save the binned NC-coherent-1g reweighting from the dataframe.

    Accepts an eager polars DataFrame or a LazyFrame. Writes
    ``coh_1g_reweighting.parquet`` and ``coh_1g_reweighting_bins.parquet`` to
    ``intermediate_files_location`` and (if ``make_plots``) all diagnostic plots to
    ``plots/coherent_1g_reweighting/``.

    The per-bin weight is H_coherent / H_iso1g_inTPC / reference_pot, with
    H_iso1g_inTPC the UNWEIGHTED (cos theta, E_gamma) histogram of the iso1g events
    whose true vertex is in the TPC active volume and reference_pot the BatchB
    POT-equivalent (see the module docstring).  No event net weights enter, so the
    result is independent of the POT/run-period weighting of the iso1g sample.
    """
    print("computing NC coherent 1g reweighting weights from the dataframe")

    lf = df if isinstance(df, pl.LazyFrame) else df.lazy()

    os.makedirs(_PLOT_DIR, exist_ok=True)

    # ── NC coherent simulation target shape ───────────────────────────────────
    nc_coherent_gamma_costheta, nc_coherent_gamma_energy, nc_coherent_nu_energy = _load_nc_coherent_simulation()
    num_coherent_simulated_events = len(nc_coherent_gamma_energy)
    reference_pot = num_coherent_simulated_events / nc_coherent_rate_per_pot_in_tpc
    print(f"  NC coherent simulated events: {num_coherent_simulated_events:,} "
          f"(POT-equivalent {reference_pot:.4e}, rate in TPC {nc_coherent_rate_per_pot_in_tpc:.4e} / POT)")
    print(f"  NC coherent true nu energy: mean {nc_coherent_nu_energy.mean():.0f} MeV, "
          f"range [{nc_coherent_nu_energy.min():.0f}, {nc_coherent_nu_energy.max():.0f}] MeV")

    # ── Iso1g events from the dataframe ───────────────────────────────────────
    iso1g_df = (
        lf.filter(pl.col("filetype") == "isotropic_one_gamma_overlay")
        .select(
            "run", "subrun", "event",
            "wc_true_leading_shower_energy", "wc_true_leading_shower_costheta",
            "wc_truth_vtxX", "wc_truth_vtxY", "wc_truth_vtxZ",
        )
        .collect()
        .with_columns(
            (pl.col("wc_truth_vtxX").is_between(*tpc_active_x, closed="none")
             & pl.col("wc_truth_vtxY").is_between(*tpc_active_y, closed="none")
             & pl.col("wc_truth_vtxZ").is_between(*tpc_active_z, closed="none")).alias("in_tpc"),
            (pl.col("wc_truth_vtxX").is_between(10.0, 246.4, closed="none")
             & pl.col("wc_truth_vtxY").is_between(-101.5, 101.5, closed="none")
             & pl.col("wc_truth_vtxZ").is_between(10.0, 986.8, closed="none")).alias("in_fv"),
        )
    )
    in_tpc = iso1g_df["in_tpc"].to_numpy()
    print(f"  isotropic 1g events: {iso1g_df.height:,} ({in_tpc.sum():,} with true vertex in the TPC)")

    iso1g_costheta = iso1g_df["wc_true_leading_shower_costheta"].to_numpy()
    iso1g_energy   = iso1g_df["wc_true_leading_shower_energy"].to_numpy()

    if make_plots:
        _plot_input_distributions(
            nc_coherent_gamma_costheta, nc_coherent_gamma_energy,
            iso1g_costheta[in_tpc], iso1g_energy[in_tpc])

    # ── 2D reweighting in-TPC iso1g -> coherent shape ─────────────────────────
    H_coherent, _, _ = np.histogram2d(
        nc_coherent_gamma_costheta, nc_coherent_gamma_energy,
        bins=[bins_costheta, bins_energy])
    H_iso1g, _, _ = np.histogram2d(
        iso1g_costheta[in_tpc], iso1g_energy[in_tpc],
        bins=[bins_costheta, bins_energy])

    lost_fraction = H_coherent[H_iso1g == 0].sum() / H_coherent.sum()
    print(f"  fraction of reference coherent events in bins without in-TPC iso1g events (lost): {lost_fraction:.2e}")

    # unitless coherent/iso1g ratio per bin; the per-POT weight divides by reference_pot below
    weight_ratios = np.where(H_iso1g > 0, H_coherent / np.where(H_iso1g > 0, H_iso1g, 1.0), 0.0)
    i_cos = np.clip(np.digitize(iso1g_costheta, bins_costheta) - 1, 0, len(bins_costheta) - 2)
    i_E   = np.clip(np.digitize(iso1g_energy,   bins_energy)   - 1, 0, len(bins_energy)   - 2)
    event_weights = weight_ratios[i_cos, i_E]

    iso1g_df = iso1g_df.with_columns(
        pl.Series("coherent_1g_weight_nopot", event_weights)
    )

    H_iso1g_after, _, _ = np.histogram2d(
        iso1g_costheta, iso1g_energy,
        bins=[bins_costheta, bins_energy], weights=event_weights)
    H_iso1g_after_unweighted, _, _ = np.histogram2d(
        iso1g_costheta, iso1g_energy,
        bins=[bins_costheta, bins_energy], weights=(event_weights > 0).astype(float))

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

    # ── True neutrino energy assignment ───────────────────────────────────────
    # Each iso1g event with positive weight draws one true E_nu at random from the
    # reference coherent events in the same (costheta, E_gamma) bin.  Those bins are
    # non-empty by construction (positive weight <=> H_coherent > 0). E_nu is
    # strongly correlated with E_gamma for coherent scattering, so this reproduces the
    # E_nu distribution of the reference simulation conditional on the photon
    # kinematics. Same rng as the keep-mask draw, so the result is reproducible.
    coh_i_cos = np.clip(np.digitize(nc_coherent_gamma_costheta, bins_costheta) - 1, 0, len(bins_costheta) - 2)
    coh_i_E   = np.clip(np.digitize(nc_coherent_gamma_energy,   bins_energy)   - 1, 0, len(bins_energy)   - 2)
    coh_bin_key = coh_i_cos * (len(bins_energy) - 1) + coh_i_E
    coh_order = np.argsort(coh_bin_key, kind="stable")
    coh_key_sorted = coh_bin_key[coh_order]
    iso_bin_key = i_cos_all * (len(bins_energy) - 1) + i_E_all
    lo_idx = np.searchsorted(coh_key_sorted, iso_bin_key, side="left")
    hi_idx = np.searchsorted(coh_key_sorted, iso_bin_key, side="right")
    n_in_bin = hi_idx - lo_idx
    if np.any(n_in_bin[mask_positive] == 0):
        raise Exception("PROBLEM! iso1g event with positive coherent weight but no reference coherent event in its bin")
    pick = lo_idx + np.floor(rng.random(iso1g_df.height) * np.maximum(n_in_bin, 1)).astype(np.int64)
    pick = np.minimum(pick, np.maximum(hi_idx - 1, 0))
    assigned_nu_energy = np.where(mask_positive, nc_coherent_nu_energy[coh_order[pick]], np.nan).astype(np.float32)
    iso1g_df = iso1g_df.with_columns(pl.Series("coherent_1g_true_nuEnergy", assigned_nu_energy))
    print(f"  assigned true nu energy to {int(mask_positive.sum()):,} iso1g events "
          f"(mean {np.nanmean(assigned_nu_energy):.0f} MeV vs reference {nc_coherent_nu_energy.mean():.0f} MeV)")
    print(f"  Target fraction: {target_fraction:.2f}  ({target_kept:,.0f} events)")
    print(f"  Events kept:     {keep_mask.sum():,} / {n_positive:,}  ({keep_mask.sum() / n_positive:.3f})")

    if make_plots:
        _plot_sampling(iso1g_costheta, iso1g_energy, keep_mask,
                       H_iso1g_after_unweighted, H_iso1g_after, target_fraction)

    # ── Per-POT weight ────────────────────────────────────────────────────────
    # The target POT is factored out: the saved weight is per unit POT, and
    # apply_nc_coh_1g_reweighting multiplies by the desired normalizing POT
    # (coherent_1g_weight = coherent_1g_weight_per_pot * normalizing_POT).
    iso1g_df = iso1g_df.with_columns(
        (pl.col("coherent_1g_weight_nopot") / reference_pot).alias("coherent_1g_weight_per_pot")
    )

    paper_coherent_num_tpc_687e20 = coherent_num_sel / (coherent_eff_topological * coherent_eff_rel_to_topological)
    reference_num_tpc_687e20 = nc_coherent_rate_per_pot_in_tpc * 6.87e20
    df_coherent_num_tpc_687e20 = iso1g_df.filter("in_tpc")["coherent_1g_weight_per_pot"].sum() * 6.87e20
    df_coherent_num_fv_687e20 = iso1g_df.filter("in_fv")["coherent_1g_weight_per_pot"].sum() * 6.87e20
    df_coherent_num_all_687e20 = iso1g_df["coherent_1g_weight_per_pot"].sum() * 6.87e20
    print(f"  NC coherent 1g events in the TPC per 6.87e20 POT: paper {paper_coherent_num_tpc_687e20:.3f}, "
          f"reference simulation {reference_num_tpc_687e20:.3f}, our iso1g in TPC {df_coherent_num_tpc_687e20:.3f}")
    print(f"  our iso1g in FV {df_coherent_num_fv_687e20:.3f}, all iso1g (incl. outside the TPC) {df_coherent_num_all_687e20:.3f}")

    # ── Save weights (cell 9) ─────────────────────────────────────────────────
    weights_to_save = iso1g_df.filter(
        pl.col("coherent_1g_weight_per_pot") > 0
    ).select(["run", "subrun", "event", "coherent_1g_weight_per_pot", "coherent_1g_keep",
              "coherent_1g_true_nuEnergy"])

    out_path = f"{intermediate_files_location}/coh_1g_reweighting.parquet"
    weights_to_save.write_parquet(out_path)
    print(f"  saved {out_path}")

    # per-bin version for the DetVar iso1g samples (see module docstring); the per-POT
    # weight is constant within a bin, so (rate / n_iso1g_events) is the nominal per-event weight
    bin_table = (
        iso1g_df.with_columns(pl.Series("coherent_1g_bin", iso_bin_key.astype(np.int32)))
        .filter(pl.col("coherent_1g_weight_per_pot") > 0)
        .group_by("coherent_1g_bin")
        .agg(pl.col("coherent_1g_weight_per_pot").sum().alias("coherent_1g_rate_per_pot"),
             pl.len().alias("n_iso1g_events"))
        .sort("coherent_1g_bin")
    )
    bins_out_path = f"{intermediate_files_location}/coh_1g_reweighting_bins.parquet"
    bin_table.write_parquet(bins_out_path)
    print(f"  saved {bins_out_path} ({bin_table.height:,} non-empty bins)")
    print(f"  {weights_to_save.height:,} / {iso1g_df.height:,} events have valid weights")

    # ── Write a log of this run into the plot directory ──────────────────────
    now = datetime.now()
    log_lines = [
        "NC coherent 1g reweighting log",
        "==============================",
        f"created: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.astimezone().tzname()})",
        "see: ipynb_notebooks/nc_coherent_1g_file_comparison.ipynb",
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
        "normalization:",
        f"  reference_nc_coherent_pot = {reference_nc_coherent_pot:.6g} (Mark, slack 2026-02-24)",
        f"  reference_nc_coherent_num_generated = {reference_nc_coherent_num_generated}",
        f"  NC coherent 1g rate in the TPC = {nc_coherent_rate_per_pot_in_tpc:.6g} / POT",
        f"  reference file POT-equivalent = {reference_pot:.6g}",
        f"  H_iso1g: unweighted, iso1g with true vertex in the TPC ({int(in_tpc.sum()):,} events)",
        f"  reference fraction in bins without in-TPC iso1g events: {lost_fraction:.3e}",
        f"  normalizing (target) POT: not applied here -- applied later by apply_nc_coh_1g_reweighting",
        f"  NC coherent 1g events in the TPC per 6.87e20 POT, paper (arxiv 2502.06091): {paper_coherent_num_tpc_687e20:.4f}",
        f"  NC coherent 1g events in the TPC per 6.87e20 POT, reference simulation: {reference_num_tpc_687e20:.4f}",
        f"  NC coherent 1g events per 6.87e20 POT from our iso1g: TPC {df_coherent_num_tpc_687e20:.4f}, "
        f"FV {df_coherent_num_fv_687e20:.4f}, all (incl. outside the TPC) {df_coherent_num_all_687e20:.4f}",
        "",
        "outputs:",
        f"  weights parquet: {out_path}",
        f"  valid-weight events (coherent_1g_weight_per_pot > 0): {weights_to_save.height:,} / {iso1g_df.height:,}",
        f"  saved columns: run, subrun, event, coherent_1g_weight_per_pot, coherent_1g_keep, coherent_1g_true_nuEnergy",
        f"  per-bin table (for DetVar): {bins_out_path}",
        f"  make_plots: {make_plots} (plots in this directory)",
    ]
    log_path = os.path.join(_PLOT_DIR, "reweighting_log.txt")
    with open(log_path, "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
    print(f"  wrote log {log_path}")

    return weights_to_save


def _plot_input_distributions(nc_coherent_gamma_costheta, nc_coherent_gamma_energy,
                              iso1g_costheta, iso1g_energy):
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
               norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.xlabel("cos(theta)")
    plt.ylabel("Energy (MeV)")
    plt.title("Isotropic 1g Gamma Energy and Costheta (true vertex in TPC, unweighted)")
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
        (axes[0], H_iso1g,       "Iso 1g in TPC (unweighted)\n[before coherent_1g_weight]"),
        (axes[1], H_coherent,    "NC Coherent 1g simulation (target)"),
        (axes[2], H_iso1g_after, "Iso 1g (all) after coherent_1g_weight_nopot"),
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
