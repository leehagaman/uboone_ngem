"""Blip reconstruction response for low-energy photons in the isotropic 1g sample.

Extracts the truth-matched blips (blip_true_g4id >= 0; in the isotropic
one-gamma overlay the only truth is the generated photon and its secondaries,
so every truth-matched blip is photon-induced -- the cosmic-overlay blips have
no truth match) for events with true photon energy below E_EXTRACT_MAX_MEV
from the raw checkout ROOT files, and measures how photons at nuclear
de-excitation energies (0-12 MeV, see deexcitation_photon_study.py) would be
reconstructed:

- P(>= 1 truth-matched blip) and blip multiplicity vs true photon energy,
- summed reconstructed blip energy vs true photon energy,
- distance of each truth-matched blip from the photon origin point,
- the same quantities reweighted to the GENIE de-excitation spectrum.

True photon energy, origin (generation vertex), direction, and the FV flag
come from all_df.parquet (wc_true_leading_shower_energy, wc_truth_vtx*,
wc_true_leading_shower_costheta/phi, wc_truth_inFV), matched to the raw files
on (run, subrun, event). Blip positions are reconstructed coordinates while
the origin is the true position, so distances carry cm-level space-charge
offsets; that is fine at the tens-of-cm scale of photon transport.

All response plots use the boundary-unbiased "clean" truth selection (see
iso1g_deex_boundary_check.py for the bias study): photon origin at least
VTX_WALL_MIN_CM from every TPC wall AND at least RAY_IN_TPC_MIN_CM of
projected photon ray inside the TPC before exiting.

The per-blip table is saved to intermediate_files as
iso1g_lowE_truth_blips.parquet and is reused if it already exists; plots go
to plots/deexcitation_photons/.

Run with:
    source ../uv_base/bin/activate
    python src/deex_testing/iso1g_deex_blip_response.py
"""

# study script living in src/deex_testing/: put src/ on the path so the
# pipeline modules (file_locations, blip_postprocessing, ...) import cleanly
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import glob
import os

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import uproot

from blip_postprocessing import (TPC_X_MIN, TPC_X_MAX, TPC_Y_MIN, TPC_Y_MAX,
                                 TPC_Z_MIN, TPC_Z_MAX)
from file_locations import data_files_location, intermediate_files_location
from iso1g_deex_stats_check import (
    load_deex_spectrum, DEEX_SPECTRUM_MAX_MEV, REWEIGHT_BINS_MEV)
from deexcitation_photon_study import PLOT_DIR

E_EXTRACT_MAX_MEV = 50.0  # extract below this (de-excitation range + sideband)

# clean (boundary-unbiased) truth selection
VTX_WALL_MIN_CM = 50.0     # origin at least this far from every TPC wall
RAY_IN_TPC_MIN_CM = 100.0  # at least this much projected photon ray in the TPC

RAW_FILE_PATTERN = os.path.join(
    data_files_location, "checkout_isotropic_one_gamma_run45_reco2_prod_reco2_hist_*.root")
BLIP_BRANCHES = ["blip_x", "blip_y", "blip_z", "blip_energy", "blip_nplanes",
                 "blip_true_g4id", "blip_true_pdg", "blip_true_energy"]
OUT_PARQUET = os.path.join(intermediate_files_location, "iso1g_lowE_truth_blips.parquet")


def load_lowE_truth():
    """(run, subrun, event) -> true photon energy / origin / FV flag, from all_df."""
    lf = pl.scan_parquet(os.path.join(intermediate_files_location, "all_df.parquet"))
    df = (
        lf.filter(
            (pl.col("filetype") == "isotropic_one_gamma_overlay")
            & (pl.col("wc_true_leading_shower_energy") < E_EXTRACT_MAX_MEV)
        )
        .select("run", "subrun", "event", "wc_true_leading_shower_energy",
                "wc_truth_vtxX", "wc_truth_vtxY", "wc_truth_vtxZ", "wc_truth_inFV")
        .collect()
    )
    print(f"isotropic 1g events with true E < {E_EXTRACT_MAX_MEV:.0f} MeV: {df.height:,}")
    truth = {}
    for row in df.iter_rows():
        run, subrun, event, E, vx, vy, vz, in_fv = row
        truth[(run, subrun, event)] = (E, vx, vy, vz, bool(in_fv))
    if len(truth) != df.height:
        print(f"  WARNING: {df.height - len(truth)} duplicate (run,subrun,event) keys")
    return truth


def extract_truth_blips(truth):
    """One output row per selected event, with lists of its truth-matched blips."""
    rows = []
    n_found = 0
    for path in sorted(glob.glob(RAW_FILE_PATTERN)):
        print(f"  scanning {os.path.basename(path)}")
        for chunk in uproot.iterate(
            {path: "nuselection/NeutrinoSelectionFilter"},
            ["run", "sub", "evt"] + BLIP_BRANCHES,
            step_size="500 MB",
        ):
            runs = np.asarray(chunk["run"])
            subs = np.asarray(chunk["sub"])
            evts = np.asarray(chunk["evt"])
            for i in range(len(runs)):
                key = (int(runs[i]), int(subs[i]), int(evts[i]))
                info = truth.get(key)
                if info is None:
                    continue
                n_found += 1
                E, vx, vy, vz, in_fv = info
                matched = np.asarray(chunk["blip_true_g4id"][i]) >= 0
                bx = np.asarray(chunk["blip_x"][i])[matched]
                by = np.asarray(chunk["blip_y"][i])[matched]
                bz = np.asarray(chunk["blip_z"][i])[matched]
                dist = np.sqrt((bx - vx) ** 2 + (by - vy) ** 2 + (bz - vz) ** 2)
                rows.append({
                    "run": key[0], "subrun": key[1], "event": key[2],
                    "true_gamma_energy": E,
                    "true_vtx_x": vx, "true_vtx_y": vy, "true_vtx_z": vz,
                    "truth_inFV": in_fv,
                    "blip_x": bx.tolist(), "blip_y": by.tolist(), "blip_z": bz.tolist(),
                    "blip_dist_to_origin": dist.tolist(),
                    "blip_energy": np.asarray(chunk["blip_energy"][i])[matched].tolist(),
                    "blip_nplanes": np.asarray(chunk["blip_nplanes"][i])[matched].tolist(),
                    "blip_true_pdg": np.asarray(chunk["blip_true_pdg"][i])[matched].tolist(),
                    "blip_true_energy": np.asarray(chunk["blip_true_energy"][i])[matched].tolist(),
                })
    print(f"  matched {n_found:,} of {len(truth):,} selected events in the raw files")
    df = pl.DataFrame(rows)
    df.write_parquet(OUT_PARQUET)
    print(f"  per-event truth-blip table saved to {OUT_PARQUET}")
    return df


def wilson_yerr(p, n, z=1.0):
    """Asymmetric 1-sigma (z=1) Wilson-interval errors for efficiencies.
    Returns a (2, N) array usable as matplotlib yerr. For weighted
    efficiencies pass the effective sample size as n."""
    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    # clip tiny negative values from floating-point rounding at p = 0 or 1
    return np.clip(np.vstack([p - (center - half), (center + half) - p]), 0, None)


def weighted_mean_sem(vals, weights=None):
    """(mean, standard error of the mean); weighted uses the effective N."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    if weights is None:
        weights = np.ones(len(vals))
    mean = np.average(vals, weights=weights)
    var = np.average((vals - mean) ** 2, weights=weights)
    n_eff = weights.sum() ** 2 / (weights**2).sum()
    return mean, np.sqrt(var / n_eff) if n_eff > 1 else np.nan


def step_hist_with_errors(ax, data, bins, weights=None, density=False,
                          color="#000000", label=None, linestyle="-"):
    """Step histogram plus sqrt(sum w^2) statistical error bars per bin."""
    data = np.asarray(data, dtype=float)
    counts, _ = np.histogram(data, bins=bins, weights=weights)
    w2 = None if weights is None else np.asarray(weights) ** 2
    sumw2, _ = np.histogram(data, bins=bins, weights=w2)
    err = np.sqrt(sumw2)
    if density:
        widths = np.diff(bins)
        norm = counts.sum() * widths
        with np.errstate(divide="ignore", invalid="ignore"):
            counts = np.where(norm > 0, counts / norm, 0.0)
            err = np.where(norm > 0, err / norm, 0.0)
    ax.stairs(counts, bins, color=color, label=label, linestyle=linestyle)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax.errorbar(centers, counts, yerr=err, fmt="none", ecolor=color,
                elinewidth=1, capsize=0, alpha=0.7)


def signed_distance_to_wall(x, y, z):
    """Min distance from (x, y, z) to the six TPC planes; negative outside."""
    return np.minimum.reduce([
        x - TPC_X_MIN, TPC_X_MAX - x,
        y - TPC_Y_MIN, TPC_Y_MAX - y,
        z - TPC_Z_MIN, TPC_Z_MAX - z,
    ])


def ray_length_in_tpc(x, y, z, ux, uy, uz):
    """Distance from an in-TPC point (x, y, z) along direction (ux, uy, uz)
    to where the ray exits the TPC box. Zero for points outside the TPC."""
    t_exit = np.full(len(x), np.inf)
    for pos, u, lo, hi in [(x, ux, TPC_X_MIN, TPC_X_MAX),
                           (y, uy, TPC_Y_MIN, TPC_Y_MAX),
                           (z, uz, TPC_Z_MIN, TPC_Z_MAX)]:
        with np.errstate(divide="ignore"):
            t = np.where(u > 0, (hi - pos) / u,
                         np.where(u < 0, (lo - pos) / u, np.inf))
        t_exit = np.minimum(t_exit, t)
    inside = signed_distance_to_wall(x, y, z) > 0
    return np.where(inside, t_exit, 0.0)


def load_directions():
    """(run, subrun, event) -> photon direction unit vector, from all_df.
    postprocessing.py convention: costheta = pz/|p|, phi = arctan2(px, py)
    in degrees."""
    lf = pl.scan_parquet(os.path.join(intermediate_files_location, "all_df.parquet"))
    df = (
        lf.filter(
            (pl.col("filetype") == "isotropic_one_gamma_overlay")
            & (pl.col("wc_true_leading_shower_energy") < E_EXTRACT_MAX_MEV)
        )
        .select("run", "subrun", "event",
                "wc_true_leading_shower_costheta", "wc_true_leading_shower_phi")
        .collect()
    )
    costheta = df["wc_true_leading_shower_costheta"].to_numpy()
    phi_rad = np.radians(df["wc_true_leading_shower_phi"].to_numpy())
    sintheta = np.sqrt(np.clip(1 - costheta**2, 0, 1))
    ux = sintheta * np.sin(phi_rad)
    uy = sintheta * np.cos(phi_rad)
    uz = costheta
    return {
        (r, s, e): (ux[i], uy[i], uz[i])
        for i, (r, s, e) in enumerate(zip(df["run"], df["subrun"], df["event"]))
    }


def clean_selection_mask(df):
    """Boundary-unbiased truth selection: origin >= VTX_WALL_MIN_CM from every
    TPC wall and >= RAY_IN_TPC_MIN_CM of projected photon ray in the TPC."""
    vx = df["true_vtx_x"].to_numpy()
    vy = df["true_vtx_y"].to_numpy()
    vz = df["true_vtx_z"].to_numpy()
    d_wall = signed_distance_to_wall(vx, vy, vz)
    directions = load_directions()
    dirs = np.array([
        directions[(r, s, e)]
        for r, s, e in zip(df["run"], df["subrun"], df["event"])
    ])
    ray_len = ray_length_in_tpc(vx, vy, vz, dirs[:, 0], dirs[:, 1], dirs[:, 2])
    return (d_wall >= VTX_WALL_MIN_CM) & (ray_len >= RAY_IN_TPC_MIN_CM)


def deex_bin_weights(df, clean):
    """Per-event weights reweighting the clean-selection < 12 MeV events to the
    GENIE de-excitation spectrum shape (1 MeV bins, as iso1g_deex_stats_check)."""
    deex_E = load_deex_spectrum()
    E = df["true_gamma_energy"].to_numpy()
    sel = clean & (E < DEEX_SPECTRUM_MAX_MEV)
    src_counts, _ = np.histogram(E[sel], bins=REWEIGHT_BINS_MEV)
    tgt_counts, _ = np.histogram(deex_E, bins=REWEIGHT_BINS_MEV)
    with np.errstate(divide="ignore", invalid="ignore"):
        bin_w = np.where(src_counts > 0,
                         (tgt_counts / tgt_counts.sum()) / (src_counts / src_counts.sum()),
                         0.0)
    w = np.zeros(len(E))
    w[sel] = bin_w[np.digitize(E[sel], REWEIGHT_BINS_MEV) - 1]
    return w


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    if os.path.exists(OUT_PARQUET):
        df = pl.read_parquet(OUT_PARQUET)
        print(f"loaded {df.height:,} events from existing {OUT_PARQUET}")
    else:
        truth = load_lowE_truth()
        df = extract_truth_blips(truth)

    E = df["true_gamma_energy"].to_numpy()
    clean = clean_selection_mask(df)
    print(f"clean truth selection (d_wall >= {VTX_WALL_MIN_CM:.0f} cm, "
          f"ray >= {RAY_IN_TPC_MIN_CM:.0f} cm in TPC): {clean.sum():,} of {len(E):,} "
          f"events, {(clean & (E < DEEX_SPECTRUM_MAX_MEV)).sum()} below 12 MeV")
    n_blips = df["blip_dist_to_origin"].list.len().to_numpy()
    sum_blip_E = df["blip_energy"].list.sum().fill_null(0.0).to_numpy()
    w_deex = deex_bin_weights(df, clean)

    print_deex_summary(E, clean, n_blips, sum_blip_E, df, w_deex)
    plot_response_vs_energy(E, clean, n_blips, sum_blip_E)
    plot_deex_multiplicity_and_energy(E, clean, n_blips, sum_blip_E, df, w_deex)
    plot_blip_distances(df, clean, w_deex)
    plot_blip_3d_and_projections(df, clean)
    print(f"plots saved to {os.path.abspath(PLOT_DIR)}")


def print_deex_summary(E, clean, n_blips, sum_blip_E, df, w_deex):
    sel = clean & (E < DEEX_SPECTRUM_MAX_MEV)
    w = w_deex[sel]
    print(f"\nclean-selection events with true E < {DEEX_SPECTRUM_MAX_MEV:.0f} MeV: {sel.sum()}")
    print("de-excitation-spectrum-weighted blip response (clean selection):")
    print(f"  P(>= 1 truth-matched blip): unweighted {(n_blips[sel] > 0).mean()*100:.1f}%, "
          f"weighted {np.average(n_blips[sel] > 0, weights=w)*100:.1f}%")
    print(f"  mean multiplicity: unweighted {n_blips[sel].mean():.2f}, "
          f"weighted {np.average(n_blips[sel], weights=w):.2f}")
    has = sel & (n_blips > 0)
    wh = w_deex[has]
    print(f"  mean summed blip energy (events with >= 1 blip): "
          f"unweighted {sum_blip_E[has].mean():.2f} MeV, "
          f"weighted {np.average(sum_blip_E[has], weights=wh):.2f} MeV")
    dist_lists = df["blip_dist_to_origin"].to_list()
    dist = np.concatenate(
        [np.asarray(dist_lists[i]) for i in np.where(sel)[0] if dist_lists[i]]
    ) if has.sum() else np.array([])
    if len(dist):
        print(f"  blip distance to origin: median {np.median(dist):.1f} cm, "
              f"90% within {np.percentile(dist, 90):.1f} cm")


def plot_response_vs_energy(E, clean, n_blips, sum_blip_E):
    bins = np.concatenate([np.arange(0, 15, 1.5), [17, 20, 25, 30, 40, 50]])
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for mask, color, label in [(clean, "#009E73", "clean truth selection")]:
        p1, mult, mult_err, meanE, meanE_err, ns = \
            (np.full(len(centers), np.nan) for _ in range(6))
        for i in range(len(centers)):
            s = mask & (E >= bins[i]) & (E < bins[i + 1])
            if s.sum() < 15:
                continue
            ns[i] = s.sum()
            p1[i] = (n_blips[s] > 0).mean()
            mult[i], mult_err[i] = weighted_mean_sem(n_blips[s])
            h = s & (n_blips > 0)
            if h.sum() >= 10:
                meanE[i], meanE_err[i] = weighted_mean_sem(sum_blip_E[h])
        axes[0].errorbar(centers, p1, yerr=wilson_yerr(p1, ns), marker="o",
                         markersize=4, color=color, label=label)
        axes[1].errorbar(centers, mult, yerr=mult_err, marker="o",
                         markersize=4, color=color, label=label)
        axes[2].errorbar(centers, meanE, yerr=meanE_err, marker="o",
                         markersize=4, color=color, label=label)

    axes[0].set_ylabel("P(>= 1 truth-matched blip)")
    axes[1].set_ylabel("Mean truth-matched blip multiplicity")
    axes[2].set_ylabel("Mean summed blip energy [MeV]\n(events with >= 1 blip)")
    axes[2].plot([0, 50], [0, 50], color="#999999", linestyle=":", linewidth=1)
    for ax in axes:
        ax.axvspan(0, DEEX_SPECTRUM_MAX_MEV, color="#E69F00", alpha=0.15)
        ax.set_xlabel("True photon energy [MeV]")
        ax.set_ylim(bottom=0)
    axes[0].legend()
    fig.suptitle("Truth-matched blip response vs photon energy, isotropic 1$\\gamma$ sample, "
                 f"clean truth selection (d_wall >= {VTX_WALL_MIN_CM:.0f} cm, "
                 f"ray >= {RAY_IN_TPC_MIN_CM:.0f} cm in TPC; shaded: de-excitation range)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "iso1g_blip_response_vs_energy.png"), dpi=150)
    plt.close(fig)


def plot_deex_multiplicity_and_energy(E, clean, n_blips, sum_blip_E, df, w_deex):
    sel = clean & (E < DEEX_SPECTRUM_MAX_MEV)
    w = w_deex[sel]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    bins_n = np.arange(-0.5, 8.5, 1)
    step_hist_with_errors(axes[0], n_blips[sel], bins_n, density=True,
                          color="#0072B2", label="clean < 12 MeV, unweighted")
    step_hist_with_errors(axes[0], n_blips[sel], bins_n, weights=w, density=True,
                          color="#D55E00", label="de-excitation weighted")
    axes[0].set_xlabel("Truth-matched blips per photon")
    axes[0].set_ylabel("Fraction of events")
    axes[0].legend(fontsize=9)

    has = sel & (n_blips > 0)
    bins_e = np.arange(0, 13, 0.5)
    step_hist_with_errors(axes[1], sum_blip_E[has], bins_e, density=True,
                          color="#0072B2", label="unweighted")
    step_hist_with_errors(axes[1], sum_blip_E[has], bins_e, weights=w_deex[has],
                          density=True, color="#D55E00", label="de-excitation weighted")
    axes[1].set_xlabel("Summed truth-matched blip energy [MeV]")
    axes[1].set_ylabel("Fraction of events with >= 1 blip / 0.5 MeV")
    axes[1].legend(fontsize=9)

    idx = np.where(has)[0]
    Eh = E[idx]
    sums = sum_blip_E[idx]
    axes[2].scatter(Eh, sums, s=4, alpha=0.4, color="#0072B2")
    axes[2].plot([0, 12], [0, 12], color="#999999", linestyle=":", linewidth=1)
    axes[2].set_xlabel("True photon energy [MeV]")
    axes[2].set_ylabel("Summed truth-matched blip energy [MeV]")
    axes[2].set_title("Per-event energy response (clean, >= 1 blip)")

    fig.suptitle("De-excitation-range blip response (clean truth selection, < 12 MeV)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "iso1g_deex_blip_multiplicity_energy.png"), dpi=150)
    plt.close(fig)


def plot_blip_distances(df, clean, w_deex):
    E = df["true_gamma_energy"].to_numpy()
    dist_lists = df["blip_dist_to_origin"].to_list()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bins = np.arange(0, 205, 10)
    for elo, ehi, color in [(0, 4, "#0072B2"), (4, 8, "#E69F00"), (8, 12, "#009E73"),
                            (12, 50, "#CC79A7")]:
        idx = np.where(clean & (E >= elo) & (E < ehi))[0]
        dists = [np.asarray(dist_lists[i]) for i in idx if dist_lists[i]]
        if not dists:
            continue
        dist = np.concatenate(dists)
        step_hist_with_errors(axes[0], dist, bins, density=True, color=color,
                              label=f"{elo}-{ehi} MeV ({len(dist)} blips)")
    axes[0].set_xlabel("Blip distance to photon origin [cm]")
    axes[0].set_ylabel("Area-normalized blips / 10 cm")
    axes[0].legend(fontsize=9, title="true photon energy")

    # de-excitation-weighted distance CDF, with the analysis sphere radii marked
    sel_idx = np.where(clean & (E < DEEX_SPECTRUM_MAX_MEV))[0]
    dist_w, dist_all = [], []
    for i in sel_idx:
        for d in dist_lists[i]:
            dist_all.append(d)
            dist_w.append(w_deex[i])
    dist_all = np.asarray(dist_all)
    dist_w = np.asarray(dist_w)
    order = np.argsort(dist_all)
    cdf = np.cumsum(dist_w[order]) / dist_w.sum()
    n_eff = dist_w.sum() ** 2 / (dist_w**2).sum()
    band = np.sqrt(np.clip(cdf * (1 - cdf), 0, None) / n_eff)
    axes[1].fill_between(dist_all[order], cdf - band, cdf + band,
                         color="#D55E00", alpha=0.25, linewidth=0,
                         label=f"$\\pm 1\\sigma$ ($N_{{eff}}$ = {n_eff:.0f} blips)")
    axes[1].plot(dist_all[order], cdf, color="#D55E00",
                 label="de-excitation weighted")
    for r in (25, 50, 75, 100):
        axes[1].axvline(r, color="#999999", linestyle=":", linewidth=1)
        frac = cdf[np.searchsorted(dist_all[order], r)] if (dist_all <= r).any() else 0
        axes[1].text(r, 0.03, f" {frac*100:.0f}% < {r} cm", rotation=90, fontsize=8,
                     va="bottom", color="#555555")
    axes[1].set_xlim(0, 200)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Blip distance to photon origin [cm]")
    axes[1].set_ylabel("Cumulative fraction of blips")
    axes[1].legend(fontsize=9)

    fig.suptitle("Truth-matched blip distance from the photon origin point "
                 "(clean truth selection)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "iso1g_blip_distance_to_origin.png"), dpi=150)
    plt.close(fig)


def plot_blip_3d_and_projections(df, clean):
    """Per-blip 3D scatter of (true photon energy, reco blip energy, reco blip
    distance to origin) for clean-selection de-excitation-range events, with
    all 2D and 1D projections. Unweighted, so every extracted blip is shown."""
    E = df["true_gamma_energy"].to_numpy()
    dist_lists = df["blip_dist_to_origin"].to_list()
    reco_lists = df["blip_energy"].to_list()
    idx = np.where(clean & (E < DEEX_SPECTRUM_MAX_MEV))[0]
    true_E = np.array([E[i] for i in idx for _ in dist_lists[i]])
    reco_E = np.array([e for i in idx for e in reco_lists[i]])
    dist = np.array([d for i in idx for d in dist_lists[i]])
    print(f"3D blip plot: {len(dist)} truth-matched blips from "
          f"{len(idx)} clean events < {DEEX_SPECTRUM_MAX_MEV:.0f} MeV")

    fig = plt.figure(figsize=(19, 8.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.6, 1, 1, 1])

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    sc = ax3d.scatter(true_E, reco_E, dist, c=dist, cmap="viridis", s=14,
                      alpha=0.85, linewidths=0)
    ax3d.set_xlabel("True photon energy [MeV]")
    ax3d.set_ylabel("Reco blip energy [MeV]")
    ax3d.set_zlabel("Blip distance to origin [cm]")
    fig.colorbar(sc, ax=ax3d, shrink=0.6, pad=0.1,
                 label="Blip distance to origin [cm]")

    # 2D projections
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(true_E, reco_E, s=8, alpha=0.6, color="#0072B2")
    ax.plot([0, 12], [0, 12], color="#999999", linestyle=":", linewidth=1)
    ax.set_xlabel("True photon energy [MeV]")
    ax.set_ylabel("Reco blip energy [MeV]")

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(true_E, dist, s=8, alpha=0.6, color="#0072B2")
    ax.set_xlabel("True photon energy [MeV]")
    ax.set_ylabel("Blip distance to origin [cm]")

    ax = fig.add_subplot(gs[0, 3])
    ax.scatter(reco_E, dist, s=8, alpha=0.6, color="#0072B2")
    ax.set_xlabel("Reco blip energy [MeV]")
    ax.set_ylabel("Blip distance to origin [cm]")

    # 1D projections
    for col, (vals, bins, xlabel) in enumerate([
        (true_E, np.arange(0, 12.5, 0.5), "True photon energy [MeV]"),
        (reco_E, np.arange(0, 12.5, 0.5), "Reco blip energy [MeV]"),
        (dist, np.arange(0, 210, 10), "Blip distance to origin [cm]"),
    ]):
        ax = fig.add_subplot(gs[1, col + 1])
        step_hist_with_errors(ax, vals, bins, color="#0072B2")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Blips / bin")

    fig.suptitle("Per-blip reco energy and distance vs true photon energy, "
                 "clean truth selection, < 12 MeV (unweighted)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "iso1g_deex_blip_3d_projections.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
