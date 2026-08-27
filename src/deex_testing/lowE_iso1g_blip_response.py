"""Blip response from the dedicated low-energy isotropic 1-gamma production.

Full-statistics version of lowE_iso1g_blip_check.py, mirroring the plot set of
iso1g_deex_blip_response.py (which used the sparse low-E tail of the big
run-4/5 isotropic sample): efficiency turn-on, multiplicity, energy response,
blip distance to the photon origin, and the per-blip 3D scatter with
projections. All response plots use the clean truth selection (photon origin
>= 50 cm from every TPC wall and >= 1 m of projected ray in the TPC), with the
photon direction taken directly from the true momentum.

The 1000-job sample has photons generated ~flat in 0-15 MeV, isotropic,
vertices throughout the TPC. Truth from wcpselection/T_PFeval (primary photon,
truth_mother==0); truth-matched blips (blip_true_g4id >= 0) from
nuselection/NeutrinoSelectionFilter; alignment verified before trusting row
order. The per-event table is saved to intermediate_files as
lowE_iso1g_truth_blips.parquet.

De-excitation-weighted numbers are reported for both emission models: the old
GENIE AR23 NucDeExcitationSim spectrum (also used in the weighted plots, for
continuity with the earlier study) and the GENIE+INCL+MARLEY spectrum.

Run with:
    source ../uv_base/bin/activate
    python src/deex_testing/lowE_iso1g_blip_response.py
"""

# study script living in src/deex_testing/: put src/ on the path so the
# pipeline modules (file_locations, blip_postprocessing, ...) import cleanly
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import uproot

from check_ntuple_alignment import check_ntuple_alignment
from file_locations import intermediate_files_location
from iso1g_deex_stats_check import load_deex_spectrum, DEEX_SPECTRUM_MAX_MEV
from iso1g_deex_blip_response import (
    OUT_PARQUET as BIG_SAMPLE_PARQUET, clean_selection_mask,
    signed_distance_to_wall, ray_length_in_tpc, VTX_WALL_MIN_CM,
    RAY_IN_TPC_MIN_CM, wilson_yerr, weighted_mean_sem, step_hist_with_errors)
from deexcitation_photon_study_marley import (
    GST_FILE as MARLEY_GST_FILE, DEEX_E_MAX as MARLEY_DEEX_E_MAX)

NTUPLE = ("/nevis/riverside/data/leehagaman/ngem/other_files/lowE_1g_files/"
          "lowE_iso1g_1000job_ntuple.root")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "plots", "lowE_iso1g")
OUT_TABLE = os.path.join(intermediate_files_location, "lowE_iso1g_truth_blips.parquet")

E_MAX_MEV = 15.0     # plots cover the full generated range
REWEIGHT_BINS = np.arange(0.0, DEEX_SPECTRUM_MAX_MEV + 1e-9, 0.5)
SPHERE_RADIUS_CM = 75.0  # analysis sphere radius for the contained-efficiency curve


def load_sample():
    f = uproot.open(NTUPLE)
    bad = [r["tree"] for r in check_ntuple_alignment(f)
           if not r["aligned"] and "eventweight" not in r["tree"]]
    if bad:
        raise RuntimeError(f"misaligned trees in {NTUPLE}: {bad}")

    pf = f["wcpselection/T_PFeval"].arrays(
        ["truth_pdg", "truth_mother", "truth_startMomentum", "truth_startXYZT"])
    prim = (pf["truth_pdg"] == 22) & (pf["truth_mother"] == 0)
    assert np.all(np.asarray(ak.sum(prim, axis=1)) == 1)
    mom = np.asarray(ak.flatten(pf["truth_startMomentum"][prim]))
    pos = np.asarray(ak.flatten(pf["truth_startXYZT"][prim]))
    E = mom[:, 3] * 1000  # MeV
    p = np.linalg.norm(mom[:, :3], axis=1)
    u = mom[:, :3] / p[:, None]
    vtx = pos[:, :3]

    ns = f["nuselection/NeutrinoSelectionFilter"].arrays(
        ["blip_x", "blip_y", "blip_z", "blip_energy", "blip_nplanes",
         "blip_true_pdg", "blip_true_energy", "blip_true_g4id"])
    matched = ns["blip_true_g4id"] >= 0
    bx, by, bz = (ns[k][matched] for k in ("blip_x", "blip_y", "blip_z"))
    dist = np.sqrt((bx - vtx[:, 0]) ** 2 + (by - vtx[:, 1]) ** 2
                   + (bz - vtx[:, 2]) ** 2)

    df = pl.DataFrame({
        "true_gamma_energy": E,
        "true_vtx_x": vtx[:, 0], "true_vtx_y": vtx[:, 1], "true_vtx_z": vtx[:, 2],
        "true_dir_x": u[:, 0], "true_dir_y": u[:, 1], "true_dir_z": u[:, 2],
        "blip_dist_to_origin": dist.to_list(),
        "blip_energy": ns["blip_energy"][matched].to_list(),
        "blip_nplanes": ns["blip_nplanes"][matched].to_list(),
        "blip_true_pdg": ns["blip_true_pdg"][matched].to_list(),
        "blip_true_energy": ns["blip_true_energy"][matched].to_list(),
    })
    df.write_parquet(OUT_TABLE)
    print(f"loaded {df.height:,} events "
          f"({int(ak.sum(matched)):,} truth-matched blips); table saved to {OUT_TABLE}")
    return df


def clean_mask(df):
    d_wall = signed_distance_to_wall(df["true_vtx_x"].to_numpy(),
                                     df["true_vtx_y"].to_numpy(),
                                     df["true_vtx_z"].to_numpy())
    ray = ray_length_in_tpc(df["true_vtx_x"].to_numpy(),
                            df["true_vtx_y"].to_numpy(),
                            df["true_vtx_z"].to_numpy(),
                            df["true_dir_x"].to_numpy(),
                            df["true_dir_y"].to_numpy(),
                            df["true_dir_z"].to_numpy())
    return (d_wall >= VTX_WALL_MIN_CM) & (ray >= RAY_IN_TPC_MIN_CM)


def load_marley_spectrum():
    """De-excitation photon energies [MeV] from the GENIE+INCL+MARLEY gst."""
    tree = uproot.open(MARLEY_GST_FILE)["gst"]
    arrays = tree.arrays(["pdgf", "Ef"])
    photon_E = arrays["Ef"][arrays["pdgf"] == 22]
    return np.asarray(ak.flatten(photon_E[photon_E < MARLEY_DEEX_E_MAX])) * 1000


def spectrum_weights(E, sel, target_E):
    """Per-event weights so the selected events match the target spectrum
    shape in REWEIGHT_BINS; target photons outside the bins are dropped."""
    src_counts, _ = np.histogram(E[sel], bins=REWEIGHT_BINS)
    tgt_counts, _ = np.histogram(target_E, bins=REWEIGHT_BINS)
    with np.errstate(divide="ignore", invalid="ignore"):
        bin_w = np.where(src_counts > 0,
                         (tgt_counts / tgt_counts.sum()) / (src_counts / src_counts.sum()),
                         0.0)
    w = np.zeros(len(E))
    w[sel] = bin_w[np.digitize(E[sel], REWEIGHT_BINS) - 1]
    return w


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = load_sample()
    E = df["true_gamma_energy"].to_numpy()
    clean = clean_mask(df)
    n_blips = df["blip_dist_to_origin"].list.len().to_numpy()
    sum_blip_E = df["blip_energy"].list.sum().fill_null(0.0).to_numpy()
    dist_lists = df["blip_dist_to_origin"].to_list()

    n_blips_sphere = np.array([
        int(np.sum(np.asarray(d) < SPHERE_RADIUS_CM)) if d else 0
        for d in dist_lists
    ])
    n_blips_3pl = np.array([
        sum(1 for n in l if n == 3) for l in df["blip_nplanes"].to_list()
    ])

    sel = clean & (E < DEEX_SPECTRUM_MAX_MEV)
    print(f"clean truth selection: {clean.sum():,} of {df.height:,} events, "
          f"{sel.sum():,} below {DEEX_SPECTRUM_MAX_MEV:.0f} MeV")

    w_genie = spectrum_weights(E, sel, load_deex_spectrum())
    w_marley = spectrum_weights(E, sel, load_marley_spectrum())

    plateau = clean & (E >= 3) & (E < E_MAX_MEV)
    print("\nper-photon blip efficiency (clean selection):")
    print(f"  plateau (3-{E_MAX_MEV:.0f} MeV): "
          f"{(n_blips[plateau] > 0).mean()*100:.1f}% any blip, "
          f"{(n_blips_sphere[plateau] > 0).mean()*100:.1f}% within "
          f"{SPHERE_RADIUS_CM:.0f} cm ({plateau.sum()} events)")
    for name, w in [("AR23-weighted", w_genie), ("MARLEY-weighted", w_marley)]:
        ww = w[sel]
        print(f"  {name}: {np.average(n_blips[sel] > 0, weights=ww)*100:.1f}% "
              f"any blip, "
              f"{np.average(n_blips_sphere[sel] > 0, weights=ww)*100:.1f}% "
              f"within {SPHERE_RADIUS_CM:.0f} cm")

    print("\nde-excitation-weighted response (clean selection, < 12 MeV):")
    for name, w in [("old GENIE AR23", w_genie), ("GENIE+INCL+MARLEY", w_marley)]:
        ww = w[sel]
        n_eff = ww.sum() ** 2 / (ww**2).sum()
        p1 = np.average(n_blips[sel] > 0, weights=ww)
        mult = np.average(n_blips[sel], weights=ww)
        has = sel & (n_blips > 0)
        meanE = np.average(sum_blip_E[has], weights=w[has])
        print(f"  {name:18s}: P(>=1 blip) = {p1*100:.1f}%, mean mult = {mult:.2f}, "
              f"mean summed E = {meanE:.2f} MeV (N_eff = {n_eff:.0f})")

    plot_response_vs_energy(E, clean, n_blips, sum_blip_E)
    plot_efficiency(E, clean, n_blips, n_blips_sphere, n_blips_3pl)
    plot_multiplicity_fractions_vs_energy(E, clean, n_blips)
    plot_deex_multiplicity_and_energy(E, clean, n_blips, sum_blip_E, w_genie)
    plot_blip_distances(E, clean, dist_lists, w_genie)
    plot_blip_3d_and_projections(E, clean, df, n_blips)
    print(f"plots saved to {os.path.abspath(PLOT_DIR)}")


def big_sample_turnon(bins):
    """Clean-selection turn-on from the big run-4/5 isotropic sample."""
    df = pl.read_parquet(BIG_SAMPLE_PARQUET)
    E = df["true_gamma_energy"].to_numpy()
    clean = clean_selection_mask(df)
    n_blips = df["blip_dist_to_origin"].list.len().to_numpy()
    eff = np.full(len(bins) - 1, np.nan)
    ns = np.full(len(bins) - 1, np.nan)
    for i in range(len(eff)):
        s = clean & (E >= bins[i]) & (E < bins[i + 1])
        if s.sum() >= 10:
            eff[i] = (n_blips[s] > 0).mean()
            ns[i] = s.sum()
    return eff, ns


def plot_response_vs_energy(E, clean, n_blips, sum_blip_E):
    bins = np.arange(0, E_MAX_MEV + 1e-9, 1.0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    p1, mult, mult_err, meanE, meanE_err, ns = \
        (np.full(len(centers), np.nan) for _ in range(6))
    for i in range(len(centers)):
        s = clean & (E >= bins[i]) & (E < bins[i + 1])
        if s.sum() < 20:
            continue
        ns[i] = s.sum()
        p1[i] = (n_blips[s] > 0).mean()
        mult[i], mult_err[i] = weighted_mean_sem(n_blips[s])
        h = s & (n_blips > 0)
        if h.sum() >= 10:
            meanE[i], meanE_err[i] = weighted_mean_sem(sum_blip_E[h])

    # the big-sample overlay needs all_df.parquet (for the photon directions);
    # skip the comparison gracefully while a production is in flux
    try:
        eff_big, ns_big = big_sample_turnon(bins)
        axes[0].errorbar(centers, eff_big, yerr=wilson_yerr(eff_big, ns_big),
                         marker="s", markersize=4, color="#999999",
                         label="big run-4/5 sample (clean)")
    except Exception as e:
        print(f"  skipping big-sample comparison overlay ({type(e).__name__}: {e})")
    axes[0].errorbar(centers, p1, yerr=wilson_yerr(p1, ns), marker="o",
                     markersize=4, color="#009E73", label="low-E sample (clean)")
    axes[0].set_ylabel("P(>= 1 truth-matched blip)")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=9)
    axes[1].errorbar(centers, mult, yerr=mult_err, marker="o", markersize=4,
                     color="#009E73")
    axes[1].set_ylabel("Mean truth-matched blip multiplicity")
    axes[1].set_ylim(bottom=0)
    axes[2].errorbar(centers, meanE, yerr=meanE_err, marker="o", markersize=4,
                     color="#009E73")
    axes[2].plot([0, E_MAX_MEV], [0, E_MAX_MEV], color="#999999",
                 linestyle=":", linewidth=1)
    axes[2].set_ylabel("Mean summed blip energy [MeV]\n(events with >= 1 blip)")
    axes[2].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("True photon energy [MeV]")
    fig.suptitle("Truth-matched blip response vs photon energy, low-E isotropic "
                 f"1$\\gamma$ production, clean truth selection "
                 f"(d_wall >= {VTX_WALL_MIN_CM:.0f} cm, ray >= "
                 f"{RAY_IN_TPC_MIN_CM:.0f} cm in TPC)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "lowE_iso1g_response_vs_energy.png"), dpi=150)
    plt.close(fig)



def plot_multiplicity_fractions_vs_energy(E, clean, n_blips):
    """How often a true photon reconstructs as 0, 1, 2, 3, or >= 4
    truth-matched blips, as a function of true photon energy."""
    bins = np.arange(0, E_MAX_MEV + 1e-9, 1.0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    categories = [(0, "0 blips", "#999999"), (1, "1 blip", "#0072B2"),
                  (2, "2 blips", "#E69F00"), (3, "3 blips", "#009E73"),
                  (4, ">= 4 blips", "#D55E00")]
    n_capped = np.minimum(n_blips, 4)

    fracs = {n: np.full(len(centers), np.nan) for n, _, _ in categories}
    ns_bin = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        sel = clean & (E >= bins[i]) & (E < bins[i + 1])
        if sel.sum() < 20:
            continue
        ns_bin[i] = sel.sum()
        for n, _, _ in categories:
            fracs[n][i] = (n_capped[sel] == n).mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    for n, label, color in categories:
        axes[0].errorbar(centers, fracs[n], yerr=wilson_yerr(fracs[n], ns_bin),
                         marker="o", markersize=4, color=color, label=label)
    axes[0].set_xlabel("True photon energy [MeV]")
    axes[0].set_ylabel("Fraction of photons")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=9)

    bottom = np.zeros(len(centers))
    for n, label, color in categories:
        vals = np.nan_to_num(fracs[n])
        axes[1].bar(centers, vals, width=np.diff(bins), bottom=bottom,
                    color=color, label=label, edgecolor="white", linewidth=0.4)
        bottom += vals
    axes[1].set_xlabel("True photon energy [MeV]")
    axes[1].set_ylabel("Fraction of photons (stacked)")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend(fontsize=9, loc="center right")

    fig.suptitle("Truth-matched blip multiplicity fractions vs true photon "
                 "energy, low-E isotropic 1$\\gamma$ production (clean truth "
                 "selection)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "lowE_iso1g_multiplicity_vs_energy.png"),
                dpi=150)
    plt.close(fig)


def plot_efficiency(E, clean, n_blips, n_blips_sphere, n_blips_3pl):
    """Per-photon efficiency: how often a true photon has >= 1 truth-matched
    reco blip, finely binned. All saved blips have nplanes of 2 or 3, and the
    analysis accepts both (blip_postprocessing requires nplanes > 1), so the
    default curves count both; the 3-plane-only curve shows the high-purity
    subset."""
    bins = np.arange(0, E_MAX_MEV + 1e-9, 0.5)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for counts, color, marker, label in [
        (n_blips, "#009E73", "o", "any truth-matched blip (2 or 3 planes)"),
        (n_blips_sphere, "#0072B2", "s",
         f"truth-matched blip within {SPHERE_RADIUS_CM:.0f} cm of origin"),
        (n_blips_3pl, "#CC79A7", "^", "3-plane truth-matched blip only"),
    ]:
        eff = np.full(len(centers), np.nan)
        ns = np.full(len(centers), np.nan)
        for i in range(len(centers)):
            s = clean & (E >= bins[i]) & (E < bins[i + 1])
            if s.sum() >= 20:
                eff[i] = (counts[s] > 0).mean()
                ns[i] = s.sum()
        ax.errorbar(centers, eff, yerr=wilson_yerr(eff, ns), marker=marker,
                    markersize=4, color=color, label=label)
    plateau = clean & (E >= 3) & (E < E_MAX_MEV)
    ax.axhline((n_blips[plateau] > 0).mean(), color="#009E73", linestyle=":",
               linewidth=1, alpha=0.7)
    ax.text(0.98, (n_blips[plateau] > 0).mean() + 0.015,
            f"plateau {((n_blips[plateau] > 0).mean())*100:.1f}%",
            transform=ax.get_yaxis_transform(), ha="right", fontsize=9,
            color="#00744F")
    ax.set_xlabel("True photon energy [MeV]")
    ax.set_ylabel("Per-photon blip efficiency")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Efficiency for a true photon to have truth-matched reco blips\n"
                 "(clean truth selection, 0.5 MeV bins)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "lowE_iso1g_efficiency.png"), dpi=150)
    plt.close(fig)


def plot_deex_multiplicity_and_energy(E, clean, n_blips, sum_blip_E, w_deex):
    sel = clean & (E < E_MAX_MEV)
    w = w_deex[sel]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    bins_n = np.arange(-0.5, 8.5, 1)
    step_hist_with_errors(axes[0], n_blips[sel], bins_n, density=True,
                          color="#0072B2",
                          label=f"clean < {E_MAX_MEV:.0f} MeV, unweighted")
    step_hist_with_errors(axes[0], n_blips[sel], bins_n, weights=w, density=True,
                          color="#D55E00", label="de-excitation weighted (AR23)")
    axes[0].set_xlabel("Truth-matched blips per photon")
    axes[0].set_ylabel("Fraction of events")
    axes[0].legend(fontsize=9)

    has = sel & (n_blips > 0)
    bins_e = np.arange(0, E_MAX_MEV + 0.5, 0.5)
    step_hist_with_errors(axes[1], sum_blip_E[has], bins_e, density=True,
                          color="#0072B2", label="unweighted")
    step_hist_with_errors(axes[1], sum_blip_E[has], bins_e, weights=w_deex[has],
                          density=True, color="#D55E00",
                          label="de-excitation weighted (AR23)")
    axes[1].set_xlabel("Summed truth-matched blip energy [MeV]")
    axes[1].set_ylabel("Fraction of events with >= 1 blip / 0.5 MeV")
    axes[1].legend(fontsize=9)

    axes[2].scatter(E[has], sum_blip_E[has], s=3, alpha=0.35, color="#0072B2")
    axes[2].plot([0, E_MAX_MEV], [0, E_MAX_MEV], color="#999999",
                 linestyle=":", linewidth=1)
    axes[2].set_xlabel("True photon energy [MeV]")
    axes[2].set_ylabel("Summed truth-matched blip energy [MeV]")
    axes[2].set_title("Per-event energy response (clean, >= 1 blip)")

    fig.suptitle("Blip response, low-E isotropic 1$\\gamma$ production "
                 f"(clean truth selection, < {E_MAX_MEV:.0f} MeV; AR23 weights "
                 "cover < 12 MeV)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "lowE_iso1g_multiplicity_energy.png"), dpi=150)
    plt.close(fig)


def plot_blip_distances(E, clean, dist_lists, w_deex):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bins = np.arange(0, 205, 5)
    for elo, ehi, color in [(0, 4, "#0072B2"), (4, 8, "#E69F00"),
                            (8, 12, "#009E73"), (12, 15, "#CC79A7")]:
        idx = np.where(clean & (E >= elo) & (E < ehi))[0]
        dists = [np.asarray(dist_lists[i]) for i in idx if dist_lists[i]]
        if not dists:
            continue
        dist = np.concatenate(dists)
        step_hist_with_errors(axes[0], dist, bins, density=True, color=color,
                              label=f"{elo}-{ehi} MeV ({len(dist)} blips)")
    axes[0].set_xlabel("Blip distance to photon origin [cm]")
    axes[0].set_ylabel("Area-normalized blips / 5 cm")
    axes[0].legend(fontsize=9, title="true photon energy")

    sel_idx = np.where(clean & (E < DEEX_SPECTRUM_MAX_MEV))[0]
    dist_all = np.array([d for i in sel_idx for d in dist_lists[i]])
    dist_w = np.array([w_deex[i] for i in sel_idx for _ in dist_lists[i]])
    order = np.argsort(dist_all)
    cdf = np.cumsum(dist_w[order]) / dist_w.sum()
    n_eff = dist_w.sum() ** 2 / (dist_w**2).sum()
    band = np.sqrt(np.clip(cdf * (1 - cdf), 0, None) / n_eff)
    axes[1].fill_between(dist_all[order], cdf - band, cdf + band,
                         color="#D55E00", alpha=0.25, linewidth=0,
                         label=f"$\\pm 1\\sigma$ ($N_{{eff}}$ = {n_eff:.0f} blips)")
    axes[1].plot(dist_all[order], cdf, color="#D55E00",
                 label="de-excitation weighted (AR23)")
    for r in (25, 50, 75, 100):
        axes[1].axvline(r, color="#999999", linestyle=":", linewidth=1)
        frac = cdf[min(np.searchsorted(dist_all[order], r), len(cdf) - 1)]
        axes[1].text(r, 0.03, f" {frac*100:.0f}% < {r} cm", rotation=90,
                     fontsize=8, va="bottom", color="#555555")
    axes[1].set_xlim(0, 200)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Blip distance to photon origin [cm]")
    axes[1].set_ylabel("Cumulative fraction of blips")
    axes[1].legend(fontsize=9)

    fig.suptitle("Truth-matched blip distance from the photon origin, low-E "
                 "isotropic 1$\\gamma$ production (clean truth selection)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "lowE_iso1g_blip_distance_to_origin.png"),
                dpi=150)
    plt.close(fig)


def plot_blip_3d_and_projections(E, clean, df, n_blips):
    dist_lists = df["blip_dist_to_origin"].to_list()
    reco_lists = df["blip_energy"].to_list()
    idx = np.where(clean & (E < E_MAX_MEV))[0]
    true_E = np.array([E[i] for i in idx for _ in dist_lists[i]])
    reco_E = np.array([e for i in idx for e in reco_lists[i]])
    dist = np.array([d for i in idx for d in dist_lists[i]])
    print(f"3D blip plot: {len(dist)} truth-matched blips from "
          f"{len(idx)} clean events < {E_MAX_MEV:.0f} MeV")

    fig = plt.figure(figsize=(19, 8.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.6, 1, 1, 1])

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    sc = ax3d.scatter(true_E, reco_E, dist, c=dist, cmap="viridis", s=6,
                      alpha=0.6, linewidths=0)
    ax3d.set_xlabel("True photon energy [MeV]")
    ax3d.set_ylabel("Reco blip energy [MeV]")
    ax3d.set_zlabel("Blip distance to origin [cm]")
    fig.colorbar(sc, ax=ax3d, shrink=0.6, pad=0.1,
                 label="Blip distance to origin [cm]")

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(true_E, reco_E, s=3, alpha=0.35, color="#0072B2")
    ax.plot([0, E_MAX_MEV], [0, E_MAX_MEV], color="#999999", linestyle=":", linewidth=1)
    ax.set_xlabel("True photon energy [MeV]")
    ax.set_ylabel("Reco blip energy [MeV]")

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(true_E, dist, s=3, alpha=0.35, color="#0072B2")
    ax.set_xlabel("True photon energy [MeV]")
    ax.set_ylabel("Blip distance to origin [cm]")

    ax = fig.add_subplot(gs[0, 3])
    ax.scatter(reco_E, dist, s=3, alpha=0.35, color="#0072B2")
    ax.set_xlabel("Reco blip energy [MeV]")
    ax.set_ylabel("Blip distance to origin [cm]")

    for col, (vals, bins, xlabel) in enumerate([
        (true_E, np.arange(0, E_MAX_MEV + 0.5, 0.5), "True photon energy [MeV]"),
        (reco_E, np.arange(0, E_MAX_MEV + 0.25, 0.25), "Reco blip energy [MeV]"),
        (dist, np.arange(0, 210, 5), "Blip distance to origin [cm]"),
    ]):
        ax = fig.add_subplot(gs[1, col + 1])
        step_hist_with_errors(ax, vals, bins, color="#0072B2")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Blips / bin")

    fig.suptitle("Per-blip reco energy and distance vs true photon energy, "
                 "low-E isotropic 1$\\gamma$ production, clean truth selection, "
                 f"< {E_MAX_MEV:.0f} MeV (unweighted)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "lowE_iso1g_blip_3d_projections.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
