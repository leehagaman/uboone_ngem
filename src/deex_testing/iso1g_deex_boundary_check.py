"""Detector-boundary bias in the low-energy-photon blip response.

Photons generated near the TPC edge can leave the argon before Compton
scattering, which suppresses P(>= 1 blip) and truncates the blip
distance-to-origin distribution. The isotropic 1g sample is generated well
beyond the TPC, so the response can be measured as a function of the photon
origin's signed distance to the nearest TPC wall (d_wall, negative = outside),
and a "deep" subsample (d_wall >= DEEP_CM) gives boundary-unbiased true-energy
and distance distributions.

The boundary-unbiased ("clean") selection requires the photon origin to be at
least VTX_WALL_MIN_CM from every TPC wall AND the projected photon ray to have
at least RAY_IN_TPC_MIN_CM of path inside the TPC before exiting. The photon
direction comes from wc_true_leading_shower_costheta/phi in all_df.parquet
(costheta = pz/|p|, phi = arctan2(px, py) in degrees, see postprocessing.py).

Uses the per-event truth-matched-blip table produced by
iso1g_deex_blip_response.py (iso1g_lowE_truth_blips.parquet). De-excitation
weights here are computed over ALL events below 12 MeV (not just in-FV ones)
so the response can be followed all the way through the wall.

Run with:
    source ../uv_base/bin/activate
    python src/deex_testing/iso1g_deex_boundary_check.py
"""

# study script living in src/deex_testing/: put src/ on the path so the
# pipeline modules (file_locations, blip_postprocessing, ...) import cleanly
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from deexcitation_photon_study import PLOT_DIR
from iso1g_deex_stats_check import (
    load_deex_spectrum, DEEX_SPECTRUM_MAX_MEV, REWEIGHT_BINS_MEV)
from iso1g_deex_blip_response import (
    OUT_PARQUET, VTX_WALL_MIN_CM, RAY_IN_TPC_MIN_CM,
    signed_distance_to_wall, ray_length_in_tpc, load_directions,
    wilson_yerr, weighted_mean_sem, step_hist_with_errors)


def deex_weights_all(E):
    """De-excitation-spectrum weights for ALL events with E < 12 MeV."""
    deex_E = load_deex_spectrum()
    sel = E < DEEX_SPECTRUM_MAX_MEV
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
    df = pl.read_parquet(OUT_PARQUET)
    print(f"loaded {df.height:,} events from {OUT_PARQUET}")

    E = df["true_gamma_energy"].to_numpy()
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

    n_blips = df["blip_dist_to_origin"].list.len().to_numpy()
    sum_reco_E = df["blip_energy"].list.sum().fill_null(0.0).to_numpy()
    dist_lists = df["blip_dist_to_origin"].to_list()
    true_E_lists = df["blip_true_energy"].to_list()
    w_deex = deex_weights_all(E)

    deex = E < DEEX_SPECTRUM_MAX_MEV
    clean = (d_wall >= VTX_WALL_MIN_CM) & (ray_len >= RAY_IN_TPC_MIN_CM)
    print(f"\nevents < 12 MeV: {deex.sum()}  by origin d_wall:")
    for lo, hi in [(-1e9, 0), (0, 25), (25, 50), (50, 100), (100, 1e9)]:
        m = deex & (d_wall >= lo) & (d_wall < hi)
        lo_s = "outside" if lo < -1e8 else f"{lo:.0f}"
        print(f"  d_wall {lo_s:>7s} to {min(hi, 999):3.0f} cm: {m.sum():5d} events, "
              f"P(>=1 blip) = {(n_blips[m] > 0).mean()*100 if m.sum() else 0:5.1f}%")
    print(f"\nclean selection (d_wall >= {VTX_WALL_MIN_CM:.0f} cm and "
          f">= {RAY_IN_TPC_MIN_CM:.0f} cm of projected ray in the TPC):")
    print(f"  events < 12 MeV passing d_wall cut alone: "
          f"{(deex & (d_wall >= VTX_WALL_MIN_CM)).sum()}")
    print(f"  events < 12 MeV passing both cuts: {(deex & clean).sum()}")

    print_clean_summary(deex, clean, n_blips, w_deex, dist_lists, true_E_lists)
    plot_response_vs_dwall(d_wall, deex, n_blips, sum_reco_E, w_deex)
    plot_unbiased_true_energy_distance(E, d_wall, deex, clean, n_blips, w_deex,
                                       dist_lists, true_E_lists)
    print(f"plots saved to {os.path.abspath(PLOT_DIR)}")


def print_clean_summary(deex, clean, n_blips, w_deex, dist_lists, true_E_lists):
    print("\nde-excitation-weighted response, clean (boundary-unbiased) selection:")
    sel_clean = np.where(deex & clean)[0]
    w = w_deex[sel_clean]
    p1 = np.average(n_blips[sel_clean] > 0, weights=w)
    mult = np.average(n_blips[sel_clean], weights=w)
    print(f"  clean sample ({len(sel_clean)} events): P(>=1 blip) = {p1*100:.1f}%, "
          f"mean multiplicity = {mult:.2f}")
    dist = np.array([d for i in sel_clean for d in dist_lists[i]])
    dw = np.array([w_deex[i] for i in sel_clean for _ in dist_lists[i]])
    if len(dist):
        order = np.argsort(dist)
        cdf = np.cumsum(dw[order]) / dw.sum()
        med = dist[order][np.searchsorted(cdf, 0.5)]
        p90 = dist[order][np.searchsorted(cdf, 0.9)]
        print(f"  clean-sample blip distance to origin: median {med:.1f} cm, "
              f"90% within {p90:.1f} cm")
        for r in (25, 50, 75, 100):
            frac = cdf[min(np.searchsorted(dist[order], r), len(cdf) - 1)]
            print(f"    fraction within {r:3d} cm: {frac*100:.0f}%")
    tE = np.array([te for i in sel_clean for te in true_E_lists[i]])
    if len(tE):
        print(f"  clean-sample per-blip true deposited energy: median {np.median(tE):.2f} MeV")


def plot_response_vs_dwall(d_wall, deex, n_blips, sum_reco_E, w_deex):
    bins = np.array([-100, -50, -25, 0, 25, 50, 75, 100, 150, 200, 300, 500])
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    p1 = np.full(len(centers), np.nan)
    mult = np.full(len(centers), np.nan)
    mult_err = np.full(len(centers), np.nan)
    n_eff = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        s = deex & (d_wall >= bins[i]) & (d_wall < bins[i + 1])
        if s.sum() < 20:
            continue
        w = w_deex[s]
        p1[i] = np.average(n_blips[s] > 0, weights=w)
        n_eff[i] = w.sum() ** 2 / (w**2).sum()
        mult[i], mult_err[i] = weighted_mean_sem(n_blips[s], w)
    axes[0].errorbar(centers, p1, yerr=wilson_yerr(p1, n_eff), marker="o",
                     color="#0072B2", label="P(>= 1 truth-matched blip)")
    axes[0].errorbar(centers, mult, yerr=mult_err, marker="s", color="#D55E00",
                     label="mean blip multiplicity")
    axes[0].axvline(0, color="#999999", linestyle="--", linewidth=1)
    axes[0].text(0, 0.97, " TPC wall", transform=axes[0].get_xaxis_transform(),
                 va="top", color="#555555", fontsize=9)
    axes[0].axvspan(VTX_WALL_MIN_CM, bins[-1], color="#009E73", alpha=0.12)
    axes[0].text(0.98, 0.62, "clean-selection\nd_wall range", transform=axes[0].transAxes,
                 ha="right", color="#00744F")
    axes[0].set_xlabel("Photon origin signed distance to nearest TPC wall [cm]")
    axes[0].set_ylabel("De-excitation-weighted response")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()

    # where the events actually sit in d_wall (weighting relevance)
    bins_dw = np.arange(-200, 520, 20)
    step_hist_with_errors(axes[1], d_wall[deex], bins_dw,
                          color="#000000", label="photons < 12 MeV, all")
    step_hist_with_errors(axes[1], d_wall[deex & (n_blips > 0)], bins_dw,
                          color="#0072B2", label="with >= 1 blip")
    axes[1].axvline(0, color="#999999", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Photon origin signed distance to nearest TPC wall [cm]")
    axes[1].set_ylabel("Events / 20 cm")
    axes[1].legend(fontsize=9)

    fig.suptitle("Boundary dependence of the de-excitation-photon blip response")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "iso1g_blip_response_vs_dwall.png"), dpi=150)
    plt.close(fig)


def plot_unbiased_true_energy_distance(E, d_wall, deex, clean, n_blips, w_deex,
                                       dist_lists, true_E_lists):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # efficiency turn-on vs true photon energy: clean selection vs near-wall
    bins_E = np.arange(0, 13, 1.5)
    centers_E = 0.5 * (bins_E[:-1] + bins_E[1:])
    for mask, color, label in [
        (clean, "#009E73",
         f"clean (d_wall >= {VTX_WALL_MIN_CM:.0f} cm, ray >= {RAY_IN_TPC_MIN_CM:.0f} cm)"),
        ((d_wall >= 0) & (d_wall < 50), "#CC79A7", "near wall (0-50 cm)"),
    ]:
        eff = np.full(len(centers_E), np.nan)
        ns = np.full(len(centers_E), np.nan)
        for i in range(len(centers_E)):
            s = mask & (E >= bins_E[i]) & (E < bins_E[i + 1])
            if s.sum() >= 20:
                eff[i] = (n_blips[s] > 0).mean()
                ns[i] = s.sum()
        axes[0].errorbar(centers_E, eff, yerr=wilson_yerr(eff, ns), marker="o",
                         color=color, label=label)
    axes[0].set_xlabel("True photon energy [MeV]")
    axes[0].set_ylabel("P(>= 1 truth-matched blip)")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=9)

    # blip distance to origin: clean selection vs near-wall (de-excitation weighted)
    bins_d = np.arange(0, 205, 10)
    for mask, color, label in [
        (clean, "#009E73", "clean selection"),
        ((d_wall >= 0) & (d_wall < 50), "#CC79A7", "near wall (0-50 cm)"),
    ]:
        idx = np.where(deex & mask)[0]
        dist = np.array([d for i in idx for d in dist_lists[i]])
        dw = np.array([w_deex[i] for i in idx for _ in dist_lists[i]])
        if len(dist) < 50:
            continue
        step_hist_with_errors(axes[1], dist, bins_d, weights=dw, density=True,
                              color=color, label=f"{label} ({len(dist)} blips)")
    axes[1].set_xlabel("Blip distance to photon origin [cm]")
    axes[1].set_ylabel("Area-normalized blips / 10 cm (de-exc. weighted)")
    axes[1].legend(fontsize=9)

    # per-blip true deposited energy, clean selection, de-excitation weighted
    idx = np.where(deex & clean)[0]
    tE = np.array([te for i in idx for te in true_E_lists[i]])
    tw = np.array([w_deex[i] for i in idx for _ in true_E_lists[i]])
    step_hist_with_errors(axes[2], tE, np.arange(0, 6.25, 0.25), weights=tw,
                          density=True, color="#009E73",
                          label="true deposited energy")
    axes[2].set_xlabel("Per-blip true deposited energy [MeV]")
    axes[2].set_ylabel("Area-normalized blips / 0.25 MeV (de-exc. weighted)")
    axes[2].set_title("Clean selection, boundary-unbiased")
    axes[2].legend(fontsize=9)

    fig.suptitle("Boundary-unbiased de-excitation blip response: "
                 "true energy and distance to origin")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "iso1g_deex_unbiased_true_energy_distance.png"),
                dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
