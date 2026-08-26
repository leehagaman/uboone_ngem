"""First look at the dedicated low-energy isotropic single-photon sample.

Checks the 10-job test ntuple (132 events, photons generated ~uniformly in
0-15 MeV, isotropic, vertices throughout the TPC):
- truth distributions: photon energy, direction (costheta, phi), and vertex
  distance to the nearest TPC wall,
- truth-matched reco blips (blip_true_g4id >= 0, all photon-induced since the
  cosmic overlay has no truth): efficiency turn-on vs true energy compared to
  the big run-4/5 isotropic sample, multiplicity, energy response, and
  distance to the photon origin.

Truth comes from wcpselection/T_PFeval (primary photon: truth_pdg==22 with
truth_mother==0); blips from nuselection/NeutrinoSelectionFilter. Tree
alignment is verified with check_ntuple_alignment before trusting row order.

Run with:
    source ../uv_base/bin/activate
    python src/deex_testing/lowE_iso1g_blip_check.py
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
from iso1g_deex_blip_response import (
    OUT_PARQUET, signed_distance_to_wall, wilson_yerr, step_hist_with_errors)

NTUPLE = ("/nevis/riverside/data/leehagaman/ngem/other_files/lowE_1g_files/"
          "lowE_iso1g_10job_test_ntuple.root")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "plots", "lowE_iso1g")

E_MAX_MEV = 15.0
EFF_BINS = np.arange(0, E_MAX_MEV + 1e-9, 2.5)


def load_new_sample():
    f = uproot.open(NTUPLE)
    bad = [r["tree"] for r in check_ntuple_alignment(f)
           if not r["aligned"] and "eventweight" not in r["tree"]]
    if bad:
        raise RuntimeError(f"misaligned trees in {NTUPLE}: {bad}")

    pf = f["wcpselection/T_PFeval"].arrays(
        ["truth_pdg", "truth_mother", "truth_startMomentum", "truth_startXYZT"])
    prim = (pf["truth_pdg"] == 22) & (pf["truth_mother"] == 0)
    mom = np.asarray(ak.flatten(pf["truth_startMomentum"][prim]))
    pos = np.asarray(ak.flatten(pf["truth_startXYZT"][prim]))
    E = mom[:, 3] * 1000  # MeV
    p = np.linalg.norm(mom[:, :3], axis=1)
    costheta = mom[:, 2] / p
    phi = np.degrees(np.arctan2(mom[:, 0], mom[:, 1]))  # postprocessing.py convention
    vtx = pos[:, :3]

    ns = f["nuselection/NeutrinoSelectionFilter"].arrays(
        ["blip_x", "blip_y", "blip_z", "blip_energy", "blip_true_g4id"])
    matched = ns["blip_true_g4id"] >= 0
    bx, by, bz = (ns[k][matched] for k in ("blip_x", "blip_y", "blip_z"))
    bE = ns["blip_energy"][matched]
    dist = np.sqrt((bx - vtx[:, 0]) ** 2 + (by - vtx[:, 1]) ** 2
                   + (bz - vtx[:, 2]) ** 2)
    print(f"loaded {len(E)} events; {int(ak.sum(matched))} truth-matched blips")
    return E, costheta, phi, vtx, bE, dist


def old_sample_turnon():
    """Blip turn-on from the big run-4/5 isotropic sample, in-TPC origins
    (matching the new file's generation volume)."""
    df = pl.read_parquet(OUT_PARQUET)
    E = df["true_gamma_energy"].to_numpy()
    d_wall = signed_distance_to_wall(df["true_vtx_x"].to_numpy(),
                                    df["true_vtx_y"].to_numpy(),
                                    df["true_vtx_z"].to_numpy())
    n_blips = df["blip_dist_to_origin"].list.len().to_numpy()
    in_tpc = d_wall > 0
    eff = np.full(len(EFF_BINS) - 1, np.nan)
    ns = np.full(len(EFF_BINS) - 1, np.nan)
    for i in range(len(eff)):
        s = in_tpc & (E >= EFF_BINS[i]) & (E < EFF_BINS[i + 1])
        if s.sum() >= 20:
            eff[i] = (n_blips[s] > 0).mean()
            ns[i] = s.sum()
    return eff, ns


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    E, costheta, phi, vtx, bE, dist = load_new_sample()
    d_wall = signed_distance_to_wall(vtx[:, 0], vtx[:, 1], vtx[:, 2])
    n_blips = np.asarray(ak.num(bE))

    print(f"true E: min {E.min():.2f}, median {np.median(E):.2f}, "
          f"max {E.max():.2f} MeV")
    print(f"events with >= 1 matched blip: {(n_blips > 0).sum()} / {len(E)} "
          f"({(n_blips > 0).mean()*100:.0f}%)")
    print(f"in-TPC (d_wall > 0): {(d_wall > 0).sum()}, "
          f"d_wall >= 50 cm: {(d_wall >= 50).sum()}")

    plot_truth_distributions(E, costheta, phi, d_wall)
    plot_blip_response(E, n_blips, bE, dist)
    print(f"plots saved to {os.path.abspath(PLOT_DIR)}")


def plot_truth_distributions(E, costheta, phi, d_wall):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    step_hist_with_errors(axes[0], E, np.arange(0, E_MAX_MEV + 1.0, 1.0),
                          color="#0072B2")
    axes[0].set_xlabel("True photon energy [MeV]")
    axes[0].set_ylabel("Events / MeV")

    step_hist_with_errors(axes[1], costheta, np.linspace(-1, 1, 11),
                          color="#0072B2")
    axes[1].set_xlabel(r"True photon $\cos\theta$")
    axes[1].set_ylabel("Events / bin")

    step_hist_with_errors(axes[2], phi, np.linspace(-180, 180, 13),
                          color="#0072B2")
    axes[2].set_xlabel(r"True photon $\phi$ [deg]")
    axes[2].set_ylabel("Events / bin")

    step_hist_with_errors(axes[3], d_wall, np.arange(0, 130, 10),
                          color="#0072B2")
    axes[3].set_xlabel("Vertex distance to nearest TPC wall [cm]")
    axes[3].set_ylabel("Events / 10 cm")

    for ax in axes[1:3]:
        ax.set_ylim(bottom=0)
    fig.suptitle(f"Low-energy isotropic 1$\\gamma$ test sample truth "
                 f"distributions ({len(E)} events)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "lowE_iso1g_truth_distributions.png"), dpi=150)
    plt.close(fig)


def plot_blip_response(E, n_blips, bE, dist):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    centers = 0.5 * (EFF_BINS[:-1] + EFF_BINS[1:])

    # efficiency turn-on, with the big-sample in-TPC curve for comparison
    eff_old, ns_old = old_sample_turnon()
    axes[0].errorbar(centers, eff_old, yerr=wilson_yerr(eff_old, ns_old),
                     marker="s", markersize=4, color="#999999",
                     label="big run-4/5 sample (in TPC)")
    eff = np.full(len(centers), np.nan)
    ns = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        s = (E >= EFF_BINS[i]) & (E < EFF_BINS[i + 1])
        if s.sum() >= 5:
            eff[i] = (n_blips[s] > 0).mean()
            ns[i] = s.sum()
    axes[0].errorbar(centers, eff, yerr=wilson_yerr(eff, ns), marker="o",
                     markersize=4, color="#0072B2", label="this low-E sample")
    axes[0].set_xlabel("True photon energy [MeV]")
    axes[0].set_ylabel("P(>= 1 truth-matched blip)")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)

    step_hist_with_errors(axes[1], n_blips, np.arange(-0.5, 6.5, 1),
                          color="#0072B2")
    axes[1].set_xlabel("Truth-matched blips per photon")
    axes[1].set_ylabel("Events")

    sum_bE = np.asarray(ak.sum(bE, axis=1))
    has = n_blips > 0
    axes[2].scatter(E[has], sum_bE[has], s=14, alpha=0.8, color="#0072B2")
    axes[2].plot([0, E_MAX_MEV], [0, E_MAX_MEV], color="#999999",
                 linestyle=":", linewidth=1)
    axes[2].set_xlabel("True photon energy [MeV]")
    axes[2].set_ylabel("Summed truth-matched blip energy [MeV]")

    dist_flat = np.asarray(ak.flatten(dist))
    step_hist_with_errors(axes[3], dist_flat, np.arange(0, 210, 15),
                          color="#0072B2")
    axes[3].set_xlabel("Blip distance to photon origin [cm]")
    axes[3].set_ylabel("Blips / 15 cm")

    fig.suptitle("Truth-matched blip response, low-energy isotropic 1$\\gamma$ "
                 "test sample")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "lowE_iso1g_blip_response.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
