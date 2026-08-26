"""Check isotropic-one-gamma statistics in the nuclear de-excitation energy range.

The isotropic_one_gamma_overlay sample was generated with a broad photon energy
spectrum. To study how GENIE-style de-excitation photons (see
deexcitation_photon_study.py: 0-12 MeV, median 3.1 MeV) would be reconstructed
as blips, we would reweight the low-energy part of the isotropic sample to the
de-excitation spectrum shape. This script checks whether there are enough
isotropic events at those energies for that to be viable:

- plots the isotropic-sample true photon energy spectrum, zoomed on the
  de-excitation range, with event counts (all and in-FV),
- computes the per-MeV-bin reweighting to the GENIE de-excitation spectrum and
  the resulting effective sample size.

Note that nblips_saved in the dataframe is dominated by the ~84 cosmic-overlay
blips per event, so the actual reconstruction study needs the truth-matched
per-blip branches (blip_true_pdg / blip_true_g4id / blip_true_energy) from the
raw checkout ROOT files; those are not carried into all_df.parquet.

Run with:
    source ../uv_base/bin/activate
    python src/deex_testing/iso1g_deex_stats_check.py
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

from file_locations import intermediate_files_location
from deexcitation_photon_study import FLAT_FILE, DEEX_E_MAX, PLOT_DIR

DEEX_SPECTRUM_MAX_MEV = 12.0  # de-excitation photons end here (max seen: 11.75 MeV)
REWEIGHT_BINS_MEV = np.arange(0.0, DEEX_SPECTRUM_MAX_MEV + 1e-9, 1.0)


def load_iso1g():
    lf = pl.scan_parquet(os.path.join(intermediate_files_location, "all_df.parquet"))
    df = (
        lf.filter(pl.col("filetype") == "isotropic_one_gamma_overlay")
        .select("wc_true_leading_shower_energy", "wc_truth_inFV")
        .collect()
    )
    E = df["wc_true_leading_shower_energy"].to_numpy()  # MeV
    in_fv = df["wc_truth_inFV"].to_numpy().astype(bool)
    print(f"isotropic 1g events: {len(E):,} ({in_fv.mean()*100:.1f}% in FV)")
    return E, in_fv


def load_deex_spectrum():
    """GENIE de-excitation photon energies [MeV] from the AR23 flat tree."""
    tree = uproot.open(FLAT_FILE)["FlatTree_VARS"]
    arrays = tree.arrays(["pdg", "E"])
    photon_E = arrays["E"][arrays["pdg"] == 22]
    deex_E = np.asarray(ak.flatten(photon_E[photon_E < DEEX_E_MAX])) * 1000
    print(f"GENIE de-excitation photons: {len(deex_E):,}")
    return deex_E


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    iso_E, in_fv = load_iso1g()
    deex_E = load_deex_spectrum()

    print("\nisotropic 1g event counts at de-excitation energies:")
    for lo, hi in [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 20)]:
        m = (iso_E >= lo) & (iso_E < hi)
        print(f"  {lo:3d}-{hi:3d} MeV: {m.sum():5d} events, {(m & in_fv).sum():5d} in FV")
    m_deex = iso_E < DEEX_SPECTRUM_MAX_MEV
    print(f"  total < {DEEX_SPECTRUM_MAX_MEV:.0f} MeV: {m_deex.sum()} events, "
          f"{(m_deex & in_fv).sum()} in FV")

    weights_fv, n_eff = reweighting_effective_stats(iso_E, in_fv, deex_E)
    plot_spectrum_and_reweighting(iso_E, in_fv, deex_E, weights_fv, n_eff)
    print(f"plots saved to {os.path.abspath(PLOT_DIR)}")


def reweighting_effective_stats(iso_E, in_fv, deex_E):
    """Per-event weights making the in-FV isotropic sample match the
    de-excitation spectrum shape in 1 MeV bins, and the effective sample size
    N_eff = (sum w)^2 / sum w^2."""
    sel = in_fv & (iso_E < DEEX_SPECTRUM_MAX_MEV)
    src_counts, _ = np.histogram(iso_E[sel], bins=REWEIGHT_BINS_MEV)
    tgt_counts, _ = np.histogram(deex_E, bins=REWEIGHT_BINS_MEV)
    src_frac = src_counts / src_counts.sum()
    tgt_frac = tgt_counts / tgt_counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        bin_w = np.where(src_counts > 0, tgt_frac / src_frac, 0.0)

    idx = np.digitize(iso_E[sel], REWEIGHT_BINS_MEV) - 1
    w = bin_w[idx]
    n_eff = w.sum() ** 2 / (w**2).sum()
    print(f"\nreweighting in-FV isotropic events (< {DEEX_SPECTRUM_MAX_MEV:.0f} MeV) "
          f"to the de-excitation spectrum shape, {len(REWEIGHT_BINS_MEV)-1} bins:")
    print(f"  events used: {sel.sum()}, effective sample size: {n_eff:.0f}")
    print("  bin weights (relative):", np.array2string(bin_w, precision=2))
    weights_fv = np.zeros(len(iso_E))
    weights_fv[sel] = w
    return weights_fv, n_eff


def plot_spectrum_and_reweighting(iso_E, in_fv, deex_E, weights_fv, n_eff):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # full spectrum
    bins_full = np.logspace(np.log10(0.5), np.log10(2e4), 100)
    axes[0].hist(iso_E, bins=bins_full, histtype="step", color="#000000",
                 label="all")
    axes[0].hist(iso_E[in_fv], bins=bins_full, histtype="step", color="#0072B2",
                 label="in FV")
    axes[0].axvspan(0.5, DEEX_SPECTRUM_MAX_MEV, color="#E69F00", alpha=0.2)
    axes[0].text(2.5, 0.93, "de-excitation\nrange", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="top", color="#B07600")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("True photon energy [MeV]")
    axes[0].set_ylabel("Events")
    axes[0].set_title("Isotropic 1$\\gamma$ sample, full spectrum")
    axes[0].legend()

    # zoom on the de-excitation range
    bins_zoom = np.arange(0, 25.5, 1.0)
    axes[1].hist(iso_E, bins=bins_zoom, histtype="step", color="#000000", label="all")
    axes[1].hist(iso_E[in_fv], bins=bins_zoom, histtype="step", color="#0072B2",
                 label="in FV")
    axes[1].axvspan(0, DEEX_SPECTRUM_MAX_MEV, color="#E69F00", alpha=0.2)
    n_lt12 = (iso_E < DEEX_SPECTRUM_MAX_MEV).sum()
    n_lt12_fv = ((iso_E < DEEX_SPECTRUM_MAX_MEV) & in_fv).sum()
    axes[1].text(0.97, 0.95, f"< 12 MeV:\n{n_lt12} events\n{n_lt12_fv} in FV",
                 transform=axes[1].transAxes, ha="right", va="top")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("True photon energy [MeV]")
    axes[1].set_ylabel("Events / MeV")
    axes[1].set_title("Zoom on the de-excitation range")
    axes[1].legend()

    # reweighted shape vs the GENIE de-excitation target
    bins_rw = np.arange(0, DEEX_SPECTRUM_MAX_MEV + 0.25, 0.25)
    sel = weights_fv > 0
    axes[2].hist(deex_E, bins=bins_rw, histtype="step", density=True,
                 color="#000000", label="GENIE de-excitation target")
    axes[2].hist(iso_E[sel], bins=bins_rw, density=True, histtype="step",
                 color="#0072B2", label="iso 1$\\gamma$ in FV, unweighted")
    axes[2].hist(iso_E[sel], bins=bins_rw, weights=weights_fv[sel], density=True,
                 histtype="step", color="#D55E00",
                 label="iso 1$\\gamma$ reweighted (1 MeV bins)")
    axes[2].text(0.97, 0.72, f"N used = {int(sel.sum())}\n$N_{{eff}}$ = {n_eff:.0f}",
                 transform=axes[2].transAxes, ha="right", va="top")
    axes[2].set_xlabel("True photon energy [MeV]")
    axes[2].set_ylabel("Area-normalized events / 0.25 MeV")
    axes[2].set_title("Reweighting to the de-excitation spectrum")
    axes[2].legend(fontsize=8)

    fig.suptitle("Isotropic single-photon sample statistics for de-excitation reweighting")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "iso1g_deex_stats_check.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
