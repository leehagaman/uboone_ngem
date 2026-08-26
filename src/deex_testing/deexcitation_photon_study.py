"""Study of nuclear de-excitation photons in a GENIE v3.6.2 AR23_20i_00_000 sample.

The MicroBooNE production GENIE tune (G18_10a_02_11a) did not simulate nuclear
de-excitation photons for argon. This script investigates an alternative
generator sample (numu CC on Ar-40, BNB-like flux) where they are present, to
inform adding blip-based systematics to single-photon studies.

De-excitation photons are identified as final-state photons with E < 15 MeV:
argon de-excitation lines all end at 12 MeV, and there is a real gap up to
~20 MeV where the other final-state photon sources in these files (eta decays,
radiative events) begin. Note that pi0s are NOT decayed at the generator level here,
so pi0 decay photons do not contaminate the sample. The NUISANCE "vertex"
particle list also contains the de-excitation photons (the nuclear model emits
them as primaries), so an energy threshold rather than vertex-vs-final-state
comparison is the correct discriminator.

Input: NUISANCE flat tree (FlatTree_VARS) from the .flat.root file.
The files were downloaded from
/pnfs/uboone/persistent/users/apapadop/GENIETweakedSamples/v3_6_2_AR23_20i_00_000_again_wtf
to /nevis/riverside/data/leehagaman/ngem/other_files/generator_files/.

Outputs plots to plots/deexcitation_photons/ and a per-q0-bin multiplicity
probability table (useful for eventually applying these photons to NC pi0
events as a function of energy transfer).

Run with:
    source ../uv_base/bin/activate
    python src/deex_testing/deexcitation_photon_study.py
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
import uproot

FLAT_FILE = (
    "/nevis/riverside/data/leehagaman/ngem/other_files/generator_files/"
    "v3_6_2_AR23_20i_00_000_again_wtf/14_1000180400_CC_v3_6_2_AR23_20i_00_000.flat.root"
)
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "plots", "deexcitation_photons")

DEEX_E_MAX = 0.015  # GeV, threshold separating de-excitation photons from eta decay etc.
                    # (de-excitation ends at 12 MeV, next population starts ~20 MeV)

# Okabe-Ito colorblind-safe palette, fixed assignment per interaction mode group
MODE_GROUPS = [
    ("CCQE", [1], "#0072B2"),
    ("2p2h", [2], "#E69F00"),
    ("RES 1pi", [11, 12, 13], "#009E73"),
    ("multi-pi / DIS", [21, 26], "#D55E00"),
    ("other", None, "#CC79A7"),  # coh, eta, kaon, ...
]
TOTAL_COLOR = "#000000"


def load_arrays():
    tree = uproot.open(FLAT_FILE)["FlatTree_VARS"]
    arrays = tree.arrays(["Mode", "Enu_true", "q0", "pdg", "E"])
    print(f"loaded {len(arrays)} events from {FLAT_FILE}")
    return arrays


def mode_group_masks(mode):
    """Returns list of (label, color, event mask) covering all events exactly once."""
    mode = np.asarray(mode)
    masks = []
    grouped = np.zeros(len(mode), dtype=bool)
    for label, codes, color in MODE_GROUPS:
        if codes is None:
            mask = ~grouped
        else:
            mask = np.isin(mode, codes)
            grouped |= mask
        masks.append((label, color, mask))
    return masks


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    arrays = load_arrays()

    is_photon = arrays["pdg"] == 22
    photon_E = arrays["E"][is_photon]  # GeV, jagged per event
    is_deex = photon_E < DEEX_E_MAX
    deex_E = photon_E[is_deex]

    n_deex = np.asarray(ak.num(deex_E))
    total_deex_E = np.asarray(ak.sum(deex_E, axis=1))
    all_photon_E_flat = np.asarray(ak.flatten(photon_E))
    deex_E_flat = np.asarray(ak.flatten(deex_E))

    q0 = np.asarray(arrays["q0"])
    groups = mode_group_masks(arrays["Mode"])
    n_proton = np.asarray(ak.sum(arrays["pdg"] == 2212, axis=1))
    n_neutron = np.asarray(ak.sum(arrays["pdg"] == 2112, axis=1))

    print_summary(all_photon_E_flat, deex_E_flat, n_deex, total_deex_E, groups)
    print_res_struck_nucleon_check(arrays["Mode"], n_deex)

    plot_full_photon_spectrum(all_photon_E_flat)
    plot_deex_spectrum(deex_E_flat)
    plot_deex_spectrum_by_mode(arrays, is_photon, groups)
    plot_deex_spectrum_every_mode_fine(arrays, is_photon)
    plot_multiplicity_by_mode(n_deex, groups)
    plot_total_energy(total_deex_E)
    plot_vs_q0(q0, n_deex, total_deex_E, groups)
    plot_vs_nucleon_counts(n_proton, n_neutron, n_deex, groups)
    plot_spectrum_by_nucleon_count(deex_E, n_proton, n_neutron, groups)
    save_multiplicity_vs_q0_table(q0, n_deex, total_deex_E)

    print(f"plots saved to {os.path.abspath(PLOT_DIR)}")


def print_summary(all_photon_E, deex_E, n_deex, total_deex_E, groups):
    n_events = len(n_deex)
    print(f"\ntotal events: {n_events}")
    print(f"final-state photons: {len(all_photon_E)}")
    print(f"  de-excitation (E < {DEEX_E_MAX*1000:.0f} MeV): {len(deex_E)}")
    print(f"  higher-energy (eta decays etc.): {len(all_photon_E) - len(deex_E)}")
    print(f"events with >= 1 de-excitation photon: {(n_deex > 0).sum()} "
          f"({(n_deex > 0).mean()*100:.1f}%)")
    print(f"mean de-excitation photon multiplicity: {n_deex.mean():.3f}")
    print(f"de-excitation photon energy: median {np.median(deex_E)*1000:.2f} MeV, "
          f"mean {deex_E.mean()*1000:.2f} MeV, max {deex_E.max()*1000:.2f} MeV")
    print(f"mean total de-excitation energy per event: {total_deex_E.mean()*1000:.2f} MeV")
    print("\nper interaction mode group:")
    for label, _, mask in groups:
        frac = (n_deex[mask] > 0).mean() if mask.sum() else float("nan")
        print(f"  {label:15s} {mask.sum():7d} events ({mask.mean()*100:5.1f}%), "
              f"P(>=1 deex gamma) = {frac*100:5.1f}%, "
              f"mean N = {n_deex[mask].mean():.3f}, "
              f"mean total E = {total_deex_E[mask].mean()*1000:6.2f} MeV")


def print_res_struck_nucleon_check(mode, n_deex):
    """NEUT CC RES codes tag the struck nucleon: 11 is nu p -> mu- p pi+
    (proton struck), 12 is nu n -> mu- p pi0 and 13 is nu n -> mu- n pi+
    (neutron struck). Comparing them tests for a struck-nucleon-type effect."""
    mode = np.asarray(mode)
    print("\nRES struck-nucleon check:")
    for code, label in [(11, "mode 11 (struck p)"), (12, "mode 12 (struck n)"),
                        (13, "mode 13 (struck n)")]:
        mask = mode == code
        print(f"  {label}: {mask.sum():7d} events, "
              f"P(>=1 deex gamma) = {(n_deex[mask] > 0).mean()*100:5.1f}%")


def plot_vs_nucleon_counts(n_proton, n_neutron, n_deex, groups):
    """De-excitation photon probability vs the number of nucleons knocked out
    of the nucleus (final-state proton / neutron counts, post-FSI)."""
    max_n = 7  # last bin is >= max_n
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    all_mask = np.ones(len(n_deex), dtype=bool)
    has_deex = n_deex > 0
    for counts, ax, xlabel in [(n_proton, axes[0], "Final-state protons"),
                               (n_neutron, axes[1], "Final-state neutrons")]:
        counts_capped = np.minimum(counts, max_n)
        for label, color, mask, lw in (
            [("all CC events", TOTAL_COLOR, all_mask, 2.0)]
            + [(lbl, c, m, 1.2) for lbl, c, m in groups]
        ):
            xs, ps = [], []
            for n in range(max_n + 1):
                sel = mask & (counts_capped == n)
                if sel.sum() >= 500:
                    xs.append(n)
                    ps.append(has_deex[sel].mean())
            ax.plot(xs, ps, marker="o", markersize=4, color=color, linewidth=lw,
                    label=label)
        ax.set_xlabel(f"{xlabel} (last bin: >= {max_n})")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("P(>= 1 de-excitation photon)")
    axes[0].legend()
    fig.suptitle("De-excitation photon probability vs knocked-out nucleon counts\n"
                 "(flat within each mode: emission depends on the primary interaction, "
                 "not the FSI cascade)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_vs_nucleon_counts.png"), dpi=150)
    plt.close(fig)


def plot_spectrum_by_nucleon_count(deex_E, n_proton, n_neutron, groups):
    """Does the photon *energy* depend on how many nucleons were knocked out?
    Compared within CCQE so mode composition cannot fake a dependence."""
    qe_mask = next(mask for label, _, mask in groups if label == "CCQE")
    n_nucleon = n_proton + n_neutron
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.arange(0, 12.5, 0.25)
    for n_sel, label, color in [
        (n_nucleon == 1, "1 FS nucleon", "#0072B2"),
        (n_nucleon == 2, "2 FS nucleons", "#E69F00"),
        ((n_nucleon >= 3) & (n_nucleon <= 4), "3-4 FS nucleons", "#009E73"),
        (n_nucleon >= 5, ">= 5 FS nucleons", "#D55E00"),
    ]:
        E = np.asarray(ak.flatten(deex_E[qe_mask & n_sel])) * 1000
        if len(E) < 500:
            continue
        ax.hist(E, bins=bins, histtype="step", density=True, color=color,
                label=f"{label} ({len(E)} photons)")
    ax.set_yscale("log")
    ax.set_xlabel("De-excitation photon energy [MeV]")
    ax.set_ylabel("Area-normalized photons / 0.25 MeV")
    ax.set_title("CCQE de-excitation photon spectrum vs knocked-out nucleon count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_spectrum_by_nucleon_count.png"), dpi=150)
    plt.close(fig)


def plot_full_photon_spectrum(all_photon_E):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.logspace(np.log10(0.1), np.log10(2000), 120)  # MeV
    ax.hist(all_photon_E * 1000, bins=bins, histtype="step", color=TOTAL_COLOR)
    ax.axvline(DEEX_E_MAX * 1000, color="#D55E00", linestyle="--")
    ax.text(DEEX_E_MAX * 1000 * 1.15, 0.6, f"{DEEX_E_MAX*1000:.0f} MeV threshold",
            transform=ax.get_xaxis_transform(), color="#D55E00", rotation=90, va="top")
    ax.text(3, 0.85, "nuclear de-excitation", transform=ax.get_xaxis_transform(),
            ha="center", color="#0072B2")
    ax.text(250, 0.35, "eta decays,\nradiative, ...", transform=ax.get_xaxis_transform(),
            ha="center", color="#0072B2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Final-state photon energy [MeV]")
    ax.set_ylabel("Photons")
    ax.set_title("All final-state photons, GENIE v3.6.2 AR23, "
                 r"$\nu_\mu$ CC on Ar (pi0s not decayed)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "photon_energy_spectrum_full.png"), dpi=150)
    plt.close(fig)


def plot_deex_spectrum(deex_E):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    E_MeV = deex_E * 1000

    axes[0].hist(E_MeV, bins=np.arange(0, 11.0, 0.05), histtype="step", color=TOTAL_COLOR)
    axes[0].set_xlabel("De-excitation photon energy [MeV]")
    axes[0].set_ylabel("Photons / 0.05 MeV")
    axes[0].set_title("Fine binning (discrete nuclear lines)")

    axes[1].hist(E_MeV, bins=np.arange(0, 11.0, 0.05), histtype="step", color=TOTAL_COLOR)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("De-excitation photon energy [MeV]")
    axes[1].set_ylabel("Photons / 0.05 MeV")
    axes[1].set_title("Same, log scale")

    fig.suptitle("Nuclear de-excitation photon spectrum, GENIE v3.6.2 AR23, "
                 r"$\nu_\mu$ CC on Ar")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_photon_energy_spectrum.png"), dpi=150)
    plt.close(fig)


def plot_deex_spectrum_by_mode(arrays, is_photon, groups):
    """Shape comparison of the photon spectrum across interaction modes.

    If the spectrum shape is mode-independent, applying it to NC pi0 events
    based only on a multiplicity model is a reasonable approximation.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.arange(0, 11.0, 0.2)
    deex_E_per_event = arrays["E"][is_photon]
    deex_E_per_event = deex_E_per_event[deex_E_per_event < DEEX_E_MAX]
    for label, color, mask in groups:
        E = np.asarray(ak.flatten(deex_E_per_event[mask])) * 1000
        if len(E) < 100:
            continue
        ax.hist(E, bins=bins, histtype="step", density=True, color=color,
                label=f"{label} ({len(E)} photons)")
    ax.set_yscale("log")
    ax.set_xlabel("De-excitation photon energy [MeV]")
    ax.set_ylabel("Area-normalized photons / 0.2 MeV")
    ax.set_title("De-excitation photon spectrum shape by interaction mode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_photon_energy_by_mode.png"), dpi=150)
    plt.close(fig)


NEUT_CC_MODE_NAMES = {
    1: "CCQE", 2: "2p2h",
    11: r"RES $p\pi^+$", 12: r"RES $p\pi^0$", 13: r"RES $n\pi^+$",
    16: r"coh $\pi$", 17: r"1$\gamma$", 21: r"multi-$\pi$",
    22: r"$\eta$", 23: "kaon", 26: "DIS",
}


def plot_deex_spectrum_every_mode_fine(arrays, is_photon, min_photons=200):
    """Finely-binned (0.05 MeV) de-excitation photon spectrum for every
    individual interaction mode, each compared to the scaled all-mode shape."""
    deex_E_per_event = arrays["E"][is_photon]
    deex_E_per_event = deex_E_per_event[deex_E_per_event < DEEX_E_MAX]
    mode = np.asarray(arrays["Mode"])
    all_E = np.asarray(ak.flatten(deex_E_per_event)) * 1000
    bins = np.arange(0, 12.05, 0.05)
    all_counts, _ = np.histogram(all_E, bins=bins)

    modes = sorted(np.unique(mode))
    panels = []
    for m in modes:
        E = np.asarray(ak.flatten(deex_E_per_event[mode == m])) * 1000
        if len(E) >= min_photons:
            panels.append((m, E))
        else:
            print(f"  mode {m} ({NEUT_CC_MODE_NAMES.get(m, '?')}): only {len(E)} "
                  f"de-excitation photons, no fine-binned panel")

    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows),
                             sharex=True, squeeze=False)
    for ax, (m, E) in zip(axes.flat, panels):
        counts, _ = np.histogram(E, bins=bins)
        scale = counts.sum() / all_counts.sum()
        ax.stairs(all_counts * scale, bins, fill=True, color="#BBBBBB",
                  label="all modes (scaled)")
        ax.stairs(counts, bins, color="#0072B2",
                  label=f"mode {m}: {NEUT_CC_MODE_NAMES.get(m, '?')}")
        ax.text(0.97, 0.92, f"{len(E)} photons", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylabel("Photons / 0.05 MeV")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("De-excitation photon energy [MeV]")
    fig.suptitle("De-excitation photon spectrum per interaction mode, "
                 "fine binning (gray: all-mode spectrum scaled to panel)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_photon_energy_every_mode_fine.png"), dpi=150)
    plt.close(fig)


def plot_multiplicity_by_mode(n_deex, groups):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.arange(-0.5, 10.5, 1)
    ax.hist(n_deex, bins=bins, histtype="step", density=True, color=TOTAL_COLOR,
            linewidth=1.5, label="all CC events")
    for label, color, mask in groups:
        if mask.sum() < 1000:
            continue
        # RES and multi-pi/DIS overlap almost exactly; dash one so both stay visible
        linestyle = "--" if label == "RES 1pi" else "-"
        ax.hist(n_deex[mask], bins=bins, histtype="step", density=True, color=color,
                linestyle=linestyle, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("De-excitation photons per event")
    ax.set_ylabel("Fraction of events")
    ax.set_title("De-excitation photon multiplicity by interaction mode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_multiplicity_by_mode.png"), dpi=150)
    plt.close(fig)


def plot_total_energy(total_deex_E):
    fig, ax = plt.subplots(figsize=(7, 5))
    has_deex = total_deex_E > 0
    ax.hist(total_deex_E[has_deex] * 1000, bins=np.arange(0, 25.0, 0.25),
            histtype="step", color=TOTAL_COLOR)
    ax.set_yscale("log")
    ax.set_xlabel("Total de-excitation photon energy per event [MeV]")
    ax.set_ylabel("Events / 0.25 MeV")
    ax.set_title(f"Events with >= 1 de-excitation photon "
                 f"({has_deex.mean()*100:.1f}% of CC events)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_total_energy.png"), dpi=150)
    plt.close(fig)


def plot_vs_q0(q0, n_deex, total_deex_E, groups):
    """The key plot for extrapolating to NC pi0: de-excitation activity vs
    energy transfer to the nucleus, split by interaction mode. If the curves
    agree across modes at fixed q0, q0 is a good universal predictor."""
    q0_bins = np.arange(0, 2.5 + 1e-9, 0.1)
    centers = 0.5 * (q0_bins[:-1] + q0_bins[1:])

    def binned_mean(values, mask):
        means = np.full(len(centers), np.nan)
        idx = np.digitize(q0[mask], q0_bins) - 1
        vals = values[mask]
        for i in range(len(centers)):
            sel = idx == i
            if sel.sum() >= 200:
                means[i] = vals[sel].mean()
        return means

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    all_mask = np.ones(len(q0), dtype=bool)
    for label, color, mask, lw in (
        [("all CC events", TOTAL_COLOR, all_mask, 2.0)]
        + [(lbl, c, m, 1.2) for lbl, c, m in groups]
    ):
        axes[0].plot(centers, binned_mean((n_deex > 0).astype(float), mask),
                     color=color, linewidth=lw, label=label)
        axes[1].plot(centers, binned_mean(n_deex.astype(float), mask),
                     color=color, linewidth=lw)
        axes[2].plot(centers, binned_mean(total_deex_E * 1000, mask),
                     color=color, linewidth=lw)

    axes[0].set_ylabel("P(>= 1 de-excitation photon)")
    axes[1].set_ylabel("Mean de-excitation photon multiplicity")
    axes[2].set_ylabel("Mean total de-excitation energy [MeV]")
    for ax in axes:
        ax.set_xlabel(r"Energy transfer $q_0$ [GeV]")
        ax.set_ylim(bottom=0)
    axes[0].legend()
    fig.suptitle("De-excitation photon activity vs energy transfer, by interaction mode")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_vs_q0.png"), dpi=150)
    plt.close(fig)


def save_multiplicity_vs_q0_table(q0, n_deex, total_deex_E):
    """P(N de-excitation photons | q0 bin), the ingredient for applying this
    to other samples (e.g. NC pi0) as a function of energy transfer."""
    q0_bins = np.arange(0, 2.5 + 1e-9, 0.1)
    max_n = 6
    header = ("q0_low_GeV,q0_high_GeV,n_events,"
              + ",".join(f"P_{n}gamma" for n in range(max_n))
              + f",P_ge{max_n}gamma,mean_total_E_MeV")
    lines = [header]
    idx = np.digitize(q0, q0_bins) - 1
    for i in range(len(q0_bins) - 1):
        sel = idx == i
        n_ev = sel.sum()
        if n_ev == 0:
            continue
        n_sel = n_deex[sel]
        probs = [(n_sel == n).mean() for n in range(max_n)] + [(n_sel >= max_n).mean()]
        mean_E = total_deex_E[sel].mean() * 1000
        lines.append(f"{q0_bins[i]:.1f},{q0_bins[i+1]:.1f},{n_ev},"
                     + ",".join(f"{p:.5f}" for p in probs)
                     + f",{mean_E:.4f}")
    out_path = os.path.join(PLOT_DIR, "deex_multiplicity_vs_q0.csv")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"multiplicity table saved to {out_path}")


if __name__ == "__main__":
    main()
