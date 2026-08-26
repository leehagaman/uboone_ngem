"""Study of de-excitation photons in the GENIE+INCL+++MARLEY sample.

Companion to deexcitation_photon_study.py (which analyzed the GENIE v3.6.2
AR23 sample with the simple NucDeExcitationSim leading-gamma model). This
sample instead runs INCL++ for the intranuclear cascade and the full MARLEY
model for remnant de-excitation, so photon cascades (multiplicity > 1) and
genuine kinematic dependence are possible.

Sample: 500k numu on Ar-40 with the MicroBooNE flux, tune ARINCL23_20i_00_000,
CC+NC with QE/MEC/RES/DIS (no COH), downloaded 2026-08-24 from
/pnfs/uboone/persistent/users/gardiner/deex/ to
/nevis/riverside/data/leehagaman/ngem/other_files/generator_files/deex/.
See sample_details.txt there; generator code from
github.com/S81D/Generator-DeExaaS (branch incl-hepmc3-marley-SBND-ready),
MARLEY v2 GENIE-de-excite branch.

De-excitation photons here are final-state photons with E < 30 MeV: the
spectrum falls steeply to ~10 MeV with a sparse tail to ~30 MeV, the 30-50 MeV
range is completely empty, and the tiny >50 MeV population is meson decays
(pi0s are not decayed at generator level). Final states also contain nuclear
fragments (d/t/3He/alpha/Li/...) from INCL cluster emission and MARLEY
evaporation.

Run with:
    source ../uv_base/bin/activate
    python src/deex_testing/deexcitation_photon_study_marley.py
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

GST_FILE = (
    "/nevis/riverside/data/leehagaman/ngem/other_files/generator_files/deex/"
    "GENIE_INCL_MARLEY_numu_flux_500k.gst.root"
)
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                        "plots", "deexcitation_photons_marley")

DEEX_E_MAX = 0.030  # GeV; 30-50 MeV is completely empty, >50 MeV is meson decays

# process gets the color, current gets the linestyle
PROC_GROUPS = [("QE", "qel", "#0072B2"), ("MEC", "mec", "#E69F00"),
               ("RES", "res", "#009E73"), ("DIS", "dis", "#D55E00")]
CURRENTS = [("CC", "cc", "-"), ("NC", "nc", "--")]
TOTAL_COLOR = "#000000"


def load_arrays():
    tree = uproot.open(GST_FILE)["gst"]
    arrays = tree.arrays(["pdgf", "Ef", "pxf", "pyf", "pzf",
                          "qel", "mec", "res", "dis", "cc", "nc",
                          "Ev", "El", "pxv", "pyv", "pzv", "pxl", "pyl", "pzl",
                          "hitnuc"])
    print(f"loaded {len(arrays)} events from {GST_FILE}")
    return arrays


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    arrays = load_arrays()

    is_photon = arrays["pdgf"] == 22
    photon_E = arrays["Ef"][is_photon]
    is_deex = photon_E < DEEX_E_MAX
    deex_E = photon_E[is_deex]

    n_deex = np.asarray(ak.num(deex_E))
    total_deex_E = np.asarray(ak.sum(deex_E, axis=1))
    all_photon_E_flat = np.asarray(ak.flatten(photon_E))
    deex_E_flat = np.asarray(ak.flatten(deex_E))

    q0 = np.asarray(arrays["Ev"] - arrays["El"])  # El = FS neutrino for NC
    n_proton = np.asarray(ak.sum(arrays["pdgf"] == 2212, axis=1))
    n_neutron = np.asarray(ak.sum(arrays["pdgf"] == 2112, axis=1))
    groups = make_groups(arrays)

    print_summary(all_photon_E_flat, deex_E_flat, n_deex, total_deex_E,
                  groups, arrays)

    plot_full_photon_spectrum(all_photon_E_flat)
    plot_deex_spectrum(deex_E_flat)
    plot_multiplicity(n_deex, total_deex_E, groups)
    plot_vs_q0(q0, n_deex, total_deex_E, groups)
    plot_vs_nucleon_counts(n_proton, n_neutron, n_deex, groups)
    plot_vs_total_nucleons(n_proton, n_neutron, n_deex, total_deex_E, deex_E, groups)
    plot_energy_vs_pn_counts(n_proton, n_neutron, n_deex, deex_E, groups)
    plot_spectrum_by_total_nucleons(n_proton + n_neutron, deex_E)
    plot_spectrum_per_mode_fine(deex_E, groups)
    plot_angular_correlations(arrays, is_photon, is_deex)
    print(f"plots saved to {os.path.abspath(PLOT_DIR)}")


def make_groups(arrays):
    """[(label, color, linestyle, event mask)] for the 8 process x current
    combinations that were generated."""
    groups = []
    for proc_label, proc_key, color in PROC_GROUPS:
        proc = np.asarray(arrays[proc_key])
        for cur_label, cur_key, ls in CURRENTS:
            mask = proc & np.asarray(arrays[cur_key])
            groups.append((f"{cur_label} {proc_label}", color, ls, mask))
    return groups


def print_summary(all_photon_E, deex_E, n_deex, total_deex_E, groups, arrays):
    n_events = len(n_deex)
    print(f"\ntotal events: {n_events}")
    print(f"final-state photons: {len(all_photon_E)}, de-excitation "
          f"(E < {DEEX_E_MAX*1000:.0f} MeV): {len(deex_E)}")
    print(f"events with >= 1 de-excitation photon: {(n_deex > 0).sum()} "
          f"({(n_deex > 0).mean()*100:.1f}%)")
    print(f"mean multiplicity: {n_deex.mean():.2f}; "
          f"photon energy median {np.median(deex_E)*1000:.2f} MeV, "
          f"mean {deex_E.mean()*1000:.2f} MeV")
    print(f"mean total de-excitation energy per event: "
          f"{total_deex_E.mean()*1000:.2f} MeV")
    print("\nper interaction mode:")
    for label, _, _, mask in groups:
        if mask.sum() == 0:
            continue
        print(f"  {label:7s} {mask.sum():7d} events ({mask.mean()*100:5.1f}%), "
              f"P(>=1 gamma) = {(n_deex[mask] > 0).mean()*100:5.1f}%, "
              f"mean N = {n_deex[mask].mean():.2f}, "
              f"mean total E = {total_deex_E[mask].mean()*1000:5.2f} MeV")
    hitnuc = np.asarray(arrays["hitnuc"])
    qe = np.asarray(arrays["qel"])
    print("\nstruck-nucleon check (QE only):")
    for pdg, name in [(2212, "proton"), (2112, "neutron")]:
        m = qe & (hitnuc == pdg)
        if m.sum():
            print(f"  struck {name}: {m.sum():7d} events, "
                  f"P(>=1 gamma) = {(n_deex[m] > 0).mean()*100:5.1f}%, "
                  f"mean N = {n_deex[m].mean():.2f}")


def plot_full_photon_spectrum(all_photon_E):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.logspace(np.log10(0.02), np.log10(2000), 130)
    ax.hist(all_photon_E * 1000, bins=bins, histtype="step", color=TOTAL_COLOR)
    ax.axvline(DEEX_E_MAX * 1000, color="#D55E00", linestyle="--")
    ax.text(DEEX_E_MAX * 1000 * 1.15, 0.6, f"{DEEX_E_MAX*1000:.0f} MeV threshold",
            transform=ax.get_xaxis_transform(), color="#D55E00", rotation=90, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Final-state photon energy [MeV]")
    ax.set_ylabel("Photons")
    ax.set_title("All final-state photons, GENIE+INCL+MARLEY, "
                 r"$\nu_\mu$ CC+NC on Ar (pi0s not decayed)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "photon_energy_spectrum_full.png"), dpi=150)
    plt.close(fig)


def plot_deex_spectrum(deex_E):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    E_MeV = deex_E * 1000

    axes[0].hist(E_MeV, bins=np.arange(0, 10.0, 0.05), histtype="step",
                 color=TOTAL_COLOR)
    axes[0].set_xlabel("De-excitation photon energy [MeV]")
    axes[0].set_ylabel("Photons / 0.05 MeV")
    axes[0].set_title("Fine binning, 0-10 MeV")

    axes[1].hist(E_MeV, bins=np.arange(0, 30.0, 0.1), histtype="step",
                 color=TOTAL_COLOR)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("De-excitation photon energy [MeV]")
    axes[1].set_ylabel("Photons / 0.1 MeV")
    axes[1].set_title("Full de-excitation range, log scale")

    fig.suptitle("De-excitation photon spectrum, GENIE+INCL+MARLEY")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_photon_energy_spectrum.png"), dpi=150)
    plt.close(fig)


def plot_multiplicity(n_deex, total_deex_E, groups):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bins_n = np.arange(-0.5, 10.5, 1)
    axes[0].hist(n_deex, bins=bins_n, histtype="step", density=True,
                 color=TOTAL_COLOR, linewidth=1.8, label="all events")
    for label, color, ls, mask in groups:
        if mask.sum() < 1000:
            continue
        axes[0].hist(n_deex[mask], bins=bins_n, histtype="step", density=True,
                     color=color, linestyle=ls, label=label)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("De-excitation photons per event")
    axes[0].set_ylabel("Fraction of events")
    axes[0].legend(fontsize=8, ncol=2)

    has = total_deex_E > 0
    axes[1].hist(total_deex_E[has] * 1000, bins=np.arange(0, 40.0, 0.25),
                 histtype="step", color=TOTAL_COLOR)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Total de-excitation photon energy per event [MeV]")
    axes[1].set_ylabel("Events / 0.25 MeV")
    axes[1].set_title(f"Events with >= 1 photon ({has.mean()*100:.1f}%)")

    fig.suptitle("De-excitation photon multiplicity and total energy, "
                 "GENIE+INCL+MARLEY")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_multiplicity_total_energy.png"), dpi=150)
    plt.close(fig)


def plot_vs_q0(q0, n_deex, total_deex_E, groups):
    """The key physics question vs the old sample: does the INCL+MARLEY chain
    introduce a real dependence on energy transfer within each mode?"""
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
    for label, color, ls, mask, lw in (
        [("all events", TOTAL_COLOR, "-", all_mask, 2.0)]
        + [(lbl, c, ls, m, 1.2) for lbl, c, ls, m in groups]
    ):
        axes[0].plot(centers, binned_mean((n_deex > 0).astype(float), mask),
                     color=color, linestyle=ls, linewidth=lw, label=label)
        axes[1].plot(centers, binned_mean(n_deex.astype(float), mask),
                     color=color, linestyle=ls, linewidth=lw)
        axes[2].plot(centers, binned_mean(total_deex_E * 1000, mask),
                     color=color, linestyle=ls, linewidth=lw)

    axes[0].set_ylabel("P(>= 1 de-excitation photon)")
    axes[1].set_ylabel("Mean de-excitation photon multiplicity")
    axes[2].set_ylabel("Mean total de-excitation energy [MeV]")
    for ax in axes:
        ax.set_xlabel(r"Energy transfer $q_0$ [GeV]")
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("De-excitation photon activity vs energy transfer, "
                 "GENIE+INCL+MARLEY (solid: CC, dashed: NC)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_vs_q0.png"), dpi=150)
    plt.close(fig)


def plot_vs_nucleon_counts(n_proton, n_neutron, n_deex, groups):
    """With a real cascade + statistical de-excitation, the photon yield can
    now correlate with how many nucleons were knocked out."""
    max_n = 8
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    all_mask = np.ones(len(n_deex), dtype=bool)
    for counts, ax, xlabel in [(n_proton, axes[0], "Final-state protons"),
                               (n_neutron, axes[1], "Final-state neutrons")]:
        counts_capped = np.minimum(counts, max_n)
        for label, color, ls, mask, lw in (
            [("all events", TOTAL_COLOR, "-", all_mask, 2.0)]
            + [(lbl, c, l, m, 1.2) for lbl, c, l, m in groups]
        ):
            xs, ys = [], []
            for n in range(max_n + 1):
                sel = mask & (counts_capped == n)
                if sel.sum() >= 500:
                    xs.append(n)
                    ys.append(n_deex[sel].mean())
            ax.plot(xs, ys, marker="o", markersize=4, color=color, linestyle=ls,
                    linewidth=lw, label=label)
        ax.set_xlabel(f"{xlabel} (last bin: >= {max_n})")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Mean de-excitation photon multiplicity")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("De-excitation photon multiplicity vs knocked-out nucleon "
                 "counts, GENIE+INCL+MARLEY (solid: CC, dashed: NC)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_vs_nucleon_counts.png"), dpi=150)
    plt.close(fig)


def _binned_mean_sem(counts, values, n_bins, mask=None, min_events=500):
    """Mean and SEM of `values` in integer bins of `counts` (capped at
    n_bins-1). Bins with fewer than min_events entries return NaN."""
    if mask is None:
        mask = np.ones(len(values), dtype=bool)
    capped = np.minimum(counts[mask], n_bins - 1)
    vals = values[mask]
    xs = np.arange(n_bins, dtype=float)
    means = np.full(n_bins, np.nan)
    sems = np.full(n_bins, np.nan)
    for n in range(n_bins):
        sel = capped == n
        if sel.sum() >= min_events:
            means[n] = vals[sel].mean()
            sems[n] = vals[sel].std() / np.sqrt(sel.sum())
    return xs, means, sems


def plot_vs_total_nucleons(n_proton, n_neutron, n_deex, total_deex_E, deex_E,
                           groups):
    """Everything vs the total number of ejected nucleons (protons+neutrons):
    photon probability, multiplicity, total energy, and mean per-photon energy."""
    max_n = 13  # last bin: >= 12
    n_tot = n_proton + n_neutron
    n_deex_f = n_deex.astype(float)
    has_deex = (n_deex > 0).astype(float)

    # per-photon arrays for the mean single-photon energy panel
    photon_E = np.asarray(ak.flatten(deex_E)) * 1000
    photon_ntot = np.repeat(n_tot, n_deex)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    all_mask = np.ones(len(n_deex), dtype=bool)
    for label, color, ls, mask, lw in (
        [("all events", TOTAL_COLOR, "-", all_mask, 2.0)]
        + [(lbl, c, l, m, 1.2) for lbl, c, l, m in groups]
    ):
        xs, p, p_sem = _binned_mean_sem(n_tot, has_deex, max_n, mask)
        axes[0, 0].errorbar(xs, p, yerr=p_sem, color=color, linestyle=ls,
                            linewidth=lw, marker="o", markersize=3, label=label)
        xs, m, m_sem = _binned_mean_sem(n_tot, n_deex_f, max_n, mask)
        axes[0, 1].errorbar(xs, m, yerr=m_sem, color=color, linestyle=ls,
                            linewidth=lw, marker="o", markersize=3)
        xs, e, e_sem = _binned_mean_sem(n_tot, total_deex_E * 1000, max_n, mask)
        axes[1, 0].errorbar(xs, e, yerr=e_sem, color=color, linestyle=ls,
                            linewidth=lw, marker="o", markersize=3)
        gmask = np.repeat(mask, n_deex)
        xs, g, g_sem = _binned_mean_sem(photon_ntot, photon_E, max_n, gmask,
                                        min_events=200)
        axes[1, 1].errorbar(xs, g, yerr=g_sem, color=color, linestyle=ls,
                            linewidth=lw, marker="o", markersize=3)

    axes[0, 0].set_ylabel("P(>= 1 de-excitation photon)")
    axes[0, 1].set_ylabel("Mean photon multiplicity")
    axes[1, 0].set_ylabel("Mean total de-excitation energy [MeV]")
    axes[1, 1].set_ylabel("Mean single-photon energy [MeV]")
    for ax in axes.flat:
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel(f"Final-state protons + neutrons (last bin: >= {max_n-1})")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("De-excitation photons vs total ejected nucleons, "
                 "GENIE+INCL+MARLEY (solid: CC, dashed: NC)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_vs_total_nucleons.png"), dpi=150)
    plt.close(fig)


def plot_energy_vs_pn_counts(n_proton, n_neutron, n_deex, deex_E, groups):
    """Mean single-photon energy vs proton count and vs neutron count."""
    max_n = 9  # last bin: >= 8
    photon_E = np.asarray(ak.flatten(deex_E)) * 1000
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    all_mask = np.ones(len(n_deex), dtype=bool)
    for counts, ax, xlabel in [(n_proton, axes[0], "Final-state protons"),
                               (n_neutron, axes[1], "Final-state neutrons")]:
        photon_counts = np.repeat(counts, n_deex)
        for label, color, ls, mask, lw in (
            [("all events", TOTAL_COLOR, "-", all_mask, 2.0)]
            + [(lbl, c, l, m, 1.2) for lbl, c, l, m in groups]
        ):
            gmask = np.repeat(mask, n_deex)
            xs, g, g_sem = _binned_mean_sem(photon_counts, photon_E, max_n,
                                            gmask, min_events=200)
            ax.errorbar(xs, g, yerr=g_sem, color=color, linestyle=ls,
                        linewidth=lw, marker="o", markersize=3, label=label)
        ax.set_xlabel(f"{xlabel} (last bin: >= {max_n-1})")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Mean single-photon energy [MeV]")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Mean de-excitation photon energy vs knocked-out nucleon "
                 "counts, GENIE+INCL+MARLEY (solid: CC, dashed: NC)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_energy_vs_nucleon_counts.png"), dpi=150)
    plt.close(fig)


def plot_spectrum_by_total_nucleons(n_tot, deex_E):
    """Does the photon spectrum shape change with how many nucleons left?"""
    photon_E = np.asarray(ak.flatten(deex_E)) * 1000
    photon_ntot = np.repeat(n_tot, np.asarray(ak.num(deex_E)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bins_fine = np.arange(0, 10.05, 0.1)
    bins_coarse = np.arange(0, 10.25, 0.25)
    n_groups = [(0, 1, "#0072B2"), (2, 3, "#E69F00"), (4, 6, "#009E73"),
                (7, 99, "#D55E00")]
    for lo, hi, color in n_groups:
        E = photon_E[(photon_ntot >= lo) & (photon_ntot <= hi)]
        label = (f"{lo}-{hi} nucleons" if hi < 99 else f">= {lo} nucleons")
        label += f" ({len(E)} photons)"
        axes[0].hist(E, bins=bins_fine, histtype="step", density=True,
                     color=color, label=label)
        axes[1].hist(E, bins=bins_coarse, histtype="step", density=True,
                     color=color, label=label)
    axes[0].set_xlabel("De-excitation photon energy [MeV]")
    axes[0].set_ylabel("Area-normalized photons / 0.1 MeV")
    axes[0].legend(fontsize=8, title="FS protons + neutrons")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("De-excitation photon energy [MeV]")
    axes[1].set_ylabel("Area-normalized photons / 0.25 MeV")
    fig.suptitle("De-excitation photon spectrum shape vs total ejected "
                 "nucleons, GENIE+INCL+MARLEY")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_spectrum_by_nucleon_count.png"), dpi=150)
    plt.close(fig)


def plot_spectrum_per_mode_fine(deex_E, groups, min_photons=500):
    """Finely-binned spectrum per mode, vs the scaled all-mode shape."""
    all_E = np.asarray(ak.flatten(deex_E)) * 1000
    bins = np.arange(0, 10.05, 0.05)
    all_counts, _ = np.histogram(all_E, bins=bins)

    panels = []
    for label, color, ls, mask in groups:
        E = np.asarray(ak.flatten(deex_E[mask])) * 1000
        if len(E) >= min_photons:
            panels.append((label, E))
        else:
            print(f"  {label}: only {len(E)} de-excitation photons, no panel")

    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows),
                             sharex=True, squeeze=False)
    for ax, (label, E) in zip(axes.flat, panels):
        counts, _ = np.histogram(E, bins=bins)
        scale = counts.sum() / all_counts.sum()
        ax.stairs(all_counts * scale, bins, fill=True, color="#BBBBBB",
                  label="all modes (scaled)")
        ax.stairs(counts, bins, color="#0072B2", label=label)
        ax.text(0.97, 0.92, f"{len(E)} photons", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylabel("Photons / 0.05 MeV")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("De-excitation photon energy [MeV]")
    fig.suptitle("De-excitation photon spectrum per interaction mode, "
                 "GENIE+INCL+MARLEY (gray: all-mode spectrum scaled to panel)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_photon_energy_every_mode_fine.png"),
                dpi=150)
    plt.close(fig)


def plot_angular_correlations(arrays, is_photon, is_deex):
    """Is the photon emission isotropic? Three tests:
    - photon direction w.r.t. the beam axis (z),
    - photon direction w.r.t. the momentum-transfer direction q-hat
      (any remnant-recoil boost or emission anisotropy would show up here),
    - opening angle between photon pairs within the same event (real nuclear
      cascades have gamma-gamma angular correlations; independent isotropic
      emission gives a flat distribution)."""
    sel = lambda b: arrays[b][is_photon][is_deex]  # per-de-ex-photon jagged
    px, py, pz = sel("pxf"), sel("pyf"), sel("pzf")
    p = np.sqrt(px**2 + py**2 + pz**2)
    ux, uy, uz = px / p, py / p, pz / p

    cos_beam = np.asarray(ak.flatten(uz))

    qx = arrays["pxv"] - arrays["pxl"]
    qy = arrays["pyv"] - arrays["pyl"]
    qz = arrays["pzv"] - arrays["pzl"]
    qmag = np.sqrt(qx**2 + qy**2 + qz**2)
    cos_q = np.asarray(ak.flatten(
        (ux * qx + uy * qy + uz * qz) / qmag))

    dirs = ak.zip({"x": ux, "y": uy, "z": uz})
    pairs = ak.combinations(dirs, 2, fields=["a", "b"])
    cos_gg = np.asarray(ak.flatten(
        pairs.a.x * pairs.b.x + pairs.a.y * pairs.b.y + pairs.a.z * pairs.b.z))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    bins = np.linspace(-1, 1, 41)
    for ax, vals, xlabel in [
        (axes[0], cos_beam, r"$\cos\theta$(photon, beam axis)"),
        (axes[1], cos_q, r"$\cos\theta$(photon, $\vec{q}$ direction)"),
        (axes[2], cos_gg, r"$\cos\theta$(photon pair opening angle)"),
    ]:
        counts, _ = np.histogram(vals, bins=bins)
        dens = counts / counts.sum() / np.diff(bins)
        err = np.sqrt(counts) / counts.sum() / np.diff(bins)
        centers = 0.5 * (bins[:-1] + bins[1:])
        ax.errorbar(centers, dens, yerr=err, fmt="o", markersize=3,
                    color="#0072B2")
        ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1,
                   label="isotropic (0.5)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density")
        ax.set_ylim(0.45, 0.55)
        ax.legend(fontsize=9)
    axes[2].set_title(f"{len(cos_gg)} pairs from multi-photon events")
    fig.suptitle("De-excitation photon angular distributions, GENIE+INCL+MARLEY "
                 "(zoomed y axis: 0.45-0.55)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "deex_angular_correlations.png"), dpi=150)
    plt.close(fig)

    for name, vals in [("cos(beam)", cos_beam), ("cos(q)", cos_q),
                       ("cos(gamma-gamma)", cos_gg)]:
        print(f"  {name}: mean = {vals.mean():+.5f} +- "
              f"{vals.std()/np.sqrt(len(vals)):.5f}  (isotropic: 0)")


if __name__ == "__main__":
    main()
