"""Debugging / validation / physics plots for the pion-FSI (hA2025) reweighting.

Produces, in ``plots/pion_fsi_reweighting/``:

  Validation against GENIE's own ``hA2025reweight`` branch (the macro output):
    * validation_residuals.png       -- histogram of (mine - GENIE), log-y.
    * validation_weight_dist.png     -- overlaid weight distributions.

  The interpolation (all GENIE-exact):
    * fate_fractions_vs_KE.png       -- hA2018 vs hA2025 fate fractions vs KE for
      carbon and argon.
    * fraction_maps_<fate>.png       -- the interpolated hA2018 and hA2025 fate
      fraction over (KE, A), side by side, for each fate.
    * ratio_map_<fate>.png           -- the hA2025/hA2018 reweight factor over
      (KE, A) for each fate.
    * pion_phase_space_KE_A.png      -- 2D histogram of FSI'd pion KE vs remnant
      A (where the reweight is sampled; same axes as the maps).
    * pion_ke_distribution.png       -- KE distribution (counts) of the FSI'd
      pions, by fate.

  The deliverable comparison:
    * pi0_momentum_reweighting.png   -- true final-state pi0 momentum spectrum,
      unweighted vs hA2025-reweighted, with a ratio panel.

Run:  python src/pion_fsi_reweighting_plots.py [FILE.root]
The default FILE is the downloaded run4b nu-overlay FSIrwgt reference, which has
both the GENIE ``hA2025reweight`` branch (for validation) and the
``mc_generator_*`` truth (for both the weight recomputation and the pi0 plot).
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pion_fsi_reweighting as R

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLOT_DIR = os.path.join(_REPO_ROOT, "plots", "pion_fsi_reweighting")

DEFAULT_FILE = (
    "/nevis/riverside/data/leehagaman/ngem/other_files/"
    "checkout_MCC9.10_Run4b_v10_04_07_20_BNB_nu_overlay_retuple_retuple_hist_FSIrwgt.root"
)

_FATES = [(R.FATE_CEX, "charge exchange"), (R.FATE_INELAS, "inelastic"),
          (R.FATE_ABS, "absorption"), (R.FATE_PIPRO, "pi production")]


def _save(fig, name):
    os.makedirs(_PLOT_DIR, exist_ok=True)
    path = os.path.join(_PLOT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


_KE_MAX = 999.0  # FracADep clamps KE to [1, 999] MeV


def _set_ke_axis(ax, label=True):
    ax.set_xlim(0, _KE_MAX)
    if label:
        ax.set_xlabel("pion KE [MeV]")


def _ke_mesh(n=480):
    return np.linspace(0, _KE_MAX, n)


# ─────────────────────────────────────────────────────────────────────────────
# Validation against GENIE's hA2025reweight branch
# ─────────────────────────────────────────────────────────────────────────────
def plot_validation(file_path, tree_path="nuselection/NeutrinoSelectionFilter",
                    entry_stop=None):
    mine, _add, q = R.compute_pion_fsi_weights(file_path, tree_path=tree_path,
                                               entry_stop=entry_stop,
                                               return_queries=True)
    with uproot.open(file_path) as f:
        ref = f[tree_path]["hA2025reweight"].array(
            entry_stop=entry_stop, library="np").astype(float)
    n = len(mine)
    diff = mine - ref
    ad = np.abs(diff)
    n_exact = int((ad < 1e-6).sum())
    frac_exact = 100.0 * n_exact / n

    # residuals ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    m = ad > 0
    if m.any():
        ax.hist(diff[m], bins=120, color="C3")
    ax.set_yscale("log")
    ax.set_xlabel("this code - GENIE  (per-event weight)")
    ax.set_ylabel("events (nonzero residual only)")
    ax.set_title("Weight residuals: %d/%d exact, %d differ "
                 "(max |Δ|=%.2e, mean |Δ|=%.2e)"
                 % (n_exact, n, int(m.sum()), ad.max(), ad.mean()))
    _save(fig, "validation_residuals.png")

    # 3) weight distributions --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(min(mine.min(), ref.min()), np.percentile(ref, 99.9), 120)
    ax.hist(ref, bins=bins, histtype="step", lw=2, label="GENIE", color="k")
    ax.hist(mine, bins=bins, histtype="step", lw=1.3, ls="--",
            label="this code", color="C0")
    ax.set_yscale("log")
    ax.set_xlabel("hA2025reweight")
    ax.set_ylabel("events")
    ax.set_title("Weight distribution (means: mine %.5f, GENIE %.5f)"
                 % (mine.mean(), ref.mean()))
    ax.legend()
    _save(fig, "validation_weight_dist.png")

    return mine, ref, q


# ─────────────────────────────────────────────────────────────────────────────
# Debugging: the FracADep fate fractions and their ratio
# ─────────────────────────────────────────────────────────────────────────────
def plot_fate_fractions():
    ke = np.linspace(1, 999, 400)
    nuclei = [(12, "carbon (A=12)"), (40, "argon (A=40)")]

    # fractions vs KE
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for (fate, fname), ax in zip(_FATES, axes.ravel()):
        for (A, alabel), col in zip(nuclei, ("C0", "C1")):
            fa = np.full_like(ke, fate, dtype=int)
            AA = np.full_like(ke, A, dtype=float)
            f18 = R._frac_adep_2018(fa, ke, AA)
            f25 = R._frac_adep_2025(fa, ke, AA)
            ax.plot(ke, f18, color=col, ls="--", lw=1.4,
                    label="%s hA2018" % alabel)
            ax.plot(ke, f25, color=col, ls="-", lw=1.8,
                    label="%s hA2025" % alabel)
        ax.set_title("%s fraction" % fname)
        ax.set_ylabel("fate fraction")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("pion KE [MeV]")
    fig.suptitle("INTRANUKE pion fate fractions: hA2018 (dashed) vs hA2025 (solid)")
    _save(fig, "fate_fractions_vs_KE.png")


# The three quantities we can map: each hA model's fate fraction, and their
# ratio.  ``cmap``/``norm`` and colorbar label per quantity; 0 -> NaN (white) so
# the "no data / outside hull" region reads the same in all three.
_QUANTITIES = {
    "f2018": dict(label="hA2018 %s fraction", cmap="viridis",
                  norm=lambda: matplotlib.colors.Normalize(0, 1)),
    "f2025": dict(label="hA2025 %s fraction", cmap="viridis",
                  norm=lambda: matplotlib.colors.Normalize(0, 1)),
    "ratio": dict(label="%s ratio hA2025/hA2018", cmap="RdBu_r",
                  norm=lambda: matplotlib.colors.LogNorm(0.2, 5.0)),
}


def _quantity_grid(fate, quantity, engine, phantom, ke, A):
    """(KE, A) mesh of ``quantity`` in {f2018, f2025, ratio} for ``fate``."""
    t2025, t2018 = R.make_tables(engine, phantom)
    KE, AA = np.meshgrid(ke, A)
    fa = np.full(KE.size, fate, dtype=int)
    if quantity == "ratio":
        f18 = R._frac_adep_2018(fa, KE.ravel(), AA.ravel(), t2018)
        f25 = R._frac_adep_2025(fa, KE.ravel(), AA.ravel(), t2025)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(f18 != 0, f25 / f18, np.nan)
    elif quantity == "f2018":
        v = R._frac_adep_2018(fa, KE.ravel(), AA.ravel(), t2018)
        v = np.where(v == 0.0, np.nan, v)  # 0 == outside hull -> white
    else:  # f2025
        v = R._frac_adep_2025(fa, KE.ravel(), AA.ravel(), t2025)
        v = np.where(v == 0.0, np.nan, v)
    return v.reshape(KE.shape)


def _ratio_grid(fate, engine, phantom, ke, A):
    return _quantity_grid(fate, "ratio", engine, phantom, ke, A)


def plot_pion_phase_space(q, fate=None, fname="all FSI'd"):
    """2D histogram of FSI'd pion KE vs pre-FSI remnant A -- where in the (KE, A)
    plane the reweight is actually sampled.  Same axes as the ratio maps, so it
    can be read alongside them.  ``fate=None`` uses all reweighted pions."""
    if fate is None:
        ke, A = q["ke"], q["A"]
        suffix, key = "all", "all"
    else:
        m = q["fate"] == fate
        ke, A = q["ke"][m], q["A"][m]
        suffix, key = fname, R._FATE_TO_KEY[fate]
    kebins = np.arange(0, _KE_MAX + 20, 20)  # uniform 20 MeV bins
    abins = np.linspace(0, 209, 80)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    h = ax.hist2d(ke, A, bins=[kebins, abins], cmin=1, cmap="viridis",
                  norm=matplotlib.colors.LogNorm())
    fig.colorbar(h[3], ax=ax, label="FSI'd pions / bin")
    for n in R._NUCLEI_2025["cex"]:
        ax.axhline(n, color="w", lw=0.4, alpha=0.35)
    ax.axhline(40, color="r", ls=":", lw=1)
    ax.text(700, 44, "argon", color="r", fontsize=9)
    _set_ke_axis(ax)
    ax.set_ylabel("pre-FSI remnant A")
    ax.set_title("FSI'd pion phase space: KE vs remnant A (%s, %d pions)"
                 % (suffix, len(ke)))
    _save(fig, "pion_phase_space_KE_A.png" if fate is None
          else "pion_phase_space_%s.png" % key)


def plot_ratio_map(fate, fname):
    """GENIE-exact reweight factor over (KE, A) for one fate."""
    key = R._FATE_TO_KEY[fate]
    ke = _ke_mesh()
    A = np.linspace(3, 208, 260)
    ratio = _ratio_grid(fate, "root628", True, ke, A)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.pcolormesh(ke, A, ratio, cmap="RdBu_r",
                       norm=matplotlib.colors.LogNorm(vmin=0.2, vmax=5.0),
                       shading="auto")
    for n in R._NUCLEI_2025[key]:
        ax.axhline(n, color="0.3", lw=0.3, alpha=0.4)
    ax.axhline(40, color="k", ls=":", lw=1)
    ax.text(700, 44, "argon", fontsize=9)
    ax.set_ylabel("target / remnant A")
    ax.set_title("%s reweight factor over (KE, A)\n"
                 "(grey = 2025 data nuclei; white = no hA2018 data -> no reweight)"
                 % fname.capitalize())
    fig.colorbar(im, ax=ax, label="%s ratio hA2025/hA2018" % fname)
    _set_ke_axis(ax)
    _save(fig, "ratio_map_%s.png" % key)


def plot_pion_ke_hist(q):
    """KE distribution (raw counts) of the FSI'd pions that enter the reweight,
    broken down by fate, on the same KE axis as the maps."""
    hbins = np.arange(0, _KE_MAX + 20, 20)  # uniform 20 MeV bins (counts compare)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(q["ke"], bins=hbins, histtype="step", lw=2, color="k",
            label="all fates (%d)" % len(q["ke"]))
    for (fate, fname), col in zip(_FATES, ("C0", "C1", "C2", "C3")):
        kf = q["ke"][q["fate"] == fate]
        ax.hist(kf, bins=hbins, histtype="step", lw=1.4, color=col,
                label="%s (%d)" % (fname, len(kf)))
    ax.set_ylabel("FSI'd pions / bin")
    ax.set_title("KE distribution of FSI'd pions entering the reweight")
    ax.legend(fontsize=8)
    _set_ke_axis(ax)
    _save(fig, "pion_ke_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Deliverable: true final-state pi0 momentum, unweighted vs reweighted
# ─────────────────────────────────────────────────────────────────────────────
def plot_pi0_momentum(file_path, tree_path="nuselection/NeutrinoSelectionFilter",
                      entry_stop=None):
    weights, additional = R.compute_pion_fsi_weights(file_path, tree_path=tree_path,
                                                     entry_stop=entry_stop)
    weights_c = weights * additional   # hA2025c = hA2025 * additional charge factor
    with uproot.open(file_path) as f:
        a = f[tree_path].arrays(
            ["mc_generator_pdg", "mc_generator_statuscode",
             "mc_generator_px", "mc_generator_py", "mc_generator_pz"],
            entry_stop=entry_stop, library="np")
    pdg = a["mc_generator_pdg"]
    st = a["mc_generator_statuscode"]
    px, py, pz = a["mc_generator_px"], a["mc_generator_py"], a["mc_generator_pz"]

    # true final-state pi0: pdg 111, status kIStStableFinalState (1)
    p_list, w_list, wc_list = [], [], []
    for i in range(len(pdg)):
        sel = (pdg[i] == 111) & (st[i] == 1)
        if not sel.any():
            continue
        mom = np.sqrt(px[i][sel] ** 2 + py[i][sel] ** 2 + pz[i][sel] ** 2)
        p_list.append(mom.astype(float))
        w_list.append(np.full(sel.sum(), weights[i], dtype=float))
        wc_list.append(np.full(sel.sum(), weights_c[i], dtype=float))
    p = np.concatenate(p_list) if p_list else np.array([])
    w = np.concatenate(w_list) if w_list else np.array([])
    wc = np.concatenate(wc_list) if wc_list else np.array([])

    bins = np.linspace(0, 1.2, 49)
    cent = 0.5 * (bins[1:] + bins[:-1])
    h_unw, _ = np.histogram(p, bins=bins)
    h_rw, _ = np.histogram(p, bins=bins, weights=w)
    h_rwc, _ = np.histogram(p, bins=bins, weights=wc)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.5, 6.5), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
    ax.hist(p, bins=bins, histtype="step", lw=2, color="k",
            label="no reweighting (CV)")
    ax.hist(p, bins=bins, weights=w, histtype="step", lw=2, color="C3",
            label="hA2025 pion-FSI reweight")
    ax.hist(p, bins=bins, weights=wc, histtype="step", lw=2, color="C0",
            label="hA2025c (+ pion-charge)")
    ax.set_ylabel("true final-state pi0 / bin")
    ax.set_title("True pi0 momentum spectrum, pion-FSI reweighting (%d true pi0)\n"
                 "sum of weights / count: hA2025 %.4f, hA2025c %.4f"
                 % (len(p), w.sum() / max(len(p), 1), wc.sum() / max(len(p), 1)))
    ax.legend()
    ax.grid(alpha=0.3)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(h_unw > 0, h_rw / h_unw, np.nan)
        ratio_c = np.where(h_unw > 0, h_rwc / h_unw, np.nan)
    axr.plot(cent, ratio, drawstyle="steps-mid", color="C3", label="hA2025")
    axr.plot(cent, ratio_c, drawstyle="steps-mid", color="C0", label="hA2025c")
    axr.axhline(1.0, color="k", lw=0.8)
    axr.set_ylim(0.8, 1.2)
    axr.set_ylabel("reweighted / CV")
    axr.set_xlabel("true pi0 momentum [GeV]")
    axr.grid(alpha=0.3)
    axr.legend(fontsize=8, ncol=2)
    _save(fig, "pi0_momentum_reweighting.png")


# 35 MeV true final-state proton KE threshold for the Np / 0p split.
_PROTON_KE_THRESHOLD_GEV = 0.035


def _true_mass(E, px, py, pz):
    return np.sqrt(np.clip(E * E - (px * px + py * py + pz * pz), 0.0, None))


def plot_pion_momentum_by_category(file_path, pion_pdg, pion_label, out_name,
                                   tree_path="nuselection/NeutrinoSelectionFilter",
                                   entry_stop=None):
    """True final-state pion (``pion_pdg``) momentum spectrum -- CV vs hA2025 vs
    hA2025c -- in a 2x2 grid split by interaction current (numuCC / NC) and
    proton topology (Np / 0p, with a 35 MeV true final-state proton KE
    threshold).  Classification and kinematics come from the mc_generator_*
    truth stack (status==1 = stable final state)."""
    weights, additional = R.compute_pion_fsi_weights(file_path, tree_path=tree_path,
                                                     entry_stop=entry_stop)
    weights_c = weights * additional
    with uproot.open(file_path) as f:
        a = f[tree_path].arrays(
            ["mc_generator_pdg", "mc_generator_statuscode", "mc_generator_E",
             "mc_generator_px", "mc_generator_py", "mc_generator_pz"],
            entry_stop=entry_stop, library="np")
    pdg, st, E = a["mc_generator_pdg"], a["mc_generator_statuscode"], a["mc_generator_E"]
    px, py, pz = a["mc_generator_px"], a["mc_generator_py"], a["mc_generator_pz"]

    # per (current, topology) -> [momenta, hA2025 weights, hA2025c weights]
    cats = {(c, t): ([], [], []) for c in ("numuCC", "NC") for t in ("Np", "0p")}
    for i in range(len(pdg)):
        p_i, s_i = pdg[i], st[i]
        fs = s_i == 1
        absp = np.abs(p_i)
        if np.any(fs & (absp == 13)):
            cur = "numuCC"                       # final-state muon -> numu CC
        elif np.any(fs & ((absp == 12) | (absp == 14) | (absp == 16))):
            cur = "NC"                           # final-state neutrino -> NC
        else:
            continue                             # e.g. nueCC -> neither bin
        prot = fs & (p_i == 2212)
        n_p = 0
        if prot.any():
            ke = (E[i][prot].astype(float)
                  - _true_mass(E[i][prot].astype(float), px[i][prot].astype(float),
                               py[i][prot].astype(float), pz[i][prot].astype(float)))
            n_p = int(np.sum(ke > _PROTON_KE_THRESHOLD_GEV))
        topo = "Np" if n_p >= 1 else "0p"
        pi = fs & (p_i == pion_pdg)
        if not pi.any():
            continue
        mom = np.sqrt(px[i][pi] ** 2 + py[i][pi] ** 2 + pz[i][pi] ** 2).astype(float)
        mlist, wlist, wclist = cats[(cur, topo)]
        mlist.append(mom)
        wlist.append(np.full(pi.sum(), weights[i], dtype=float))
        wclist.append(np.full(pi.sum(), weights_c[i], dtype=float))

    bins = np.linspace(0, 1.2, 49)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for (cur, row) in (("numuCC", 0), ("NC", 1)):
        for (topo, col) in (("Np", 0), ("0p", 1)):
            ax = axes[row, col]
            mlist, wlist, wclist = cats[(cur, topo)]
            p = np.concatenate(mlist) if mlist else np.array([])
            w = np.concatenate(wlist) if wlist else np.array([])
            wc = np.concatenate(wclist) if wclist else np.array([])
            ax.hist(p, bins=bins, histtype="step", lw=2, color="k",
                    label="no reweighting (CV)")
            ax.hist(p, bins=bins, weights=w, histtype="step", lw=2, color="C3",
                    label="hA2025")
            ax.hist(p, bins=bins, weights=wc, histtype="step", lw=2, color="C0",
                    label="hA2025c (+ pion-charge)")
            ax.set_title("%s, %s  (%d true %s)" % (cur, topo, len(p), pion_label),
                         fontsize=10)
            ax.grid(alpha=0.3)
            if row == 1:
                ax.set_xlabel("true %s momentum [GeV]" % pion_label)
            if col == 0:
                ax.set_ylabel("true final-state %s / bin" % pion_label)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("True %s momentum, pion-FSI reweighting, by current and proton "
                 "topology\n(Np/0p: >=1 / 0 true final-state protons with "
                 "KE > 35 MeV)" % pion_label)
    _save(fig, out_name)


# ─────────────────────────────────────────────────────────────────────────────
# Per-fate interpolated fate fractions (GENIE-exact)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fraction_maps(fate, fname):
    """Default-method (bit-exact) interpolated fate fraction over (KE, A) for
    hA2018 and hA2025 side by side -- the two models that go into the ratio."""
    key = R._FATE_TO_KEY[fate]
    ke = _ke_mesh()
    A = np.linspace(3, 208, 260)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for quantity, ax in zip(("f2018", "f2025"), axes):
        cfg = _QUANTITIES[quantity]
        grid = _quantity_grid(fate, quantity, "root628", True, ke, A)
        im = ax.pcolormesh(ke, A, grid, cmap=cfg["cmap"], norm=cfg["norm"](),
                           shading="auto")
        nuc = R._NUCLEI_2018[key] if quantity == "f2018" else R._NUCLEI_2025[key]
        for n in nuc:
            ax.axhline(n, color="w", lw=0.4, alpha=0.4)
        _set_ke_axis(ax)
        ax.set_title(cfg["label"] % fname)
        fig.colorbar(im, ax=ax, label="fate fraction", shrink=0.85)
    axes[0].set_ylabel("target / remnant A")
    fig.suptitle("%s fate fraction, interpolated (white lines = data nuclei "
                 "anchoring each model)" % fname.capitalize())
    _save(fig, "fraction_maps_%s.png" % key)



def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    entry_stop = int(sys.argv[2]) if len(sys.argv) > 2 else None
    print("input:", file_path, " entry_stop:", entry_stop)
    _, _, q = plot_validation(file_path, entry_stop=entry_stop)
    plot_fate_fractions()
    plot_pi0_momentum(file_path, entry_stop=entry_stop)
    # four-panel (numuCC/NC x Np/0p) momentum spectra for each pion species
    for pdg, label, out in ((111, "pi0", "pi0_momentum_4panel.png"),
                            (211, "pi+", "piplus_momentum_4panel.png"),
                            (-211, "pi-", "piminus_momentum_4panel.png")):
        plot_pion_momentum_by_category(file_path, pdg, label, out, entry_stop=entry_stop)
    plot_pion_phase_space(q)   # KE vs remnant A density (same axes as ratio maps)
    plot_pion_ke_hist(q)       # FSI'd pion KE distribution (counts) by fate
    # per-fate GENIE-exact maps: hA2018 / hA2025 fractions and their ratio
    for fate, fname in _FATES:
        plot_ratio_map(fate, fname)
        plot_fraction_maps(fate, fname)
    print("\nall plots in", _PLOT_DIR)


if __name__ == "__main__":
    main()
