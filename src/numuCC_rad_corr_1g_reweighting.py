"""Compute per-event radiative-correction weights for delete-one-gamma numuCC events.

This module replicates the weight-producing logic of
``ipynb_notebooks/rad_corrections_reweighting.ipynb`` so that
``numuCC_rad_corr_1g_reweighting.parquet`` can be regenerated as part of the main
dataframe-creation pipeline instead of by hand.

Reference: Tomalak et al., Phys. Rev. D 106, 093006 (2022)
https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.093006

The produced parquet has columns:
    run, subrun, event, fix_del1g_weight, x_eta_uniform_weight,
    rad_frac_x_eta, wc_muon_gamma_opening_angle
Only events with all weights > 0 are saved.
"""

import os
from datetime import datetime

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from file_locations import intermediate_files_location
from rad_corrections import (
    alpha, m_e, m_mu, Delta_E,
    j_collinear, rad_frac_integrated_energy, rad_frac_x_eta, rad_frac_kinematic,
)

# plots/ folder lives at the repo root (parent of this src/ directory)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLOT_DIR = os.path.join(_REPO_ROOT, "plots", "rad_corrections_reweighting")

# ── Shared plotting parameters (from notebook) ───────────────────────────────
Delta_theta_max_deg = 60.0
colors     = ['royalblue', 'darkorange', 'forestgreen', 'crimson']
linestyles = ['-', '--', '-.', ':']

# ── Binning parameters used by the weight computation ────────────────────────
Etree_edges = np.array([0, 200, 500, 750, 1000, 1250, 1500, 2000, 3000, 5000], dtype=float)
N_x   = 20
N_eta = 20

_RELEVANT_VARS = [
    "run",
    "subrun",
    "event",
    "wc_truth_muonMomentum_0",
    "wc_truth_muonMomentum_1",
    "wc_truth_muonMomentum_2",
    "wc_truth_muonMomentum_3",
    "wc_true_leading_shower_energy",
    "wc_true_leading_shower_costheta",
    "wc_true_leading_shower_phi",
    "wc_true_sum_prim_proton_energy",
]


def _reproduction_and_theory_plots():
    """Theory/reference plots (notebook cells 3, 5, 6, 7) -- no data needed."""
    # ── Reproduction of Figure 4 (cell 3) ────────────────────────────────────
    Delta_theta = np.linspace(0.0, 15.0, 3000)
    energies   = [600, 2000, 6000]
    fig4_ls    = [':',  '--', '-']
    fig4_labs  = [r'$E_\ell + E_\gamma = 0.6\,\mathrm{GeV}$',
                  r'$E_\ell + E_\gamma = 2\,\mathrm{GeV}$',
                  r'$E_\ell + E_\gamma = 6\,\mathrm{GeV}$']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, m_ell, flav, ylim in [(axes[0], m_e, 'e', 0.20), (axes[1], m_mu, r'\mu', 0.05)]:
        for E_tree, ls, lab in zip(energies, fig4_ls, fig4_labs):
            ax.plot(Delta_theta, rad_frac_integrated_energy(Delta_theta, E_tree, m_ell),
                    ls, color='k', linewidth=1.4, label=lab)
        ax.set_xlim(0, 15)
        ax.set_ylim(0, ylim)
        ax.set_xlabel(r'$\Delta\theta\,{}^\circ$', fontsize=13)
        ax.set_ylabel(r'$d\sigma^\gamma/d\sigma_{\rm LO}$', fontsize=13)
        ax.set_title(rf'static limit, ${flav}$ flavor', fontsize=12)
        ax.text(5.5, 0.06 * ylim, r'$E_\gamma < 20\,\mathrm{MeV}$', fontsize=10)
        ax.legend(fontsize=9, loc='lower right')
        ax.tick_params(direction='in', which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(_PLOT_DIR, "fig4_reproduction.jpeg"), dpi=400)
    plt.close(fig)

    # ── 1D photon energy and angle distributions (cell 5) ────────────────────
    E_nu = [200, 500, 1000, 2000]
    nu_labels = [rf'$E_\mu^{{LO}} = {E}\,\mathrm{{MeV}}$' for E in E_nu]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for E_tree, col, ls, lab in zip(E_nu, colors, linestyles, nu_labels):
        E_gamma_max = E_tree - m_mu
        if E_gamma_max <= Delta_E:
            continue
        E_gamma = np.linspace(Delta_E * 1.005, E_gamma_max * 0.998, 800)
        x        = 1.0 - E_gamma / E_tree
        eta_max  = np.deg2rad(Delta_theta_max_deg) * E_tree / m_mu
        spectrum = j_collinear(x, eta_max) / E_tree
        ax1.plot(E_gamma, spectrum, ls, color=col, lw=1.5, label=lab)
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$E_\gamma\;[\mathrm{MeV}]$', fontsize=13)
    ax1.set_ylabel(r'$d\sigma^\gamma / (dE_\gamma\,d\sigma_{\rm LO})\;[\mathrm{MeV}^{-1}]$', fontsize=11)
    ax1.set_title(rf'$\nu_\mu$ photon energy spectrum  ($\Delta\theta < {Delta_theta_max_deg:.0f}^\circ$)', fontsize=11)
    ax1.set_ylim(bottom=0)
    ax1.text(0.03, 0.97, r'$\Delta E = 20\,\mathrm{MeV}$', transform=ax1.transAxes, va='top', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.tick_params(direction='in', which='both')

    Delta_theta_arr = np.linspace(0.05, Delta_theta_max_deg, 3000)
    dth = 1e-4
    for E_tree, col, ls, lab in zip(E_nu, colors, linestyles, nu_labels):
        f_p = rad_frac_integrated_energy(Delta_theta_arr + dth, E_tree, m_mu)
        f_m = rad_frac_integrated_energy(Delta_theta_arr - dth, E_tree, m_mu)
        ax2.plot(Delta_theta_arr, (f_p - f_m) / (2.0 * dth), ls, color=col, lw=1.5, label=lab)
    ax2.set_xlabel(r'$\Delta\theta\;[{}^\circ]$', fontsize=13)
    ax2.set_ylabel(r'$(d\sigma^\gamma/d\sigma_{\rm LO})\,/\,d(\Delta\theta^\circ)$', fontsize=11)
    ax2.set_title(r'$\nu_\mu$ photon angle distribution  ($E_\gamma > 20\,\mathrm{MeV}$)', fontsize=11)
    ax2.set_xlim(0, Delta_theta_max_deg)
    ax2.set_ylim(bottom=0)
    ax2.text(0.97, 0.97, r'$\Delta E = 20\,\mathrm{MeV}$', transform=ax2.transAxes, va='top', ha='right', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.tick_params(direction='in', which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(_PLOT_DIR, "gamma_1d_distributions.jpeg"), dpi=400)
    plt.close(fig)

    # ── 2D distribution in x, eta (cell 6) ────────────────────────────────────
    fig = plt.figure(figsize=(12, 9))
    ax = plt.gca()
    x = np.linspace(0, 1, 100)
    eta = np.linspace(0, 50, 100)
    X, ETA = np.meshgrid(x, eta)
    Z = rad_frac_x_eta(X, ETA)
    vmax = float(Z.max())
    vmin = vmax * 1e-3
    im = ax.pcolormesh(X, ETA, Z, cmap='viridis', shading='gouraud',
                       norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 7)
    ax.contour(X, ETA, Z, levels=levels, colors='white', linewidths=0.6, alpha=0.4)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'$d^2\sigma^\gamma\!/\!(dx d\eta\ d\sigma_{\rm LO})$', fontsize=8)
    ax.set_xlabel(r'x = $E_\mu/(E_\mu + E_\gamma)$', fontsize=11)
    ax.set_ylabel(r'$\eta = \Delta\theta \cdot (E_\mu + E_\gamma) / m_\mu$', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 50)
    ax.tick_params(direction='in', which='both')
    fig.savefig(os.path.join(_PLOT_DIR, "rad_corr_double_differential.jpeg"), dpi=400)
    plt.close(fig)

    # ── 2D kinematic distribution (cell 7) ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, E_tree in zip(axes.ravel(), E_nu):
        E_gamma_min = Delta_E * 1.01
        E_gamma_max = (E_tree - m_mu) * 0.999
        E_gamma = np.logspace(np.log10(E_gamma_min), np.log10(E_gamma_max), 500)
        theta   = np.linspace(0.2, Delta_theta_max_deg, 500)
        EE, TT  = np.meshgrid(E_gamma, theta)
        Z = np.clip(rad_frac_kinematic(EE, TT, E_tree), 0.0, None)
        vmax = float(Z.max())
        vmin = vmax * 1e-3
        im = ax.pcolormesh(EE, TT, Z, cmap='viridis', shading='gouraud',
                           norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
        levels = np.logspace(np.log10(vmin), np.log10(vmax), 7)
        ax.contour(EE, TT, Z, levels=levels, colors='white', linewidths=0.6, alpha=0.4)
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(
            r'$d^2\sigma^\gamma\!/\!(dE_\gamma\,d(\Delta\theta)^\circ\,d\sigma_{\rm LO})$'
            r'  [MeV$^{-1}$deg$^{-1}$]', fontsize=8)
        theta_char = np.rad2deg(m_mu / (E_tree))
        ax.axhline(theta_char, color='red', lw=1.2, ls='--', alpha=0.85)
        ax.text(E_gamma_min * 1.3, theta_char + 1.5,
                rf'$m_\mu/E_\mathrm{{tree}} = {theta_char:.1f}^\circ$', color='red', fontsize=8)
        ax.set_xscale('log')
        ax.set_xlabel(r'$E_\gamma\;[\mathrm{MeV}]$', fontsize=11)
        ax.set_ylabel(r'$\Delta\theta\;[{}^\circ]$', fontsize=11)
        ax.set_xlim(E_gamma_min, E_gamma_max)
        ax.set_ylim(0, Delta_theta_max_deg)
        ax.set_title(rf'$E_\mu^{{LO}} = {int(E_tree)}\,\mathrm{{MeV}}$', fontsize=12)
        ax.tick_params(direction='in', which='both')
    fig.suptitle(
        r'$\nu_\mu$ 2D photon distribution  '
        r'$d^2\sigma^\gamma\!/\!(dE_\gamma\,d(\Delta\theta^\circ)\,d\sigma_{\rm LO})$'
        r'    [$\Delta E = 20\,\mathrm{MeV}$,  $\Delta\theta < 60^\circ$]', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_PLOT_DIR, "gamma_2d_kinematic.jpeg"), dpi=400)
    plt.close(fig)


def compute_1g1mu_rad_corr_reweighting(df, make_plots=True, net_weight_var="wc_net_weight_open_data"):
    """Compute and save the binned 1g1mu radiative-correction reweighting from the dataframe.

    Replicates ``rad_corrections_reweighting.ipynb``. Accepts an eager
    polars DataFrame or a LazyFrame. Writes
    ``numuCC_rad_corr_1g_reweighting.parquet`` to ``intermediate_files_location``
    and (if ``make_plots``) all diagnostic plots to
    ``plots/rad_corrections_reweighting/``.

    ``net_weight_var`` is the per-config net-weight column used to weight events
    when building the binned reweighting (defaults to the open-data weighting).
    The saved binned weights are POT-independent (the x_eta and fix_del1g factors
    carry canceling POT dependence), so this choice only affects the relative
    weighting of events within each bin, not the final per-config normalization
    (that is applied later by apply_1g1mu_rad_corr_reweighting).
    """
    print(f"computing del1g radiative-correction weights from the dataframe (net_weight_var={net_weight_var})")

    lf = df if isinstance(df, pl.LazyFrame) else df.lazy()

    os.makedirs(_PLOT_DIR, exist_ok=True)
    if make_plots:
        _reproduction_and_theory_plots()

    select_vars = _RELEVANT_VARS + [net_weight_var]

    # ── Load del1g and normal numuCC events (cell 9) ──────────────────────────
    del1g_numuCC_df = (
        lf.filter(
            (pl.col("filetype") == "delete_one_gamma_overlay")
            & (pl.col("wc_truth_muonMomentum_3") > 0.0)
        )
        .select(select_vars)
        .collect()
    )
    normal_numuCC_df = (
        lf.filter(
            (pl.col("filetype") != "delete_one_gamma_overlay")
            & (pl.col("wc_truth_muonMomentum_3") > 0.0)
        )
        .select(select_vars)
        .collect()
    )
    del1g_numuCC_df_input_count = del1g_numuCC_df.height
    print(f"  del1g numuCC events: {del1g_numuCC_df.height:,}")
    print(f"  normal numuCC events: {normal_numuCC_df.height:,}")

    # ── Per-event derived quantities (vectorized; matches notebook loop) ──────
    mu0 = del1g_numuCC_df["wc_truth_muonMomentum_0"].to_numpy()
    mu1 = del1g_numuCC_df["wc_truth_muonMomentum_1"].to_numpy()
    mu2 = del1g_numuCC_df["wc_truth_muonMomentum_2"].to_numpy()
    mu3 = del1g_numuCC_df["wc_truth_muonMomentum_3"].to_numpy()

    shw_E = del1g_numuCC_df["wc_true_leading_shower_energy"].to_numpy()
    shw_costheta = del1g_numuCC_df["wc_true_leading_shower_costheta"].to_numpy()
    shw_phi = del1g_numuCC_df["wc_true_leading_shower_phi"].to_numpy() * np.pi / 180.0

    sintheta = np.sqrt(np.clip(1.0 - shw_costheta * shw_costheta, 0.0, None))
    shw0 = shw_E * sintheta * np.cos(shw_phi) / 1000.0
    shw1 = shw_E * sintheta * np.sin(shw_phi) / 1000.0
    shw2 = shw_E * shw_costheta / 1000.0
    shw3 = shw_E / 1000.0

    true_muon_costheta = mu2 / mu3

    muon_dir = np.stack([mu0 / mu3, mu1 / mu3, mu2 / mu3], axis=1)
    gamma_dir = np.stack([shw0 / shw3, shw1 / shw3, shw2 / shw3], axis=1)
    dot_product = np.clip(np.sum(muon_dir * gamma_dir, axis=1), -1.0, 1.0)
    muon_gamma_opening_angle = np.arccos(dot_product) * 180.0 / np.pi

    rad_corr_E_tree = (mu3 + shw3) * 1000.0
    rad_corr_x = mu3 * 1000.0 / rad_corr_E_tree
    rad_corr_eta = np.arccos(shw_costheta) * rad_corr_E_tree / m_mu
    rad_frac_x_eta_vals = rad_frac_x_eta(rad_corr_x, rad_corr_eta)

    del1g_numuCC_df = del1g_numuCC_df.with_columns(
        pl.Series("wc_true_leading_shower_0", shw0),
        pl.Series("wc_true_leading_shower_1", shw1),
        pl.Series("wc_true_leading_shower_2", shw2),
        pl.Series("wc_true_leading_shower_3", shw3),
        pl.Series("wc_true_muon_costheta", true_muon_costheta),
        pl.Series("wc_muon_gamma_opening_angle", muon_gamma_opening_angle),
        pl.Series("rad_corr_E_tree", rad_corr_E_tree),
        pl.Series("rad_corr_x", rad_corr_x),
        pl.Series("rad_corr_eta", rad_corr_eta),
        pl.Series("rad_frac_x_eta", rad_frac_x_eta_vals),
    )

    normal_numuCC_df = normal_numuCC_df.with_columns(
        pl.Series(
            "wc_true_muon_costheta",
            normal_numuCC_df["wc_truth_muonMomentum_2"].to_numpy()
            / normal_numuCC_df["wc_truth_muonMomentum_3"].to_numpy(),
        )
    )

    # opening-angle cut (cell 9)
    del1g_numuCC_df = del1g_numuCC_df.filter(pl.col("wc_muon_gamma_opening_angle") < 60)

    # ── x_eta_uniform_weight (cells 11, 12) ───────────────────────────────────
    rad_corr_E_tree_arr = np.array(del1g_numuCC_df["rad_corr_E_tree"])
    rad_corr_x_arr      = np.array(del1g_numuCC_df["rad_corr_x"])
    rad_corr_eta_arr    = np.array(del1g_numuCC_df["rad_corr_eta"])
    weights_arr         = np.array(del1g_numuCC_df[net_weight_var])

    x_eta_uniform_weight = np.zeros(len(rad_corr_E_tree_arr))
    for panel_idx in range(len(Etree_edges) - 1):
        Etree_lo, Etree_hi = Etree_edges[panel_idx], Etree_edges[panel_idx + 1]
        mask = (rad_corr_E_tree_arr >= Etree_lo) & (rad_corr_E_tree_arr < Etree_hi)
        if mask.sum() == 0:
            continue
        x_sel   = rad_corr_x_arr[mask]
        eta_sel = rad_corr_eta_arr[mask]
        w_sel   = weights_arr[mask]
        x_edges   = np.linspace(0, 1, N_x + 1)
        eta_edges = np.linspace(0, np.max(eta_sel), N_eta + 1)
        H, _, _ = np.histogram2d(x_sel, eta_sel, bins=[x_edges, eta_edges], weights=w_sel)
        i_x   = np.clip(np.digitize(x_sel,   x_edges)   - 1, 0, N_x   - 1)
        i_eta = np.clip(np.digitize(eta_sel, eta_edges) - 1, 0, N_eta - 1)
        bin_vals = H[i_x, i_eta]
        uw = np.where(bin_vals > 0, 1.0 / bin_vals, 0.0)
        x_eta_uniform_weight[mask] = uw

    del1g_numuCC_df = del1g_numuCC_df.with_columns(
        pl.Series("x_eta_uniform_weight", x_eta_uniform_weight)
    )

    if make_plots:
        _plot_3d_x_eta(rad_corr_E_tree_arr, rad_corr_x_arr, rad_corr_eta_arr,
                       weights_arr, x_eta_uniform_weight)

    # ── fix_del1g_weight (cell 14) ────────────────────────────────────────────
    del1g_KE = np.array(del1g_numuCC_df["wc_truth_muonMomentum_3"]) * 1000.0 - m_mu
    del1g_proton_E = np.array(del1g_numuCC_df["wc_true_sum_prim_proton_energy"])
    del1g_w = np.array(del1g_numuCC_df[net_weight_var])
    del1g_xeta_w = np.array(del1g_numuCC_df["x_eta_uniform_weight"])

    normal_KE = np.array(normal_numuCC_df["wc_truth_muonMomentum_3"]) * 1000.0 - m_mu
    normal_proton_E = np.array(normal_numuCC_df["wc_true_sum_prim_proton_energy"])
    normal_w = np.array(normal_numuCC_df[net_weight_var])

    bins_mu_ke = np.concatenate([np.linspace(0, 2000, 41), [1e6]])
    bins_proton_e = np.concatenate([np.linspace(0, 1000, 41), [1e6]])

    H_del1g, _, _ = np.histogram2d(
        del1g_KE, del1g_proton_E, bins=[bins_mu_ke, bins_proton_e],
        weights=del1g_w * del1g_xeta_w)
    H_normal, _, _ = np.histogram2d(
        normal_KE, normal_proton_E, bins=[bins_mu_ke, bins_proton_e],
        weights=normal_w)

    weight_ratios_2d = np.where(H_del1g > 0, H_normal / H_del1g, 0.0)
    i_mu = np.clip(np.digitize(del1g_KE, bins_mu_ke) - 1, 0, len(bins_mu_ke) - 2)
    i_proton = np.clip(np.digitize(del1g_proton_E, bins_proton_e) - 1, 0, len(bins_proton_e) - 2)
    event_weight_ratios = weight_ratios_2d[i_mu, i_proton]

    del1g_numuCC_df = del1g_numuCC_df.with_columns(
        pl.Series("fix_del1g_weight", event_weight_ratios)
    )

    if make_plots:
        _plot_energy_diagnostics(
            del1g_numuCC_df, normal_numuCC_df,
            del1g_KE, del1g_proton_E, del1g_w, del1g_xeta_w, event_weight_ratios,
            normal_KE, normal_proton_E, normal_w, normal_numuCC_df["wc_true_muon_costheta"].to_numpy(),
            bins_mu_ke, bins_proton_e, H_del1g, H_normal)
        _plot_muon_gamma_diagnostics(del1g_numuCC_df, net_weight_var)

    # ── Save weights (cell 15) ────────────────────────────────────────────────
    weights_to_save = del1g_numuCC_df.filter(
        (pl.col("fix_del1g_weight") > 0)
        & (pl.col("x_eta_uniform_weight") > 0)
        & (pl.col("rad_frac_x_eta") > 0)
    ).select([
        "run", "subrun", "event",
        "fix_del1g_weight", "x_eta_uniform_weight", "rad_frac_x_eta",
        "wc_muon_gamma_opening_angle",
    ])

    out_path = f"{intermediate_files_location}/numuCC_rad_corr_1g_reweighting.parquet"
    weights_to_save.write_parquet(out_path)
    print(f"  saved {out_path}")
    print(f"  {weights_to_save.height:,} / {del1g_numuCC_df.height:,} events have valid weights")

    # ── Write a log of this run into the plot directory ──────────────────────
    now = datetime.now()
    log_lines = [
        "del1g radiative-correction reweighting log",
        "==========================================",
        f"created: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.astimezone().tzname()})",
        "replicates: ipynb_notebooks/rad_corrections_reweighting.ipynb",
        "reference: Tomalak et al., Phys. Rev. D 106, 093006 (2022)",
        "",
        "inputs:",
        f"  del1g numuCC events (filetype=delete_one_gamma_overlay, muonMomentum_3>0): {del1g_numuCC_df_input_count:,}",
        f"  normal numuCC events (filetype!=delete_one_gamma_overlay, muonMomentum_3>0): {normal_numuCC_df.height:,}",
        f"  muon-gamma opening angle cut applied to del1g: < 60 deg",
        f"  del1g events after opening-angle cut: {del1g_numuCC_df.height:,}",
        "",
        "binning:",
        f"  m_mu = {m_mu} MeV",
        f"  x_eta_uniform_weight: E_tree panel edges = {Etree_edges.tolist()}",
        f"  x_eta_uniform_weight: N_x = {N_x}, N_eta = {N_eta} (eta_edges per panel up to max eta)",
        f"  fix_del1g_weight muon-KE bins = linspace(0,2000,41) + overflow at 1e6",
        f"  fix_del1g_weight proton-E bins = linspace(0,1000,41) + overflow at 1e6",
        "",
        "outputs:",
        f"  weights parquet: {out_path}",
        f"  valid-weight events (all of fix_del1g_weight, x_eta_uniform_weight, rad_frac_x_eta > 0): "
        f"{weights_to_save.height:,} / {del1g_numuCC_df.height:,}",
        f"  saved columns: run, subrun, event, fix_del1g_weight, x_eta_uniform_weight, "
        f"rad_frac_x_eta, wc_muon_gamma_opening_angle",
        f"  make_plots: {make_plots} (plots in this directory)",
    ]
    log_path = os.path.join(_PLOT_DIR, "reweighting_log.txt")
    with open(log_path, "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
    print(f"  wrote log {log_path}")

    return weights_to_save


def _plot_3d_x_eta(rad_corr_E_tree_arr, rad_corr_x_arr, rad_corr_eta_arr,
                   weights_arr, x_eta_uniform_weight):
    """3D (E_tree panels) x-eta plots before and after uniform reweighting (cells 11, 12)."""
    # Before uniform reweighting
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, Etree_lo, Etree_hi in zip(axes.ravel(), Etree_edges[:-1], Etree_edges[1:]):
        mask = (rad_corr_E_tree_arr >= Etree_lo) & (rad_corr_E_tree_arr < Etree_hi)
        x_sel   = rad_corr_x_arr[mask]
        eta_sel = rad_corr_eta_arr[mask]
        w_sel   = weights_arr[mask]
        if eta_sel.size == 0:
            continue
        x_edges   = np.linspace(0, 1, N_x + 1)
        eta_edges = np.linspace(0, np.max(eta_sel), N_eta + 1)
        H, _, _ = np.histogram2d(x_sel, eta_sel, bins=[x_edges, eta_edges], weights=w_sel)
        # robust to empty / all-zero / NaN histograms (e.g. small --frac_events):
        # use only finite positive bins so LogNorm always gets a valid vmin < vmax
        finite_pos = H[np.isfinite(H) & (H > 0)]
        vmax = float(finite_pos.max()) if finite_pos.size else 1.0
        vmin = vmax * 1e-3
        H_plot = np.where(H > 0, H, np.nan)
        im = ax.pcolormesh(x_edges, eta_edges, H_plot.T, cmap='plasma', shading='flat',
                           norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(r'Weighted events / bin', fontsize=7)
        cbar.ax.tick_params(labelsize=7)
        n_events = mask.sum()
        ax.set_xlabel(r'$x = E_\mu / E_\mathrm{tree}$', fontsize=9)
        ax.set_ylabel(r'$\eta = \Delta\theta \cdot E_\mathrm{tree} / m_\mu$', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, np.max(eta_edges))
        ax.set_title(
            rf'$E_\mu^{{LO}} \in [{int(Etree_lo)},{int(Etree_hi)}]\,\mathrm{{MeV}}$'
            rf'  —  {n_events:,} events', fontsize=8.5)
        ax.tick_params(direction='in', which='both', labelsize=8)
    fig.suptitle(
        r'Del1g overlay: $\nu_\mu$CC events in $(x,\,\eta)$ plane'
        r'  binned by $E_\mu^{LO} = E_\mu + E_\gamma$  [weighted by $w_\mathrm{net}$]', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_PLOT_DIR, "x_eta_uniform_reweighting.jpeg"), dpi=400)
    plt.close(fig)

    # After uniform reweighting (validation)
    combined_weights = weights_arr * x_eta_uniform_weight
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, Etree_lo, Etree_hi in zip(axes.ravel(), Etree_edges[:-1], Etree_edges[1:]):
        mask = (rad_corr_E_tree_arr >= Etree_lo) & (rad_corr_E_tree_arr < Etree_hi)
        x_sel   = rad_corr_x_arr[mask]
        eta_sel = rad_corr_eta_arr[mask]
        if eta_sel.size == 0:
            continue
        x_edges   = np.linspace(0, 1, N_x + 1)
        eta_edges = np.linspace(0, np.max(eta_sel), N_eta + 1)
        H, _, _ = np.histogram2d(x_sel, eta_sel, bins=[x_edges, eta_edges],
                                 weights=combined_weights[mask])
        H_plot = np.where(H > 0, H, np.nan)
        im = ax.pcolormesh(x_edges, eta_edges, H_plot.T, cmap='plasma', shading='flat',
                           vmin=0.9, vmax=1.1)
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(r'Weighted events / bin', fontsize=7)
        cbar.ax.tick_params(labelsize=7)
        ax.set_xlabel(r'$x = E_\mu / E_\mathrm{tree}$', fontsize=9)
        ax.set_ylabel(r'$\eta = \Delta\theta \cdot E_\mathrm{tree} / m_\mu$', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, np.max(eta_edges))
        ax.set_title(rf'$E_\mu^{{LO}} \in [{int(Etree_lo)},{int(Etree_hi)}]\,\mathrm{{MeV}}$', fontsize=8.5)
        ax.tick_params(direction='in', which='both', labelsize=8)
    fig.suptitle(
        r'Validation: uniform-reweighted Del1g events in $(x,\,\eta)$ plane'
        r'  [each nonzero bin $\to$ 1]', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_PLOT_DIR, "x_eta_uniform_reweighting_validation.jpeg"), dpi=400)
    plt.close(fig)


def _plot_energy_diagnostics(del1g_numuCC_df, normal_numuCC_df,
                             del1g_KE, del1g_proton_E, del1g_w, del1g_xeta_w, del1g_fix_w,
                             normal_KE, normal_proton_E, normal_w, normal_muon_costheta,
                             bins_mu_ke, bins_proton_e, H_del1g, H_normal):
    """2D and 1D reweighting diagnostics (cell 14)."""
    H_del1g_after, _, _ = np.histogram2d(
        del1g_KE, del1g_proton_E, bins=[bins_mu_ke, bins_proton_e],
        weights=del1g_w * del1g_xeta_w * del1g_fix_w)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, H, title in [
        (axes[0], H_del1g,       "Del1g (x-η uniform weighted)\n[before fix_del1g_weight]"),
        (axes[1], H_normal,      "Normal overlay (target)"),
        (axes[2], H_del1g_after, "Del1g after fix_del1g_weight"),
    ]:
        # robust to empty / all-zero / NaN histograms (e.g. small --frac_events):
        # use only finite positive bins so LogNorm always gets a valid vmin < vmax
        finite_pos = H[np.isfinite(H) & (H > 0)]
        vmax = float(finite_pos.max()) if finite_pos.size else 1.0
        vmin = vmax * 1e-3
        H_plot = np.where(H > 0, H, np.nan)
        im = ax.pcolormesh(bins_mu_ke, bins_proton_e, H_plot.T,
                           cmap='plasma', shading='flat',
                           norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
        plt.colorbar(im, ax=ax, label='Weighted events / bin')
        ax.set_xlabel("True muon kinetic energy [MeV]\nwith overflow bin")
        ax.set_ylabel("True summed primary proton kinetic energy [MeV]\nwith overflow bin")
        ax.set_xlim(0, 2100)
        ax.set_ylim(0, 1100)
        ax.set_title(title)
    fig.suptitle("2D (muon KE, proton energy) distribution before and after reweighting")
    fig.tight_layout()
    fig.savefig(os.path.join(_PLOT_DIR, "muon_ke_proton_energy_reweighting.jpeg"), dpi=400)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ratio = np.where(H_normal > 0, H_del1g_after / H_normal, np.nan)
    im = ax.pcolormesh(bins_mu_ke, bins_proton_e, ratio.T,
                       cmap='RdBu_r', shading='flat', vmin=0.5, vmax=1.5)
    plt.colorbar(im, ax=ax, label='Del1g (after) / Normal overlay')
    ax.set_xlabel("True muon kinetic energy [MeV]\nwith overflow bin")
    ax.set_ylabel("True summed primary proton kinetic energy [MeV]\nwith overflow bin")
    ax.set_xlim(0, 2100)
    ax.set_ylim(0, 1100)
    ax.set_title("Ratio: del1g (after reweighting) / normal overlay\n(should be ~1.0 everywhere)")
    fig.tight_layout()
    fig.savefig(os.path.join(_PLOT_DIR, "muon_ke_proton_energy_reweighting_ratio.jpeg"), dpi=400)
    plt.close(fig)

    # 1D validation plots
    bins_ke_1d = np.linspace(0, 2000, 101)
    fig = plt.figure(figsize=(10, 6))
    plt.hist(del1g_KE, weights=del1g_w, bins=bins_ke_1d, histtype="step", label="Del1g overlay")
    plt.hist(del1g_KE, weights=del1g_w * del1g_xeta_w, bins=bins_ke_1d, histtype="step", label="Del1g overlay (x-eta uniform weighted)")
    plt.hist(normal_KE, weights=normal_w, bins=bins_ke_1d, histtype="step", label="normal overlay")
    plt.hist(del1g_KE, weights=del1g_w * del1g_xeta_w * del1g_fix_w, bins=bins_ke_1d, histtype="step", label="Normal-BNB NC $\pi^0$ overlay reweighted Del1g overlay")
    plt.hist(del1g_KE, weights=del1g_w * del1g_fix_w * np.array(del1g_numuCC_df["rad_frac_x_eta"]), bins=bins_ke_1d, histtype="step", label="Radiative Reweighted Del1g overlay")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("True leading muon kinetic energy [MeV]")
    plt.ylabel("Number of events")
    plt.title("All true numuCC Del1g Events")
    fig.savefig(os.path.join(_PLOT_DIR, "muon_ke_reweighting.jpeg"), dpi=400)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    bins = np.linspace(-1, 1, 101)
    plt.hist(normal_muon_costheta, weights=normal_w, bins=bins, histtype="step", label="normal overlay")
    plt.hist(np.array(del1g_numuCC_df["wc_true_muon_costheta"]), weights=del1g_w * del1g_xeta_w * del1g_fix_w, bins=bins, histtype="step", label="Normal-BNB NC $\pi^0$ overlay reweighted Del1g overlay")
    plt.legend()
    plt.xlabel("True muon costheta")
    plt.ylabel("Number of events")
    plt.title("All true numuCC Events")
    fig.savefig(os.path.join(_PLOT_DIR, "muon_angle_reweighting.jpeg"), dpi=400)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 500, 101)
    plt.hist(del1g_numuCC_df["wc_true_leading_shower_energy"], weights=del1g_w, bins=bins, histtype="step", label="Del1g overlay")
    plt.hist(del1g_numuCC_df["wc_true_leading_shower_energy"], weights=del1g_w * del1g_xeta_w * del1g_fix_w, bins=bins, histtype="step", label="Normal-BNB NC $\pi^0$ overlay reweighted Del1g overlay")
    plt.hist(del1g_numuCC_df["wc_true_leading_shower_energy"], weights=del1g_w * del1g_xeta_w * del1g_fix_w * np.array(del1g_numuCC_df["rad_frac_x_eta"]), bins=bins, histtype="step", label="Radiative Reweighted Del1g overlay")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("True leading shower energy [MeV]")
    plt.ylabel("Number of events")
    plt.title("All true numuCC Events")
    fig.savefig(os.path.join(_PLOT_DIR, "photon_energy_reweighting.jpeg"), dpi=400)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    bins = np.linspace(-1, 1, 101)
    plt.hist(del1g_numuCC_df["wc_true_leading_shower_costheta"], weights=del1g_w, bins=bins, histtype="step", label="Del1g overlay")
    plt.hist(del1g_numuCC_df["wc_true_leading_shower_costheta"], weights=del1g_w * del1g_xeta_w * del1g_fix_w, bins=bins, histtype="step", label="Normal-BNB NC $\pi^0$ overlay reweighted Del1g overlay")
    plt.hist(del1g_numuCC_df["wc_true_leading_shower_costheta"], weights=del1g_w * del1g_xeta_w * del1g_fix_w * np.array(del1g_numuCC_df["rad_frac_x_eta"]), bins=bins, histtype="step", label="Radiative Reweighted Del1g overlay")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("True leading shower costheta")
    plt.ylabel("Number of events")
    plt.title("All true numuCC Events")
    fig.savefig(os.path.join(_PLOT_DIR, "photon_angle_reweighting.jpeg"), dpi=400)
    plt.close(fig)


def _plot_muon_gamma_diagnostics(del1g_numuCC_df, net_weight_var="wc_net_weight_open_data"):
    """x / eta / rad_frac_x_eta distributions for fully reweighted events (cell 17)."""
    full_w = (del1g_numuCC_df[net_weight_var] * del1g_numuCC_df["x_eta_uniform_weight"]
              * del1g_numuCC_df["fix_del1g_weight"] * del1g_numuCC_df["rad_frac_x_eta"])

    fig = plt.figure(figsize=(10, 6))
    plt.hist(del1g_numuCC_df["rad_corr_x"], weights=full_w, bins=np.linspace(0, 1, 101), histtype="step")
    plt.xlabel("x = E_ℓ / E_tree")
    plt.ylabel("Number of events")
    plt.title("All true numuCC Del1g Events\nRadiative Correction Reweighted")
    fig.savefig(os.path.join(_PLOT_DIR, "rad_corr_x.jpeg"), dpi=400)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    plt.hist(del1g_numuCC_df["rad_corr_eta"], weights=full_w, bins=np.linspace(0, 50, 101), histtype="step")
    plt.xlabel("eta = Δθ [rad] · E_tree / m_ℓ")
    plt.ylabel("Number of events")
    plt.title("All true numuCC Del1g Events\nRadiative Correction Reweighted")
    fig.savefig(os.path.join(_PLOT_DIR, "rad_corr_eta.jpeg"), dpi=400)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    plt.hist(del1g_numuCC_df["rad_frac_x_eta"], weights=full_w, bins=np.linspace(0, 0.2, 101), histtype="step")
    plt.xlabel("rad_frac_x_eta")
    plt.ylabel("Number of events")
    plt.title("All true numuCC Del1g Events\nRadiative Correction Reweighted")
    plt.yscale("log")
    fig.savefig(os.path.join(_PLOT_DIR, "rad_frac_x_eta.jpeg"), dpi=400)
    plt.close(fig)
