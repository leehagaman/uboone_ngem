"""Angle-sliced data/MC check of calorimetric and PID BDT input variables.

For each variable, the events where the variable's reconstructed object has a direction are split by
alpha, the angle between that object's direction and the drift direction x (0 deg = along the drift,
90 deg = parallel to the wire planes).  The variable is histogrammed with the same bins in each alpha
slice, and the slices are joined end to end into one histogram.  Two chi2 tests are made, with the full
reweightable + DetVar covariance, on exactly the same events:
  - "1D": the variable without the angle split (the slices summed), and
  - "2D": the angle-sliced histogram.
A large 2D chi2 relative to the 1D one means the data/MC disagreement depends on the angle, which the
1D scan (plot_many_variables.py) can't see.  This was first seen in wc_cosmict_2_dQ_dx_end, see
ipynb_notebooks/cosmict_2_dQ_dx_end_study.ipynb.

Pages go to plots/angle_sliced_scan/<var>.pdf with one CSV per variable; finished variables are skipped
on a rerun, so an interrupted run can be resumed.  The covariance cache needs write access.
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from pypdf import PdfWriter

from src.file_locations import intermediate_files_location
from src.plot_helpers import make_histogram_plot
from src.signal_categories import del1g_detailed_category_queries
from src.df_helpers import format_duration

plt.rcParams.update({"font.size": 12})

ALPHA_EDGES = [0, 30, 50, 70, 90]
N_BINS_PER_SLICE = 12

# How to get each object's direction.  kinds:
#   theta_phi_deg / theta_phi_rad: polar angle from the beam (z) and azimuth atan2(y, x)
#   costheta_phi_deg: cos(polar angle from z) and azimuth in degrees
#   momentum: px, py, pz columns
#   cosz_cosy: direction cosines along z and y (LANTERN CosTheta, CosThetaY)
ANGLE_DEFINITIONS = {
    "wc_cosmict_2": dict(kind="theta_phi_deg", cols=["wc_cosmict_2_theta", "wc_cosmict_2_phi"], valid=pl.col("wc_cosmict_2_filled") == 1, extra=["wc_cosmict_2_filled"]),
    "wc_cosmict_3": dict(kind="theta_phi_deg", cols=["wc_cosmict_3_theta", "wc_cosmict_3_phi"], valid=pl.col("wc_cosmict_3_filled") == 1, extra=["wc_cosmict_3_filled"]),
    "wc_cosmict_7": dict(kind="theta_phi_deg", cols=["wc_cosmict_7_theta", "wc_cosmict_7_phi"], valid=pl.col("wc_cosmict_7_filled") == 1, extra=["wc_cosmict_7_filled"]),
    "wc_reco_shower": dict(kind="theta_phi_deg", cols=["wc_reco_shower_theta", "wc_reco_shower_phi"], valid=pl.col("wc_reco_shower_theta").is_not_null(), extra=[]),
    "wc_reco_max_prim_proton": dict(kind="costheta_phi_deg", cols=["wc_reco_max_prim_proton_costheta", "wc_reco_max_prim_proton_phi"],
                                    valid=pl.col("wc_reco_max_prim_proton_costheta") >= -1, extra=[]),
    "pandora_trk": dict(kind="theta_phi_rad", cols=["pandora_trk_theta", "pandora_trk_phi"], valid=pl.col("pandora_trk_theta").abs() < 10, extra=[]),
    "pandora_max_len_trk": dict(kind="theta_phi_rad", cols=["pandora_max_len_trk_trk_theta_v", "pandora_max_len_trk_trk_phi_v"],
                                valid=pl.col("pandora_max_len_trk_trk_theta_v").is_not_null() & (pl.col("pandora_max_len_trk_trk_theta_v").abs() < 10), extra=[]),
    "pandora_max2_len_trk": dict(kind="theta_phi_rad", cols=["pandora_max2_len_trk_trk_theta_v", "pandora_max2_len_trk_trk_phi_v"],
                                 valid=pl.col("pandora_max2_len_trk_trk_theta_v").is_not_null() & (pl.col("pandora_max2_len_trk_trk_theta_v").abs() < 10), extra=[]),
    "pandora_max3_len_trk": dict(kind="theta_phi_rad", cols=["pandora_max3_len_trk_trk_theta_v", "pandora_max3_len_trk_trk_phi_v"],
                                 valid=pl.col("pandora_max3_len_trk_trk_theta_v").is_not_null() & (pl.col("pandora_max3_len_trk_trk_theta_v").abs() < 10), extra=[]),
    "pandora_shr": dict(kind="theta_phi_rad", cols=["pandora_shr_theta", "pandora_shr_phi"], valid=pl.col("pandora_shr_theta").abs() < 10, extra=[]),
}
for _obj in ["max_electron_shower", "second_max_electron_shower", "max_photon_shower", "second_max_photon_shower",
             "max_nonprimary_shower", "second_max_nonprimary_shower"]:
    ANGLE_DEFINITIONS[f"lantern_{_obj}"] = dict(kind="cosz_cosy", cols=[f"lantern_{_obj}_CosTheta", f"lantern_{_obj}_CosThetaY"],
                                                valid=pl.col(f"lantern_{_obj}_CosTheta").is_not_null(), extra=[])

# variable -> the object whose direction defines alpha
VARIABLE_ANGLES = {}
for _v in ["wc_cosmict_2_dQ_dx_end", "wc_cosmict_2_dQ_dx_front"]:
    VARIABLE_ANGLES[_v] = "wc_cosmict_2"
for _v in ["wc_cosmict_3_dQ_dx_end", "wc_cosmict_3_dQ_dx_front"]:
    VARIABLE_ANGLES[_v] = "wc_cosmict_3"
for _v in ["wc_cosmict_7_dQ_dx_end", "wc_cosmict_7_dQ_dx_front"]:
    VARIABLE_ANGLES[_v] = "wc_cosmict_7"
# Wire-Cell shower dQ/dx variables, with the reconstructed shower direction (approximately the shower these describe)
for _v in ["wc_shw_sp_hol_2_medium_dQ_dx", "wc_shw_sp_br3_8_max_dQ_dx", "wc_shw_sp_lem_e_charge", "wc_shw_sp_lem_e_dQdx",
           "wc_spt_max_dQ_dx", "wc_shw_sp_max_dQ_dx_sample", "wc_shw_sp_lowest_dQ_dx", "wc_shw_sp_highest_dQ_dx",
           "wc_shw_sp_medium_dQ_dx", "wc_shw_sp_min_dQ_dx_5", "wc_vis_2_min_medium_dQ_dx",
           "wc_shw_sp_vec_median_dedx", "wc_shw_sp_vec_mean_dedx"] + [f"wc_shw_sp_vec_dQ_dx_{i}" for i in range(20)]:
    VARIABLE_ANGLES[_v] = "wc_reco_shower"
for _v in ["wc_shw_sp_proton_dqdx_1", "wc_shw_sp_proton_dqdx_2"]:
    VARIABLE_ANGLES[_v] = "wc_reco_max_prim_proton"
VARIABLE_ANGLES["pandora_trkpid"] = "pandora_trk"
VARIABLE_ANGLES["pandora_max_len_trk_trk_llr_pid_score_v"] = "pandora_max_len_trk"
VARIABLE_ANGLES["pandora_max2_len_trk_trk_llr_pid_score_v"] = "pandora_max2_len_trk"
VARIABLE_ANGLES["pandora_max3_len_trk_trk_llr_pid_score_v"] = "pandora_max3_len_trk"
for _v in ["pandora_shr_dedx_Y", "pandora_shr_dedx_V", "pandora_shr_dedx_U",
           "pandora_shr_dedx_Y_cali", "pandora_shr_dedx_V_cali", "pandora_shr_dedx_U_cali",
           "pandora_shr_llrpid_dedx", "pandora_shr_llrpid_dedx_U", "pandora_shr_llrpid_dedx_V", "pandora_shr_llrpid_dedx_Y",
           "pandora_shr_bragg_p", "pandora_shr_bragg_mu", "pandora_shr_bragg_mip", "pandora_shr_bragg_pion", "pandora_shr_bragg_kaon",
           "pandora_shr_tkfit_2cm_dedx_U", "pandora_shr_tkfit_2cm_dedx_V", "pandora_shr_tkfit_2cm_dedx_Y",
           "pandora_shr_tkfit_dedx_U_alt", "pandora_shr_tkfit_dedx_V_alt", "pandora_shr_tkfit_dedx_Y_alt", "pandora_shr_tkfit_dedx_max",
           "pandora_shr_tkfit_gap05_dedx_U", "pandora_shr_tkfit_gap05_dedx_V", "pandora_shr_tkfit_gap05_dedx_Y",
           "pandora_shr_tkfit_gap10_dedx_U", "pandora_shr_tkfit_gap10_dedx_V", "pandora_shr_tkfit_gap10_dedx_Y"]:
    VARIABLE_ANGLES[_v] = "pandora_shr"
for _obj in ["max_electron_shower", "second_max_electron_shower", "max_photon_shower", "second_max_photon_shower",
             "max_nonprimary_shower", "second_max_nonprimary_shower"]:
    for _score in ["PhScore", "ElScore", "MuScore", "PiScore", "PrScore", "ph_normedscore", "FromChargedScore"]:
        VARIABLE_ANGLES[f"lantern_{_obj}_{_score}"] = f"lantern_{_obj}"

PRED_FILETYPES = ["nu_overlay", "nue_overlay", "dirt_overlay", "ext", "nc_pi0_overlay", "numucc_pi0_overlay",
                  "ncdeltarad_overlay", "NC_coherent_1g_reweighted", "numucc_rad_corr_sim"]


def alpha_expr(angle_name):
    """Angle (degrees) between the object's direction and the drift direction x, folded to [0, 90]."""
    d = ANGLE_DEFINITIONS[angle_name]
    a, b = (pl.col(c).cast(pl.Float64) for c in d["cols"])
    if d["kind"] == "theta_phi_deg":
        cos_x = (a * np.pi / 180).sin() * (b * np.pi / 180).cos()
    elif d["kind"] == "theta_phi_rad":
        cos_x = a.sin() * b.cos()
    elif d["kind"] == "costheta_phi_deg":
        cos_x = (1 - a.clip(-1, 1) ** 2).sqrt() * (b * np.pi / 180).cos()
    elif d["kind"] == "cosz_cosy":
        cos_x = (1 - a ** 2 - b ** 2).clip(0, 1).sqrt()
    else:
        raise ValueError(d["kind"])
    return cos_x.abs().clip(0, 1).arccos() * 180 / np.pi


def slice_index_expr(alpha):
    k = pl.lit(None, dtype=pl.Float64)
    n = len(ALPHA_EDGES) - 1
    for i, (lo, hi) in enumerate(zip(ALPHA_EDGES[:-1], ALPHA_EDGES[1:])):
        upper = (alpha <= hi) if i == n - 1 else (alpha < hi)
        k = pl.when((alpha >= lo) & upper).then(pl.lit(float(i))).otherwise(k)
    return k


def choose_bins(vals):
    """N_BINS_PER_SLICE bins between the 2nd and 98th percentiles (fewer for discrete variables)."""
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        return None
    edges = np.unique(np.percentile(vals[(vals >= lo) & (vals <= hi)], np.linspace(0, 100, N_BINS_PER_SLICE + 1)))
    if len(edges) < 4:  # discrete-ish: fall back to equal-width bins
        edges = np.linspace(lo, hi, N_BINS_PER_SLICE + 1)
    edges[-1] = hi + 1e-6 * max(abs(hi), 1)
    return edges


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--vars", type=str, default=None, help="Comma-separated subset of the variables in VARIABLE_ANGLES")
    parser.add_argument("--output-name", type=str, default="angle_sliced_scan")
    args = parser.parse_args()

    vars_ = list(VARIABLE_ANGLES) if args.vars is None else [v.strip() for v in args.vars.split(",") if v.strip()]
    unknown = [v for v in vars_ if v not in VARIABLE_ANGLES]
    if unknown:
        raise ValueError(f"no angle definition for {unknown}")

    presel_path = f"{intermediate_files_location}/presel_df_train_vars.parquet"
    detvar_path = f"{intermediate_files_location}/detvar_presel_df_train_vars.parquet"
    presel_schema = pl.read_parquet_schema(presel_path)
    detvar_schema = pl.read_parquet_schema(detvar_path)

    category_columns = set()
    for query in del1g_detailed_category_queries:
        category_columns |= set(eval(query, {"pl": pl, "__builtins__": {}}).meta.root_names())
    angle_columns = {c for name in {VARIABLE_ANGLES[v] for v in vars_} for c in ANGLE_DEFINITIONS[name]["cols"] + ANGLE_DEFINITIONS[name]["extra"]}
    base_columns = (
        {"filename", "run", "subrun", "event", "filetype", "vartype",
         "wc_net_weight_open_data", "wc_net_weight", "wc_net_weight_sce_recomb2", "pandora_weightTune",
         "normal_overlay", "del1g_overlay", "iso1g_overlay"}
        | {c for c in list(presel_schema) + list(detvar_schema) if c.startswith(("norm_goal_pot_", "normalizing_run_period_"))}
        | category_columns | angle_columns
    )
    print("loading base columns...")
    presel_base = pl.read_parquet(presel_path, columns=[c for c in presel_schema if c in base_columns])
    presel_base = presel_base.filter(pl.col("filetype").is_in(PRED_FILETYPES + ["data"]))
    detvar_base = pl.read_parquet(detvar_path, columns=[c for c in detvar_schema if c in base_columns])
    print("loading presel_weights_df.parquet...")
    weights_df = pl.read_parquet(f"{intermediate_files_location}/presel_weights_df.parquet")
    print(presel_base.shape, detvar_base.shape, weights_df.shape)

    out_dir = project_root / "plots" / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, pages = [], []
    n_slices = len(ALPHA_EDGES) - 1

    for i, var in enumerate(vars_):
        page, csv = out_dir / f"{var}.pdf", out_dir / f"{var}.csv"
        if page.exists() and csv.exists():
            print(f"skipping {var} ({i + 1}/{len(vars_)}), already done")
            rows.append(pd.read_csv(csv).iloc[0].to_dict()); pages.append(page)
            continue
        print(f"\n{var} ({i + 1}/{len(vars_)})", flush=True)
        angle_name = VARIABLE_ANGLES[var]
        adef = ANGLE_DEFINITIONS[angle_name]

        presel_col = pl.read_parquet(presel_path, columns=[var, "filetype"]).filter(pl.col("filetype").is_in(PRED_FILETYPES + ["data"])).get_column(var)
        detvar_col = pl.read_parquet(detvar_path, columns=[var]).get_column(var)
        if presel_col.len() != presel_base.height or detvar_col.len() != detvar_base.height:
            raise RuntimeError(f"row count mismatch reading {var}")

        x = pl.col(var).cast(pl.Float64)
        valid_var = x.is_finite() & (x.abs() < 1e10) & (x > -900)
        alpha = alpha_expr(angle_name)

        def select(base, col):
            d = base.with_columns(col.alias(var))
            d = d.filter(adef["valid"] & valid_var.fill_null(False))
            return d.with_columns(alpha.alias("_alpha")).with_columns(slice_index_expr(pl.col("_alpha")).alias("_slice")).filter(pl.col("_slice").is_not_null())

        d = select(presel_base, presel_col)
        dv = select(detvar_base, detvar_col)
        pred_mask = pl.col("filetype").is_in(PRED_FILETYPES)
        vals = d.get_column(var).to_numpy().astype(float)
        bins = choose_bins(vals)
        n_data = d.filter(pl.col("filetype") == "data").height
        if bins is None or n_data < 200:
            print(f"  skipping {var}: too few events or no spread ({n_data} data events)")
            continue
        lo, hi = bins[0], bins[-1]
        width = hi - lo
        var_1d, var_2d = f"{var}__clipped", f"{var}__alpha_slices"
        clipped = pl.col(var).cast(pl.Float64).clip(lo, hi - 1e-9 * max(abs(hi), 1))

        def add_cols(df):
            return df.with_columns(clipped.alias(var_1d), (clipped + pl.col("_slice") * width).alias(var_2d))
        d, dv = add_cols(d), add_cols(dv)
        bins_2d = np.concatenate([bins[:-1] + k * width for k in range(n_slices)] + [[bins[-1] + (n_slices - 1) * width]])

        common = dict(pred_sel_df=d.filter(pred_mask), data_sel_df=d.filter(pl.col("filetype") == "data"),
                      include_overflow=False, include_underflow=False, include_legend=False,
                      selname=f"angle_scan__{var}", include_ratio=True, include_decomposition=True,
                      use_rw_systematics=True, use_detvar_systematics=True, detvar_df=dv, weights_df=weights_df,
                      require_all_rw_events=True, show=False, return_p_value_info=True)
        info_1d = make_histogram_plot(bins=bins, var=var_1d, title=f"{var} (1D, same events)", **common)
        plt.close("all")
        info_2d = make_histogram_plot(bins=bins_2d, var=var_2d, display_var=f"{var}, in slices of α (angle of {angle_name} to drift)",
                                      title=f"{var} in α slices", include_sys_breakdown_panels=True,
                                      vertical_lines=[lo + k * width for k in range(1, n_slices)], **common)
        fig = plt.gcf()
        main_ax = next(a for a in fig.axes if a.get_ylabel().startswith("Counts"))
        for k, (a0, a1) in enumerate(zip(ALPHA_EDGES[:-1], ALPHA_EDGES[1:])):
            main_ax.text(lo + (k + 0.5) * width, 0.80, f"{a0}° ≤ α < {a1}°", transform=main_ax.get_xaxis_transform(),
                         ha="center", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.8, lw=0))
        ticks = [lo + k * width + f * width for k in range(n_slices) for f in (0.25, 0.75)]
        for a in fig.axes:
            x0, x1 = a.get_xlim()
            if abs(x0 - lo) < 0.05 * width and abs(x1 - (lo + n_slices * width)) < 0.05 * width:
                a.set_xticks(ticks, [f"{(t - lo) % width + lo:.3g}" for t in ticks])
        fig.text(0.5, 0.92, f"α = angle between the {angle_name} direction ({', '.join(adef['cols'])}) and the drift direction x; "
                 f"{var} clipped to [{lo:.3g}, {hi:.3g}] in each slice; 1D on the same events: "
                 f"χ²/ndf = {info_1d['tot_chi2']:.1f}/{info_1d['ndf']} with DetVar",
                 ha="center", va="bottom", fontsize=11)
        fig.savefig(page, bbox_inches="tight")
        plt.close("all")

        row = dict(var=var, angle=angle_name, n_data=n_data, n_pred=float(d.filter(pred_mask).get_column("wc_net_weight_open_data").sum()))
        for tag, info in [("1d", info_1d), ("2d", info_2d)]:
            row[f"{tag}_ndf"] = info["ndf"]
            for cov in ["nodetvar", "tot"]:
                row[f"{tag}_{cov}_chi2"] = info[f"{cov}_chi2"]
        pd.DataFrame([row]).to_csv(csv, index=False)
        rows.append(row); pages.append(page)
        print(f"  1D: {info_1d['tot_chi2']:.1f}/{info_1d['ndf']}   2D: {info_2d['tot_chi2']:.1f}/{info_2d['ndf']}  (with DetVar)", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(project_root / "plots" / f"{args.output_name}_summary.csv", index=False)
    writer = PdfWriter()
    for p in pages:
        writer.append(str(p))
    with open(project_root / "plots" / f"{args.output_name}.pdf", "wb") as f:
        writer.write(f)
    print(f"\ndone: {len(summary)} variables in {format_duration(time.time() - start)}")


if __name__ == "__main__":
    main()
