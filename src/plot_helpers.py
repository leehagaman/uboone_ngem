import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.signal_categories import del1g_detailed_category_labels, del1g_detailed_category_labels_latex, del1g_detailed_category_colors, del1g_detailed_category_hatches
from src.signal_categories import filetype_category_labels, filetype_category_labels_latex, filetype_category_colors, filetype_category_hatches

from src.systematics import get_frac_cov_matrices

from src.file_locations import intermediate_files_location

def get_vals(df, var):
    if var == "(wc_flash_measPe - wc_flash_predPe) / wc_flash_predPe":
        vals = (df.get_column("wc_flash_measPe") - df.get_column("wc_flash_predPe")) / df.get_column("wc_flash_predPe")
        vals = vals.to_numpy()
    elif var == "wc_WCPMTInfoChi2 / wc_WCPMTInfoNDF":
        vals = df.get_column("wc_WCPMTInfoChi2") / df.get_column("wc_WCPMTInfoNDF")
        vals = np.nan_to_num(vals.to_numpy(), nan=-1, posinf=-1, neginf=-1)
    else:
        vals = df.get_column(var)
        vals = vals.to_numpy()
    return vals

def make_plot(pred_sel_df=None, data_sel_df=None, pred_and_data_sel_df=None, bins=None, var=None, display_var=None, breakdown_type="del1g_detailed", 
        iso1g_norm_factor=None, del1g_norm_factor=None, title=None, include_overflow=True, include_underflow=False, log_x=False, log_y=False, savename=None,
        plot_rw_systematics=False, dont_use_systematic_cache=False):

    if pred_and_data_sel_df is not None:
        pred_sel_df = pred_and_data_sel_df.filter(pl.col("filetype") != "data")
        data_sel_df = pred_and_data_sel_df.filter(pl.col("filetype") == "data")
    elif pred_sel_df is not None and data_sel_df is not None:
        pass
    else:
        raise ValueError("Either pred_sel_df and data_sel_df or pred_and_data_sel_df must be provided")    

    if bins is None:
        vals = get_vals(pred_sel_df, var)
        min_val = np.min(vals[np.isfinite(vals)])
        max_val = np.max(vals[np.isfinite(vals)])
        if min_val < -1e10:
            min_val = -1e10
        if max_val > 1e10:
            max_val = 1e10
        del vals
        if min_val > 0 and max_val > 0 and np.log10(max_val) - np.log10(min_val) > 2:
            bins = np.logspace(0.5 * np.log10(min_val), 2 * np.log10(max_val), 21)
            log_x = True
        else:
            bins = np.linspace(min_val, max_val, 21)

    if include_overflow and include_underflow:
        bin_width = bins[-1] - bins[-2]
        display_bins = np.concatenate([bins, [bins[-1] + bin_width]])
        bins = np.concatenate([bins, [np.inf]])
        bin_width = bins[1] - bins[0]
        display_bins = np.concatenate([[bins[0] - bin_width], display_bins])
        bins = np.concatenate([-np.inf, bins])
    elif include_overflow:
        bin_width = bins[-1] - bins[-2]
        display_bins = np.concatenate([bins, [bins[-1] + bin_width]])
        bins = np.concatenate([bins, [np.inf]])
    elif include_underflow:
        bin_width = bins[1] - bins[0]
        display_bins = np.concatenate([[bins[0] - bin_width], display_bins])
        bins = np.concatenate([[-np.inf], bins])
    else:
        display_bins = bins

    # check if bins is sorted
    if not np.all(np.diff(bins) > 0):
        raise ValueError(f"bins is not sorted: {bins}")

    if log_x:
        display_bin_centers = np.sqrt(display_bins[:-1] * display_bins[1:])
    else:
        display_bin_centers = (display_bins[:-1] + display_bins[1:]) / 2

    if breakdown_type == "del1g_detailed":
        breakdown_labels = del1g_detailed_category_labels
        breakdown_labels_latex = del1g_detailed_category_labels_latex
        breakdown_colors = del1g_detailed_category_colors
        breakdown_hatches = del1g_detailed_category_hatches
        breakdown_queries = []
        for label_i in range(len(breakdown_labels)):
            breakdown_queries.append(pl.col("del1g_detailed_signal_category") == label_i)
    elif breakdown_type == "filetype":
        breakdown_labels = filetype_category_labels
        breakdown_labels_latex = filetype_category_labels
        breakdown_colors = filetype_category_colors
        breakdown_hatches = filetype_category_hatches
        breakdown_queries = []
        for label_i in range(len(breakdown_labels)):
            breakdown_queries.append(pl.col("filetype_signal_category") == label_i)
    else:
        raise ValueError(f"Invalid breakdown type: {breakdown_type}")

    pred_vals = get_vals(pred_sel_df, var)

    if plot_rw_systematics:

        print("creating reweightable systematic covariance matrix...")
        print("loading weights_df from parquet file...")
        weights_df = pl.read_parquet(f"{intermediate_files_location}/presel_weights_df.parquet")

        print("getting covariance matrices...")
        genie_frac_cov, flux_frac_cov, reint_frac_cov = get_frac_cov_matrices(
            pred_sel_df, weights_df, pred_vals, bins, dont_use_systematic_cache=dont_use_systematic_cache
        )

        combined_frac_cov = genie_frac_cov + flux_frac_cov + reint_frac_cov
        rw_sys_frac_errors = np.sqrt(np.diag(combined_frac_cov))

    breakdown_counts = []
    unweighted_breakdown_counts = []
    for breakdown_i, breakdown_label in enumerate(breakdown_labels):
        curr_df = pred_sel_df.filter(breakdown_queries[breakdown_i])
        vals = get_vals(curr_df, var)
        breakdown_counts.append(np.histogram(vals, weights=curr_df.get_column("wc_net_weight").to_numpy(), bins=bins)[0])
        unweighted_breakdown_counts.append(np.histogram(vals, bins=bins)[0])

    data_counts = np.histogram(get_vals(data_sel_df, var), bins=bins)[0]

    plt.figure(figsize=(10, 6))

    bottom = np.zeros(len(bins)-1)
    tot_pred = 0
    tot_unweighted_pred = 0
    for breakdown_i, (breakdown_label, breakdown_count, unweighted_breakdown_count, breakdown_color, breakdown_hatch, breakdown_label_latex) in enumerate(
                            zip(breakdown_labels, breakdown_counts, unweighted_breakdown_counts, breakdown_colors, breakdown_hatches, breakdown_labels_latex)):
        if "iso1g" in breakdown_label:
            if iso1g_norm_factor == None:
                continue
            breakdown_count = breakdown_count * iso1g_norm_factor
        elif "del1g" in breakdown_label:
            if del1g_norm_factor == None:
                continue
            breakdown_count = breakdown_count * del1g_norm_factor
        elif "data" in breakdown_label:
            continue

        curr_breakdown_label = f"{breakdown_label_latex}, {np.sum(breakdown_count):.1f} ({np.sum(unweighted_breakdown_count):.0f})"

        n, _, _ = plt.hist(display_bin_centers, weights=breakdown_count, bins=display_bins, bottom=bottom if breakdown_i > 0 else None, 
                            color=breakdown_color, hatch=breakdown_hatch, label=curr_breakdown_label)

        plt.hist(display_bin_centers, weights=breakdown_count, bins=display_bins, bottom=bottom if breakdown_i > 0 else None, histtype="step", color="k", lw=0.5)

        tot_pred += np.sum(breakdown_count)
        tot_unweighted_pred += np.sum(unweighted_breakdown_count)

        if breakdown_i == 0:
            bottom = n
        else:
            bottom += n

    if plot_rw_systematics:
        total_counts = bottom + n
        ax = plt.gca()
        for i in range(len(total_counts)):
            abs_err = rw_sys_frac_errors[i] * total_counts[i]
            left = display_bins[i]
            width = display_bins[i+1] - display_bins[i]
            bottom_y = total_counts[i] - abs_err
            rect = Rectangle(
                (left, bottom_y),
                width,
                2 * abs_err,
                hatch="/////",
                fill=False,
                edgecolor="k",
                linewidth=0,
                label="Syst. Uncert." if i == 0 else None,
            )
            ax.add_patch(rect)

    plt.plot([], [], c="k", lw=0.5, label=f"Total Pred: {tot_pred:.1f} ({tot_unweighted_pred:.0f})")

    max_pred = np.max(bottom)
    max_data = np.max(data_counts)

    plt.errorbar(display_bin_centers, data_counts, yerr=np.sqrt(data_counts), fmt="o", color="k", lw=0.5, 
                capsize=2, capthick=1, markersize=2, label=f"3.33e19 POT Run 4b Data ({np.sum(data_counts)})")

    if display_var is None:
        plt.xlabel(var)
    else:
        plt.xlabel(display_var)
    plt.ylabel("Counts (weighted\nto 3.33e19 POT)")
    plt.title(title)
    plt.xlim(display_bins[0], display_bins[-1])
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
        plt.ylim(0.01, max(max_pred, max_data) * 10)
    else:
        plt.ylim(0, max(max_pred, max_data) * 1.2)
    plt.legend(ncol=2, loc='upper right', fontsize=6)
    if savename is not None:
        plt.savefig(f"../plots/{savename}.pdf")
        plt.savefig(f"../plots/{savename}.png")
    plt.show()