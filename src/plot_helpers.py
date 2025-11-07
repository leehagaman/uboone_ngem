import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.signal_categories import del1g_detailed_category_labels, del1g_detailed_category_labels_latex, del1g_detailed_category_colors, del1g_detailed_category_hatches
from src.signal_categories import filetype_category_labels, filetype_category_colors, filetype_category_hatches

from src.systematics import get_rw_sys_frac_cov_matrices, get_data_stat_cov, get_pred_stat_cov
from src.systematics import get_significance

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
        plot_rw_systematics=False, dont_load_from_systematic_cache=False,
        include_data=True, additional_scaling_factor=1.0, include_systematic_breakdown=False,
        include_legend=True, show=True, return_p_value_info=False,
        page_num=None):

    if pred_and_data_sel_df is not None:
        pred_sel_df = pred_and_data_sel_df.filter(pl.col("filetype") != "data")
        data_sel_df = pred_and_data_sel_df.filter(pl.col("filetype") == "data")
    elif pred_sel_df is not None and data_sel_df is not None:
        pass
    else:
        raise ValueError("Either pred_sel_df and data_sel_df or pred_and_data_sel_df must be provided")    

    if bins is None:
        pred_vals = get_vals(pred_sel_df, var)
        data_vals = get_vals(data_sel_df, var)
        all_vals = np.concatenate([pred_vals, data_vals])
        min_val = np.min(all_vals[np.isfinite(all_vals)])
        max_val = np.max(all_vals[np.isfinite(all_vals)])
        reasonable_vals = all_vals[all_vals > -1e9]
        reasonable_vals = reasonable_vals[reasonable_vals < 1e9]
        min_reasonable_val = np.min(reasonable_vals)
        max_reasonable_val = np.max(reasonable_vals)

        if min_val != min_reasonable_val:
            include_underflow = True
        if max_val != max_reasonable_val:
            include_overflow = True

        print(f"choosing bins automatically, min_val = {min_val:.4e}, max_val = {max_val:.4e}, min_reasonable_val = {min_reasonable_val:.4e}, max_reasonable_val = {max_reasonable_val:.4e}")
        if len(np.unique(all_vals)) == 2:
            bins = np.array([min_val - 0.5, (min_val + max_val) / 2, max_val + 0.5])
            include_overflow = False
            include_underflow = False
            print("choosing two bins:", bins)
        elif 0 < min_reasonable_val and 10_000 < max_reasonable_val and np.log10(max_reasonable_val) - np.log10(min_reasonable_val) > 3:
            bins = np.logspace(np.log10(min_reasonable_val) - 0.01, np.log10(max_reasonable_val) + 0.01, 21)
            log_x = True
            print("choosing log bins:", bins)
        else:
            width = max_reasonable_val - min_reasonable_val
            bins = np.linspace(min_reasonable_val - 0.01, max_reasonable_val + 0.01, 21)
            print("choosing linear bins:", bins)

    if include_overflow and include_underflow:
        if log_x:
            bin_width_log = np.log10(bins[-1]) - np.log10(bins[-2])
            print("bin_width_log:", bin_width_log)
            display_bins = np.concatenate([bins, [10**(np.log10(bins[-1]) + bin_width_log)]])
            bins = np.concatenate([bins, [np.inf]])
            display_bins = np.concatenate([[10**(np.log10(bins[0]) - bin_width_log)], display_bins])
            bins = np.concatenate([[-np.inf], bins])
        else:
            bin_width = bins[-1] - bins[-2]
            display_bins = np.concatenate([bins, [bins[-1] + bin_width]])
            bins = np.concatenate([bins, [np.inf]])
            display_bins = np.concatenate([[bins[0] - bin_width], display_bins])
            bins = np.concatenate([[-np.inf], bins])
        print("including overflow and underflow")
    elif include_overflow:
        if log_x:
            bin_width_log = np.log10(bins[-1]) - np.log10(bins[-2])
            display_bins = np.concatenate([bins, [10**(np.log10(bins[-1]) + bin_width_log)]])
            bins = np.concatenate([bins, [np.inf]])
        else:
            bin_width = bins[-1] - bins[-2]
            display_bins = np.concatenate([bins, [bins[-1] + bin_width]])
            bins = np.concatenate([bins, [np.inf]])
        print("including overflow")
    elif include_underflow:
        if log_x:
            bin_width_log = np.log10(bins[-1]) - np.log10(bins[-2])
            display_bins = np.concatenate([[10**(np.log10(bins[0]) - bin_width_log)], bins])
            bins = np.concatenate([[-np.inf], bins])
        else:
            bin_width = bins[1] - bins[0]
            display_bins = np.concatenate([[bins[0] - bin_width], bins])
            bins = np.concatenate([[-np.inf], bins])
        print("including underflow")
    else:
        display_bins = bins

    print("display_bins:", display_bins)

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

    breakdown_counts = []
    unweighted_breakdown_counts = []
    for breakdown_i, breakdown_label in enumerate(breakdown_labels):
        curr_df = pred_sel_df.filter(breakdown_queries[breakdown_i])
        vals = get_vals(curr_df, var)
        breakdown_counts.append(np.histogram(vals, weights=curr_df.get_column("wc_net_weight").to_numpy()*additional_scaling_factor, bins=bins)[0])
        unweighted_breakdown_counts.append(np.histogram(vals, bins=bins)[0])

    data_counts = np.histogram(get_vals(data_sel_df, var), bins=bins)[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05})

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

        n, _, _ = ax1.hist(display_bin_centers, weights=breakdown_count, bins=display_bins, bottom=bottom if breakdown_i > 0 else None, 
                            color=breakdown_color, hatch=breakdown_hatch, label=curr_breakdown_label)

        ax1.hist(display_bin_centers, weights=breakdown_count, bins=display_bins, bottom=bottom if breakdown_i > 0 else None, histtype="step", color="k", lw=0.5)

        tot_pred += np.sum(breakdown_count)
        tot_unweighted_pred += np.sum(unweighted_breakdown_count)

        if breakdown_i == 0:
            bottom = n
        else:
            bottom += n

    pred_counts = bottom

    if plot_rw_systematics:

        rw_sys_frac_cov_dic = get_rw_sys_frac_cov_matrices(
            pred_sel_df.filter(pl.col("filetype") != "ext"), var, bins, dont_load_from_systematic_cache=dont_load_from_systematic_cache
        )

        combined_rw_sys_frac_cov = np.zeros((len(bins)-1, len(bins)-1))
        for rw_sys_frac_cov_name, rw_sys_frac_cov in rw_sys_frac_cov_dic.items():
            combined_rw_sys_frac_cov += rw_sys_frac_cov

        combined_rw_sys_cov = combined_rw_sys_frac_cov * np.outer(pred_counts, pred_counts)
        # using Pearson data stat cov
        data_stat_cov = get_data_stat_cov(data_counts, pred_counts)
        mc_stat_cov = get_pred_stat_cov(get_vals(pred_sel_df, var), pred_sel_df.get_column("wc_net_weight").to_numpy(), bins)
        tot_cov = combined_rw_sys_cov + data_stat_cov + mc_stat_cov

        tot_pred_cov = combined_rw_sys_cov + mc_stat_cov
        tot_pred_frac_cov = tot_pred_cov / np.outer(pred_counts, pred_counts)
        tot_pred_frac_errors = np.sqrt(np.diag(tot_pred_frac_cov))

        for i in range(len(pred_counts)):
            abs_err = tot_pred_frac_errors[i] * pred_counts[i]
            left = display_bins[i]
            width = display_bins[i+1] - display_bins[i]
            bottom_y = pred_counts[i] - abs_err
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
            ax1.add_patch(rect)

    ax1.plot([], [], c="k", lw=0.5, label=f"Total Pred: {tot_pred:.1f} ({tot_unweighted_pred:.0f})")

    max_pred = np.max(bottom)
    max_data = np.max(data_counts)
    if include_data:
        ax1.errorbar(display_bin_centers, data_counts, yerr=np.sqrt(data_counts), fmt="o", color="k", lw=0.5, 
                    capsize=2, capthick=1, markersize=2, label=f"3.33e19 POT Run 4b Data ({np.sum(data_counts)})")

    diff = data_counts - pred_counts

    # removing bins with low data counts
    empty_indices = np.where(data_counts <= 1)[0]
    if len(empty_indices) > 0:
        print("removing bins with 0 or 1 data counts at indices:", empty_indices)
        diff = np.delete(diff, empty_indices)
        tot_cov = np.delete(tot_cov, empty_indices, axis=0)
        tot_cov = np.delete(tot_cov, empty_indices, axis=1)

    try:
        tot_cov_inv = np.linalg.inv(tot_cov)
        inverse_success = True
    except:
        inverse_success = False
        print(f"WARNING: tot_cov is not invertible, using pseudo-inverse")
        tot_cov_inv = np.linalg.pinv(tot_cov)
    chi2 = diff @ tot_cov_inv @ diff
    # don't include totally empty bins in the ndf
    ndf = len(diff)
    p_value, sigma = get_significance(chi2, ndf)

    s = f"$\chi^2/ndf$ = {chi2:.2f}/{ndf}, p-value = {p_value:.2e}, $\sigma$ = {sigma:.2f}"
    if inverse_success:
        ax1.text(0.05, 0.95, s, transform=ax1.transAxes, fontsize=8, ha="left", va="top")
    else:
        ax1.text(0.05, 0.95, "WARNING: tot_cov is not invertible, using pseudo-inverse\n" + s, transform=ax1.transAxes, fontsize=8, ha="left", va="top")

    if display_var is None:
        display_var = var
    
    if additional_scaling_factor != 1.0:
        ax1.set_ylabel(f"Counts (weighted\nto {additional_scaling_factor*3.33e19:.2e} POT)")
    else:
        ax1.set_ylabel("Counts (weighted\nto 3.33e19 POT)")
    ax1.set_title(title)
    ax1.set_xlim(display_bins[0], display_bins[-1])
    if log_x:
        ax1.set_xscale("log")
    if log_y:
        ax1.set_yscale("log")
        ax1.set_ylim(0.01, max(max_pred, max_data) * 10)
    else:
        ax1.set_ylim(0, max(max_pred, max_data) * 1.2)
    if include_legend:
        ax1.legend(ncol=2, loc='upper right', fontsize=6)
    
    # Remove x-axis labels from top plot
    ax1.set_xticklabels([])
    
    # Create ratio plot
    ratio = np.zeros_like(pred_counts)
    ratio_err_data = np.zeros_like(pred_counts)
    ratio_err_pred = np.zeros_like(pred_counts)
    
    for i in range(len(pred_counts)):
        if pred_counts[i] > 0:
            ratio[i] = data_counts[i] / pred_counts[i]
            # Error on data/pred
            ratio_err_data[i] = np.sqrt(data_counts[i]) / pred_counts[i]
            # Error on pred (if systematics are available)
            if plot_rw_systematics:
                ratio_err_pred[i] = tot_pred_frac_errors[i] * ratio[i]
        else:
            ratio[i] = np.nan
            ratio_err_data[i] = np.nan
            ratio_err_pred[i] = np.nan
    
    # Plot data points with error bars
    if include_data:
        ax2.errorbar(display_bin_centers, ratio, yerr=ratio_err_data, fmt="o", color="k", lw=0.5,
                    capsize=2, capthick=1, markersize=2, label="Data/Pred")
    
    # Plot prediction uncertainty band if systematics are available
    if plot_rw_systematics:
        for i in range(len(pred_counts)):
            if pred_counts[i] > 0:
                left = display_bins[i]
                width = display_bins[i+1] - display_bins[i]
                bottom_y = 1 - tot_pred_frac_errors[i]
                rect = Rectangle(
                    (left, bottom_y),
                    width,
                    2 * tot_pred_frac_errors[i],
                    hatch="/////",
                    fill=False,
                    edgecolor="k",
                    linewidth=0,
                )
                ax2.add_patch(rect)
    
    # Draw horizontal line at 1
    ax2.axhline(y=1, color='k', linestyle='--', linewidth=1)
    
    ax2.set_xlabel(display_var)
    ax2.set_ylabel("Data/Pred")
    ax2.set_xlim(display_bins[0], display_bins[-1])
    ax2.set_ylim(0, 2)
    if log_x:
        ax2.set_xscale("log")

    if page_num is not None:
        ax2.text(-0.1, -0.3, f"{page_num}", transform=ax2.transAxes, fontsize=8, ha="left", va="bottom")
    
    
    if savename is not None:
        plt.savefig(f"../plots/{savename}.pdf")
        plt.savefig(f"../plots/{savename}.png")

    if show: plt.show()

    if include_systematic_breakdown:

        plt.figure(figsize=(10, 6))

        tot_pred_frac_errors_extra_val = np.nan_to_num(np.concatenate([tot_pred_frac_errors, [tot_pred_frac_errors[-1]]]), nan=0, posinf=0, neginf=0)
        plt.step(display_bins, tot_pred_frac_errors_extra_val, where="post", label="Total", ls="-", color="k")

        mc_stat_errors = np.sqrt(np.diag(mc_stat_cov))
        mc_stat_frac_errors = mc_stat_errors / pred_counts
        mc_stat_frac_errors_extra_val = np.concatenate([mc_stat_frac_errors, [mc_stat_frac_errors[-1]]])
        plt.step(display_bins, mc_stat_frac_errors_extra_val, where="post", label="MC Stat", ls="-")

        for rw_sys_name, rw_sys_frac_cov in rw_sys_frac_cov_dic.items():
            diag_frac_errors = np.sqrt(np.diag(rw_sys_frac_cov))
            diag_frac_errors_extra_val = np.concatenate([diag_frac_errors, [diag_frac_errors[-1]]])
            if rw_sys_name in [
                    "AxFFCCQEshape_UBGenie",
                    "DecayAngMEC_UBGenie",
                    "NormCCCOH_UBGenie",
                    "NormNCCOH_UBGenie",
                    "RPA_CCQE_UBGenie",
                    "ThetaDelta2NRad_UBGenie",
                    "Theta_Delta2Npi_UBGenie",
                    "VecFFCCQEshape_UBGenie",
                    "XSecShape_CCMEC_UBGenie",
                    "xsr_scc_Fa3_SCC",
                    "xsr_scc_Fv3_SCC",
                ]:
                ls = "--"
            else:
                ls = "-"
            plt.step(display_bins, diag_frac_errors_extra_val, where="post", label=rw_sys_name, ls=ls)

        if include_legend:
            plt.legend(ncol=1, loc='upper right', fontsize=10)

        if log_x:
            plt.xscale("log")

        plt.xlabel(display_var)
        plt.ylabel("Fractional Error")
        plt.title("Systematic Breakdown")
        plt.xlim(display_bins[0], display_bins[-1])
        plt.ylim(0, min(1, np.max(tot_pred_frac_errors_extra_val) * 1.2))

        if savename is not None:
            plt.savefig(f"../plots/{savename}_systematic_breakdown.pdf")
        if show: plt.show()
        

    if return_p_value_info:
        return chi2, ndf, p_value, sigma, inverse_success
