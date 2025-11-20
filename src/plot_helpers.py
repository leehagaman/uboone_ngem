import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

from src.signal_categories import del1g_detailed_category_labels, del1g_detailed_category_labels_latex, del1g_detailed_category_colors, del1g_detailed_category_hatches
from src.signal_categories import filetype_category_labels, filetype_category_colors, filetype_category_hatches

from src.systematics import get_rw_sys_frac_cov_matrices, get_detvar_sys_frac_cov_matrices, get_data_stat_cov, get_pred_stat_cov
from src.systematics import get_significance, get_significance_from_p_value, chi2_decomposition

from src.df_helpers import get_vals

from src.file_locations import intermediate_files_location


def custom_step(bins, counts, ax=None, **kwargs):
    extra_counts = np.concatenate([counts, [counts[-1]]])
    extra_counts = np.nan_to_num(extra_counts, nan=0, posinf=0, neginf=0)
    if ax is None:
        plt.step(bins, extra_counts, where="post", **kwargs)
    else:
        ax.step(bins, extra_counts, where="post", **kwargs)


def add_underflow_overflow(bins, display_bins, include_overflow, include_underflow, log_x):
    if include_overflow and include_underflow:
        if log_x:
            bin_width_log = np.log10(bins[-1]) - np.log10(bins[-2])
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
    elif include_overflow:
        if log_x:
            bin_width_log = np.log10(bins[-1]) - np.log10(bins[-2])
            display_bins = np.concatenate([bins, [10**(np.log10(bins[-1]) + bin_width_log)]])
            bins = np.concatenate([bins, [np.inf]])
        else:
            bin_width = bins[-1] - bins[-2]
            display_bins = np.concatenate([bins, [bins[-1] + bin_width]])
            bins = np.concatenate([bins, [np.inf]])
    elif include_underflow:
        if log_x:
            bin_width_log = np.log10(bins[-1]) - np.log10(bins[-2])
            display_bins = np.concatenate([[10**(np.log10(bins[0]) - bin_width_log)], bins])
            bins = np.concatenate([[-np.inf], bins])
        else:
            bin_width = bins[1] - bins[0]
            display_bins = np.concatenate([[bins[0] - bin_width], bins])
            bins = np.concatenate([[-np.inf], bins])
    else:
        display_bins = bins

    return bins, display_bins


def auto_binning(all_vals):

    min_val = np.min(all_vals)
    max_val = np.max(all_vals)

    # calculate the 10% and 90% edges of all_vals
    lower_common_edge = np.percentile(all_vals, 5)
    upper_common_edge = np.percentile(all_vals, 95)

    if min_val != lower_common_edge:
        print("including underflow")
        include_underflow = True
    if max_val != upper_common_edge:
        print("including overflow")
        include_overflow = True

    print(f"choosing bins automatically, min_val = {min_val:.4e}, max_val = {max_val:.4e}, lower_common_edge = {lower_common_edge:.4e}, upper_common_edge = {upper_common_edge:.4e}")

    # if min_val, lower_common_edge, max_val, upper_common_edge are all integers, then use integers for the bins
    if min_val.is_integer() and lower_common_edge.is_integer() and max_val.is_integer() and upper_common_edge.is_integer():
        print("choosing integer bins")
        bins = np.arange(min_val, max_val + 1)
    elif len(np.unique(all_vals)) == 2:
        print("choosing two bins")
        bins = np.array([min_val - 0.5, (min_val + max_val) / 2, max_val + 0.5])
        include_overflow = False
        include_underflow = False
    elif 0 < lower_common_edge and 10_000 < upper_common_edge and np.log10(upper_common_edge) - np.log10(lower_common_edge) > 3:
        print("choosing log bins:", bins)
        bins = np.logspace(np.log10(lower_common_edge) - 0.01, np.log10(upper_common_edge) + 0.01, 21)
        log_x = True
    else:
        width = upper_common_edge - lower_common_edge
        bins = np.linspace(lower_common_edge - 0.01, upper_common_edge + 0.01, 21)
        print("choosing linear bins:", bins)

    bins, display_bins = add_underflow_overflow(bins, display_bins, include_overflow, include_underflow, log_x)

    if not np.all(np.diff(bins) > 0):
        raise ValueError(f"bins is not sorted: {bins}")

    if log_x:
        display_bin_centers = np.sqrt(display_bins[:-1] * display_bins[1:])
    else:
        display_bin_centers = (display_bins[:-1] + display_bins[1:]) / 2

    return bins, display_bins, display_bin_centers, log_x


def make_sys_frac_error_plot(tot_sys_frac_cov, tot_pred_sys_frac_cov, rw_sys_frac_cov_dic, detvar_sys_frac_cov_dic, pred_stat_cov, data_stat_cov, mc_pred_counts, pred_counts,
        display_var, display_bins, log_x, savename=None, show=True,
        include_total=True, include_pred_stat=True, include_data_stat=False, 
        include_rw=True, just_genie_breakdown=False,
        include_detvar=True, just_detvar_breakdown=False, detvar_df=None,
        print_sys_breakdown=False):

    if include_detvar and detvar_df is None:
        raise ValueError("detvar_df must be provided if include_detvar is True")

    if just_genie_breakdown:
        if just_detvar_breakdown or include_total or include_pred_stat or include_data_stat:
            raise ValueError(f"trying to plot non-genie systematic breakdown with just_detvar_breakdown! just_detvar_breakdown = {just_detvar_breakdown}, include_total = {include_total}, include_pred_stat = {include_pred_stat}, include_data_stat = {include_data_stat}")

    if just_detvar_breakdown:
        if just_genie_breakdown or include_total or include_pred_stat or include_data_stat:
            raise ValueError(f"trying to plot non-detvar systematic breakdown with just_genie_breakdown! just_genie_breakdown = {just_genie_breakdown}, include_total = {include_total}, include_pred_stat = {include_pred_stat}, include_data_stat = {include_data_stat}")

    plt.figure(figsize=(10, 6))

    if include_total and include_data_stat:
        custom_step(display_bins, np.sqrt(np.diag(tot_sys_frac_cov)), label="Total (incl. Data Stat)", ls="-", color="k")
        if print_sys_breakdown:
            print(f"Total (incl. Data Stat): {np.sqrt(np.diag(tot_sys_frac_cov))}")
    elif include_total:
        custom_step(display_bins, np.sqrt(np.diag(tot_pred_sys_frac_cov)), label="Total (no Data Stat)", ls="-", color="k")
        if print_sys_breakdown:
            print(f"Total (no Data Stat): {np.sqrt(np.diag(tot_pred_sys_frac_cov))}")
    
    if include_pred_stat:
        pred_stat_frac_cov = pred_stat_cov / np.outer(pred_counts, pred_counts)
        pred_stat_frac_cov = np.nan_to_num(pred_stat_frac_cov, nan=0, posinf=0, neginf=0)
        custom_step(display_bins, np.sqrt(np.diag(pred_stat_frac_cov)), label="Pred Stat", ls="-")
        if print_sys_breakdown:
            print(f"Pred Stat: {np.sqrt(np.diag(pred_stat_frac_cov))}")
    if include_data_stat:
        data_stat_frac_cov = data_stat_cov / np.outer(pred_counts, pred_counts)
        data_stat_frac_cov = np.nan_to_num(data_stat_frac_cov, nan=0, posinf=0, neginf=0)
        custom_step(display_bins, np.sqrt(np.diag(data_stat_frac_cov)), label="Data Stat", ls="-")
        if print_sys_breakdown:
            print(f"Data Stat: {np.sqrt(np.diag(data_stat_frac_cov))}")

    if include_rw:
        tot_genie_frac_cov = np.zeros((len(display_bins)-1, len(display_bins)-1))
        for rw_sys_name, rw_sys_frac_cov_mc in rw_sys_frac_cov_dic.items():
            rw_sys_cov = rw_sys_frac_cov_mc * np.outer(mc_pred_counts, mc_pred_counts) # fractional uncertainty on the MC pred, not including EXT
            rw_sys_frac_cov = rw_sys_cov / np.outer(pred_counts, pred_counts)
            rw_sys_frac_cov = np.nan_to_num(rw_sys_frac_cov, nan=0, posinf=0, neginf=0)
            if rw_sys_name in [
                    "All_UBGenie",
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
                tot_genie_frac_cov += rw_sys_frac_cov
                if not just_genie_breakdown:
                    continue
            label = rw_sys_name
            if rw_sys_name == "flux":
                label = "Flux"
            elif rw_sys_name == "reinteraction":
                label = "Reinteraction"
            custom_step(display_bins, np.sqrt(np.diag(rw_sys_frac_cov)), label=label)
            if print_sys_breakdown:
                print(f"{rw_sys_name}: {np.sqrt(np.diag(rw_sys_frac_cov))}")
        if just_genie_breakdown:
            custom_step(display_bins, np.sqrt(np.diag(tot_genie_frac_cov)), label="Total GENIE", color="k")
        else:
            custom_step(display_bins, np.sqrt(np.diag(tot_genie_frac_cov)), label="Total GENIE")
        if print_sys_breakdown:
            print(f"Total GENIE: {np.sqrt(np.diag(tot_genie_frac_cov))}")
    if include_detvar:
        tot_detvar_frac_cov = np.zeros((len(display_bins)-1, len(display_bins)-1))
        for detvar_sys_name, detvar_sys_frac_cov_mc in detvar_sys_frac_cov_dic.items():
            detvar_sys_cov = detvar_sys_frac_cov_mc * np.outer(mc_pred_counts, mc_pred_counts) # fractional uncertainty on the MC pred, not including EXT
            detvar_sys_frac_cov = detvar_sys_cov / np.outer(pred_counts, pred_counts)
            detvar_sys_frac_cov = np.nan_to_num(detvar_sys_frac_cov, nan=0, posinf=0, neginf=0)
            tot_detvar_frac_cov += detvar_sys_frac_cov
            if not just_detvar_breakdown:
                continue
            custom_step(display_bins, np.sqrt(np.diag(detvar_sys_frac_cov)), label=detvar_sys_name)
            if print_sys_breakdown:
                print(f"{detvar_sys_name}: {np.sqrt(np.diag(detvar_sys_frac_cov))}")
        if just_detvar_breakdown:
            custom_step(display_bins, np.sqrt(np.diag(tot_detvar_frac_cov)), label="Total Detvar", color="k")
        else:
            custom_step(display_bins, np.sqrt(np.diag(tot_detvar_frac_cov)), label="Total Detvar")
        if print_sys_breakdown:
            print(f"Total Detvar: {np.sqrt(np.diag(tot_detvar_frac_cov))}")

    plt.legend(ncol=1, loc='upper right', fontsize=10)

    if log_x:
        plt.xscale("log")

    plt.xlabel(display_var)
    plt.ylabel("Fractional Error")
    plt.title("Systematic Breakdown")
    plt.xlim(display_bins[0], display_bins[-1])

    if include_data_stat:
        plt.ylim(0, min(1, np.max(np.sqrt(np.diag(tot_sys_frac_cov))) * 1.2))
    else:
        plt.ylim(0, min(1, np.max(np.sqrt(np.diag(tot_pred_sys_frac_cov))) * 1.2))

    if savename is not None:
        plt.savefig(f"../plots/{savename}_systematic_breakdown.pdf")
    if show: plt.show()


def make_det_variation_histogram(var, display_var, bins, display_bins, display_bin_centers, include_overflow=True, include_underflow=False, log_x=False, log_y=False,
        additional_scaling_factor=1.0, normalizing_POT=3.33e19, 
        page_num=None, savename=None, show=True, detvar_df=None):


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05})

    cv_df = detvar_df.filter(pl.col("vartype") == "CV")

    cv_counts = np.histogram(get_vals(cv_df, var), weights=cv_df.get_column("wc_net_weight").to_numpy()*additional_scaling_factor, bins=bins)[0]
    max_y = np.max(cv_counts)
    ax1.hist(display_bin_centers, weights=cv_counts, bins=display_bins, histtype="step", color="k", lw=2, zorder=-1, label="CV")

    ratios_by_var_dic = {}

    total_cv_weight = np.sum(cv_df.get_column("wc_net_weight").to_numpy())
    for vartype in ["LYAtt", "LYDown", "LYRayleigh", "WireModX", "Recomb2", "SCE"]:
        curr_df = detvar_df.filter(pl.col("vartype") == vartype)

        curr_filetype_rse_df = curr_df.select(["filetype", "run", "subrun", "event"])
        matching_cv_df = cv_df.join(curr_filetype_rse_df, on=["filetype", "run", "subrun", "event"], how="inner")

        matching_cv_weight = np.sum(matching_cv_df.get_column("wc_net_weight").to_numpy())
        match_weight = total_cv_weight / matching_cv_weight

        matching_curr_df = curr_df.join(matching_cv_df.select(["filetype", "run", "subrun", "event"]), on=["filetype", "run", "subrun", "event"], how="inner")

        matching_cv_counts = np.histogram(get_vals(matching_cv_df, var), weights=matching_cv_df.get_column("wc_net_weight").to_numpy()*additional_scaling_factor*match_weight, bins=bins)[0]
        matching_var_counts = np.histogram(get_vals(matching_curr_df, var), weights=matching_curr_df.get_column("wc_net_weight").to_numpy()*additional_scaling_factor*match_weight, bins=bins)[0]

        var_over_cv_ratio = matching_cv_counts / matching_var_counts
        var_over_cv_ratio = np.nan_to_num(var_over_cv_ratio, nan=0, posinf=0, neginf=0)
        scaled_var_over_cv_ratio = var_over_cv_ratio * cv_counts

        ax1.hist(display_bin_centers, weights=scaled_var_over_cv_ratio, bins=display_bins, label=vartype, histtype="step")

        ratios_by_var_dic[vartype] = var_over_cv_ratio

        max_y = max(max_y, np.max(scaled_var_over_cv_ratio))

    ax1.set_xticklabels([])

    for vartype in ["LYAtt", "LYDown", "LYRayleigh", "WireModX", "Recomb2", "SCE"]:
        
        ratios = ratios_by_var_dic[vartype]
        custom_step(display_bins, ratios, ax=ax2, label=vartype)                

        # Draw horizontal line at 1
        ax2.axhline(y=1, color='k', linestyle='--', linewidth=1)
        
        ax2.set_xlabel(display_var)
        ax2.set_ylabel("Var/CV")
        ax2.set_xlim(display_bins[0], display_bins[-1])
        ax2.set_ylim(0.5, 1.5)
        if log_x:
            ax2.set_xscale("log")

    ax1.set_ylabel(f"MC Pred counts (no EXT) (weighted\nto {additional_scaling_factor*normalizing_POT:.2e} POT)")
    ax1.set_xlim(display_bins[0], display_bins[-1])
    if log_x:
        ax1.set_xscale("log")
    if log_y:
        ax1.set_yscale("log")
        ax1.set_ylim(0.01, max_y * 10)
    else:
        ax1.set_ylim(0, max_y * 1.2)
    ax1.legend(ncol=1, loc='upper right', fontsize=12)

    if savename is not None:
        plt.savefig(f"../plots/{savename}_det_variation_histograms.pdf")
    if show: plt.show()

def make_histogram_plot(

        # core information for plot
        pred_sel_df=None, data_sel_df=None, pred_and_data_sel_df=None, 
        bins=None, include_overflow=True, include_underflow=False, log_x=False, log_y=False, 
        var=None, display_var=None, breakdown_type="del1g_detailed", 
        title=None, savename=None,
        iso1g_norm_factor=None, del1g_norm_factor=None, 
        include_data=True, additional_scaling_factor=1.0, normalizing_POT=3.33e19, 
        include_legend=True, show=True,
        page_num=None,
        include_ratio=True, include_decomposition=False,

        # information for optional systematics
        selname=None,
        dont_load_rw_from_systematic_cache=False, dont_load_detvar_from_systematic_cache=False,
        use_rw_systematics=False, weights_df=None,
        use_detvar_systematics=False, detvar_df=None,
        use_detvar_bootstrapping=True,
        return_p_value_info=False,

        # optional detector variation histogram plot
        plot_det_variations=False,

        # optional systematics breakdown plot
        plot_sys_breakdown=False,
        include_total=True, include_pred_stat=True, include_data_stat=False, 
        include_rw=True, just_genie_breakdown=False,
        include_detvar=True, just_detvar_breakdown=False,
        print_sys_breakdown=False,
        
        ):

    if include_decomposition and not include_ratio:
        raise ValueError("include_decomposition requires include_ratio to be True")
    if include_decomposition and not include_data:
        raise ValueError("include_decomposition requires include_data to be True")
    if include_decomposition and not use_rw_systematics:
        raise ValueError("include_decomposition requires use_rw_systematics to be True")

    if pred_and_data_sel_df is not None:
        pred_sel_df = pred_and_data_sel_df.filter(pl.col("filetype") != "data")
        data_sel_df = pred_and_data_sel_df.filter(pl.col("filetype") == "data")        
    elif pred_sel_df is None:
        raise ValueError("Either pred_sel_df and or pred_and_data_sel_df must be provided")    

    if bins is None: # automatically calculate binning
        pred_vals = get_vals(pred_sel_df, var)
        if include_data:
            data_vals = get_vals(data_sel_df, var)
            all_vals = np.concatenate([pred_vals, data_vals])
        else:
            all_vals = pred_vals

        bins, display_bins, display_bin_centers, log_x = auto_binning(all_vals)
    else:
        bins, display_bins = add_underflow_overflow(bins, bins, include_overflow, include_underflow, log_x)
        if log_x:
            display_bin_centers = np.sqrt(display_bins[:-1] * display_bins[1:])
        else:
            display_bin_centers = (display_bins[:-1] + display_bins[1:]) / 2


    # plotting the main histogram prediction, optionally with ratio and decomposition
    if include_decomposition:
        fig = plt.figure(figsize=(10, 12))
        gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 0.01, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        # gs[2] is left blank as a spacer
        ax3 = fig.add_subplot(gs[3])
    elif include_ratio:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # making the total prediction histogram with breakdown categories
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

    pred_counts = np.histogram(get_vals(pred_sel_df, var), weights=pred_sel_df.get_column("wc_net_weight").to_numpy()*additional_scaling_factor, bins=bins)[0]
    mc_pred_counts = np.histogram(get_vals(pred_sel_df.filter(pl.col("filetype") != "ext"), var), weights=pred_sel_df.filter(pl.col("filetype") != "ext").get_column("wc_net_weight").to_numpy()*additional_scaling_factor, bins=bins)[0]
    
    max_y = np.max(pred_counts)
    ax1.plot([], [], c="k", lw=0.5, label=f"Total Pred: {tot_pred:.1f} ({tot_unweighted_pred:.0f})")

    # optionally including data points
    if include_data:
        data_counts = np.histogram(get_vals(data_sel_df, var), bins=bins)[0]
        max_y = max(max_y, np.max(data_counts))

        ax1.errorbar(display_bin_centers, data_counts, yerr=np.sqrt(data_counts), fmt="o", color="k", lw=0.5, 
                    capsize=2, capthick=1, markersize=2, label=f"3.33e19 POT Run 4b Data ({np.sum(data_counts)})")

        diff = data_counts - pred_counts

    if use_detvar_systematics and not use_rw_systematics:
        raise ValueError("use_detvar_systematics requires use_rw_systematics to be True")

    # optionally including rw systematic uncertainties
    if use_rw_systematics:
        
        if selname is None:
            raise ValueError("selname must be provided if use_rw_systematics is True")

        rw_sys_frac_cov_dic = get_rw_sys_frac_cov_matrices(
            pred_sel_df.filter(pl.col("filetype") != "ext"), selname, var, bins, dont_load_rw_from_systematic_cache=dont_load_rw_from_systematic_cache, weights_df=weights_df
        )
        combined_rw_sys_frac_cov = np.zeros((len(bins)-1, len(bins)-1))
        for rw_sys_frac_cov_name, rw_sys_frac_cov in rw_sys_frac_cov_dic.items():
            combined_rw_sys_frac_cov += rw_sys_frac_cov
        combined_rw_sys_cov = combined_rw_sys_frac_cov * np.outer(mc_pred_counts, mc_pred_counts) # fractional uncertainty on the MC pred, not including EXT
        data_stat_cov = get_data_stat_cov(data_counts, pred_counts)
        pred_stat_cov = get_pred_stat_cov(get_vals(pred_sel_df, var), pred_sel_df.get_column("wc_net_weight").to_numpy(), bins)
        nodetvar_sys_cov = combined_rw_sys_cov + data_stat_cov + pred_stat_cov
        nodetvar_sys_frac_cov = nodetvar_sys_cov / np.outer(pred_counts, pred_counts)
        nodetvar_sys_frac_cov = np.nan_to_num(nodetvar_sys_frac_cov, nan=0, posinf=0, neginf=0)
        nodetvar_pred_sys_cov = combined_rw_sys_cov + pred_stat_cov
        nodetvar_pred_sys_frac_cov = nodetvar_pred_sys_cov / np.outer(pred_counts, pred_counts)
        nodetvar_pred_sys_frac_cov = np.nan_to_num(nodetvar_pred_sys_frac_cov, nan=0, posinf=0, neginf=0)
        nodetvar_pred_sys_frac_errors = np.sqrt(np.diag(nodetvar_pred_sys_frac_cov))
        for i in range(len(pred_counts)):
            abs_err = nodetvar_pred_sys_frac_errors[i] * pred_counts[i]
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
                label="No-DetVar. Syst. Uncert." if i == 0 else None,
            )
            ax1.add_patch(rect)

        # optionally including detvar systematic uncertainties
        if use_detvar_systematics:
            if detvar_df is None:
                raise ValueError("detvar_df must be provided if use_detvar_systematics is True")

            detvar_sys_frac_cov_dic = get_detvar_sys_frac_cov_matrices(
                detvar_df, selname, var, bins, dont_load_detvar_from_systematic_cache=dont_load_detvar_from_systematic_cache, use_detvar_bootstrapping=use_detvar_bootstrapping
            )
            combined_detvar_sys_frac_cov_mc = np.zeros((len(bins)-1, len(bins)-1))
            for detvar_sys_frac_cov_name, detvar_sys_frac_cov in detvar_sys_frac_cov_dic.items():
                combined_detvar_sys_frac_cov_mc += detvar_sys_frac_cov
            combined_detvar_sys_cov = combined_detvar_sys_frac_cov_mc * np.outer(mc_pred_counts, mc_pred_counts) # fractional uncertainty on the MC pred, not including EXT
            combined_detvar_sys_frac_cov = combined_detvar_sys_cov / np.outer(pred_counts, pred_counts)
            combined_detvar_sys_frac_cov = np.nan_to_num(combined_detvar_sys_frac_cov, nan=0, posinf=0, neginf=0)
            tot_sys_cov = combined_detvar_sys_cov + combined_rw_sys_cov + data_stat_cov + pred_stat_cov
            tot_sys_frac_cov = tot_sys_cov / np.outer(pred_counts, pred_counts)
            tot_sys_frac_cov = np.nan_to_num(tot_sys_frac_cov, nan=0, posinf=0, neginf=0)
            tot_pred_sys_cov = combined_detvar_sys_cov + combined_rw_sys_cov + pred_stat_cov
            tot_pred_sys_frac_cov = tot_pred_sys_cov / np.outer(pred_counts, pred_counts)
            tot_pred_sys_frac_cov = np.nan_to_num(tot_pred_sys_frac_cov, nan=0, posinf=0, neginf=0)
            tot_pred_sys_frac_errors = np.sqrt(np.diag(tot_pred_sys_frac_cov))

            for i in range(len(pred_counts)):
                abs_err = tot_pred_sys_frac_errors[i] * pred_counts[i]
                left = display_bins[i]
                width = display_bins[i+1] - display_bins[i]
                bottom_y = pred_counts[i] - abs_err
                rect = Rectangle(
                    (left, bottom_y),
                    width,
                    2 * abs_err,
                    hatch=r"\\\\\\",
                    fill=False,
                    edgecolor="k",
                    linewidth=0,
                    label="Tot. Syst. Uncert." if i == 0 else None,
                )
                ax1.add_patch(rect)

        if include_data:
            empty_indices = np.where(data_counts <= 1)[0] # removing bins with low data counts for the chi2 calculation
            if len(empty_indices) > 0:
                print("removing bins with 0 or 1 data counts at indices:", empty_indices)
                diff = np.delete(diff, empty_indices)
                nodetvar_sys_cov = np.delete(nodetvar_sys_cov, empty_indices, axis=0)
                nodetvar_sys_cov = np.delete(nodetvar_sys_cov, empty_indices, axis=1)
            
            try:
                nodetvar_sys_cov_inv = np.linalg.inv(nodetvar_sys_cov)
                nodetvar_inverse_success = True
            except:
                nodetvar_inverse_success = False
                print(f"WARNING: rw_tot_cov is not invertible, using pseudo-inverse")
                nodetvar_sys_cov_inv = np.linalg.pinv(nodetvar_sys_cov)
            nodetvar_chi2 = diff @ nodetvar_sys_cov_inv @ diff
            if np.isnan(nodetvar_chi2):
                raise ValueError("nodetvar_chi2 is nan! nodetvar_sys_cov diagonal is:", np.diag(nodetvar_sys_cov))
            ndf = len(diff) # don't include totally empty bins in the ndf
            nodetvar_p_value, nodetvar_sigma = get_significance(nodetvar_chi2, ndf)
            s = ""
            if not nodetvar_inverse_success:
                s += "WARNING: nodetvar_sys_cov is not invertible, using pseudo-inverse\n"
            s += f"No DetVar: $\chi^2/ndf$ = {nodetvar_chi2:.2f}/{ndf}, p = {nodetvar_p_value:.2e}, $\sigma$ = {nodetvar_sigma:.2f}"

            p_value_info_dic = {}
            p_value_info_dic["nodetvar_chi2"] = nodetvar_chi2
            p_value_info_dic["nodetvar_p_value"] = nodetvar_p_value
            p_value_info_dic["nodetvar_sigma"] = nodetvar_sigma
            p_value_info_dic["nodetvar_inverse_success"] = nodetvar_inverse_success

            if use_detvar_systematics:

                tot_sys_cov = np.delete(tot_sys_cov, empty_indices, axis=0)
                tot_sys_cov = np.delete(tot_sys_cov, empty_indices, axis=1)
                try:
                    tot_sys_cov_inv = np.linalg.inv(tot_sys_cov)
                    tot_inverse_success = True
                except:
                    tot_inverse_success = False
                    print(f"WARNING: tot_sys_cov is not invertible, using pseudo-inverse")
                    tot_sys_cov_inv = np.linalg.pinv(tot_sys_cov)
                tot_chi2 = diff @ tot_sys_cov_inv @ diff
                if np.isnan(tot_chi2):
                    raise ValueError("tot_chi2 is nan! tot_sys_cov diagonal is:", np.diag(tot_sys_cov))
                ndf = len(diff) # don't include totally empty bins in the ndf
                tot_p_value, tot_sigma = get_significance(tot_chi2, ndf)
                s += "\n"
                if not tot_inverse_success:
                    s += "WARNING: tot_sys_cov is not invertible, using pseudo-inverse\n"
                s += f"$\chi^2/ndf$ = {tot_chi2:.2f}/{ndf}, p = {tot_p_value:.2e}, $\sigma$ = {tot_sigma:.2f}"

                p_value_info_dic["tot_chi2"] = tot_chi2
                p_value_info_dic["tot_p_value"] = tot_p_value
                p_value_info_dic["tot_sigma"] = tot_sigma
                p_value_info_dic["tot_inverse_success"] = tot_inverse_success
                p_value_info_dic["ndf"] = ndf

            ax1.text(0.03, 0.97, s, transform=ax1.transAxes, fontsize=8, ha="left", va="top")

    if display_var is None:
        display_var = var
    
    if additional_scaling_factor != 1.0:
        ax1.set_ylabel(f"Counts (weighted\nto {additional_scaling_factor*normalizing_POT:.2e} POT)")
    else:
        ax1.set_ylabel(f"Counts (weighted\nto {normalizing_POT:.2e} POT)")
    ax1.set_title(title)
    ax1.set_xlim(display_bins[0], display_bins[-1])
    if log_x:
        ax1.set_xscale("log")
    if log_y:
        ax1.set_yscale("log")
        ax1.set_ylim(0.01, max_y * 10)
    else:
        ax1.set_ylim(0, max_y * 1.2)
    if include_legend:
        ax1.legend(ncol=2, loc='upper right', fontsize=6)
    
    if include_ratio:
        ax1.set_xticklabels([])
            
        ratio = np.zeros_like(pred_counts)
        ratio_err_data = np.zeros_like(pred_counts)
        
        for i in range(len(pred_counts)):
            if pred_counts[i] > 0:
                ratio[i] = data_counts[i] / pred_counts[i]
                ratio_err_data[i] = np.sqrt(data_counts[i]) / pred_counts[i]
        
        # Plot data points with error bars
        if include_data:
            ax2.errorbar(display_bin_centers, ratio, yerr=ratio_err_data, fmt="o", color="k", lw=0.5,
                        capsize=2, capthick=1, markersize=2, label="Data/Pred")
        
        # Plot prediction uncertainty band if systematics are available
        if use_rw_systematics:
            for i in range(len(pred_counts)):
                if pred_counts[i] > 0:
                    left = display_bins[i]
                    width = display_bins[i+1] - display_bins[i]

                    bottom_y = 1 - nodetvar_pred_sys_frac_errors[i]
                    rect = Rectangle(
                        (left, bottom_y),
                        width,
                        2 * nodetvar_pred_sys_frac_errors[i],
                        hatch="/////",
                        fill=False,
                        edgecolor="k",
                        linewidth=0,
                    )
                    ax2.add_patch(rect)

                    if use_detvar_systematics:
                        bottom_y = 1 - tot_pred_sys_frac_errors[i]
                        rect = Rectangle(
                            (left, bottom_y),
                            width,
                            2 * tot_pred_sys_frac_errors[i],
                            hatch=r"\\\\\\",
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
    else:
        # Set x-axis label on main plot when ratio panel is not included
        ax1.set_xlabel(display_var)
        
        if page_num is not None:
            ax1.text(-0.1, -0.3, f"{page_num}", transform=ax1.transAxes, fontsize=8, ha="left", va="bottom")

    if include_decomposition:

        nodetvar_epsilons = chi2_decomposition(diff, nodetvar_sys_cov)
        tot_epsilons = chi2_decomposition(diff, tot_sys_cov)
        num_components = len(nodetvar_epsilons)

        ax3.axhline(y=0, color='grey', linestyle='--', linewidth=1)

        ax3.fill_between([-0.5, num_components - 0.5], -3, -2, color='red', alpha=0.2)
        ax3.fill_between([-0.5, num_components - 0.5], -2, -1, color='gold', alpha=0.2)
        ax3.fill_between([-0.5, num_components - 0.5], -1, 1, color='lime', alpha=0.2)
        ax3.fill_between([-0.5, num_components - 0.5], 1, 2, color='gold', alpha=0.2)
        ax3.fill_between([-0.5, num_components - 0.5], 2, 3, color='red', alpha=0.2)

        ax3.scatter(np.arange(num_components), nodetvar_epsilons, color='b', label="No DetVar")
        ax3.scatter(np.arange(num_components), tot_epsilons, color='k', label="Tot")

        ax3.set_xlabel("Decomposition Index")
        ax3.set_ylabel(r"$\epsilon_i^2$")
        ax3.set_xlim(-0.5, num_components - 0.5)
        ax3.set_ylim(-5, 5)
        ax3.legend()

        nodetvar_max_local_sigma = np.max(np.abs(nodetvar_epsilons))
        tot_max_local_sigma = np.max(np.abs(tot_epsilons))

        nodetvar_max_local_p_value = get_significance(nodetvar_max_local_sigma**2, 1)[0]
        tot_max_local_p_value = get_significance(tot_max_local_sigma**2, 1)[0]

        nodetvar_global_p_value = 1 - (1 - nodetvar_max_local_p_value)**num_components
        tot_global_p_value = 1 - (1 - tot_max_local_p_value)**num_components

        nodetvar_global_sigma = get_significance_from_p_value(nodetvar_global_p_value)
        tot_global_sigma = get_significance_from_p_value(tot_global_p_value)

        #nodetvar_str = f"No DetVar: max local $\sigma$ = {nodetvar_max_local_sigma:.2f}, p = {nodetvar_max_local_p_value:.2e}; global $\sigma$ = {nodetvar_global_sigma:.2f}, p = {nodetvar_global_p_value:.2e}"
        #tot_str = f"Tot: max local $\sigma$ = {tot_max_local_sigma:.2f}, p = {tot_max_local_p_value:.2e}; global $\sigma$ = {tot_global_sigma:.2f}, p = {tot_global_p_value:.2e}"

        nodetvar_str = f"No DetVar: max local $\sigma$ = {nodetvar_max_local_sigma:.2f}, global $\sigma$ = {nodetvar_global_sigma:.2f}"
        tot_str = f"Tot: max local $\sigma$ = {tot_max_local_sigma:.2f}, global $\sigma$ = {tot_global_sigma:.2f}"

        p_value_info_dic["nodetvar_max_local_sigma"] = nodetvar_max_local_sigma
        p_value_info_dic["nodetvar_global_sigma"] = nodetvar_global_sigma
        p_value_info_dic["tot_max_local_sigma"] = tot_max_local_sigma
        p_value_info_dic["tot_global_sigma"] = tot_global_sigma
        p_value_info_dic["nodetvar_max_local_p_value"] = nodetvar_max_local_p_value
        p_value_info_dic["nodetvar_global_p_value"] = nodetvar_global_p_value
        p_value_info_dic["tot_max_local_p_value"] = tot_max_local_p_value
        p_value_info_dic["tot_global_p_value"] = tot_global_p_value

        ax3.text(0.03, 0.97, nodetvar_str + "\n" + tot_str, transform=ax3.transAxes, fontsize=8, ha="left", va="top")
    
    
    if savename is not None:
        plt.savefig(f"../plots/{savename}.pdf")
        plt.savefig(f"../plots/{savename}.png")

    if show: plt.show()

    if plot_det_variations:
        make_det_variation_histogram(var, display_var, bins, display_bins, display_bin_centers, include_overflow, include_underflow, log_x, log_y, additional_scaling_factor, normalizing_POT, page_num, savename, show, detvar_df)


    if plot_sys_breakdown:
        if not use_rw_systematics or not use_detvar_systematics:
            raise ValueError("plot_sys_breakdown requires use_rw_systematics and use_detvar_systematics to be True")
        
        make_sys_frac_error_plot(tot_sys_frac_cov, tot_pred_sys_frac_cov, rw_sys_frac_cov_dic, detvar_sys_frac_cov_dic, pred_stat_cov, data_stat_cov, 
            mc_pred_counts, pred_counts, display_var, display_bins, log_x, savename, show, 
            include_total, include_pred_stat, include_data_stat, include_rw, just_genie_breakdown, include_detvar, just_detvar_breakdown, print_sys_breakdown)

    if return_p_value_info:
        return p_value_info_dic
    
