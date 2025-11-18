import numpy as np
import os
import hashlib
from tqdm import tqdm
import polars as pl
from src.file_locations import intermediate_files_location, covariance_cache_location
from src.df_helpers import get_vals

from scipy.special import erfinv, erfcinv, erfc
from scipy.stats import chi2
from scipy.stats import poisson


def get_significance(chisquare, ndf, printout=False):
    # probability of getting a more extreme result
    p_value = 1. - chi2.cdf(chisquare, ndf)
    sigma = np.sqrt(2.) * erfcinv(p_value)
    return p_value, sigma

def get_significance_from_p_value(p_value):
    sigma = np.sqrt(2.) * erfcinv(p_value)
    return sigma

def get_poisson_significance(measured, expected):
    # probability of getting an equal or more extreme result
    p_value = 1. - poisson.cdf(measured - 0.001, expected)
    sigma = np.sqrt(2.) * erfinv(1. - p_value)
    return p_value, sigma


def get_data_stat_cov(data_counts, pred_counts, method="pearson"):
    if method == "pearson":
        return np.diag(pred_counts)
    elif method == "neyman":
        return np.diag(data_counts)
    elif method == "cnp":
        # see https://github.com/BNLIF/LEEana/blob/58ee2a420d3e6b864f2eefb96181b10f404d6c39/plot_script/plot_check_gof.C#L594
        # see equation 14 of https://arxiv.org/pdf/1903.07185
        num_bins = len(pred_counts)
        diag_cov = np.zeros(num_bins)
        for bin_i in range(num_bins):
            if data_counts[bin_i] == 0 and pred_counts[bin_i] == 0:
                diag_cov[bin_i] = 1e9
            elif pred_counts[bin_i] > 0:
                diag_cov[bin_i] = 3 / (1 / data_counts[bin_i] + 2 / pred_counts[bin_i])
        return np.diag(diag_cov)
    else:
        raise ValueError(f"Invalid data stat cov method: {method}")


def get_pred_stat_cov(pred_vals, pred_weights, bins):
    diag_cov = np.zeros(len(bins) - 1)
    for bin_i in range(len(bins) - 1):
        pred_weights_in_bin = pred_weights[(pred_vals >= bins[bin_i]) & (pred_vals < bins[bin_i+1])]
        diag_cov[bin_i] = np.sum(pred_weights_in_bin**2)
    return np.diag(diag_cov)


def create_cov_matrix(unisim_hists, cv_hist, manual_uni_count=None):
    # Similar to np.cov, but manually setting the number of univirses to divide by
    # In some of the unisims, one of the universese is actually the CV. In these cases, 
    # we need to divide by the number of universes minus one.
    curr_cov = np.zeros((len(unisim_hists), len(unisim_hists)))
    for uni_i in range(unisim_hists.shape[1]):
        diff = np.array(unisim_hists[:,uni_i]) - np.array(cv_hist)
        row_diffs = np.tile(diff, (len(unisim_hists), 1))
        col_diffs = np.tile(np.reshape(diff, (len(unisim_hists), 1)), (1, len(unisim_hists)))
        curr_cov += row_diffs * col_diffs

    if manual_uni_count is None:
        uni_count = unisim_hists.shape[1]
    else:
        uni_count = manual_uni_count

    return curr_cov / uni_count


def create_universe_histograms(vals, bins, sys_weight_arrs, other_weights, description="", quiet=False):

    num_bins = len(bins) - 1

    sys_weight_arrs = np.stack(sys_weight_arrs)

    num_unis = sys_weight_arrs.shape[1]

    # finding the bin indices once, rather than a thousand times for each weight
    bin_indices = np.searchsorted(bins, vals, side='right') - 1

    # removing events outside of the bins
    valid = (bin_indices >= 0) & (bin_indices < len(bins) - 1)
    bin_indices = bin_indices[valid]
    sys_weight_arrs = sys_weight_arrs[valid, :]
    other_weights = other_weights[valid]

    # replacing nan, inf, negative, or large weights with 1, the same way as we did for the CV in postprocessing.py
    sys_weight_arrs = np.nan_to_num(sys_weight_arrs, nan=1, posinf=1, neginf=1)
    sys_weight_arrs[sys_weight_arrs > 30] = 1
    sys_weight_arrs[sys_weight_arrs < 0] = 1

    # creating the histogram for each universe (with pre-computed bin indices)
    hists = np.zeros((num_bins, num_unis))
    for uni_i, uni_weights in tqdm(enumerate(sys_weight_arrs.T), total=num_unis, desc=f"Creating {description} universe histograms", disable=quiet):
        hists[:, uni_i] = np.bincount(bin_indices, weights=uni_weights*other_weights, minlength=num_bins)

    return hists


def create_frac_cov_matrices(mc_pred_df, var, bins, weights_df=None):

    print("creating systematic covariance matrices...")

    if weights_df is None:
        print("loading weights_df from parquet file...")
        weights_df = pl.read_parquet(f"{intermediate_files_location}/presel_weights_df.parquet")

    print("merging mc_pred_df and weights_df...")
    pred_vars = ["filename", "run", "subrun", "event", "wc_net_weight", "wc_weight_cv", "wc_weight_spline", var]
    merged_df = mc_pred_df.select(pred_vars).join(weights_df, on=["filename", "run", "subrun", "event"], how="inner")
    if merged_df.height != mc_pred_df.height:
        print(f"WARNING: missing events in weights_df, approximate reweightable systematic uncertainties! {mc_pred_df.height=}, {weights_df.height=}")

    pred_vals = get_vals(merged_df, var)

    rw_sys_frac_cov_dic = {}

    print("getting CV histogram and non-GENIE weights...")
    cv_hist = np.histogram(pred_vals, weights=merged_df.get_column("wc_net_weight").to_numpy(), bins=bins)[0]
    cv_hist = np.maximum(cv_hist, 1e-3) # avoiding nans when we divide in the next step, this bin will have large stat uncertainty anyway

    # we use these weights when we replace GENIE weight_cv with the new systematic weight
    non_genie_cv_weights = merged_df.get_column("wc_net_weight").to_numpy() / merged_df.get_column("wc_weight_cv").to_numpy()

    # we use these weights when we consider a new weight independent of the GENIE CV weights
    normal_weights = merged_df.get_column("wc_net_weight").to_numpy()

    All_UBGenie_hists = create_universe_histograms(pred_vals, bins, merged_df.get_column("All_UBGenie").to_numpy(), non_genie_cv_weights, description="All_UBGenie")
    #All_UBGenie_cov = np.cov(All_UBGenie_hists - cv_hist[:, None])
    All_UBGenie_cov = create_cov_matrix(All_UBGenie_hists, cv_hist)
    rw_sys_frac_cov_dic["All_UBGenie"] = All_UBGenie_cov / np.outer(cv_hist, cv_hist)

    flux_hists = create_universe_histograms(pred_vals, bins, merged_df.get_column("flux_all").to_numpy(), normal_weights, description="flux")
    flux_cov = np.cov(flux_hists - cv_hist[:, None])
    rw_sys_frac_cov_dic["flux"] = flux_cov / np.outer(cv_hist, cv_hist)

    reint_hists = create_universe_histograms(pred_vals, bins, merged_df.get_column("reint_all").to_numpy(), normal_weights, description="reinteraction")
    reint_cov = np.cov(reint_hists - cv_hist[:, None])
    rw_sys_frac_cov_dic["reinteraction"] = reint_cov / np.outer(cv_hist, cv_hist)

    for unisim_type in [
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

        if unisim_type == "RPA_CCQE_UBGenie":
            num_unis = 2
        elif unisim_type == "xsr_scc_Fa3_SCC" or unisim_type == "xsr_scc_Fv3_SCC":
            num_unis = 10
        else:
            num_unis = None

        if unisim_type == "xsr_scc_Fa3_SCC" or unisim_type == "xsr_scc_Fv3_SCC":
            other_weights = normal_weights
        else:
            other_weights = non_genie_cv_weights

        unisim_hists = create_universe_histograms(pred_vals, bins, merged_df.get_column(unisim_type).to_numpy(), other_weights, description=unisim_type)
        unisim_cov = create_cov_matrix(unisim_hists, cv_hist, manual_uni_count=num_unis)
        rw_sys_frac_cov_dic[unisim_type] = unisim_cov / np.outer(cv_hist, cv_hist)

    print("done getting reweightable systematic covariance matrices")

    return rw_sys_frac_cov_dic


def create_detvar_frac_cov_matrices(detvar_df, var, bins):

    print("creating detvar systematic covariance matrices...")

    if detvar_df is None:
        print("loading detvar_df from parquet file...")
        detvar_df = pl.read_parquet(f"{intermediate_files_location}/detvar_presel_df_train_vars.parquet")

    cv_df = detvar_df.filter(pl.col("vartype") == "CV")

    detvar_sys_frac_cov_dic = {}

    for vartype in ["LYAtt", "LYDown", "LYRayleigh", "WireModX", "Recomb2", "SCE"]:
        curr_df = detvar_df.filter(pl.col("vartype") == vartype)

        curr_filetype_rse_df = curr_df.select(["filetype", "run", "subrun", "event"])
        matching_cv_df = cv_df.join(curr_filetype_rse_df, on=["filetype", "run", "subrun", "event"], how="inner")

        matching_curr_df = curr_df.join(matching_cv_df.select(["filetype", "run", "subrun", "event"]), on=["filetype", "run", "subrun", "event"], how="inner")

        matching_cv_counts = np.histogram(get_vals(matching_cv_df, var), weights=matching_cv_df.get_column("wc_net_weight").to_numpy(), bins=bins)[0]
        matching_var_counts = np.histogram(get_vals(matching_curr_df, var), weights=matching_curr_df.get_column("wc_net_weight").to_numpy(), bins=bins)[0]

        diff = matching_cv_counts - matching_var_counts
        detvar_sys_frac_cov_dic[vartype] = np.cov(diff) / np.outer(matching_cv_counts, matching_cv_counts)

    print("done getting detvar systematic covariance matrices")

    return detvar_sys_frac_cov_dic

def _key_hash(var, bins):
    bins_arr = np.asarray(bins, dtype=float)
    h = hashlib.sha256()
    h.update(var.encode("utf-8"))
    h.update(b"\x00")
    h.update(bins_arr.tobytes())
    return h.hexdigest()

def get_rw_sys_frac_cov_matrices(mc_pred_df, var, bins, dont_load_from_systematic_cache=False, weights_df=None):

    if not dont_load_from_systematic_cache:
        key_h = _key_hash(var, bins)
        cache_path = f"{covariance_cache_location}/cov_{key_h}.npz"
        if os.path.exists(cache_path):
            print("loading reweightable systematic covariance matrices from cache...")
            with np.load(cache_path, allow_pickle=True) as data:
                return data["rw_sys_frac_cov_dic"].item()

    rw_sys_frac_cov_dic = create_frac_cov_matrices(mc_pred_df, var, bins, weights_df=weights_df)

    key_h = _key_hash(var, bins)
    cache_path = f"{covariance_cache_location}/cov_{key_h}.npz"
    np.savez_compressed(cache_path, rw_sys_frac_cov_dic=rw_sys_frac_cov_dic)

    return rw_sys_frac_cov_dic

def get_detvar_sys_frac_cov_matrices(var, bins, dont_load_from_systematic_cache=False, detvar_df=None):

    if not dont_load_from_systematic_cache:
        key_h = _key_hash(var, bins)
        cache_path = f"{covariance_cache_location}/detvar_cov_{key_h}.npz"
        if os.path.exists(cache_path):
            print("loading DetVar systematic covariance matrices from cache...")
            with np.load(cache_path, allow_pickle=True) as data:
                return data["detvar_sys_frac_cov_dic"].item()

    detvar_sys_frac_cov_dic = create_detvar_frac_cov_matrices(detvar_df, var, bins)

    key_h = _key_hash(var, bins)
    cache_path = f"{covariance_cache_location}/detvar_cov_{key_h}.npz"
    np.savez_compressed(cache_path, detvar_sys_frac_cov_dic=detvar_sys_frac_cov_dic)

    return detvar_sys_frac_cov_dic
