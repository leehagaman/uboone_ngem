import numpy as np
import os
import hashlib
from tqdm import tqdm
import polars as pl
from .file_locations import intermediate_files_location, covariance_cache_location
from .df_helpers import get_vals

from scipy.special import erfinv, erfcinv, erfc
from scipy.stats import chi2
from scipy.stats import poisson


def chi2_decomposition(diff, cov):
    # see https://journals.aps.org/prd/abstract/10.1103/PhysRevD.111.092010, equation 6

    cov_inv = np.linalg.inv(cov)
    simple_chi2 = diff @ cov_inv @ diff

    eigenvalues, eigenvectors = np.linalg.eig(cov)
    epsilon_values = []
    for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors.T):
        transformed_diff = diff @ eigenvector
        epsilon = transformed_diff / np.sqrt(eigenvalue)
        epsilon_values.append(epsilon)
    epsilon_values = np.array(epsilon_values)

    summed_decomp_chi2 = sum(epsilon_values**2)

    if not np.isclose(summed_decomp_chi2, simple_chi2):
        raise ValueError(f"Summed decomposition chi2 {summed_decomp_chi2} is not close to simple chi2 {simple_chi2}")

    return epsilon_values
    

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
    # Similar to np.cov, but manually setting the number of universes to divide by
    # In some of the unisims, one of the universes is actually the CV. In these cases, 
    # we need to divide by the number of universes minus one.
    curr_cov = np.zeros((len(unisim_hists), len(unisim_hists)))
    for uni_i in range(unisim_hists.shape[1]):
        diff = np.array(unisim_hists[:,uni_i]) - np.array(cv_hist)
        curr_cov += np.outer(diff, diff)

    if manual_uni_count is None:
        uni_count = unisim_hists.shape[1]
    else:
        uni_count = manual_uni_count

    return curr_cov / uni_count


def create_universe_histograms(vals, bins, sys_weight_arrs, other_weights, description="", quiet=False,
                               chunk_size=50):
    # sys_weight_arrs is an object array from .to_numpy() on a Polars List column:
    # shape (n_events,), each element a 1D float array of length n_universes.
    # We process universes in chunks of chunk_size to avoid a single large allocation
    # (e.g. flux_all: 1000 universes × 1.87M events × 4 bytes = 7.5 GB if stacked at once).

    num_bins = len(bins) - 1
    num_unis = len(sys_weight_arrs[0])

    # finding the bin indices once, rather than a thousand times for each weight
    bin_indices = np.searchsorted(bins, vals, side='right') - 1

    # removing events outside of the bins
    valid = (bin_indices >= 0) & (bin_indices < len(bins) - 1)
    bin_indices = bin_indices[valid]
    sys_weight_arrs = sys_weight_arrs[valid]
    other_weights = other_weights[valid]

    hists = np.zeros((num_bins, num_unis))

    for chunk_start in tqdm(range(0, num_unis, chunk_size), desc=f"Creating {description} universe histograms", disable=quiet, mininterval=10):
        chunk_end = min(chunk_start + chunk_size, num_unis)
        chunk = np.array([arr[chunk_start:chunk_end] for arr in sys_weight_arrs], dtype=np.float32)
        # replacing nan, inf, negative, or large weights with 1, the same way as we did for the CV in postprocessing.py
        chunk = np.nan_to_num(chunk, nan=1, posinf=1, neginf=1)
        chunk[chunk > 30] = 1
        chunk[chunk < 0] = 1
        for i, uni_weights in enumerate(chunk.T):
            hists[:, chunk_start + i] = np.bincount(bin_indices, weights=uni_weights.astype(np.float64) * other_weights, minlength=num_bins)

    return hists


def create_rw_frac_cov_matrices(mc_pred_df, var, bins, weights_df=None, net_weight_var="wc_net_weight"):

    print("creating reweightable systematic covariance matrices...")

    derived_filetypes = ["numuCC_rad_corrected", "NC_coherent_1g_reweighted"]
    if "filetype" in mc_pred_df.columns:
        derived_counts = mc_pred_df.filter(pl.col("filetype").is_in(derived_filetypes)).group_by("filetype").agg(pl.len().alias("count"))
        for row in derived_counts.collect().iter_rows(named=True):
            print(f"WARNING: {row['count']} events with filetype='{row['filetype']}' are present in mc_pred_df. "
                  "These events have no flux, cross-section, or re-interaction systematics available,"
                  " so they are assigned unit weights for all reweightable systematics.")

    # Filter mc_pred_df to events present in weights, fetching only key columns from weights
    # so no List[Float32] universe columns are loaded yet.
    join_keys = ["filename", "run", "subrun", "event"]
    pred_vars = list(dict.fromkeys(join_keys + [net_weight_var, "wc_weight_cv", "wc_weight_spline", var]))
    weights_parquet_path = f"{intermediate_files_location}/presel_weights_df.parquet"

    print("filtering mc_pred_df to events present in weights...")
    if weights_df is None:
        base_merged = (
            mc_pred_df.select(pred_vars).lazy()
            .join(pl.scan_parquet(weights_parquet_path).select(join_keys), on=join_keys, how="inner")
            .collect(engine="streaming")
        )
    else:
        base_merged = (
            mc_pred_df.select(pred_vars).lazy()
            .join(weights_df.select(join_keys).lazy(), on=join_keys, how="inner")
            .collect()
        )

    mc_pred_count = mc_pred_df.select(pl.len()).collect().item() if isinstance(mc_pred_df, pl.LazyFrame) else mc_pred_df.height
    if base_merged.height != mc_pred_count:
        print(f"WARNING: missing events in weights_df, approximate reweightable systematic uncertainties! {base_merged.height=}, mc_pred_df.height={mc_pred_count}")

    pred_vals = get_vals(base_merged, var)

    rw_sys_frac_cov_dic = {}

    print("getting CV histogram and non-GENIE weights...")
    cv_hist = np.histogram(pred_vals, weights=base_merged.get_column(net_weight_var).to_numpy(), bins=bins)[0]
    cv_hist = np.maximum(cv_hist, 1e-3) # avoiding nans when we divide in the next step, this bin will have large stat uncertainty anyway

    # we use these weights when we replace GENIE weight_cv with the new systematic weight
    non_genie_cv_weights = base_merged.get_column(net_weight_var).to_numpy() / base_merged.get_column("wc_weight_cv").to_numpy()

    # we use these weights when we consider a new weight independent of the GENIE CV weights
    normal_weights = base_merged.get_column(net_weight_var).to_numpy()

    def _get_universe_weights(col_name):
        """Fetch a single List[Float32] universe column, joining only that column."""
        if weights_df is None:
            return (
                base_merged.select(join_keys).lazy()
                .join(pl.scan_parquet(weights_parquet_path).select(join_keys + [col_name]), on=join_keys, how="inner")
                .collect(engine="streaming")
                .get_column(col_name).to_numpy()
            )
        else:
            return (
                base_merged.select(join_keys).lazy()
                .join(weights_df.select(join_keys + [col_name]).lazy(), on=join_keys, how="inner")
                .collect()
                .get_column(col_name).to_numpy()
            )

    for sys_name, col_name, event_weights in [
        ("All_UBGenie", "All_UBGenie", non_genie_cv_weights),
        ("flux",        "flux_all",    normal_weights),
        ("reinteraction", "reint_all", normal_weights),
    ]:
        uni_weights = _get_universe_weights(col_name)
        hists = create_universe_histograms(pred_vals, bins, uni_weights, event_weights, description=sys_name)
        cov = create_cov_matrix(hists, cv_hist)
        denom = np.outer(cv_hist, cv_hist)
        rw_sys_frac_cov_dic[sys_name] = np.nan_to_num(np.divide(cov, denom, out=np.zeros_like(cov), where=(denom != 0)), nan=0, posinf=0, neginf=0)
        del uni_weights

    print("creating GENIE unisim systematic covariance matrices...")

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

        event_weights = normal_weights if unisim_type in ("xsr_scc_Fa3_SCC", "xsr_scc_Fv3_SCC") else non_genie_cv_weights

        uni_weights = _get_universe_weights(unisim_type)
        unisim_hists = create_universe_histograms(pred_vals, bins, uni_weights, event_weights, description=unisim_type, quiet=True)
        del uni_weights
        unisim_cov = create_cov_matrix(unisim_hists, cv_hist, manual_uni_count=num_unis)
        denom = np.outer(cv_hist, cv_hist)
        rw_sys_frac_cov_dic[unisim_type] = np.nan_to_num(np.divide(unisim_cov, denom, out=np.zeros_like(unisim_cov), where=(denom != 0)), nan=0, posinf=0, neginf=0)

    print("done getting reweightable systematic covariance matrices")

    return rw_sys_frac_cov_dic


def create_detvar_frac_cov_matrices(detvar_df, var, bins, use_detvar_bootstrapping, num_bootstrap_rounds_detvar=5000, num_bootstrap_samples_detvar=5000):

    print("creating detvar systematic covariance matrices...")

    cv_df = detvar_df.filter(pl.col("vartype") == "CV")

    detvar_sys_frac_cov_dic = {}

    for vartype in ["LYAtt", "LYDown", "LYRayleigh", "WireModX", "WireModYZ", "Recomb2", "SCE"]:
        curr_df = detvar_df.filter(pl.col("vartype") == vartype)

        curr_filetype_rse_df = curr_df.select(["filetype", "run", "subrun", "event"])
        matching_cv_df = cv_df.join(curr_filetype_rse_df, on=["filetype", "run", "subrun", "event"], how="inner")

        matching_curr_df = curr_df.join(matching_cv_df.select(["filetype", "run", "subrun", "event"]), on=["filetype", "run", "subrun", "event"], how="inner")

        if not use_detvar_bootstrapping:
            matching_cv_counts = np.histogram(get_vals(matching_cv_df, var), weights=get_vals(matching_cv_df, "wc_net_weight"), bins=bins)[0]
            matching_var_counts = np.histogram(get_vals(matching_curr_df, var), weights=get_vals(matching_curr_df, "wc_net_weight"), bins=bins)[0]
            diff = matching_cv_counts - matching_var_counts
            curr_cov = np.outer(diff, diff)
            denom = np.outer(matching_cv_counts, matching_cv_counts)
            curr_frac_cov = np.divide(curr_cov, denom, out=np.zeros_like(curr_cov), where=(denom != 0))
            curr_frac_cov = np.nan_to_num(curr_frac_cov, nan=0, posinf=0, neginf=0)
        else:
            # bootstrapping to estimate the statistical uncertainty on the CV-var difference
            # see page 68 of https://microboone-docdb.fnal.gov/cgi-bin/sso/RetrieveFile?docid=33302&filename=MicroBooNE_Wire_Cell_LEE_Analysis_Internal_Note-16.pdf&version=30
            # also see code at https://github.com/BNLIF/wcp-uboone-bdt/blob/05acfe6d3c2a175ff52573669be7ce8ba77c623c/src/mcm_1.h#L77

            matching_cv_vals = get_vals(matching_cv_df, var)
            matching_cv_weights = get_vals(matching_cv_df, "wc_net_weight")
            matching_var_vals = get_vals(matching_curr_df, var)
            matching_var_weights = get_vals(matching_curr_df, "wc_net_weight")

            matching_cv_counts = np.histogram(matching_cv_vals, weights=matching_cv_weights, bins=bins)[0]
            matching_var_counts = np.histogram(matching_var_vals, weights=matching_var_weights, bins=bins)[0]
            nominal_cv_var_diff = matching_var_counts - matching_cv_counts

            cv_val_weight_pairs = list(zip(matching_cv_vals, matching_cv_weights))
            var_val_weight_pairs = list(zip(matching_var_vals, matching_var_weights))

            # sampling the CV and var spectra with replacement to get statistically plausible CV-var differences
            bootstrap_cv_var_diffs = [] # each row is a spectrum difference between CV and var for each bootstrap sample, each column is a sample
            for bootstrap_i in tqdm(range(num_bootstrap_rounds_detvar), desc=f"Bootstrapping {vartype} detvar systematic covariance matrices", mininterval=10):
                
                bootstrap_cv_indices = np.random.choice(len(matching_cv_vals), size=len(matching_cv_vals), replace=True)
                bootstrap_cv_vals = matching_cv_vals[bootstrap_cv_indices]
                bootstrap_cv_weights = matching_cv_weights[bootstrap_cv_indices]

                bootstrap_var_indices = np.random.choice(len(matching_var_vals), size=len(matching_var_vals), replace=True)
                bootstrap_var_vals = matching_var_vals[bootstrap_var_indices]
                bootstrap_var_weights = matching_var_weights[bootstrap_var_indices]

                bootstrap_cv_counts = np.histogram(bootstrap_cv_vals, weights=bootstrap_cv_weights, bins=bins)[0]
                bootstrap_var_counts = np.histogram(bootstrap_var_vals, weights=bootstrap_var_weights, bins=bins)[0]
                bootstrap_cv_var_diffs.append(bootstrap_var_counts - bootstrap_cv_counts)
            
            # building a covariance matrix to describe the statistical uncertainty on the CV-var difference, called M_R in the note
            bootstrap_cv_var_diff_cov = np.cov(bootstrap_cv_var_diffs, rowvar=False)

            # drawing samples from the bootstrap_cv_var_diff_cov covariance matrix, each called V_D in the note
            bootstrap_cv_var_diff_samples = np.random.multivariate_normal(nominal_cv_var_diff, bootstrap_cv_var_diff_cov, size=num_bootstrap_samples_detvar)

            normal_distribution_samples = np.random.normal(0, 1, size=len(bootstrap_cv_var_diff_samples))

            # each called r * V_D
            bootstrap_scaled_cv_var_diff_samples = normal_distribution_samples[:, None] * bootstrap_cv_var_diff_samples

            # called M_D in the note
            curr_cov = np.cov(bootstrap_scaled_cv_var_diff_samples, rowvar=False)

            denom = np.outer(matching_cv_counts, matching_cv_counts)
            curr_frac_cov = np.divide(curr_cov, denom, out=np.zeros_like(curr_cov), where=(denom != 0))
            curr_frac_cov = np.nan_to_num(curr_frac_cov, nan=0, posinf=0, neginf=0)

        detvar_sys_frac_cov_dic[vartype] = curr_frac_cov

    print("done getting detvar systematic covariance matrices")

    return detvar_sys_frac_cov_dic

def _key_hash(sel, var, bins):
    bins_arr = np.asarray(bins, dtype=float)
    h = hashlib.sha256()
    h.update(sel.encode("utf-8"))
    h.update(b"\x00")
    h.update(var.encode("utf-8"))
    h.update(b"\x00")
    h.update(bins_arr.tobytes())
    return h.hexdigest()

def _key_hash_detvar(sel, var, bins, use_detvar_bootstrapping, num_bootstrap_rounds_detvar, num_bootstrap_samples_detvar):
    bins_arr = np.asarray(bins, dtype=float)
    h = hashlib.sha256()
    h.update(sel.encode("utf-8"))
    h.update(b"\x00")
    h.update(var.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(use_detvar_bootstrapping).encode("utf-8"))
    h.update(b"\x00")
    h.update(str(num_bootstrap_rounds_detvar).encode("utf-8"))
    h.update(b"\x00")
    h.update(str(num_bootstrap_samples_detvar).encode("utf-8"))
    h.update(bins_arr.tobytes())
    return h.hexdigest()

def get_rw_sys_frac_cov_matrices(mc_pred_df, selname, var, bins, dont_load_rw_from_systematic_cache=False, weights_df=None, net_weight_var="wc_net_weight"):

    if not dont_load_rw_from_systematic_cache:
        key_h = _key_hash(selname, var, bins)
        cache_path = f"{covariance_cache_location}/cov_{key_h}.npz"
        if os.path.exists(cache_path):
            print("loading reweightable systematic covariance matrices from cache...")
            with np.load(cache_path, allow_pickle=True) as data:
                return data["rw_sys_frac_cov_dic"].item()

    rw_sys_frac_cov_dic = create_rw_frac_cov_matrices(mc_pred_df, var, bins, weights_df=weights_df, net_weight_var=net_weight_var)

    key_h = _key_hash(selname, var, bins)
    cache_path = f"{covariance_cache_location}/cov_{key_h}.npz"
    np.savez_compressed(cache_path, rw_sys_frac_cov_dic=rw_sys_frac_cov_dic)

    return rw_sys_frac_cov_dic


def get_detvar_sys_frac_cov_matrices(detvar_df, selname, var, bins, dont_load_detvar_from_systematic_cache=False, use_detvar_bootstrapping=True, num_bootstrap_rounds_detvar=5000, num_bootstrap_samples_detvar=5000):
    # detvar_df is not optional, it must be loaded by the user and must have the selection cuts applied to match those on mc_pred_df

    if not dont_load_detvar_from_systematic_cache:
        key_h = _key_hash_detvar(selname, var, bins, use_detvar_bootstrapping, num_bootstrap_rounds_detvar, num_bootstrap_samples_detvar)
        cache_path = f"{covariance_cache_location}/detvar_cov_{key_h}.npz"
        if os.path.exists(cache_path):
            print("loading DetVar systematic covariance matrices from cache...")
            with np.load(cache_path, allow_pickle=True) as data:
                return data["detvar_sys_frac_cov_dic"].item()

    detvar_sys_frac_cov_dic = create_detvar_frac_cov_matrices(detvar_df, var, bins, use_detvar_bootstrapping, num_bootstrap_rounds_detvar=num_bootstrap_rounds_detvar, num_bootstrap_samples_detvar=num_bootstrap_samples_detvar)

    key_h = _key_hash_detvar(selname, var, bins, use_detvar_bootstrapping, num_bootstrap_rounds_detvar, num_bootstrap_samples_detvar)
    cache_path = f"{covariance_cache_location}/detvar_cov_{key_h}.npz"
    np.savez_compressed(cache_path, detvar_sys_frac_cov_dic=detvar_sys_frac_cov_dic)

    return detvar_sys_frac_cov_dic
