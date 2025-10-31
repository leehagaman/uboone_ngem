import numpy as np
import os
import hashlib
from tqdm import tqdm
from src.file_locations import covariance_cache_location

def create_universe_histograms(vals, bins, sys_weight_arrs, other_weights):

    num_bins = len(bins) - 1
    num_unis = weight_arrs.shape[1]

    # finding the bin indices once, rather than a thousand times for each weight
    bin_indices = np.searchsorted(bins, vals, side='right') - 1

    # removing events outside of the bins
    valid = (bin_indices >= 0) & (bin_indices < len(bins) - 1)
    bin_indices = bin_indices[valid]
    weight_arrs = weight_arrs[valid]

    # creating the histogram for each universe (with pre-computed bin indices)
    hists = np.zeros((num_bins, num_unis))
    for uni_i, uni_weights in tqdm(enumerate(sys_weight_arrs.T), total=num_unis, desc="Creating universe histograms"):
        hists[:, uni_i] = np.bincount(bin_indices, weights=uni_weights*other_weights, minlength=num_bins)

    return hists


def create_frac_cov_matrices(pred_df, weights_df, pred_vals, bins):

    print("creating systematic covariance matrices...")

    print("merging pred_df and weights_df...")
    pred_vars = ["filetype", "run", "subrun", "event", "wc_net_weight"]
    merged_df = pred_df.select(pred_vars).join(weights_df, on=["filename", "run", "subrun", "event"], how="inner")
    if merged_df.height != pred_df.height:
        raise ValueError("merged_df height does not match pred_df height, missing events in weights_df?")

    print("getting CV histogram and non-GENIE weights...")
    cv_hist = np.histogram(pred_vals, weights=merged_df.get_column("wc_net_weight").to_numpy(), bins=bins)[0]
    other_weights = merged_df.get_column("wc_net_weight").to_numpy() / merged_df.get_column("wc_weight_cv").to_numpy()

    print("creating GENIE histograms...")
    genie_hists = create_universe_histograms(pred_vals, bins, weights_df.get_column("weights_genie").to_numpy(), other_weights)
    genie_cov = np.cov(genie_hists - cv_hist[:, None])
    genie_frac_cov = genie_cov / np.outer(cv_hist, cv_hist)

    print("creating flux histograms...")
    flux_hists = create_universe_histograms(pred_vals, bins, weights_df.get_column("weights_flux").to_numpy(), other_weights)
    flux_cov = np.cov(flux_hists - cv_hist[:, None])
    flux_frac_cov = flux_cov / np.outer(cv_hist, cv_hist)

    print("creating reinteraction histograms...")
    reint_hists = create_universe_histograms(pred_vals, bins, weights_df.get_column("weights_reint").to_numpy(), other_weights)
    reint_cov = np.cov(reint_hists - cv_hist[:, None])
    reint_frac_cov = reint_cov / np.outer(cv_hist, cv_hist)

    return genie_frac_cov, flux_frac_cov, reint_frac_cov


def _key_hash(var, bins):
    bins_arr = np.asarray(bins, dtype=float)
    h = hashlib.sha256()
    h.update(var.encode("utf-8"))
    h.update(b"\x00")
    h.update(bins_arr.tobytes())
    return h.hexdigest()

def get_frac_cov_matrices(pred_df, weights_df, var, bins, dont_use_systematic_cache=False):

    if not dont_use_systematic_cache:
        key_h = _key_hash(var, bins)
        cache_path = f"{covariance_cache_location}/cov_{key_h}.npz"
        if os.path.exists(cache_path):
            print("loading systematic covariance matrices from cache...")
            with np.load(cache_path, allow_pickle=False) as data:
                return (data["genie_frac_cov"], data["flux_frac_cov"], data["reint_frac_cov"]) 

    result = create_frac_cov_matrices(pred_df, weights_df, var, bins)

    key_h = _key_hash(var, bins)
    cache_path = f"{covariance_cache_location}/cov_{key_h}.npz"
    np.savez_compressed(cache_path, genie_frac_cov=result[0], flux_frac_cov=result[1], reint_frac_cov=result[2])

    return result