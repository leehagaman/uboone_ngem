import polars as pl
import numpy as np
import os
import hashlib
from tqdm import tqdm
from src.file_locations import covariance_cache_location

import ROOT


def get_rw_sys_weights_dic(
    file_path: str,
    tree_path: str = "nuselection/NeutrinoSelectionFilter",
    branch_name: str = "weights",
    max_entries: int = -1):
    """
    Gets systematic weights from Pandora Tree using PyROOT.
    Returns a list of dictionaries, one per event.
    
    Parameters:
    -----------
    file_path : str
        Path to the ROOT file
    tree_path : str
        Path to the tree within the ROOT file (default: "nuselection/NeutrinoSelectionFilter")
    branch_name : str
        Name of the branch containing weights (default: "weights")
    max_entries : int
        Maximum number of entries to process. If -1, process all entries (default: -1)
    
    Returns:
    --------
    list of dict
        List of dictionaries, one per event. Each dictionary maps systematic names to lists of weights.
    """
    
    # Ensure ROOT dictionary is generated for the map type
    ROOT.gInterpreter.GenerateDictionary("map<string,vector<double>>", "map;string;vector")
    
    # Open the ROOT file
    root_file = ROOT.TFile.Open(os.path.abspath(file_path))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Cannot open file: {file_path}")
    
    # Get the tree
    tree = root_file.Get(tree_path)
    if not tree:
        raise RuntimeError(f"Tree not found at path: {tree_path}")
    # Check if it's actually a TTree by checking the class name
    if tree.ClassName() != "TTree":
        raise RuntimeError(f"Object at path {tree_path} is not a TTree (got {tree.ClassName()})")
    
    # Get the branch
    branch = tree.GetBranch(branch_name)
    if not branch:
        raise RuntimeError(f"Branch not found: {branch_name}")
    
    # Set up branch address - The branch stores a pointer to std::map<string, vector<double>>
    # In PyROOT, we create the map object and pass it to SetBranchAddress
    # PyROOT will handle the pointer conversion automatically
    weight_map = ROOT.std.map('string', ROOT.std.vector('double'))()
    tree.SetBranchAddress(branch_name, weight_map)
    
    # Determine number of entries to process
    nentries = tree.GetEntries()
    if max_entries >= 0 and max_entries < nentries:
        nentries = max_entries
    
    rows = []
    
    # Process each event
    for i in range(nentries):
        entry = tree.GetEntry(i)
        if entry <= 0:
            rows.append({})
            continue
        
        # The weight_map object is updated by GetEntry
        # Convert to Python dictionary
        event_dict = {}
        
        # Iterate over the map
        # In PyROOT, std::map can be iterated directly
        # If the map is empty (or pointer was null), this loop simply won't execute
        for key_pair in weight_map:
            key = key_pair.first
            weights_vec = key_pair.second
            
            # Convert vector<double> to Python list
            weights_list = []
            for j in range(weights_vec.size()):
                weights_list.append(weights_vec[j])
            
            event_dict[str(key)] = weights_list
        
        rows.append(event_dict)

    root_file.Close()
    
    return rows
    

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