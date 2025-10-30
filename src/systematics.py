import polars as pl
import numpy as np
import os
import hashlib
from tqdm import tqdm
from file_locations import covariance_cache_location

import tempfile
import subprocess
import ast
import json
import re

def _cxx_escape(s):
    """Escapes strings for use in C++ code."""
    return s.replace('\\', '\\\\').replace('"', '\\"')

# gets systematic weights from Pandora Tree using a ROOT c++ macro
# then loads thems into a python dictionary
def get_rw_sys_weights_dic(
    file_path: str,
    tree_path: str = "nuselection/NeutrinoSelectionFilter",
    branch_name: str = "weights",
    max_entries: int = -1,
    root_bin: str = "root"):
    
    cpp_macro = r'''
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TSystem.h>
#include <TROOT.h>
#include <TInterpreter.h> // Header needed for gInterpreter
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <iomanip>

// Define the exact type expected by the branch (confirmed as double by your last code)
using WeightMap_t = std::map<std::string, std::vector<double>>;

void extract_event_weights(const char* filePath, const char* treePath, const char* branchName, Long64_t maxEntries=-1) {
    
    gInterpreter->GenerateDictionary("map<string,vector<double>>", "map;string;vector");
    
    TFile* f = TFile::Open(filePath);
    if (!f || f->IsZombie()) { std::cout << "__ERROR__ Cannot open file\n"; return; }

    TObject* obj = f->Get(treePath);
    TTree* t = dynamic_cast<TTree*>(obj);
    if (!t) { std::cout << "__ERROR__ Tree not found at path\n"; f->Close(); return; }
    
    TBranch* br = t->GetBranch(branchName);
    if (!br) { std::cout << "__ERROR__ Branch not found\n"; f->Close(); return; }

    WeightMap_t* weight_map_ptr = nullptr;
    t->SetBranchAddress(branchName, &weight_map_ptr);

    Long64_t nentries = t->GetEntries();
    if (maxEntries >= 0 && maxEntries < nentries) nentries = maxEntries;

    std::cout << "__BEGIN__\n";
    for (Long64_t i = 0; i < nentries; ++i) {
        
        if (t->GetEntry(i) <= 0) { 
            std::cout << "{}\n";
            continue;
        }

        if (!weight_map_ptr) {
            std::cout << "{}\n";
            continue;
        }

        const auto& weight_map = *weight_map_ptr;

        std::cout << "{";
        
        bool first_pair = true;
        for (const auto& pair : weight_map) {
            if (!first_pair) {
                std::cout << ", ";
            }
            
            const std::string& key = pair.first;
            const std::vector<double>& weights = pair.second;

            std::cout << "\"" << key << "\": [";
            
            for (size_t j = 0; j < weights.size(); ++j) {
                if (j) std::cout << ",";
                std::cout << std::scientific << weights.at(j); 
            }
            std::cout << "]";
            
            first_pair = false;
        }
        
        std::cout << "}\n";
    }
    std::cout << "__END__\n";
    f->Close();
}
'''
    # --- Python Execution Logic (using the env-injected subprocess call) ---
    file_path_cxx = _cxx_escape(os.path.abspath(file_path))
    tree_path_cxx = _cxx_escape(tree_path)
    branch_name_cxx = _cxx_escape(branch_name)

    with tempfile.TemporaryDirectory() as td:
        macro_path = os.path.join(td, "extract_event_weights.C")
        with open(macro_path, "w") as f:
            f.write(cpp_macro)

        arg_expr = f'{macro_path}("{file_path_cxx}","{tree_path_cxx}","{branch_name_cxx}",{int(max_entries)})'
        cmd = [root_bin, "-l", "-b", "-q", arg_expr]

        # Use the environment-injecting subprocess.run
        proc = subprocess.run(
            cmd, 
            text=True, 
            capture_output=True,
            env=os.environ.copy() # Keeps the path for the 'root' executable
        )
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        
        # print(combined)
        
        begin = "__BEGIN__"
        end = "__END__"
        
        if begin not in combined or end not in combined:
            raise RuntimeError(f"Failed to parse ROOT output or macro crashed.\nReturn code: {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")

        payload = combined.split(begin, 1)[1].split(end, 1)[0]
        rows = []

        for line in payload.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            
            # Replace inf/nan with valid JSON values (JSON doesn't support inf/nan natively)
            # Handle various forms: inf, Inf, INF, +inf, -inf, nan, NaN, NAN, -nan, +nan
            # Use 1e308 (close to max double) for inf, -1e308 for -inf to avoid overflow issues
            # Note: Order matters - handle -nan and +nan before nan, and -inf/+inf before inf
            line = re.sub(r'-\s*inf\b', '-1e308', line, flags=re.IGNORECASE)
            line = re.sub(r'\+\s*inf\b', '1e308', line, flags=re.IGNORECASE)
            line = re.sub(r'\binf\b', '1e308', line, flags=re.IGNORECASE)
            line = re.sub(r'-\s*nan\b', 'null', line, flags=re.IGNORECASE)  # -nan is still null
            line = re.sub(r'\+\s*nan\b', 'null', line, flags=re.IGNORECASE)  # +nan is still null
            line = re.sub(r'\bnan\b', 'null', line, flags=re.IGNORECASE)
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                # Try ast.literal_eval as fallback
                try:
                    data = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    # Show context around the error position
                    # JSONDecodeError.pos gives 0-indexed character position
                    error_pos = e.pos if hasattr(e, 'pos') else None
                    if error_pos is not None:
                        start = max(0, error_pos - 50)
                        end = min(len(line), error_pos + 50)
                        context = line[start:end]
                        marker_pos = error_pos - start
                        context_with_marker = context[:marker_pos] + '>>>' + context[marker_pos:marker_pos+1] + '<<<' + context[marker_pos+1:]
                        raise ValueError(f"Could not parse line (length={len(line)}, error at pos {error_pos}):\n...{context_with_marker}...\nJSON error: {e}")
                    else:
                        raise ValueError(f"Could not parse line (length={len(line)}): {line[:200]}...\nJSON error: {e}")
            
            if isinstance(data, dict):
                rows.append(data)
            elif data == {}:
                rows.append({})
                
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