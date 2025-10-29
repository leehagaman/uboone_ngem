
import uproot
import numpy as np
import pandas as pd
import polars as pl
import sys
import os
import time
import argparse
import ast
import tempfile
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from postprocessing import do_orthogonalization_and_POT_weighting, add_extra_true_photon_variables, do_spacepoint_postprocessing, add_signal_categories
from postprocessing import do_wc_postprocessing, do_pandora_postprocessing, do_blip_postprocessing, do_lantern_postprocessing, do_combined_postprocessing, do_glee_postprocessing
from postprocessing import remove_vector_variables

from file_locations import data_files_location, intermediate_files_location

from df_helpers import align_columns_for_concat, compress_df
from memory_monitoring import start_memory_logger

def _cxx_escape(s: str) -> str:
    return s.replace('\\', '\\\\').replace('"', '\\"')

def get_weights(
    file_path: str,
    tree_path: str = "nuselection/NeutrinoSelectionFilter",
    branch_name: str = "weightsGenie",
    max_entries: int = -1,
    root_bin: str = "root",
) -> list[list[int]]:
    cpp_macro = r'''
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TROOT.h>
#include <TSystem.h>
#include <vector>
#include <iostream>

void extract_vector_ushort(const char* filePath, const char* treePath, const char* branchName, Long64_t maxEntries=-1) {
    TFile* f = TFile::Open(filePath);
    if (!f || f->IsZombie()) { std::cout << "__ERROR__ Cannot open file\n"; return; }
    TObject* obj = f->Get(treePath);
    TTree* t = dynamic_cast<TTree*>(obj);
    if (!t) { std::cout << "__ERROR__ Tree not found at path\n"; return; }
    TBranch* br = t->GetBranch(branchName);
    if (!br) { std::cout << "__ERROR__ Branch not found\n"; return; }

    std::vector<unsigned short>* vec = nullptr;
    t->SetBranchAddress(branchName, &vec);

    Long64_t nentries = t->GetEntries();
    if (maxEntries >= 0 && maxEntries < nentries) nentries = maxEntries;

    std::cout << "__BEGIN__\n";
    for (Long64_t i = 0; i < nentries; ++i) {
        t->GetEntry(i);
        if (!vec) { std::cout << "[]\n"; continue; }
        std::cout << "[";
        for (size_t j = 0; j < vec->size(); ++j) {
            if (j) std::cout << ",";
            std::cout << static_cast<unsigned int>(vec->at(j));
        }
        std::cout << "]\n";
    }
    std::cout << "__END__\n";
}
'''
    file_path_cxx = _cxx_escape(os.path.abspath(file_path))
    tree_path_cxx = _cxx_escape(tree_path)
    branch_name_cxx = _cxx_escape(branch_name)

    with tempfile.TemporaryDirectory() as td:
        macro_path = os.path.join(td, "extract_vector_ushort.C")
        with open(macro_path, "w") as f:
            f.write(cpp_macro)

        arg_expr = f'{macro_path}("{file_path_cxx}","{tree_path_cxx}","{branch_name_cxx}",{int(max_entries)})'
        cmd = [root_bin, "-l", "-b", "-q", arg_expr]

        proc = subprocess.run(cmd, text=True, capture_output=True)
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

        begin = "__BEGIN__"
        end = "__END__"
        if begin not in combined or end not in combined:
            raise RuntimeError(f"Failed to parse ROOT output.\nReturn code: {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")

        payload = combined.split(begin, 1)[1].split(end, 1)[0]
        rows: list[list[int]] = []
        for line in payload.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                arr = ast.literal_eval(line)
                if isinstance(arr, list):
                    # factor of 1000 for integer -> float conversion exists here: 
                    # https://github.com/ubneutrinos/searchingfornues/blob/c1d8558e1990d9553b874daf9807c15e6ad8dc5e/Selection/AnalysisTools/EventWeightTree_tool.cc#L245
                    rows.append([np.float32(int(x) / 1000.0) for x in arr])
            except Exception:
                pass
        return rows


def process_root_file(filename, frac_events = 1):

    if "beam_off" in filename.lower() or "beamoff" in filename.lower() or "ext" in filename.lower(): # EXT file
        filetype = "ext"
    elif "nu_overlay" in filename.lower():
        filetype = "nu_overlay"
    elif "nue_overlay" in filename.lower():
        filetype = "nue_overlay"
    elif "dirt" in filename.lower():
        filetype = "dirt_overlay"
    elif "nc_pi0" in filename.lower() or "ncpi0" in filename.lower() or "nc_pio" in filename.lower() or "ncpio" in filename.lower():
        filetype = "nc_pi0_overlay"
    elif "delete_one_gamma" in filename.lower():
        filetype = "delete_one_gamma_overlay"
    elif "isotropic_one_gamma" in filename.lower():
        filetype = "isotropic_one_gamma_overlay"
    elif "beam_on" in filename.lower():
        filetype = "data"
    else:
        raise ValueError("Unknown filetype!", filename)

    if filetype == "data" or filetype == "ext" or filetype == "isotropic_one_gamma_overlay" or filetype == "delete_one_gamma_overlay":
        raise ValueError("Data, EXT, and 1g overlay files don't have systematics variables!")
    
    root_file_size_gb = os.path.getsize(f"{data_files_location}/{filename}") / 1024**3

    start_time = time.time()

    print(f"loading {filename}...")

    f = uproot.open(f"{data_files_location}/{filename}")

    # determine how many events to read based on requested fraction
    if not (0.0 < frac_events <= 1.0):
        raise ValueError("--frac_events/-f must be in the interval (0, 1].")
    total_entries = f["wcpselection"]["T_eval"].num_entries
    n_events = total_entries if frac_events >= 1.0 else max(1, int(total_entries * frac_events))

    print("loading run, subrun, event, and CV weights using uproot...")
    slice_kwargs = {} if n_events >= total_entries else {"entry_stop": n_events}
    dic = f["nuselection"]["NeutrinoSelectionFilter"].arrays(["run", "sub", "evt", "weightSpline", "weightTune", "weightSplineTimesTune"], library="np", **slice_kwargs)
    curr_weights_df = pl.DataFrame({col: dic[col].tolist() for col in dic})
    curr_weights_df = curr_weights_df.rename({"sub": "subrun", "evt": "event"})

    print("loading wc_kine_reco_Enu for preselection using uproot...")
    dic = f["wcpselection"]["T_KINEvars"].arrays(["kine_reco_Enu"], library="np", **slice_kwargs)
    curr_weights_df = curr_weights_df.with_columns(pl.Series(name="wc_kine_reco_Enu", values=dic["kine_reco_Enu"].tolist()))
    del f
    del dic

    print("loading systematic weights using ROOT c++...")
    # Using ROOT c++ to load the systematic weights and convert them to python
    if "genie" in args.weight_types:
        print("loading genie weights...")
        weights_genie = get_weights(
            f"{data_files_location}/{filename}",
            branch_name="weightsGenie",
            max_entries=n_events,
        )
        curr_weights_df = curr_weights_df.with_columns(
            pl.Series(name="weights_genie", values=weights_genie, dtype=pl.List(pl.Float32))
        )
        del weights_genie
    if "flux" in args.weight_types:
        print("loading flux weights...")
        weights_flux = get_weights(
            f"{data_files_location}/{filename}",
            branch_name="weightsFlux",
            max_entries=n_events,
        )
        curr_weights_df = curr_weights_df.with_columns(
            pl.Series(name="weights_flux", values=weights_flux, dtype=pl.List(pl.Float32))
        )
        del weights_flux
    if "reint" in args.weight_types:
        print("loading reint weights...")
        weights_reint = get_weights(
            f"{data_files_location}/{filename}",
            branch_name="weightsReint",
            max_entries=n_events,
        )
        curr_weights_df = curr_weights_df.with_columns(
            pl.Series(name="weights_reint", values=weights_reint, dtype=pl.List(pl.Float32))
        )
        del weights_reint
    previous_num_events = curr_weights_df.height
    curr_weights_df = curr_weights_df.filter(pl.col("wc_kine_reco_Enu") > 0)
    print(f"kept {curr_weights_df.height}/{previous_num_events} events with after preselection using wc_kine_reco_Enu > 0")
    
    detailed_run_period = "?"
    if "4a.root" in filename:
        detailed_run_period = "4a"
    elif "4b.root" in filename:
        detailed_run_period = "4b"
    elif "4c.root" in filename:
        detailed_run_period = "4c"
    elif "4d.root" in filename:
        detailed_run_period = "4d"
    elif "5.root" in filename:
        detailed_run_period = "5"
    elif "run4b" in filename.lower(): # if the filename doesn't end with the run period, look for run strings in the file names
        detailed_run_period = "4b"
    elif "run4c" in filename.lower():
        detailed_run_period = "4c"
    elif "run4d" in filename.lower():
        detailed_run_period = "4d"
    elif "run5" in filename.lower():
        detailed_run_period = "5"
    elif "run45" in filename.lower():
        detailed_run_period = "45"
    else:
        raise ValueError("Invalid detailed run period!", filename)

    curr_weights_df = curr_weights_df.with_columns(pl.Series(name="detailed_run_period", values=[detailed_run_period] * curr_weights_df.shape[0]))
    curr_weights_df = curr_weights_df.with_columns(pl.Series(name="filename", values=[filename] * curr_weights_df.shape[0]))
    curr_weights_df = curr_weights_df.with_columns(pl.Series(name="filetype", values=[filetype] * curr_weights_df.shape[0]))

    end_time = time.time()

    progress_str = f"\nloaded {filetype:<30}   Run {detailed_run_period:<4} {curr_weights_df.shape[0]:>10,d} events {root_file_size_gb:>6.2f} GB {end_time - start_time:>6.2f} s"
    if frac_events < 1.0:
        progress_str += f" (f={frac_events})"
    print(progress_str)

    return filetype, curr_weights_df


if __name__ == "__main__":
    main_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Create merged dataframe from SURPRISE 4b ROOT files")
    parser.add_argument("-f", "--frac_events", type=float, default=1.0,
                        help="Fraction of events (and POT) to load from each file, in (0,1]. Default: 1.0")
    parser.add_argument("-m", "--memory_logger", action="store_true", default=False,
                        help="Start a memory logger thread")
    parser.add_argument("-w", "--weight_types", type=str, default="genie,flux,reint",
                        help="Comma-separated list of weight types to load. Default: genie,flux,reint")
    parser.add_argument("--just_one_file", action="store_true", default=False,
                        help="Only process one file for debugging purposes")
    args = parser.parse_args()

    if args.memory_logger:
        start_memory_logger(10)

    if args.frac_events < 1.0:
        print(f"Loading {args.frac_events} fraction of events from each file")

    for file in os.listdir(intermediate_files_location):
        if file.startswith("presel_weights_df") and file.endswith(".parquet"):
            os.remove(f"{intermediate_files_location}/{file}")
    print("Deleted intermediate presel_weights_df*.parquet files")

    print("Starting loop over root files...")
    weights_dfs_pl = pl.DataFrame()

    filenames = os.listdir(data_files_location)
    filenames.sort()
    # sorting these puts an NC Pi0 overlay first, which will have all the WCPMTInfo and truth variables present, 
    # so it can be used to add columns to future dataframes with missing values
    
    for file_num, filename in enumerate(filenames):

        if "UNUSED" in filename or "older_downloads" in filename:
            continue

        if "beam_on" in filename.lower() or "beamon" in filename.lower():
            continue
        if "beam_off" in filename.lower() or "beamoff" in filename.lower() or "ext" in filename.lower():
            continue
        if "one_gamma" in filename.lower():
            continue

        filetype, curr_presel_weights_df = process_root_file(filename, frac_events=args.frac_events)

        print(f"curr_presel_weights_df size: {curr_presel_weights_df.estimated_size() / 1e9:.2f} GB")
        curr_presel_weights_df.write_parquet(f"{intermediate_files_location}/presel_weights_df_{file_num}.parquet")
        print("saved to parquet file")
        del curr_presel_weights_df

        if args.just_one_file:
            break

    print("loading polars dataframes from parquet files...")

    presel_weights_dfs = []
    for file in os.listdir(intermediate_files_location):
        if file.startswith("presel_weights_df") and file.endswith(".parquet"):
            presel_weights_dfs.append(pl.read_parquet(f"{intermediate_files_location}/{file}"))
    presel_weights_df = pl.concat(presel_weights_dfs, how="vertical")
    del presel_weights_dfs

    if presel_weights_df.is_empty():
        raise ValueError("No events in the dataframe!")
    
    print(f"finished looping over root files, presel_weights_df.shape={presel_weights_df.height}")

    print(f"saving {intermediate_files_location}/presel_weights_df.parquet...", end="", flush=True)
    start_time = time.time()
    presel_weights_df.write_parquet(f"{intermediate_files_location}/presel_weights_df.parquet")
    end_time = time.time()
    file_size_gb = os.path.getsize(f"{intermediate_files_location}/presel_weights_df.parquet") / 1024**3
    print(f"done, {file_size_gb:.2f} GB, {end_time - start_time:.2f} seconds")
    main_end_time = time.time()
    print(f"Total time to create weights dataframe: {main_end_time - main_start_time:.2f} seconds")
    