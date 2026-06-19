
import gc
import ctypes
import uproot
import numpy as np
import pandas as pd
import polars as pl
import os
import time
import argparse


from ntuple_variables.variables import wc_T_BDT_including_training_vars, wc_T_KINEvars_including_training_vars, wc_training_only_vars
from ntuple_variables.variables import wc_T_spacepoints_vars, wc_T_eval_vars, wc_T_pf_vars, wc_T_pf_data_vars, wc_T_eval_data_vars
from ntuple_variables.variables import blip_vars, pandora_vars, glee_vars, glee_eventweight_vars, lantern_vars, vector_columns
from postprocessing import do_orthogonalization_and_POT_weighting, apply_rootino_correction, add_extra_true_photon_variables, do_spacepoint_postprocessing, add_signal_categories
from postprocessing import do_wc_postprocessing, do_pandora_postprocessing, do_lantern_postprocessing, do_combined_postprocessing, do_glee_postprocessing
from blip_postprocessing import do_blip_postprocessing
from postprocessing import remove_vector_variables, change_dtypes
from postprocessing import apply_1g1mu_rad_corr_reweighting, apply_nc_coh_1g_reweighting
from numuCC_rad_corr_1g_reweighting import compute_1g1mu_rad_corr_reweighting
from coh_1g_reweighting import compute_nc_coh_1g_reweighting
from pi0_dalitz_reweighting import compute_pi0_dalitz_reweighting, apply_pi0_dalitz_reweighting
from pion_fsi_reweighting import compute_pion_fsi_weights_from_arrays

from file_locations import data_files_location, intermediate_files_location

from pot_and_trigger_numbers import (
    open_data_POT, open_data_num_triggers, ext_num_triggers,
    ext_pot_normalizing_period, expected_full_dataset_data_POT,
)

from memory_monitoring import start_memory_logger


def get_weight_configs():
    """The standard set of POT-weighting configs passed to
    do_orthogonalization_and_POT_weighting.

    Each produces its own net-weight column (and suffixed helper columns) so the
    same dataframe can be histogrammed at several normalizations at once.  See the
    do_orthogonalization_and_POT_weighting docstring for the config schema.  The
    per-run-period POT/trigger numbers live in pot_and_trigger_numbers.py.

    run_period_map maps the file run period to the normalizing run period (data run period).
    POT weighting is done independently in each normalizing run period.
    """
    return [
        # Runs 1-5 full prediction: overlays (+ EXT, dirt) normalized to the
        # expected full-dataset data POT in each period.  Real data is excluded
        # (only the small open-data subset exists); NuWro fake data is excluded.
        # Run 4a is kept separate; runs 4b/4c/4d/4bcd are grouped as "4nota".
        dict(
            name="full_pred",
            weight_col="wc_net_weight_full_pred",
            run_period_map={
                "1": "1", "2": "2", "3": "3", "4a": "4a",
                "4b": "4nota", "4c": "4nota", "4d": "4nota", "4bcd": "4nota", "5": "5",
            },
            goal_pot=expected_full_dataset_data_POT,
            goal_pot_filetypes=None,
            total_pot=None,
            exclude_filetypes=["data", "nuwro_fake_data"],
        ),
        # Runs 1-5 open data: overlays normalized per-group to the open data POT.
        #   run 1 overlays      -> run 1 open data
        #   runs 2-3 overlays    -> run 3 open data (run 2 has no open data)
        #   run 4a overlays      -> run 4a open data
        #   runs 4nota5 overlays   -> run 4b open data
        dict(
            name="open_data",
            weight_col="wc_net_weight_open_data",
            run_period_map={
                "1": "1", "2": "23", "3": "23", "4a": "4a",
                "4b": "4nota5", "4c": "4nota5", "4d": "4nota5", "4bcd": "4nota5", "5": "4nota5",
            },
            goal_pot=None,
            goal_pot_filetypes=["data"],
            total_pot=None,
            exclude_filetypes=["nuwro_fake_data"],
        ),
        # NuWro fake data: overlays normalized per-group to the NuWro fake-data POT
        # (NuWro plays the data role).  NuWro exists for periods 1, 2, 3, 4a, 4c, 5;
        # runs 4b/4d/4bcd overlays are folded into the run 4c group to boost MC
        # stats there.  EXT, dirt, and real data are excluded (not part of the
        # fake-data study).
        dict(
            name="nuwro",
            weight_col="wc_net_weight_nuwro",
            run_period_map={
                "1": "1", "2": "2", "3": "3", "4a": "4a",
                "4b": "4c", "4c": "4c", "4d": "4c", "4bcd": "4c", "5": "5",
            },
            goal_pot=None,
            goal_pot_filetypes=["nuwro_fake_data"],
            total_pot=None,
            exclude_filetypes=["data", "ext", "dirt_overlay"],
        ),
        # Legacy run-4b-only weighting (kept for notebooks that still use
        # run4b_only_wc_net_weight): only run 4b events get a weight.
        dict(
            name="run4b_only",
            weight_col="run4b_only_wc_net_weight",
            run_period_map={"4b": "4b"},
            goal_pot=None,
            goal_pot_filetypes=["data"],
            total_pot=None,
            exclude_filetypes=[],
        ),
    ]


def _detailed_run_period_from_filename(filename):
    """Infer the detailed_run_period (e.g. "1", "4a", "4bcd") from a filename.

    Checks explicit run-period filename suffixes first, then a few special open
    data filenames, then falls back to substring matches on the lowercased name.
    """
    # special-case the runs 1-3 open data files, whose "..._N_<X>e19opendata.root"
    # names contain misleading digits ("5e19", "1e19") that would trip the
    # generic "<N>.root" suffix checks below.
    if "_1_5e19opendata.root" in filename:
        return "1"
    if "_3_1e19opendata.root" in filename:
        return "3"

    if "1.root" in filename:
        return "1"
    elif "2.root" in filename:
        return "2"
    elif "3.root" in filename:
        return "3"
    elif "4a.root" in filename:
        return "4a"
    elif "4b.root" in filename:
        return "4b"
    elif "4c.root" in filename:
        return "4c"
    elif "4d.root" in filename:
        return "4d"
    elif "4bcd.root" in filename:
        return "4bcd"
    elif "5.root" in filename:
        return "5"
    # if the filename doesn't end with the run period, look for run strings
    elif "4a" in filename.lower():
        return "4a"
    elif "run4b" in filename.lower():
        return "4b"
    elif "run4c" in filename.lower():
        return "4c"
    elif "run4d" in filename.lower():
        return "4d"
    elif "run4bcd" in filename.lower():
        return "4bcd"
    elif "run5" in filename.lower():
        return "5"
    else:
        raise ValueError("Invalid detailed run period!", filename)


def _filetype_from_filename(filename):
    fn = filename.lower()
    if "beam_off" in fn or "beamoff" in fn or "ext" in fn:
        return "ext"
    if "nuwro" in fn:
        return "nuwro_fake_data"
    if "nu_overlay" in fn:
        return "nu_overlay"
    if "nue_overlay" in fn:
        return "nue_overlay"
    if "dirt" in fn:
        return "dirt_overlay"
    if "nc_pi0" in fn or "ncpi0" in fn or "nc_pio" in fn or "ncpio" in fn:
        return "nc_pi0_overlay"
    if "ccpi0" in fn:
        return "numucc_pi0_overlay"
    if "delete_one_gamma" in fn:
        return "delete_one_gamma_overlay"
    if "isotropic_one_gamma" in fn:
        return "isotropic_one_gamma_overlay"
    if "beam_on" in fn:
        return "data"
    raise ValueError("Unknown filetype!", filename)


def _get_file_metadata(filename, frac_events=1):
    """Open a ROOT file briefly to collect per-file metadata without reading branch data.

    Returns a dict with keys:
        filetype, detailed_run_period, file_POT, n_events,
        root_file_size_gb, curr_wc_T_BDT_including_training_vars, curr_wc_T_pf_vars
    """
    filetype = _filetype_from_filename(filename)

    root_file_size_gb = os.path.getsize(f"{data_files_location}/{filename}") / 1024**3

    if not (0.0 < frac_events <= 1.0):
        raise ValueError("--frac_events/-f must be in the interval (0, 1].")

    f = uproot.open(f"{data_files_location}/{filename}")
    total_entries = f["wcpselection"]["T_eval"].num_entries
    n_events = total_entries if frac_events >= 1.0 else max(1, int(total_entries * frac_events))

    print(f"{total_entries=}, {frac_events=}, {n_events=}")

    curr_wc_T_pf_vars = wc_T_pf_vars
    curr_wc_T_BDT_including_training_vars = wc_T_BDT_including_training_vars
    if (("v10_04_07_09" in filename) or (filename == "checkout_MCC9.10_Run4b_v10_04_07_20_BNB_beam_off_metapatch_retuple_retuple_hist.root")
                 or (filename == "checkout_MCC9.10_Run4b_v10_04_07_20_BNB_nu_overlay_retuple_retuple_hist.root")):
        print(f"    TEMPORARY: NOT LOADING WCPMTInfo VARIABLES FOR {filetype}")
        curr_wc_T_BDT_including_training_vars = [var for var in wc_T_BDT_including_training_vars if "WCPMTInfo" not in var]

    detailed_run_period = _detailed_run_period_from_filename(filename)

    file_POT_total = np.sum(f["wcpselection"]["T_pot"].arrays("pot_tor875good", library="np")["pot_tor875good"])
    f.close()

    # overwrite POT saved in the file for EXT and data, these need to instead use Zarko's tool
    # see src/pot_and_trigger_numbers.py and ipynb_notebooks/pot_and_trigger_processing.ipynb for more details
    if filetype == "ext":
        if detailed_run_period not in ext_num_triggers:
            raise ValueError("EXT file num triggers not found!", filename, detailed_run_period)
        norm_period = ext_pot_normalizing_period[detailed_run_period]
        data_pot_per_trigger = open_data_POT[norm_period] / open_data_num_triggers[norm_period]
        file_POT_total = ext_num_triggers[detailed_run_period] * data_pot_per_trigger
    elif filetype == "data":
        if detailed_run_period not in open_data_POT:
            raise ValueError("Beam-on data file POT not found!", filename, detailed_run_period)
        file_POT_total = open_data_POT[detailed_run_period]

    file_POT = file_POT_total * frac_events

    return {
        "filetype": filetype,
        "detailed_run_period": detailed_run_period,
        "file_POT": file_POT,
        "n_events": n_events,
        "root_file_size_gb": root_file_size_gb,
        "curr_wc_T_BDT_including_training_vars": curr_wc_T_BDT_including_training_vars,
        "curr_wc_T_pf_vars": curr_wc_T_pf_vars,
    }


def _arrays_filling_missing(tree, varlist, slice_kwargs, n_rows, filename, tree_label):
    """Read varlist from an uproot tree, tolerating branches absent in this file.

    Some files in the heterogeneous set (e.g. the Run123 run-1/run-2 ntuples) lack
    branches that others have (e.g. the Pandora _closestNuCosmicDist).  Reading a
    missing branch raises KeyInFileError, so instead we read only the branches that
    exist and fill the missing ones with NaN, keeping the output column schema
    consistent across all files.
    """
    available = set(tree.keys())
    present = [v for v in varlist if v in available]
    missing = [v for v in varlist if v not in available]
    dic = {}
    if present:
        dic.update(tree.arrays(present, library="np", **slice_kwargs))
    if missing:
        print(f"    WARNING: {tree_label}: {len(missing)} branch(es) missing in {filename}, "
              f"filling with NaN: {missing}")
        for v in missing:
            dic[v] = np.full(n_rows, np.nan)
    return dic


def _load_chunk(filename, filetype, detailed_run_period, file_POT,
                curr_wc_T_BDT_including_training_vars, curr_wc_T_pf_vars,
                entry_start, entry_stop, **_):
    """Load events [entry_start, entry_stop) from a ROOT file and return a DataFrame.

    **_ absorbs unused metadata keys (n_events, root_file_size_gb) so callers can
    pass the full _get_file_metadata dict via **meta.
    """
    f = uproot.open(f"{data_files_location}/{filename}")
    slice_kwargs = {"entry_start": entry_start, "entry_stop": entry_stop}

    # loading Wire-Cell variables
    dic = {}
    dic.update(f["wcpselection"]["T_BDTvars"].arrays(curr_wc_T_BDT_including_training_vars, library="np", **slice_kwargs))
    dic.update(f["wcpselection"]["T_KINEvars"].arrays(wc_T_KINEvars_including_training_vars, library="np", **slice_kwargs))
    dic.update(f["wcpselection"]["T_spacepoints"].arrays(wc_T_spacepoints_vars, library="np", **slice_kwargs))
    if filetype == "ext" or filetype == "data":
        dic.update(f["wcpselection"]["T_PFeval"].arrays(wc_T_pf_data_vars, library="np", **slice_kwargs))
        dic.update(f["wcpselection"]["T_eval"].arrays(wc_T_eval_data_vars, library="np", **slice_kwargs))
    else:
        dic.update(f["wcpselection"]["T_PFeval"].arrays(curr_wc_T_pf_vars, library="np", **slice_kwargs))
        dic.update(f["wcpselection"]["T_eval"].arrays(wc_T_eval_vars, library="np", **slice_kwargs))
    all_df = pd.DataFrame({col: arr.tolist() if arr.ndim != 1 else arr for col, arr in dic.items()}).add_prefix("wc_")
    del dic
    all_df["wc_file_POT"] = file_POT

    n_rows = len(all_df)

    # loading blip variables (blip variables already have the "blip_" prefix)
    dic = _arrays_filling_missing(f["nuselection"]["NeutrinoSelectionFilter"], blip_vars,
                                  slice_kwargs, n_rows, filename, "blip")
    blip_df = pd.DataFrame({col: arr.tolist() if arr.ndim != 1 else arr for col, arr in dic.items()})
    del dic
    all_df = pd.concat([all_df, blip_df], axis=1)
    del blip_df

    # loading pandora variables
    dic = _arrays_filling_missing(f["nuselection"]["NeutrinoSelectionFilter"], pandora_vars,
                                  slice_kwargs, n_rows, filename, "pandora")
    pandora_df = pd.DataFrame({col: arr.tolist() if arr.ndim != 1 else arr for col, arr in dic.items()}).add_prefix("pandora_")
    del dic
    all_df = pd.concat([all_df, pandora_df], axis=1)
    del pandora_df

    # hA2025 pion-FSI reweight: a standalone per-event weight computed (in pure
    # Python, no GENIE/ROOT) from the same nuselection mc_generator_* truth slice.
    # It reweights GENIE's INTRANUKE hA2018->hA2025, so it is skipped for data/EXT
    # (no truth), NuWro fake data (a different generator), and the isotropic
    # one-gamma overlay; those are left without the column (-> filled 1.0 when it
    # is folded into wc_net_weight below).
    if filetype not in ("data", "ext", "nuwro_fake_data", "isotropic_one_gamma_overlay"):
        mcg = f["nuselection"]["NeutrinoSelectionFilter"].arrays(
            ["mc_generator_pdg", "mc_generator_mother", "mc_generator_rescatter",
             "mc_generator_statuscode", "mc_generator_E", "mc_generator_px",
             "mc_generator_py", "mc_generator_pz"], library="np", **slice_kwargs)
        # one pass returns both the hA2025 weight and the additional hA2025c
        # factor (hA2025 + pion-charge effects).  hA2025_pion_fsi_rw_weight is the
        # default (folded into wc_net_weight below); multiply it by
        # additional_hA2025c_weight to get the hA2025c variant.  Both stored
        # separately -- additional_hA2025c_weight is NOT folded into wc_net_weight.
        hA2025_w, additional_hA2025c_w = compute_pion_fsi_weights_from_arrays(
            mcg["mc_generator_pdg"], mcg["mc_generator_mother"],
            mcg["mc_generator_rescatter"], mcg["mc_generator_statuscode"],
            mcg["mc_generator_E"], mcg["mc_generator_px"],
            mcg["mc_generator_py"], mcg["mc_generator_pz"])
        all_df["hA2025_pion_fsi_rw_weight"] = hA2025_w
        all_df["additional_hA2025c_weight"] = additional_hA2025c_w
        del mcg

    # loading gLEE variables
    dic = _arrays_filling_missing(f["singlephotonana"]["vertex_tree"], glee_vars,
                                  slice_kwargs, n_rows, filename, "glee vertex_tree")
    if filetype != "ext" and filetype != "data":
        dic.update(_arrays_filling_missing(f["singlephotonana"]["eventweight_tree"], glee_eventweight_vars,
                                           slice_kwargs, n_rows, filename, "glee eventweight_tree"))
    glee_df = pd.DataFrame({col: arr.tolist() if arr.ndim != 1 else arr for col, arr in dic.items()}).add_prefix("glee_")
    del dic
    all_df = pd.concat([all_df, glee_df], axis=1)
    del glee_df

    # loading LANTERN variables
    dic = _arrays_filling_missing(f["lantern"]["EventTree"], lantern_vars,
                                  slice_kwargs, n_rows, filename, "lantern")
    lantern_df = pd.DataFrame({col: arr.tolist() if arr.ndim != 1 else arr for col, arr in dic.items()}).add_prefix("lantern_")
    del dic
    all_df = pd.concat([all_df, lantern_df], axis=1)
    del lantern_df

    del f

    # remove some of these prefixes, for things that should be universal
    all_df.rename(columns={"wc_run": "run", "wc_subrun": "subrun", "wc_event": "event"}, inplace=True)

    all_df["detailed_run_period"] = detailed_run_period
    all_df["filename"] = filename
    all_df["filetype"] = filetype

    return all_df


def process_root_file(filename, frac_events=1):
    """Load an entire ROOT file as a single DataFrame. Thin wrapper around _get_file_metadata + _load_chunk."""
    start_time = time.time()
    print(f"loading {filename}...")
    meta = _get_file_metadata(filename, frac_events)
    all_df = _load_chunk(filename, entry_start=0, entry_stop=meta["n_events"], **meta)
    end_time = time.time()
    events_per_POT = all_df.shape[0] / (meta["file_POT"] / 1e19)
    progress_str = (
        f"\nloaded {meta['filetype']:<30}   Run {meta['detailed_run_period']:<4} "
        f"{all_df.shape[0]:>10,d} events {meta['file_POT']:>10.2e} POT "
        f"{events_per_POT:>6.2f} events / 1e19 POT "
        f"{meta['root_file_size_gb']:>6.2f} GB {end_time - start_time:>6.2f} s"
    )
    if frac_events < 1.0:
        progress_str += f" (f={frac_events})"
    print(progress_str)
    return meta["filetype"], meta["detailed_run_period"], all_df, meta["file_POT"]


if __name__ == "__main__":
    main_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Create merged dataframe from SURPRISE 4b ROOT files")
    parser.add_argument("-f", "--frac_events", type=float, default=1.0,
                        help="Fraction of events (and POT) to load from each file, in (0,1]. Default: 1.0")
    parser.add_argument("-m", "--memory_logger", action="store_true", default=False,
                        help="Start a memory logger thread")
    parser.add_argument("--just_one_file", action="store_true", default=False,
                        help="Only process one file for debugging purposes")
    parser.add_argument("--create_file_dfs", action="store_true", default=False,
                        help="Create file-level dataframes for each file")
    parser.add_argument("--merge_file_dfs", action="store_true", default=False,
                        help="Merge file-level dataframes into a single dataframe")
    parser.add_argument("--chunk_size", type=int, default=100_000,
                        help="Number of events per chunk when reading ROOT files. Default: 100000")
    args = parser.parse_args()

    if args.memory_logger:
        start_memory_logger(1)

    if args.create_file_dfs:
        print("Creating file-level dataframes for each file...")

        if args.frac_events < 1.0:
            print(f"Loading {args.frac_events} fraction of events from each file")

        for file in os.listdir(intermediate_files_location):
            if (file.startswith("curr_df_pl_") and file.endswith(".parquet")) or \
               (file.startswith("chunk_") and file.endswith(".parquet")):
                os.remove(f"{intermediate_files_location}/{file}")
        print("Deleted intermediate df parquet files")

        print(f"Starting loop over root files (chunk_size={args.chunk_size:,})...")

        filenames_with_unused = os.listdir(data_files_location)
        filenames_with_unused.sort()
        # sorting these puts an NC Pi0 overlay first, which will have all the WCPMTInfo and truth variables present, 
        # so it can be used to add columns to future dataframes with missing values

        filenames = []
        for filename in filenames_with_unused:
            if not filename.endswith(".root"):  # skip .csv / .npz / other non-ROOT files in the directory
                continue

            if args.just_one_file and "checkout_MCC9.10_Run4c4d5_v10_04_07_13_BNB_NCpi0_overlay_surprise_reco2_hist_4c.root" not in filename:
                continue

            if "UNUSED" in filename or "older_downloads" in filename:
                continue

            if "nfs000000150070ba7d00000751" in filename: # TEMPORARY: weird file in directory now
                continue

            if "detvar" in filename.lower():
                continue

            filenames.append(filename)

        pot_dic = {}

        for file_num, filename in enumerate(filenames):

            print(f"Processing file {file_num} / {len(filenames)}")
            print(f"loading {filename}...")
            file_start_time = time.time()

            meta = _get_file_metadata(filename, frac_events=args.frac_events)
            filetype = meta["filetype"]
            detailed_run_period = meta["detailed_run_period"]

            file_POT = meta["file_POT"]
            n_events = meta["n_events"]

            # each (filetype, detailed_run_period) should come from exactly one file
            if (filetype, detailed_run_period) in pot_dic:
                raise ValueError(
                    f"Multiple files map to the same (filetype, detailed_run_period) "
                    f"key ({filetype}, {detailed_run_period})! Latest file: {filename}"
                )
            pot_dic[(filetype, detailed_run_period)] = file_POT

            n_chunks = (n_events + args.chunk_size - 1) // args.chunk_size
            chunk_parquet_paths = []

            for chunk_idx, chunk_start in enumerate(range(0, n_events, args.chunk_size)):
                chunk_stop = min(chunk_start + args.chunk_size, n_events)
                print(f"  chunk {chunk_idx + 1}/{n_chunks}: events {chunk_start}-{chunk_stop}...")

                curr_df = _load_chunk(filename, entry_start=chunk_start, entry_stop=chunk_stop, **meta)

                print("  doing post-processing that requires vector variables...")

                curr_df = do_wc_postprocessing(curr_df)
                curr_df = add_extra_true_photon_variables(curr_df)
                curr_df = do_spacepoint_postprocessing(curr_df)
                curr_df = do_pandora_postprocessing(curr_df)
                curr_df = do_blip_postprocessing(curr_df)
                curr_df = do_lantern_postprocessing(curr_df)
                curr_df = do_glee_postprocessing(curr_df)

                curr_df = remove_vector_variables(curr_df)

                # converting to polars
                curr_df_pl = pl.from_pandas(curr_df)
                del curr_df

                # Validate filetype column after conversion to polars
                filetype_values = curr_df_pl["filetype"].unique().to_list()
                if '' in filetype_values or None in filetype_values:
                    empty_count = curr_df_pl.filter(pl.col("filetype") == '').height
                    null_count = curr_df_pl.filter(pl.col("filetype").is_null()).height
                    if empty_count > 0 or null_count > 0:
                        raise ValueError(f"filetype column has empty/null values after polars conversion for {filename} chunk {chunk_idx}: {empty_count} empty, {null_count} null")

                curr_df_pl = curr_df_pl.with_columns([pl.col(pl.Float64).cast(pl.Float32)])
                curr_df_pl = curr_df_pl.with_columns([pl.col(pl.Int32).cast(pl.Int64)])

                chunk_path = f"{intermediate_files_location}/chunk_{file_num}_{chunk_idx}.parquet"
                curr_df_pl.write_parquet(chunk_path)
                chunk_parquet_paths.append(chunk_path)
                del curr_df_pl

            # combine chunks into the per-file parquet
            parquet_path = f"{intermediate_files_location}/curr_df_pl_{file_num}.parquet"
            if len(chunk_parquet_paths) == 1:
                os.rename(chunk_parquet_paths[0], parquet_path)
                print("single chunk, renamed to final parquet")
            else:
                print(f"combining {len(chunk_parquet_paths)} chunks into {parquet_path}...")
                pl.concat(
                    [pl.read_parquet(p) for p in chunk_parquet_paths],
                    how="diagonal_relaxed",
                ).write_parquet(parquet_path)
                for p in chunk_parquet_paths:
                    os.remove(p)
            print(f"curr_df_pl size: {os.path.getsize(parquet_path) / 1e9:.2f} GB (on disk)")
            print("saved to parquet file")

            print(f"Reloading {parquet_path} to ensure on-disk integrity...")
            reloaded_df = pl.read_parquet(parquet_path)
            if "filetype" not in reloaded_df.columns:
                raise ValueError(f"{parquet_path} is missing the filetype column after writing!")
            empty_count = reloaded_df.filter(pl.col("filetype") == '').height
            null_count = reloaded_df.filter(pl.col("filetype").is_null()).height
            if empty_count > 0 or null_count > 0:
                raise ValueError(
                    f"{parquet_path} has corrupted filetype values after writing: "
                    f"{empty_count} empty strings, {null_count} nulls"
                )
            del reloaded_df

            file_end_time = time.time()
            events_per_POT = n_events / (file_POT / 1e19)
            progress_str = (
                f"\nloaded {filetype:<30}   Run {detailed_run_period:<4} "
                f"{n_events:>10,d} events {file_POT:>10.2e} POT "
                f"{events_per_POT:>6.2f} events / 1e19 POT "
                f"{meta['root_file_size_gb']:>6.2f} GB {file_end_time - file_start_time:>6.2f} s"
            )
            if args.frac_events < 1.0:
                progress_str += f" (f={args.frac_events})"
            print(progress_str)

                # TODO: When we have more files, do weighting to make each set of run fractions match the run fractions in data


        print("saving pot_dic to csv file...")
        if os.path.exists(f"{intermediate_files_location}/pot_dic.csv"):
            os.remove(f"{intermediate_files_location}/pot_dic.csv")
        with open(f"{intermediate_files_location}/pot_dic.csv", "w") as f:
            for key, value in pot_dic.items():
                f.write(f"{key[0]},{key[1]},{value}\n")

        print("done creating file-level dataframes for each file")

    if args.merge_file_dfs:
        print("Merging file-level dataframes into a single dataframe...")

        print("loading pot_dic from csv file...")
        with open(f"{intermediate_files_location}/pot_dic.csv", "r") as f:
            pot_dic = {}
            for line in f:
                filetype, detailed_run_period, value = line.strip().split(",")
                pot_dic[(filetype, detailed_run_period)] = float(value)

        for file in os.listdir(intermediate_files_location):
            if file == "presel_df_train_vars.parquet" or file == "all_df.parquet":
                os.remove(f"{intermediate_files_location}/{file}")
        print("Deleted final df parquet files")

        print("loading polars dataframes from parquet files...")

        parquet_files = sorted([
            f"{intermediate_files_location}/{file}"
            for file in os.listdir(intermediate_files_location)
            if file.startswith("curr_df_pl_") and file.endswith(".parquet")
        ])
        print(f"Found {len(parquet_files)} parquet files")

        # Files have differing schemas (overlays carry the wc_truth_* / hA2025 /
        # WCPMTInfo columns that data/EXT lack, while data/EXT carry wc_evtTimeNS
        # that overlays lack), so no single file's schema is a superset.  Union all
        # columns with a diagonal concat (missing entries filled with null) rather
        # than scan_parquet + extra_columns="ignore", which keys off the first
        # file's schema and would silently drop columns absent from it (e.g. all
        # wc_truth_* when a Run123 EXT file happens to sort first).
        all_df = pl.concat(
            [pl.scan_parquet(p) for p in parquet_files],
            how="diagonal_relaxed",
        ).collect()
        print(f"all_df size: {all_df.estimated_size() / 1e9:.2f} GB")
        
        # Validate filetype column immediately after concatenation
        known_filetypes = {
            "ext", "data", "nuwro_fake_data", "nu_overlay", "nue_overlay", "dirt_overlay",
            "nc_pi0_overlay", "numucc_pi0_overlay",
            "delete_one_gamma_overlay", "isotropic_one_gamma_overlay",
        }
        if "filetype" in all_df.columns:
            # Cast to String first so Categorical comparisons don't silently miss values
            filetype_str = all_df.get_column("filetype").cast(pl.String)
            null_count = filetype_str.is_null().sum()
            empty_count = (filetype_str == "").sum()
            unknown_vals = [v for v in filetype_str.unique().to_list() if v is not None and v not in known_filetypes]
            if null_count > 0 or empty_count > 0 or unknown_vals:
                print(f"ERROR after concat: filetype has {empty_count} empty strings, {null_count} nulls, unknown values: {unknown_vals}")
                bad_mask = filetype_str.is_null() | (filetype_str == "")
                for uv in unknown_vals:
                    bad_mask = bad_mask | (filetype_str == uv)
                bad_rows = all_df.filter(bad_mask)
                for row in bad_rows.head(5).iter_rows(named=True):
                    print(f"  Bad filetype row: filetype={row.get('filetype')!r}, filename={row.get('filename', 'N/A')}")
                    print(f"    Stale parquet detected — delete intermediate parquets and re-run --create_file_dfs")
                raise ValueError(f"filetype column corrupted after concatenation: {empty_count} empty, {null_count} null, unknown: {unknown_vals}")

        if all_df.is_empty():
            raise ValueError("No events in the dataframe!")
        
        print(f"all_df.height={all_df.height}")

        # Defrag immediately: scan_parquet over N files produces an N-chunk df and
        # fragments the heap from many small allocations.  Write/reload once here
        # so ALL downstream postprocessing (including add_signal_categories) runs on
        # a fresh single-chunk df with a clean heap (~35 GB baseline instead of ~80 GB).
        temp_defrag_path = f"{intermediate_files_location}/_temp_defrag_all_df.parquet"
        print(f"Early defrag: writing {len(parquet_files)}-chunk df to {temp_defrag_path}...", end="", flush=True)
        _t0 = time.time()
        all_df.lazy().sink_parquet(temp_defrag_path)
        del all_df
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        all_df = pl.read_parquet(temp_defrag_path)
        os.remove(temp_defrag_path)
        gc.collect()
        print(f" done in {time.time() - _t0:.1f}s")

        # Polars sink_parquet can corrupt low-cardinality String columns (like filetype)
        # via dictionary encoding. Detect and fix any rows where filetype became ''.
        bad_filetype_mask = pl.col("filetype").is_null() | (pl.col("filetype") == "")
        bad_count = all_df.filter(bad_filetype_mask).height
        if bad_count > 0:
            print(f"WARNING: {bad_count} rows have empty/null filetype after defrag — re-inferring from filename")
            fixed_filetype = pl.Series(
                "filetype",
                [ft if (ft is not None and ft != "") else _filetype_from_filename(fn)
                 for ft, fn in zip(all_df["filetype"].to_list(), all_df["filename"].to_list())]
            )
            all_df = all_df.with_columns(fixed_filetype)
            still_bad = all_df.filter(bad_filetype_mask).height
            if still_bad > 0:
                raise ValueError(f"{still_bad} rows still have empty/null filetype after re-inference!")
            print(f"  Fixed {bad_count} rows.")

        print("doing post-processing that doesn't require vector variables using polars...")

        all_df = do_combined_postprocessing(all_df)

        # Convert dtypes early so all subsequent postprocessing works on smaller arrays.
        # Float64→Float32 and Int64→Int32 are done in batches to avoid holding two full
        # copies of the dataframe at once.
        print("Converting dtypes to reduce memory usage (before heavy postprocessing)...")
        memory_before = all_df.estimated_size() / (1024**3)
        print(f"Estimated memory usage before conversion: {memory_before:.4f} GB")

        float64_cols = [col for col, dtype in all_df.schema.items() if dtype == pl.Float64]
        int64_cols   = [col for col, dtype in all_df.schema.items() if dtype == pl.Int64]
        print(f"Converting {len(float64_cols)} Float64 columns to Float32")
        print(f"Converting {len(int64_cols)} Int64 columns to Int32 (clipping to Int32 range)")

        if float64_cols:
            all_df = all_df.with_columns([pl.col(col).cast(pl.Float32) for col in float64_cols])
            gc.collect()

        int32_min, int32_max = -2147483648, 2147483647
        batch_size = 50
        for i in range(0, len(int64_cols), batch_size):
            batch = int64_cols[i:i + batch_size]
            all_df = all_df.with_columns([
                pl.col(col).clip(int32_min, int32_max).cast(pl.Int32) for col in batch
            ])
            gc.collect()

        memory_after = all_df.estimated_size() / (1024**3)
        print(f"Estimated memory usage after conversion: {memory_after:.4f} GB")
        print(f"Memory saved: {memory_before - memory_after:.4f} GB ({(memory_before - memory_after) / memory_before * 100:.1f}%)")
        gc.collect()

        weight_configs = get_weight_configs()
        all_df = do_orthogonalization_and_POT_weighting(all_df, pot_dic, weight_configs)
        all_df = apply_rootino_correction(all_df, pot_dic, weight_configs)

        # do_orthogonalization_and_POT_weighting adds new Float64 weight columns; convert them now.
        new_float64_cols = [col for col, dtype in all_df.schema.items() if dtype == pl.Float64]
        if new_float64_cols:
            print(f"Converting {len(new_float64_cols)} new Float64 columns added by postprocessing: {new_float64_cols}")
            all_df = all_df.with_columns([pl.col(col).cast(pl.Float32) for col in new_float64_cols])
            gc.collect()

        all_df = add_signal_categories(all_df)
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            print("malloc_trim successful")
        except Exception:
            print("malloc_trim failed")

        train_test_score_bytes = (
            all_df.select(["filename", "run", "subrun", "event"])
            .hash_rows(seed=0)
            .to_numpy()
            & np.uint64(0xFF)
        ).astype(np.uint8)
        train_test_score = train_test_score_bytes.astype(np.float32) / 256.0
        train_mask = train_test_score < 0.5
        all_df = all_df.with_columns(
            pl.Series("train_test_score", train_test_score),
            pl.Series("will_use_for_50_50_training", train_mask),
        )

        print(f"Total number of events in all_df: {all_df.height}")
        print(f"Number of events in all_df with will_use_for_50_50_training == True: {all_df.select(pl.col('will_use_for_50_50_training').sum()).item()}")

        # Write all_df (with signal categories + train_test_score) to a temp parquet
        # so that apply_1g1mu_rad_corr_reweighting and apply_nc_coh_1g_reweighting
        # can use scan_parquet with predicate pushdown to collect only the small filtered
        # sub-dfs cheaply, then read the full df fresh from disk.
        temp_defrag_path = f"{intermediate_files_location}/_temp_defrag_all_df.parquet"
        print(f"Writing temp parquet to {temp_defrag_path}...", end="", flush=True)
        start_time = time.time()
        all_df.lazy().sink_parquet(temp_defrag_path)
        del all_df
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            print("malloc_trim successful")
        except Exception:
            print("malloc_trim failed")
            pass
        # Use scan_parquet (lazy) to compute the two small filtered sub-dfs with
        # predicate pushdown (reads only the relevant rows, not the full 35 GB).
        # Both add_* functions, when given a LazyFrame, collect and return only
        # the *new* rows to append as a small eager DataFrame.
        # After both small dfs are collected we read the full df once and concat.
        # Temp file is kept until after read_parquet then removed.
        print(f" done in {time.time() - start_time:.1f}s")
        all_lf = pl.scan_parquet(temp_defrag_path)

        # First compute the binned reweighting from the dataframe (replacing the
        # manual rad_corrections_reweighting.ipynb / coherent_1g_reweighting.ipynb
        # step), writing it to parquet, then apply it to produce the new rows.
        # The compute step uses the open-data weighting to build the binned shapes
        # (default net_weight_var); the apply step produces a weight column per
        # config, each normalized to that config's full goal POT.
        compute_1g1mu_rad_corr_reweighting(all_lf)
        rad_corrected_df = apply_1g1mu_rad_corr_reweighting(all_lf, pot_dic, weight_configs)
        compute_nc_coh_1g_reweighting(all_lf)
        coherent_1g_df = apply_nc_coh_1g_reweighting(all_lf, pot_dic, weight_configs)
        del all_lf

        # pi0 Dalitz Geant4->EvtGen reweighting: build the bin-by-bin weight grid from
        # the standalone-Geant4 and EvtGen pure-model samples (independent of the df).
        # apply_pi0_dalitz_reweighting below then looks up each truth Dalitz decay's
        # weight from the wc_true_pi0_dalitz_m_ee / _cos_theta_star columns already on
        # the df.  Unlike the two reweightings above, this APPENDS no rows -- it is a
        # shape correction applied in-place to existing Dalitz events below.
        compute_pi0_dalitz_reweighting()

        print("Reading full df from parquet...")
        start_read = time.time()
        all_df = pl.read_parquet(temp_defrag_path)
        os.remove(temp_defrag_path)
        gc.collect()
        print(f"  read done in {time.time() - start_read:.1f}s, all_df has {all_df.height} rows")

        print("Concatenating full df with new rows...")
        all_df = pl.concat([all_df, rad_corrected_df, coherent_1g_df], how="diagonal_relaxed")
        del rad_corrected_df, coherent_1g_df
        gc.collect()
        print(f"  all_df has {all_df.height} rows after adding rad_corr and coherent events")

        # pi0 Dalitz reweighting: adds the standalone pi0_dalitz_reweight_weight
        # column (1.0 for non-Dalitz events).
        all_df = apply_pi0_dalitz_reweighting(all_df)

        # hA2025 pion-FSI (charge-exchange etc.) reweighting.  Both per-event
        # columns are computed in _load_chunk for the GENIE overlays and absent
        # elsewhere (data/EXT/NuWro/isotropic-1g, and the derived rad-corr /
        # coherent-1g rows); default them to 1.0 wherever missing.  Only
        # hA2025_pion_fsi_rw_weight is folded into the net weights;
        # additional_hA2025c_weight is kept standalone (multiply it onto
        # hA2025_pion_fsi_rw_weight to get the hA2025c variant).
        for _col in ("hA2025_pion_fsi_rw_weight", "additional_hA2025c_weight"):
            if _col not in all_df.columns:
                all_df = all_df.with_columns(pl.lit(1.0).alias(_col))
            all_df = all_df.with_columns(pl.col(_col).fill_null(1.0).cast(pl.Float32))

        # Fold the per-event pi0-Dalitz and hA2025 pion-FSI factors into every
        # per-config net-weight column.  Both are 1.0 where not applicable, and
        # null weights (events excluded from a config) stay null.
        weight_cols = [c["weight_col"] for c in weight_configs]
        all_df = all_df.with_columns([
            (pl.col(wcol) * pl.col("pi0_dalitz_reweight_weight") * pl.col("hA2025_pion_fsi_rw_weight"))
            .cast(pl.Float32).alias(wcol)
            for wcol in weight_cols
        ])

        dup_mask = pl.struct("filetype", "run", "subrun", "event").is_duplicated()
        n_dups = all_df.select(dup_mask.sum()).item()
        if n_dups > 0:
            dups = all_df.filter(dup_mask).select(["filename", "filetype", "run", "subrun", "event", "wc_truth_nuEnergy"])
            print(f"Found {n_dups} duplicate rows, first 10:\n{dups.head(10)}")
            raise ValueError("Duplicate filename/run/subrun/event!")

        # Use sink_parquet via lazy API to stream the filtered data directly to disk
        # without materializing a second full copy of all_df in memory.
        print(f"saving {intermediate_files_location}/presel_df_train_vars.parquet...", end="", flush=True)
        start_time = time.time()
        all_df.lazy().filter(pl.col("wc_kine_reco_Enu") > 0).sink_parquet(
            f"{intermediate_files_location}/presel_df_train_vars.parquet"
        )
        end_time = time.time()
        file_size_gb = os.path.getsize(f"{intermediate_files_location}/presel_df_train_vars.parquet") / 1024**3
        print(f"done, {file_size_gb:.2f} GB, {end_time - start_time:.2f} seconds")

        print(f"saving {intermediate_files_location}/all_df.parquet...", end="", flush=True)
        start_time = time.time()
        # remove the large number of WC training-only-variables for a smaller file size

        remove_columns = wc_training_only_vars

        all_df = all_df.drop(remove_columns)

        all_df.write_parquet(f"{intermediate_files_location}/all_df.parquet")
        end_time = time.time()
        file_size_gb = os.path.getsize(f"{intermediate_files_location}/all_df.parquet") / 1024**3
        print(f"done, {file_size_gb:.2f} GB, {end_time - start_time:.2f} seconds")
        main_end_time = time.time()
        print(f"Total time to create the dataframes: {main_end_time - main_start_time:.2f} seconds")

        print("done merging file-level dataframes into a single dataframe")
    
