
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
from postprocessing import do_orthogonalization_and_POT_weighting, apply_rootino_correction, add_extra_true_photon_variables, do_spacepoint_postprocessing, add_signal_categories, verify_signal_categories
from postprocessing import do_wc_postprocessing, do_pandora_postprocessing, do_lantern_postprocessing, do_combined_postprocessing, do_glee_postprocessing
from postprocessing import add_afro_1mu1p_sel
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
    fullosc_sample_POT,
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
    if "fullosc" in fn:
        return "fullosc_overlay"
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
    elif filetype == "fullosc_overlay":
        # the fullosc file's T_pot tree is wrong (events matched from separate
        # numu and nue files), so its POT is hardcoded
        if detailed_run_period not in fullosc_sample_POT:
            raise ValueError("Fullosc sample POT not found!", filename, detailed_run_period)
        file_POT_total = fullosc_sample_POT[detailed_run_period]

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
        curr_wc_T_pf_vars = wc_T_pf_data_vars
        curr_wc_T_eval_vars = wc_T_eval_data_vars
    elif filetype == "fullosc_overlay":
            # matching flag + per-event numu-flux CV weight, only present in the fullosc file
            curr_wc_T_pf_vars = curr_wc_T_pf_vars
            curr_wc_T_eval_vars = wc_T_eval_vars + ["fullosc", "fullosc_cv_weight"]
    else:
        curr_wc_T_pf_vars = curr_wc_T_pf_vars
        curr_wc_T_eval_vars = wc_T_eval_vars
    dic.update(f["wcpselection"]["T_PFeval"].arrays(curr_wc_T_pf_vars, library="np", **slice_kwargs))
    dic.update(f["wcpselection"]["T_eval"].arrays(curr_wc_T_eval_vars, library="np", **slice_kwargs))
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

    # fullosc sample: keep only events where the numu<->nue matching succeeded;
    # unmatched events (fullosc==0) have fullosc_cv_weight==0 and are not part of
    # the oscillated prediction
    if filetype == "fullosc_overlay":
        n_before = len(all_df)
        all_df = all_df[all_df["wc_fullosc"] == 1].reset_index(drop=True)
        print(f"    fullosc matching cut: kept {len(all_df)}/{n_before} events with fullosc==1")

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
                curr_df = add_afro_1mu1p_sel(curr_df)

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
        # that overlays lack), so no single file's schema is a superset.  The old
        # code concat'd ALL files into one in-memory df, but at ~17.9M rows x ~1600
        # cols that df is ~110 GB even after Int64->Int32 downcasting and OOM-kills
        # the machine (several steps below also transiently need ~2x).  Instead we
        # process the files in row-count-bounded BATCHES, writing one processed
        # parquet per batch (Phase 1), then run the few genuinely-global steps -- the
        # rad-corr / coherent-1g / pi0-Dalitz reweightings, the duplicate check and
        # the final writes -- over lazy scans of those per-batch parquets (Phases 2-3).
        # Peak memory is then one batch (a few GB), never the whole dataset.
        weight_configs = get_weight_configs()

        # Stale per-batch parts from a previous failed run would corrupt the scans below.
        for _f in os.listdir(intermediate_files_location):
            if _f.startswith("_chunk_proc_") or _f.startswith("_chunk_final_") or _f in ("_rad_part.parquet", "_coh_part.parquet"):
                os.remove(f"{intermediate_files_location}/{_f}")

        # Global union schema: a zero-row diagonal concat resolves the unified dtypes
        # without reading data.  Every batch (and the derived rad/coh rows) is reindexed
        # to it so the orthogonalization masks and distance math always see their
        # wc_truth_* / lantern_* / pandora_* columns present (null-filled) -- exactly
        # what the old single diagonal concat provided (the masks reference columns that
        # some filetypes lack, so a data/EXT-only batch would otherwise error).
        union_schema = pl.concat(
            [pl.scan_parquet(p) for p in parquet_files], how="diagonal_relaxed"
        ).collect_schema()
        union_cols = list(union_schema.names())

        def _reindex_to_union(df):
            """Null-fill any union columns missing from df (correct dtype), order them
            canonically, and keep any extra post-processing columns at the end."""
            missing = [c for c in union_cols if c not in df.columns]
            if missing:
                df = df.with_columns([pl.lit(None).cast(union_schema[c]).alias(c) for c in missing])
            extra = [c for c in df.columns if c not in union_cols]
            return df.select(union_cols + extra)

        # Bin files so each batch is <= MAX_ROWS_PER_BATCH rows (the full df is ~6 GB
        # per 1M rows, so a 2M-row batch is ~12 GB before/after downcasting).
        MAX_ROWS_PER_BATCH = 2_000_000
        batches, _cur, _cur_rows = [], [], 0
        for _p in parquet_files:
            _n = pl.scan_parquet(_p).select(pl.len()).collect().item()
            if _cur and _cur_rows + _n > MAX_ROWS_PER_BATCH:
                batches.append(_cur); _cur, _cur_rows = [], 0
            _cur.append(_p); _cur_rows += _n
        if _cur:
            batches.append(_cur)
        print(f"Planned {len(batches)} batches (<= {MAX_ROWS_PER_BATCH:,} rows each) from {len(parquet_files)} files")

        # ── Phase 1: per-batch processing -> one _chunk_proc_{k}.parquet each ──
        # The per-row body below (filetype repair, do_combined_postprocessing, dtype
        # conversion, orthogonalization + POT weighting, ROOTino, signal categories,
        # train/test split) is unchanged from the old single-df flow -- it just runs
        # on one batch at a time.  train_test hashing is per-row, so batch-invariant.
        chunk_paths = []
        for _k, _batch in enumerate(batches):
            print(f"\n=== Phase 1 batch {_k + 1}/{len(batches)}: {len(_batch)} files ===")
            _t0 = time.time()
            all_df = pl.concat([pl.scan_parquet(p) for p in _batch], how="diagonal_relaxed").collect()
            all_df = _reindex_to_union(all_df)

            # Polars sink_parquet can corrupt low-cardinality String columns (like filetype)
            # via dictionary encoding. Detect and fix any rows where filetype became '' before
            # the strict validation below runs (otherwise sink corruption would trip it).
            bad_filetype_mask = pl.col("filetype").is_null() | (pl.col("filetype") == "")
            bad_count = all_df.filter(bad_filetype_mask).height
            if bad_count > 0:
                print(f"WARNING: {bad_count} rows have empty/null filetype after streaming concat — re-inferring from filename")
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

            # Validate filetype column immediately after concatenation
            known_filetypes = {
                "ext", "data", "nuwro_fake_data", "nu_overlay", "nue_overlay", "dirt_overlay",
                "nc_pi0_overlay", "numucc_pi0_overlay",
                "delete_one_gamma_overlay", "isotropic_one_gamma_overlay",
                "fullosc_overlay",
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

            all_df = do_orthogonalization_and_POT_weighting(all_df, pot_dic, weight_configs)
            all_df = apply_rootino_correction(all_df, pot_dic, weight_configs)

            # do_orthogonalization_and_POT_weighting adds new Float64 weight columns; convert them now.
            new_float64_cols = [col for col, dtype in all_df.schema.items() if dtype == pl.Float64]
            if new_float64_cols:
                print(f"Converting {len(new_float64_cols)} new Float64 columns added by postprocessing: {new_float64_cols}")
                all_df = all_df.with_columns([pl.col(col).cast(pl.Float32) for col in new_float64_cols])
                gc.collect()

            # Assign the signal-category columns per batch, but defer the per-category
            # yield printout + exhaustive/mutually-exclusive check to one global pass
            # over all batches (verify_signal_categories, after this loop).
            all_df = add_signal_categories(all_df, verify=False)
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

            _cp = f"{intermediate_files_location}/_chunk_proc_{_k}.parquet"
            all_df.write_parquet(_cp)
            chunk_paths.append(_cp)
            print(f"  batch {_k + 1}/{len(batches)} -> {all_df.height} rows, {time.time() - _t0:.1f}s")
            del all_df
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

        # Signal-category yields + exhaustive/mutually-exclusive check over ALL events
        # at once (a single streaming aggregation over the per-batch parquets, so it
        # never materializes the full df).  Raises if any category is non-exhaustive
        # or overlapping.
        print("\n=== Verifying signal categories globally (all events) ===")
        verify_signal_categories(pl.concat([pl.scan_parquet(p) for p in chunk_paths], how="diagonal_relaxed"))

        # ── Phase 2: global reweightings over a lazy scan of all batch parquets ──
        # Logic unchanged from the old single-df flow: the compute_* build their
        # binned shapes from the global filtered subset (predicate pushdown reads only
        # the relevant rows), and the apply_* return only the small derived rows to
        # append.  They scan the per-batch parquets now instead of one monolithic temp
        # parquet, so the full df is never materialized.
        print("\n=== Phase 2: global reweightings ===")
        all_lf = pl.concat([pl.scan_parquet(p) for p in chunk_paths], how="diagonal_relaxed")
        compute_1g1mu_rad_corr_reweighting(all_lf)
        rad_corrected_df = apply_1g1mu_rad_corr_reweighting(all_lf, pot_dic, weight_configs)
        compute_nc_coh_1g_reweighting(all_lf)
        coherent_1g_df = apply_nc_coh_1g_reweighting(all_lf, pot_dic, weight_configs)
        del all_lf
        # pi0 Dalitz Geant4->EvtGen weight grid (independent of the df), applied per
        # part in Phase 3.
        compute_pi0_dalitz_reweighting()

        # Regenerate the pi0-Dalitz monitoring plots once, globally: gather the (rare)
        # truth-Dalitz events from every batch into one small df and run the reweighting
        # with make_plots=True purely for the plot side-effect.  The Phase-3 per-part
        # calls stay make_plots=False because no single part holds the whole Dalitz set;
        # the plots depend only on the Dalitz events (+ their lab four-vector columns,
        # still present here -- apply_pi0_dalitz_reweighting drops them afterward).
        _dalitz_df = (
            pl.concat([pl.scan_parquet(p) for p in chunk_paths], how="diagonal_relaxed")
            .filter(pl.col("wc_true_has_pi0_dalitz_decay"))
            .collect()
        )
        print(f"  pi0-Dalitz monitoring plots: {_dalitz_df.height} truth-Dalitz events")
        if not _dalitz_df.is_empty():
            apply_pi0_dalitz_reweighting(_dalitz_df, make_plots=True)
        del _dalitz_df
        gc.collect()

        # Persist the small derived-row sets (reindexed to the common schema) so they
        # flow through Phase 3's per-part folds and the final concat just like the
        # batch parquets.
        _rad_path = f"{intermediate_files_location}/_rad_part.parquet"
        _coh_path = f"{intermediate_files_location}/_coh_part.parquet"
        _reindex_to_union(rad_corrected_df).write_parquet(_rad_path)
        _reindex_to_union(coherent_1g_df).write_parquet(_coh_path)
        print(f"  rad-corr rows: {rad_corrected_df.height}, coherent-1g rows: {coherent_1g_df.height}")
        del rad_corrected_df, coherent_1g_df
        gc.collect()

        # ── Phase 3: per-part pi0-Dalitz + hA2025 folds, then streaming writes ──
        # All remaining steps are per-row, so each part (a batch parquet or the rad/coh
        # derived rows) is processed eagerly one at a time -- never the full df.
        # make_plots=False here because the monitoring plots were already generated
        # once, globally, over all Dalitz events in Phase 2.
        print("\n=== Phase 3: pi0-Dalitz + hA2025 folds, streaming writes ===")
        weight_cols = [c["weight_col"] for c in weight_configs]
        final_parts = []
        _phase3_parts = chunk_paths + [_rad_path, _coh_path]
        for _k, _p in enumerate(_phase3_parts):
            print(f"\n--- Phase 3 part {_k + 1}/{len(_phase3_parts)}: {os.path.basename(_p)} ---")
            all_df = pl.read_parquet(_p)
            if all_df.is_empty():
                del all_df
                continue
            # hA2025 pion-FSI columns: present only on GENIE overlays; default to 1.0.
            for _col in ("hA2025_pion_fsi_rw_weight", "additional_hA2025c_weight"):
                if _col not in all_df.columns:
                    all_df = all_df.with_columns(pl.lit(1.0).alias(_col))
                all_df = all_df.with_columns(pl.col(_col).fill_null(1.0).cast(pl.Float32))
            # pi0 Dalitz shape reweight: adds standalone pi0_dalitz_reweight_weight (1.0
            # for non-Dalitz events).  Per-row lookup, so safe per part.
            all_df = apply_pi0_dalitz_reweighting(all_df, make_plots=False)
            # Fold the per-event pi0-Dalitz and hA2025 pion-FSI factors into every
            # per-config net-weight column (1.0 where not applicable; null stays null).
            all_df = all_df.with_columns([
                (pl.col(wcol) * pl.col("pi0_dalitz_reweight_weight") * pl.col("hA2025_pion_fsi_rw_weight"))
                .cast(pl.Float32).alias(wcol)
                for wcol in weight_cols
            ])
            _fp = f"{intermediate_files_location}/_chunk_final_{_k}.parquet"
            all_df.write_parquet(_fp)
            final_parts.append(_fp)
            del all_df
            gc.collect()

        # Global duplicate check over (filetype, run, subrun, event); reads only those
        # key columns across all parts, so it is cheap in memory.
        _keys = pl.concat(
            [pl.scan_parquet(p).select(["filename", "filetype", "run", "subrun", "event"]) for p in final_parts],
            how="diagonal_relaxed",
        ).collect()
        dup_mask = pl.struct("filetype", "run", "subrun", "event").is_duplicated()
        n_dups = _keys.select(dup_mask.sum()).item()
        if n_dups > 0:
            dups = _keys.filter(dup_mask).select(["filename", "filetype", "run", "subrun", "event"])
            print(f"Found {n_dups} duplicate rows, first 10:\n{dups.head(10)}")
            raise ValueError("Duplicate filename/run/subrun/event!")
        del _keys
        gc.collect()

        # Global event counts via cheap scans, reported pre-reweighting to match the
        # old printout (rad/coh rows are appended after these were historically shown).
        _proc_lf = pl.concat([pl.scan_parquet(p) for p in chunk_paths], how="diagonal_relaxed")
        print(f"Total number of events in all_df: {_proc_lf.select(pl.len()).collect().item()}")
        print(f"Number of events in all_df with will_use_for_50_50_training == True: "
              f"{_proc_lf.select(pl.col('will_use_for_50_50_training').sum()).collect().item()}")

        # presel (Enu>0): stream-filter all parts straight to disk (no full df in RAM).
        print(f"saving {intermediate_files_location}/presel_df_train_vars.parquet...", end="", flush=True)
        start_time = time.time()
        pl.concat([pl.scan_parquet(p) for p in final_parts], how="diagonal_relaxed") \
            .filter(pl.col("wc_kine_reco_Enu") > 0) \
            .sink_parquet(f"{intermediate_files_location}/presel_df_train_vars.parquet")
        file_size_gb = os.path.getsize(f"{intermediate_files_location}/presel_df_train_vars.parquet") / 1024**3
        print(f"done, {file_size_gb:.2f} GB, {time.time() - start_time:.2f} seconds")

        # all_df: drop the WC training-only vars and stream-combine all parts to disk.
        print(f"saving {intermediate_files_location}/all_df.parquet...", end="", flush=True)
        start_time = time.time()
        _combined_cols = pl.concat(
            [pl.scan_parquet(p) for p in final_parts], how="diagonal_relaxed"
        ).collect_schema().names()
        remove_columns = [c for c in wc_training_only_vars if c in _combined_cols]
        pl.concat([pl.scan_parquet(p) for p in final_parts], how="diagonal_relaxed") \
            .drop(remove_columns) \
            .sink_parquet(f"{intermediate_files_location}/all_df.parquet")
        file_size_gb = os.path.getsize(f"{intermediate_files_location}/all_df.parquet") / 1024**3
        print(f"done, {file_size_gb:.2f} GB, {time.time() - start_time:.2f} seconds")

        # Clean up per-batch intermediates.
        for _p in chunk_paths + [_rad_path, _coh_path] + final_parts:
            if os.path.exists(_p):
                os.remove(_p)

        main_end_time = time.time()
        print(f"Total time to create the dataframes: {main_end_time - main_start_time:.2f} seconds")

        print("done merging file-level dataframes into a single dataframe")
    
