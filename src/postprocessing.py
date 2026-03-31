import gc
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
import warnings
import copy

from collections import defaultdict

from ntuple_variables.variables import pandora_vector_vars, pandora_vector_vars_with_prefix, vector_columns
from signal_categories import topological_category_queries, topological_category_labels
from signal_categories import del1g_detailed_category_queries, del1g_detailed_category_labels
from signal_categories import del1g_simple_category_queries, del1g_simple_category_labels
from signal_categories import filetype_category_queries, filetype_category_labels
from signal_categories import erin_category_queries, erin_category_labels

from file_locations import intermediate_files_location

def change_dtypes(df):
    """
    Convert Float64 to Float32 and Int64 to Int32 to reduce memory usage.
    Clips Int64 values to Int32 range before conversion.
    """
    print("Converting dtypes to reduce memory usage...")
    memory_before = df.estimated_size() / (1024**3)
    print(f"Estimated memory usage before conversion: {memory_before:.4f} GB")
    print()

    # Get columns to convert
    float64_cols = [col for col, dtype in df.schema.items() if dtype == pl.Float64]
    int64_cols = [col for col, dtype in df.schema.items() if dtype == pl.Int64]

    print(f"Converting {len(float64_cols)} Float64 columns to Float32")
    print(f"Converting {len(int64_cols)} Int64 columns to Int32 (clipping values to Int32 range)")

    # Convert Float64 to Float32
    if float64_cols:
        df = df.with_columns([pl.col(col).cast(pl.Float32) for col in float64_cols])
        gc.collect()

    # Convert Int64 to Int32, clipping values to Int32 range
    # Int32 range: -2,147,483,648 to 2,147,483,647
    int32_min = -2147483648
    int32_max = 2147483647

    if int64_cols:
        # Clip values to Int32 range, then cast to Int32 in batches to limit peak memory
        print(f"Converting {len(int64_cols)} Int64 columns to Int32")
        batch_size = 50
        for i in range(0, len(int64_cols), batch_size):
            batch = int64_cols[i:i + batch_size]
            df = df.with_columns([
                pl.col(col).clip(int32_min, int32_max).cast(pl.Int32)
                for col in batch
            ])
            gc.collect()

    memory_after = df.estimated_size() / (1024**3)
    memory_saved_gb = memory_before - memory_after
    print(f"\nEstimated memory usage after conversion: {memory_after:.4f} GB")
    print(f"Memory saved: {memory_saved_gb:.4f} GB ({memory_saved_gb/memory_before*100:.1f}%)")

    return df



def do_orthogonalization_and_POT_weighting(df, pot_dic, normalizing_POT):

    print("pot_dic:")
    detailed_run_periods = []
    for k, v in sorted(pot_dic.items(), key=lambda item: item[0]):
        print(f"    {k}: {v}")
        filetype, detailed_run_period = k
        detailed_run_periods.append(detailed_run_period)
    detailed_run_periods = sorted(set(detailed_run_periods))
    print()

    normalizing_run_periods = ["4a", "4nota5"]

    # keep 4a separately normalized, since it's different from the rest of run 4
    # eventually, this will be used to separately normalize run 1, run 2, run 3, etc.
    df = df.with_columns([
        pl.when(
            
        pl.col("detailed_run_period") == "4a").then(pl.lit("4a"))

        .when(pl.col("detailed_run_period") == "4b").then(pl.lit("4nota5"))
        .when(pl.col("detailed_run_period") == "4c").then(pl.lit("4nota5"))
        .when(pl.col("detailed_run_period") == "4d").then(pl.lit("4nota5"))
        .when(pl.col("detailed_run_period") == "4bcd").then(pl.lit("4nota5"))
        .when(pl.col("detailed_run_period") == "5").then(pl.lit("4nota5"))

        .otherwise(pl.lit("error!"))
        .alias("normalizing_run_period")
    ])

    normalizing_run_period_pot_dic = defaultdict(float)
    for k, v in pot_dic.items():
        filetype, detailed_run_period = k

        if detailed_run_period == "4a":
            normalizing_run_period_pot_dic[(filetype, "4a")] += v
        
        elif detailed_run_period == "4b":
            normalizing_run_period_pot_dic[(filetype, "4nota5")] += v
        elif detailed_run_period == "4c":
            normalizing_run_period_pot_dic[(filetype, "4nota5")] += v
        elif detailed_run_period == "4d":
            normalizing_run_period_pot_dic[(filetype, "4nota5")] += v
        elif detailed_run_period == "4bcd":
            normalizing_run_period_pot_dic[(filetype, "4nota5")] += v
        elif detailed_run_period == "5":
            normalizing_run_period_pot_dic[(filetype, "4nota5")] += v
        
        else:
            raise ValueError("Detailed run period not found in pot_dic!", detailed_run_period)

    print("normalizing_run_period_pot_dic:")
    for k, v in sorted(normalizing_run_period_pot_dic.items(), key=lambda item: item[0]):
        print(f"    {k}: {v}")
    print()

    norm_goal_pot_dic = {}
    for normalizing_run_period in normalizing_run_periods:
        norm_goal_pot_dic[normalizing_run_period] = 0
    data_in_files = False
    for k, v in pot_dic.items():
        filetype, detailed_run_period = k
        if filetype == "data":
            data_in_files = True

            if detailed_run_period == "4a":
                norm_goal_pot_dic["4a"] += v

            elif detailed_run_period == "4b":
                norm_goal_pot_dic["4nota5"] += v
            elif detailed_run_period == "4c":
                norm_goal_pot_dic["4nota5"] += v
            elif detailed_run_period == "4d":
                norm_goal_pot_dic["4nota5"] += v
            elif detailed_run_period == "4bcd":
                norm_goal_pot_dic["4nota5"] += v
            elif detailed_run_period == "5":
                norm_goal_pot_dic["4nota5"] += v

            else:
                raise ValueError("Detailed run period not found in pot_dic!", detailed_run_period)
    if data_in_files:
        total_norm_goal_pot = sum(norm_goal_pot_dic.values())
        extra_pot_scale_factor = normalizing_POT / total_norm_goal_pot
        print(f"applying extra pot scale factor {extra_pot_scale_factor} to norm_goal_pot_dic...")
        for k, v in norm_goal_pot_dic.items():
            norm_goal_pot_dic[k] = v * extra_pot_scale_factor
    else: # no data (possibly DetVar processing), just normalize to nu_overlay CV POTs found earlier
        for k, v in normalizing_run_period_pot_dic.items():
            filetype, normalizing_run_period = k
            if filetype == 'nu_overlay':
                norm_goal_pot_dic[normalizing_run_period] += v
    
    print("norm_goal_pot_dic:")
    for k, v in sorted(norm_goal_pot_dic.items(), key=lambda item: item[0]):
        print(f"    {k}: {v}")
    print()

    # apply norm_goal_pot_dic to df
    df = df.with_columns([
        pl.when(pl.col("normalizing_run_period") == "4a").then(pl.lit(norm_goal_pot_dic["4a"]))
        .when(pl.col("normalizing_run_period") == "4nota5").then(pl.lit(norm_goal_pot_dic["4nota5"]))
        .otherwise(pl.lit(None).cast(pl.Float64))
        .alias("norm_goal_pot")
    ])

    original_length = df.height

    # these dataframes store the POTs split by norm_run_period, summed over the different filetypes
    summed_POT_nc_1pi0_dic = {}
    summed_POT_nue_cc_dic = {}
    summed_POT_numucc_pi0_dic = {}
    for normalizing_run_period in normalizing_run_periods:
        summed_POT_nc_1pi0_dic[normalizing_run_period] = 0
        summed_POT_nue_cc_dic[normalizing_run_period] = 0
        summed_POT_numucc_pi0_dic[normalizing_run_period] = 0
        
    for k, normalizing_pot in normalizing_run_period_pot_dic.items():
        filetype, normalizing_run_period = k
        if filetype == 'nc_pi0_overlay' or filetype == 'nu_overlay':
            summed_POT_nc_1pi0_dic[normalizing_run_period] += normalizing_pot
        if filetype == 'nue_overlay' or filetype == 'nu_overlay':
            summed_POT_nue_cc_dic[normalizing_run_period] += normalizing_pot
        if filetype == 'numucc_pi0_overlay' or filetype == 'nu_overlay':
            summed_POT_numucc_pi0_dic[normalizing_run_period] += normalizing_pot

    print("summed_POT_nc_1pi0_dic:")
    for k, v in sorted(summed_POT_nc_1pi0_dic.items(), key=lambda item: item[0]):
        print(f"    {k}: {v}")
    print()

    print("summed_POT_nue_cc_dic:")
    for k, v in sorted(summed_POT_nue_cc_dic.items(), key=lambda item: item[0]):
        print(f"    {k}: {v}")
    print()

    print("summed_POT_numucc_pi0_dic:")
    for k, v in sorted(summed_POT_numucc_pi0_dic.items(), key=lambda item: item[0]):
        print(f"    {k}: {v}")
    print()

    print("creating masks...")

    # Get masks for different event types using Polars expressions
    nc_pi0_overlay_true_nc_1pi0_mask = (
        (pl.col("filetype") == 'nc_pi0_overlay') & 
        (pl.col("wc_truth_isCC") == 0) & 
        (pl.col("wc_truth_NprimPio") == 1) & 
        (pl.col("wc_truth_vtxInside") == 1)
    )
    nu_overlay_true_nc_1pi0_mask = (
        (pl.col("filetype") == 'nu_overlay') & 
        (pl.col("wc_truth_isCC") == 0) & 
        (pl.col("wc_truth_NprimPio") == 1) & 
        (pl.col("wc_truth_vtxInside") == 1)
    )

    nue_overlay_true_nue_cc_mask = (
        (pl.col("filetype") == 'nue_overlay') & 
        (pl.col("wc_truth_isCC") == 1) & 
        (pl.col("wc_truth_nuPdg").abs() == 12) & 
        (pl.col("wc_truth_vtxInside") == 1)
    )
    nu_overlay_true_nue_cc_mask = (
        (pl.col("filetype") == 'nu_overlay') & 
        (pl.col("wc_truth_isCC") == 1) & 
        (pl.col("wc_truth_nuPdg").abs() == 12) & 
        (pl.col("wc_truth_vtxInside") == 1)
    )

    numucc_pi0_overlay_true_numucc_pi0_mask = (
        (pl.col("filetype") == 'numucc_pi0_overlay') & 
        (pl.col("wc_truth_isCC") == 1) & 
        (pl.col("wc_truth_nuPdg") == 14) & 
        (pl.col("wc_truth_NprimPio") == 1) & 
        (pl.col("wc_truth_vtxInside") == 1)
    )
    nu_overlay_true_numucc_pi0_mask = (
        (pl.col("filetype") == 'nu_overlay') & 
        (pl.col("wc_truth_isCC") == 1) & 
        (pl.col("wc_truth_nuPdg") == 14) & 
        (pl.col("wc_truth_NprimPio") == 1) & 
        (pl.col("wc_truth_vtxInside") == 1)
    )

    nu_overlay_other_mask = (
        (pl.col("filetype") == 'nu_overlay') & # nu_overlay
        ~((pl.col("wc_truth_isCC") == 0) & (pl.col("wc_truth_NprimPio") == 1) & (pl.col("wc_truth_vtxInside") == 1)) & # not NC 1pi0 inFV
        ~((pl.col("wc_truth_isCC") == 1) & (pl.col("wc_truth_nuPdg").abs() == 12) & (pl.col("wc_truth_vtxInside") == 1)) & # not nueCC inFV
        ~((pl.col("wc_truth_isCC") == 1) & (pl.col("wc_truth_nuPdg") == 14) & (pl.col("wc_truth_NprimPio") == 1) & (pl.col("wc_truth_vtxInside") == 1)) # not numuCC 1pi0 inFV
    )

    dirt_mask = pl.col("filetype") == 'dirt_overlay'
    ext_mask = pl.col("filetype") == 'ext'
    del1g_mask = pl.col("filetype") == 'delete_one_gamma_overlay'
    iso1g_mask = pl.col("filetype") == 'isotropic_one_gamma_overlay'
    data_mask = pl.col("filetype") == 'data'

    print("adding wc_normrunperiod_eventtype_POT variable...")

    # Building the wc_normrunperiod_eventtype_POT column using nested when-then-otherwise
    # Starting with default values from pot_dic, building a chain
    wc_normrunperiod_eventtype_POT_expr = None
    for k, normalizing_pot in sorted(normalizing_run_period_pot_dic.items(), key=lambda item: item[0]):
        filetype, normalizing_run_period = k
        if normalizing_pot == 0:
            raise ValueError(f"normalizing_pot is zero for {filetype} {normalizing_run_period}!")
        if wc_normrunperiod_eventtype_POT_expr is None:
            wc_normrunperiod_eventtype_POT_expr = pl.when((pl.col("filetype") == filetype) & (pl.col("normalizing_run_period") == normalizing_run_period)).then(pl.lit(normalizing_pot))
        else:
            wc_normrunperiod_eventtype_POT_expr = wc_normrunperiod_eventtype_POT_expr.when((pl.col("filetype") == filetype) & (pl.col("normalizing_run_period") == normalizing_run_period)).then(pl.lit(normalizing_pot))
    
    # Add the final otherwise clause (shouldn't happen if all filetypes are covered, will error if used later)
    wc_normrunperiod_eventtype_POT_expr = wc_normrunperiod_eventtype_POT_expr.otherwise(pl.lit(None).cast(pl.Float64))
    
    # Override with summed POTs for specific masks (these are checked first)
    for normalizing_run_period in normalizing_run_periods:
        wc_normrunperiod_eventtype_POT_expr = (
            pl.when((nc_pi0_overlay_true_nc_1pi0_mask) & (pl.col("normalizing_run_period") == normalizing_run_period)).then(pl.lit(summed_POT_nc_1pi0_dic[normalizing_run_period]))
            .when((nu_overlay_true_nc_1pi0_mask) & (pl.col("normalizing_run_period") == normalizing_run_period)).then(pl.lit(summed_POT_nc_1pi0_dic[normalizing_run_period]))
            .when((numucc_pi0_overlay_true_numucc_pi0_mask) & (pl.col("normalizing_run_period") == normalizing_run_period)).then(pl.lit(summed_POT_numucc_pi0_dic[normalizing_run_period]))
            .when((nu_overlay_true_numucc_pi0_mask) & (pl.col("normalizing_run_period") == normalizing_run_period)).then(pl.lit(summed_POT_numucc_pi0_dic[normalizing_run_period]))
            .when((nue_overlay_true_nue_cc_mask) & (pl.col("normalizing_run_period") == normalizing_run_period)).then(pl.lit(summed_POT_nue_cc_dic[normalizing_run_period]))
            .when((nu_overlay_true_nue_cc_mask) & (pl.col("normalizing_run_period") == normalizing_run_period)).then(pl.lit(summed_POT_nue_cc_dic[normalizing_run_period]))
            .otherwise(wc_normrunperiod_eventtype_POT_expr)
        )
    
    # Apply the expression to the dataframe
    df = df.with_columns([
        wc_normrunperiod_eventtype_POT_expr.alias("wc_normrunperiod_eventtype_POT")
    ])

    # check for null wc_normrunperiod_eventtype_POT values
    if df["wc_normrunperiod_eventtype_POT"].is_null().any() or (df["wc_normrunperiod_eventtype_POT"] == 0).any():
        num_null = df["wc_normrunperiod_eventtype_POT"].is_null().sum()
        num_zero = (df["wc_normrunperiod_eventtype_POT"] == 0).sum()

        print(f"about to run debug loop, with {min(df.height, 10)=}")

        zeros_df = df.filter(df["wc_normrunperiod_eventtype_POT"] == 0)

        # print filetype, detailed_run_period, normalizing_run_period for each zero value
        for i in range(min(zeros_df.height, 10)):
            if zeros_df["wc_normrunperiod_eventtype_POT"][i] == 0:
                wc_normrunperiod_eventtype_POT = zeros_df["wc_normrunperiod_eventtype_POT"][i]
                filetype = zeros_df["filetype"][i]
                detailed_run_period = zeros_df["detailed_run_period"][i]
                normalizing_run_period = zeros_df["normalizing_run_period"][i]
                print(f"wc_normrunperiod_eventtype_POT: {wc_normrunperiod_eventtype_POT}, filetype: {filetype}, detailed_run_period: {detailed_run_period}, normalizing_run_period: {normalizing_run_period}")

        raise ValueError(f"wc_normrunperiod_eventtype_POT has {num_null} null values and {num_zero} values equal to zero!")

    # Filter out unwanted events by keeping only the events we want
    combined_mask = (
        nc_pi0_overlay_true_nc_1pi0_mask | nu_overlay_true_nc_1pi0_mask
        | numucc_pi0_overlay_true_numucc_pi0_mask | nu_overlay_true_numucc_pi0_mask
        | nue_overlay_true_nue_cc_mask | nu_overlay_true_nue_cc_mask
        | nu_overlay_other_mask
        | dirt_mask | ext_mask | del1g_mask | iso1g_mask | data_mask
    )

    print("applying combined mask...")
    df = df.filter(combined_mask)
    gc.collect()

    print("adding net weights...")

    # Compute net weights using Polars expressions
    weight_temp = pl.col("wc_weight_cv") * pl.col("wc_weight_spline")
    
    # Set weight_temp to 1.0 for data/ext, or if weight is invalid
    weight_temp = (
        pl.when((pl.col("filetype") == "data") | (pl.col("filetype") == "ext"))
        .then(pl.lit(1.0))
        .when(
            (weight_temp <= 0.0) | (weight_temp > 30.0) | 
            weight_temp.is_nan() | weight_temp.is_infinite()
        )
        .then(pl.lit(1.0))
        .otherwise(weight_temp)
    )

    df = df.with_columns([
        weight_temp.alias("weight_cv_weight_spline")
    ])

    # check for null, nan or infinite values in weight_cv_weight_spline
    if df["weight_cv_weight_spline"].is_null().any() or df["weight_cv_weight_spline"].is_nan().any() or df["weight_cv_weight_spline"].is_infinite().any():
        num_null = df["weight_cv_weight_spline"].is_null().sum()
        num_nan = df["weight_cv_weight_spline"].is_nan().sum()
        num_infinite = df["weight_cv_weight_spline"].is_infinite().sum()
        raise ValueError(f"weight_cv_weight_spline has {num_null} null, {num_nan} nan or {num_infinite} infinite values!")
    
    # Compute final net weight
    df = df.with_columns([
        (pl.col("weight_cv_weight_spline") * (pl.col("norm_goal_pot") / pl.col("wc_normrunperiod_eventtype_POT"))).alias("wc_net_weight")
    ])

    # check for null, nan or infinite values in wc_net_weight
    if df["wc_net_weight"].is_null().any() or df["wc_net_weight"].is_nan().any() or df["wc_net_weight"].is_infinite().any():
        num_null = df["wc_net_weight"].is_null().sum()
        num_nan = df["wc_net_weight"].is_nan().sum()
        num_infinite = df["wc_net_weight"].is_infinite().sum()
        raise ValueError(f"wc_net_weight has {num_null} null, {num_nan} nan or {num_infinite} infinite values!")

    final_length = df.height

    print(f"When combining different file types, went from {original_length} to {final_length} events")

    return df


def do_wc_postprocessing(df):

    # Extra reco variables
    proton_nums_5_MeV  = []
    proton_nums_10_MeV = []
    proton_nums_15_MeV = []
    proton_nums_20_MeV = []
    proton_nums_25_MeV = []
    proton_nums_30_MeV = []
    proton_nums_35_MeV = []
    proton_nums_40_MeV = []
    proton_nums_45_MeV = []
    proton_nums_50_MeV = []
    other_track_nums_5_MeV  = []
    other_track_nums_10_MeV = []
    other_track_nums_15_MeV = []
    other_track_nums_20_MeV = []
    other_track_nums_25_MeV = []
    other_track_nums_30_MeV = []
    other_track_nums_35_MeV = []
    other_track_nums_40_MeV = []
    other_track_nums_45_MeV = []
    other_track_nums_50_MeV = []
    energy_lists = df["wc_kine_energy_particle"].to_numpy()
    pdg_lists = df["wc_kine_particle_type"].to_numpy()
    for i in tqdm(range(df.shape[0]), desc="Adding WC reco particle multiplicity", mininterval=10):
        proton_num_5_MeV = 0
        proton_num_10_MeV = 0
        proton_num_15_MeV = 0
        proton_num_20_MeV = 0
        proton_num_25_MeV = 0
        proton_num_30_MeV = 0
        proton_num_35_MeV = 0
        proton_num_40_MeV = 0
        proton_num_45_MeV = 0
        proton_num_50_MeV = 0
        other_track_num_5_MeV = 0
        other_track_num_10_MeV = 0
        other_track_num_15_MeV = 0
        other_track_num_20_MeV = 0
        other_track_num_25_MeV = 0
        other_track_num_30_MeV = 0
        other_track_num_35_MeV = 0
        other_track_num_40_MeV = 0
        other_track_num_45_MeV = 0
        other_track_num_50_MeV = 0

        energy_list = energy_lists[i]
        if isinstance(energy_list, float) and np.isnan(energy_list):
            proton_nums_5_MeV.append(np.nan)
            proton_nums_10_MeV.append(np.nan)
            proton_nums_15_MeV.append(np.nan)
            proton_nums_20_MeV.append(np.nan)
            proton_nums_25_MeV.append(np.nan)
            proton_nums_30_MeV.append(np.nan)
            proton_nums_35_MeV.append(np.nan)
            proton_nums_40_MeV.append(np.nan)
            proton_nums_45_MeV.append(np.nan)
            other_track_nums_5_MeV.append(np.nan)
            other_track_nums_10_MeV.append(np.nan)
            other_track_nums_15_MeV.append(np.nan)
            other_track_nums_20_MeV.append(np.nan)
            other_track_nums_25_MeV.append(np.nan)
            other_track_nums_30_MeV.append(np.nan)
            other_track_nums_35_MeV.append(np.nan)
            other_track_nums_40_MeV.append(np.nan)
            other_track_nums_45_MeV.append(np.nan)
            other_track_nums_50_MeV.append(np.nan)
            continue
        pdg_list = pdg_lists[i]
        for i in range(len(energy_list)):
            if abs(pdg_list[i]) == 2212:
                if energy_list[i] > 5:
                    proton_num_5_MeV += 1
                if energy_list[i] > 10:
                    proton_num_10_MeV += 1
                if energy_list[i] > 15:
                    proton_num_15_MeV += 1
                if energy_list[i] > 20:
                    proton_num_20_MeV += 1
                if energy_list[i] > 25:
                    proton_num_25_MeV += 1
                if energy_list[i] > 30:
                    proton_num_30_MeV += 1
                if energy_list[i] > 35:
                    proton_num_35_MeV += 1
                if energy_list[i] > 40:
                    proton_num_40_MeV += 1
                if energy_list[i] > 45:
                    proton_num_45_MeV += 1
                if energy_list[i] > 50:
                    proton_num_50_MeV += 1
            if abs(pdg_list[i]) == 13 or abs(pdg_list[i]) == 211: # see N_tracks at https://github.com/BNLIF/wcp-uboone-bdt/blob/main/inc/WCPLEEANA/cuts.h
                if energy_list[i] > 5:
                    other_track_num_5_MeV += 1
                if energy_list[i] > 10:
                    other_track_num_10_MeV += 1
                if energy_list[i] > 15:
                    other_track_num_15_MeV += 1
                if energy_list[i] > 20:
                    other_track_num_20_MeV += 1
                if energy_list[i] > 25:
                    other_track_num_25_MeV += 1
                if energy_list[i] > 30:
                    other_track_num_30_MeV += 1
                if energy_list[i] > 35:
                    other_track_num_35_MeV += 1
                if energy_list[i] > 40:
                    other_track_num_40_MeV += 1
                if energy_list[i] > 45:
                    other_track_num_45_MeV += 1
                if energy_list[i] > 50:
                    other_track_num_50_MeV += 1

        proton_nums_5_MeV.append(proton_num_5_MeV)
        proton_nums_10_MeV.append(proton_num_10_MeV)
        proton_nums_15_MeV.append(proton_num_15_MeV)
        proton_nums_20_MeV.append(proton_num_20_MeV)
        proton_nums_25_MeV.append(proton_num_25_MeV)
        proton_nums_30_MeV.append(proton_num_30_MeV)
        proton_nums_35_MeV.append(proton_num_35_MeV)
        proton_nums_40_MeV.append(proton_num_40_MeV)
        proton_nums_45_MeV.append(proton_num_45_MeV)
        proton_nums_50_MeV.append(proton_num_50_MeV)
        other_track_nums_5_MeV.append(other_track_num_5_MeV)
        other_track_nums_10_MeV.append(other_track_num_10_MeV)
        other_track_nums_15_MeV.append(other_track_num_15_MeV)
        other_track_nums_20_MeV.append(other_track_num_20_MeV)
        other_track_nums_25_MeV.append(other_track_num_25_MeV)
        other_track_nums_30_MeV.append(other_track_num_30_MeV)
        other_track_nums_35_MeV.append(other_track_num_35_MeV)
        other_track_nums_40_MeV.append(other_track_num_40_MeV)
        other_track_nums_45_MeV.append(other_track_num_45_MeV)
        other_track_nums_50_MeV.append(other_track_num_50_MeV)

    df["wc_reco_num_protons_5_MeV"] = proton_nums_5_MeV
    df["wc_reco_num_protons_10_MeV"] = proton_nums_10_MeV
    df["wc_reco_num_protons_15_MeV"] = proton_nums_15_MeV
    df["wc_reco_num_protons_20_MeV"] = proton_nums_20_MeV
    df["wc_reco_num_protons_25_MeV"] = proton_nums_25_MeV
    df["wc_reco_num_protons_30_MeV"] = proton_nums_30_MeV
    df["wc_reco_num_protons_35_MeV"] = proton_nums_35_MeV
    df["wc_reco_num_protons_40_MeV"] = proton_nums_40_MeV
    df["wc_reco_num_protons_45_MeV"] = proton_nums_45_MeV
    df["wc_reco_num_protons_50_MeV"] = proton_nums_50_MeV
    df["wc_reco_num_other_tracks_5_MeV"] = other_track_nums_5_MeV
    df["wc_reco_num_other_tracks_10_MeV"] = other_track_nums_10_MeV
    df["wc_reco_num_other_tracks_15_MeV"] = other_track_nums_15_MeV
    df["wc_reco_num_other_tracks_20_MeV"] = other_track_nums_20_MeV
    df["wc_reco_num_other_tracks_25_MeV"] = other_track_nums_25_MeV
    df["wc_reco_num_other_tracks_30_MeV"] = other_track_nums_30_MeV
    df["wc_reco_num_other_tracks_35_MeV"] = other_track_nums_35_MeV
    df["wc_reco_num_other_tracks_40_MeV"] = other_track_nums_40_MeV
    df["wc_reco_num_other_tracks_45_MeV"] = other_track_nums_45_MeV
    df["wc_reco_num_other_tracks_50_MeV"] = other_track_nums_50_MeV

    if "wc_truth_id" in df.columns: # don't run this for a df that only has EXT for example

        # Extra truth variables
        has_photonuclear_absorption_flags = []
        has_pi0_dalitz_decay_flags = []
        max_true_prim_proton_energies = []
        max_true_prim_proton_costhetas = []
        max_true_prim_proton_phis = []
        sum_true_prim_proton_energies = []
        max_true_prim_neutron_energies = []
        max_true_prim_neutron_costhetas = []
        max_true_prim_neutron_phis = []
        sum_true_prim_neutron_energies = []
        true_leading_shower_energies = []
        true_leading_shower_costhetas = []
        true_leading_shower_phis = []
        true_subleading_shower_energies = []
        true_subleading_shower_costhetas = []
        true_subleading_shower_phis = []
        true_leading_pi0_energies = []
        true_leading_pi0_costhetas = []
        true_leading_pi0_phis = []
        true_leading_pi0_opening_angles = []
        true_outgoing_lepton_energies = []
        true_nums_prim_protons = []
        true_nums_prim_protons_35 = []
        true_nums_prim_neutrons = []
        true_nums_prim_neutrons_35 = []
        truth_ids = df["wc_truth_id"].to_numpy()
        truth_pdgs = df["wc_truth_pdg"].to_numpy()
        truth_mothers = df["wc_truth_mother"].to_numpy()
        truth_startMomentums = df["wc_truth_startMomentum"].to_numpy()
        for i in tqdm(range(df.shape[0]), desc="Adding WC truth particle variables", mininterval=10):
            max_true_prim_proton_energy = -1.
            max_true_prim_proton_costheta = -2.
            max_true_prim_proton_phi = -999.
            sum_true_prim_proton_energy = 0.
            max_true_prim_neutron_energy = -1.
            max_true_prim_neutron_costheta = -2.
            max_true_prim_neutron_phi = -999.
            sum_true_prim_neutron_energy = 0.
            max_shower_energy = -1.
            max_shower_costheta = -2.
            max_shower_phi = -999.
            second_max_shower_energy = -1.
            second_max_shower_costheta = -2.
            second_max_shower_phi = -999.
            max_pi0_energy = -1.
            max_pi0_costheta = -2.
            max_pi0_phi = -999.
            max_pi0_opening_angle = -1.
            true_num_prim_protons = 0
            true_num_prim_protons_35 = 0
            true_num_prim_neutrons = 0
            true_num_prim_neutrons_35 = 0
            true_outgoing_lepton_energy = -1.
            truth_id_list = truth_ids[i]
            truth_pdg_list = truth_pdgs[i]
            truth_mother_list = truth_mothers[i]
            truth_startMomentum_list = truth_startMomentums[i]
            if isinstance(truth_id_list, float) and np.isnan(truth_id_list):
                num_particles = 0
            else:
                num_particles = len(truth_id_list)
            has_photonuclear_absorption = False
            has_pi0_dalitz_decay = False
            for j in range(num_particles):
                if truth_pdg_list[j] == 22: # photon
                    truth_photon_parent_id = truth_mother_list[j]
                    parent_pdg = -1 # default value if the photon has no parent
                    for k in range(len(truth_pdg_list)):
                        if truth_id_list[k] == truth_photon_parent_id:
                            parent_pdg = truth_pdg_list[k]
                    if parent_pdg == 111: # pi0 photon
                        # now, focusing on the pi0 daughter decay products
                        photon_daughter_pdgs = []
                        for k in range(num_particles):
                            if truth_mother_list[k] == truth_id_list[j]:
                                photon_daughter_pdgs.append(truth_pdg_list[k])
                        for pdg in photon_daughter_pdgs:
                            if pdg >= 1_000_000_000: # ten digit nuclear PDG code
                                has_photonuclear_absorption = True

                if truth_pdg_list[j] == 22 or abs(truth_pdg_list[j]) == 11: # photon, electron, or positron
                    if truth_startMomentum_list[j][3] * 1000. > max_shower_energy:
                        second_max_shower_energy = max_shower_energy
                        second_max_shower_costheta = max_shower_costheta
                        second_max_shower_phi = max_shower_phi
                        max_shower_energy = truth_startMomentum_list[j][3] * 1000.
                        max_shower_costheta = truth_startMomentum_list[j][2] / truth_startMomentum_list[j][3] # should be basically z / (x**2 + y**2 + z**2)**0.5
                        max_shower_phi = np.arctan2(truth_startMomentum_list[j][0], truth_startMomentum_list[j][1]) * 180. / np.pi
                    elif truth_startMomentum_list[j][3] * 1000. > second_max_shower_energy:
                        second_max_shower_energy = truth_startMomentum_list[j][3] * 1000.
                        second_max_shower_costheta = truth_startMomentum_list[j][2] / truth_startMomentum_list[j][3] # should be basically z / (x**2 + y**2 + z**2)**0.5
                        second_max_shower_phi = np.arctan2(truth_startMomentum_list[j][0], truth_startMomentum_list[j][1]) * 180. / np.pi

                if truth_pdg_list[j] == 111: # pi0
                    curr_pi0_energy = truth_startMomentum_list[j][3] * 1000. - 134.9768
                    if curr_pi0_energy > max_pi0_energy:
                        max_pi0_energy = curr_pi0_energy
                        max_tot_momentum = np.sqrt(truth_startMomentum_list[j][0]**2 + truth_startMomentum_list[j][1]**2 + truth_startMomentum_list[j][2]**2)
                        max_z_momentum = truth_startMomentum_list[j][2]
                        max_pi0_costheta = max_z_momentum / max_tot_momentum
                        max_pi0_phi = np.arctan2(truth_startMomentum_list[j][0], truth_startMomentum_list[j][1]) * 180. / np.pi

                        pi0_daughter_indices = []
                        for k in range(num_particles):
                            if truth_mother_list[k] == truth_id_list[j]:
                                pi0_daughter_indices.append(k)
                        if len(pi0_daughter_indices) == 2: # Not a Dalitz decay, or otherwise weird Geant4 info
                            pi0_daughter_1_momentum = truth_startMomentum_list[pi0_daughter_indices[0]][:3]
                            pi0_daughter_2_momentum = truth_startMomentum_list[pi0_daughter_indices[1]][:3]
                            pi0_daughter_1_dir = pi0_daughter_1_momentum / np.linalg.norm(pi0_daughter_1_momentum)
                            pi0_daughter_2_dir = pi0_daughter_2_momentum / np.linalg.norm(pi0_daughter_2_momentum)
                            dot_product = np.dot(pi0_daughter_1_dir, pi0_daughter_2_dir)
                            dot_product = np.clip(dot_product, -1.0, 1.0) # accounting for floating point errors near 1 or -1
                            max_pi0_opening_angle = np.arccos(dot_product) * 180 / np.pi

                    daughter_pdgs = []
                    for k in range(num_particles):
                        if truth_mother_list[k] == truth_id_list[j]:
                            daughter_pdgs.append(truth_pdg_list[k])
                    #print("pi0 daughter pdgs:", daughter_pdgs)
                    if 22 in daughter_pdgs and 11 in daughter_pdgs and -11 in daughter_pdgs:
                        has_pi0_dalitz_decay = True

                if truth_mother_list[j] == 0 and truth_pdg_list[j] == 2212: # primary proton
                    true_num_prim_protons += 1
                    if truth_startMomentum_list[j][3] * 1000. - 938.272088 > 35.:
                        true_num_prim_protons_35 += 1
                    if truth_startMomentum_list[j][3] * 1000. - 938.272088 > max_true_prim_proton_energy:
                        max_true_prim_proton_energy = truth_startMomentum_list[j][3] * 1000. - 938.272088
                        max_true_prim_proton_costheta = truth_startMomentum_list[j][2] / truth_startMomentum_list[j][3] # should be basically z / (x**2 + y**2 + z**2)**0.5
                        max_true_prim_proton_phi = np.arctan2(truth_startMomentum_list[j][0], truth_startMomentum_list[j][1]) * 180. / np.pi
                    sum_true_prim_proton_energy += truth_startMomentum_list[j][3] * 1000. - 938.272088

                if truth_mother_list[j] == 0 and truth_pdg_list[j] == 2112: # primary neutron
                    true_num_prim_neutrons += 1
                    if truth_startMomentum_list[j][3] * 1000. - 939.565422 > 35.:
                        true_num_prim_neutrons_35 += 1
                    if truth_startMomentum_list[j][3] * 1000. - 939.565422 > max_true_prim_neutron_energy:
                        max_true_prim_neutron_energy = truth_startMomentum_list[j][3] * 1000. - 939.565422
                        max_true_prim_neutron_costheta = truth_startMomentum_list[j][2] / truth_startMomentum_list[j][3] # should be basically z / (x**2 + y**2 + z**2)**0.5
                        max_true_prim_neutron_phi = np.arctan2(truth_startMomentum_list[j][0], truth_startMomentum_list[j][1]) * 180. / np.pi
                    sum_true_prim_neutron_energy += truth_startMomentum_list[j][3] * 1000. - 939.565422

                if truth_mother_list[j] == 0 and 11 <= abs(truth_pdg_list[j]) <= 16: # lepton
                    true_outgoing_lepton_energy = truth_startMomentum_list[j][3] * 1000.

            max_true_prim_proton_energies.append(max_true_prim_proton_energy)
            max_true_prim_proton_costhetas.append(max_true_prim_proton_costheta)
            max_true_prim_proton_phis.append(max_true_prim_proton_phi)
            sum_true_prim_proton_energies.append(sum_true_prim_proton_energy)
            max_true_prim_neutron_energies.append(max_true_prim_neutron_energy)
            max_true_prim_neutron_costhetas.append(max_true_prim_neutron_costheta)
            max_true_prim_neutron_phis.append(max_true_prim_neutron_phi)
            sum_true_prim_neutron_energies.append(sum_true_prim_neutron_energy)
            true_outgoing_lepton_energies.append(true_outgoing_lepton_energy)
            true_nums_prim_protons.append(true_num_prim_protons)
            true_nums_prim_protons_35.append(true_num_prim_protons_35)
            true_nums_prim_neutrons.append(true_num_prim_neutrons)
            true_nums_prim_neutrons_35.append(true_num_prim_neutrons_35)
            true_leading_shower_energies.append(max_shower_energy)
            true_leading_shower_costhetas.append(max_shower_costheta)
            true_leading_shower_phis.append(max_shower_phi)
            true_subleading_shower_energies.append(second_max_shower_energy)
            true_subleading_shower_costhetas.append(second_max_shower_costheta)
            true_subleading_shower_phis.append(second_max_shower_phi)
            true_leading_pi0_energies.append(max_pi0_energy)
            true_leading_pi0_costhetas.append(max_pi0_costheta)
            true_leading_pi0_phis.append(max_pi0_phi)
            true_leading_pi0_opening_angles.append(max_pi0_opening_angle)
            has_photonuclear_absorption_flags.append(has_photonuclear_absorption)
            has_pi0_dalitz_decay_flags.append(has_pi0_dalitz_decay)

        df["wc_true_max_prim_proton_energy"] = max_true_prim_proton_energies
        df["wc_true_max_prim_proton_costheta"] = max_true_prim_proton_costhetas
        df["wc_true_max_prim_proton_phi"] = max_true_prim_proton_phis
        df["wc_true_sum_prim_proton_energy"] = sum_true_prim_proton_energies
        df["wc_true_num_prim_protons"] = true_nums_prim_protons
        df["wc_true_num_prim_protons_35"] = true_nums_prim_protons_35
        df["wc_true_max_prim_neutron_energy"] = max_true_prim_neutron_energies
        df["wc_true_max_prim_neutron_costheta"] = max_true_prim_neutron_costhetas
        df["wc_true_max_prim_neutron_phi"] = max_true_prim_neutron_phis
        df["wc_true_sum_prim_neutron_energy"] = sum_true_prim_neutron_energies
        df["wc_true_num_prim_neutrons"] = true_nums_prim_neutrons
        df["wc_true_num_prim_neutrons_35"] = true_nums_prim_neutrons_35
        df["wc_true_outgoing_lepton_energy"] = true_outgoing_lepton_energies
        df["wc_true_leading_shower_energy"] = true_leading_shower_energies
        df["wc_true_leading_shower_costheta"] = true_leading_shower_costhetas
        df["wc_true_leading_shower_phi"] = true_leading_shower_phis
        df["wc_true_subleading_shower_energy"] = true_subleading_shower_energies
        df["wc_true_subleading_shower_costheta"] = true_subleading_shower_costhetas
        df["wc_true_subleading_shower_phi"] = true_subleading_shower_phis
        df["wc_true_leading_pi0_energy"] = true_leading_pi0_energies
        df["wc_true_leading_pi0_costheta"] = true_leading_pi0_costhetas
        df["wc_true_leading_pi0_phi"] = true_leading_pi0_phis
        df["wc_true_leading_pi0_opening_angle"] = true_leading_pi0_opening_angles
        df["wc_true_has_photonuclear_absorption"] = pd.Series(has_photonuclear_absorption_flags, dtype=bool)
        df["wc_true_has_pi0_dalitz_decay"] = pd.Series(has_pi0_dalitz_decay_flags, dtype=bool)

    # Ensure this column exists and is boolean for all filetypes (e.g., EXT)
    if "wc_true_has_photonuclear_absorption" not in df.columns:
        df["wc_true_has_photonuclear_absorption"] = False
        df["wc_true_has_pi0_dalitz_decay"] = False
    else:
        df["wc_true_has_photonuclear_absorption"] = df["wc_true_has_photonuclear_absorption"].fillna(False).astype(bool)
        df["wc_true_has_pi0_dalitz_decay"] = df["wc_true_has_pi0_dalitz_decay"].fillna(False).astype(bool)

    # extra primary shower position and angle variables
    shower_thetas = []
    shower_phis = []
    distances_to_boundary = []
    backwards_projected_dists = []
    reco_shower_momentum = df["wc_reco_showerMomentum"].to_numpy()
    reco_nu_vtx_x = df["wc_reco_showervtxX"].to_numpy()
    reco_nu_vtx_y = df["wc_reco_showervtxY"].to_numpy()
    reco_nu_vtx_z = df["wc_reco_showervtxZ"].to_numpy()
    for i in tqdm(range(df.shape[0]), desc="Adding WC shower position and angle variables", mininterval=10):

        if isinstance(reco_shower_momentum[i], float) and np.isnan(reco_shower_momentum[i]):
            shower_thetas.append(np.nan)
            shower_phis.append(np.nan)
            distances_to_boundary.append(np.nan)
            backwards_projected_dists.append(np.nan)
            continue

        reco_shower_momentum_0 = reco_shower_momentum[i][0]
        reco_shower_momentum_1 = reco_shower_momentum[i][1]
        reco_shower_momentum_2 = reco_shower_momentum[i][2]
        reco_shower_momentum_3 = reco_shower_momentum[i][3]
        if reco_shower_momentum_3 > 0:
            reco_shower_momentum_perp = np.sqrt(reco_shower_momentum_0 * reco_shower_momentum_0 + reco_shower_momentum_1 * reco_shower_momentum_1)
            shower_thetas.append(np.arctan2(reco_shower_momentum_perp, reco_shower_momentum_2) * 180. / np.pi)
            shower_phis.append(np.arctan2(reco_shower_momentum_0, reco_shower_momentum_1) * 180. / np.pi)

            shower_momentum_total_3d = np.sqrt(reco_shower_momentum_0 * reco_shower_momentum_0 + 
                                                reco_shower_momentum_1 * reco_shower_momentum_1 + 
                                                reco_shower_momentum_2 * reco_shower_momentum_2)
            shower_unit_vector_3d = [reco_shower_momentum_0 / shower_momentum_total_3d, 
                                        reco_shower_momentum_1 / shower_momentum_total_3d, 
                                        reco_shower_momentum_2 / shower_momentum_total_3d]
            center_x = 130.
            center_y = 0.
            center_z = 525.    
            towards_center_length = np.sqrt((reco_nu_vtx_x[i] - center_x) * (reco_nu_vtx_x[i] - center_x) + 
                                            (reco_nu_vtx_y[i] - center_y) * (reco_nu_vtx_y[i] - center_y) + 
                                            (reco_nu_vtx_z[i] - center_z) * (reco_nu_vtx_z[i] - center_z))
            towards_center_unit_vector_3d = [(center_x - reco_nu_vtx_x[i]) / towards_center_length, 
                                                (center_y - reco_nu_vtx_y[i]) / towards_center_length, 
                                                (center_z - reco_nu_vtx_z[i]) / towards_center_length]

            shower_momentum_total_2d = np.sqrt(reco_shower_momentum_0 * reco_shower_momentum_0 + 
                                                reco_shower_momentum_1 * reco_shower_momentum_1)
            shower_unit_vector_2d = [reco_shower_momentum_0 / shower_momentum_total_3d, 
                                        reco_shower_momentum_1 / shower_momentum_total_3d]
            towards_center_length = np.sqrt((reco_nu_vtx_x[i] - center_x) * (reco_nu_vtx_x[i] - center_x) + 
                                            (reco_nu_vtx_y[i] - center_y) * (reco_nu_vtx_y[i] - center_y))
            towards_center_unit_vector_2d = [(center_x - reco_nu_vtx_x[i]) / towards_center_length, 
                                                (center_y - reco_nu_vtx_y[i]) / towards_center_length]
            
            min_backwards_projected_dist = 1e9
                    
            # projecting to x walls
            if shower_unit_vector_3d[0] > 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_x[i] - (-1)) / shower_unit_vector_3d[0])
            elif shower_unit_vector_3d[0] < 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_x[i] - (254.3)) / shower_unit_vector_3d[0])
                
            # projecting to y walls
            if shower_unit_vector_3d[1] > 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_y[i] - (-115.)) / shower_unit_vector_3d[1])
            elif shower_unit_vector_3d[1] < 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_y[i] - (117.)) / shower_unit_vector_3d[1])
                
            # projecting to z walls
            if shower_unit_vector_3d[2] > 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_z[i] - (0.6)) / shower_unit_vector_3d[2])
            elif shower_unit_vector_3d[2] < 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_z[i] - (1036.4)) / shower_unit_vector_3d[2])

            backwards_projected_dists.append(min_backwards_projected_dist)
            
        else:
            shower_thetas.append(np.nan)
            shower_phis.append(np.nan)
            backwards_projected_dists.append(np.nan)
            
        distances_to_boundary.append(np.min([
            abs(reco_nu_vtx_x[i] - (-1.)),
            abs(reco_nu_vtx_x[i] - (254.3)),
            abs(reco_nu_vtx_y[i] - (-115.)),
            abs(reco_nu_vtx_y[i] - (117.)),
            abs(reco_nu_vtx_z[i] - (0.6)),
            abs(reco_nu_vtx_z[i] - (1036.4))
        ]))
    df["wc_reco_shower_theta"] = shower_thetas
    df["wc_reco_shower_phi"] = shower_phis
    df["wc_reco_distance_to_boundary"] = distances_to_boundary
    df["wc_reco_backwards_projected_dist"] = backwards_projected_dists

    max_prim_proton_energies = []
    max_prim_proton_costhetas = []
    max_prim_proton_phis = []
    max_prim_other_track_energies = []
    max_prim_other_track_costhetas = []
    max_prim_other_track_phis = []
    reco_pdg = df["wc_reco_pdg"].to_numpy()
    reco_mother = df["wc_reco_mother"].to_numpy()
    reco_startMomentum = df["wc_reco_startMomentum"].to_numpy()
    reco_startXYZT = df["wc_reco_startXYZT"].to_numpy()
    showervtx_x = df["wc_reco_showervtxX"].to_numpy()
    showervtx_y = df["wc_reco_showervtxY"].to_numpy()
    showervtx_z = df["wc_reco_showervtxZ"].to_numpy()
    nuvtx_x = df["wc_reco_nuvtxX"].to_numpy()
    nuvtx_y = df["wc_reco_nuvtxY"].to_numpy()
    nuvtx_z = df["wc_reco_nuvtxZ"].to_numpy()

    # -----------------------------------------
    # WC two-shower vertex: midpoint of closest approach between the two primary shower axes
    #
    # For each event with >= 2 showers (wc_reco_pdg == 11), identifies the two highest-energy
    # showers and computes the closest point of approach between their momentum-direction lines,
    # each anchored at the shower start position.  The vertex is the midpoint of the shortest
    # connecting segment (i.e. the average of the two closest points on the respective lines).
    # If lines are nearly parallel (denominator < 1e-10) or < 2 showers exist, fills NaN.
    # -----------------------------------------
    wc_2shw_vtx_xs = []
    wc_2shw_vtx_ys = []
    wc_2shw_vtx_zs = []
    wc_2shw_vtx_doca = []  # distance of closest approach between the two shower lines

    for i in tqdm(range(df.shape[0]), desc="Computing WC two-shower vertex", mininterval=10):
        pdg_arr = reco_pdg[i]
        if isinstance(pdg_arr, float) or not hasattr(pdg_arr, '__getitem__'):
            wc_2shw_vtx_xs.append(np.nan); wc_2shw_vtx_ys.append(np.nan); wc_2shw_vtx_zs.append(np.nan); wc_2shw_vtx_doca.append(np.nan)
            continue

        shower_indices = [j for j in range(len(pdg_arr)) if pdg_arr[j] == 11]
        if len(shower_indices) < 2:
            wc_2shw_vtx_xs.append(np.nan); wc_2shw_vtx_ys.append(np.nan); wc_2shw_vtx_zs.append(np.nan); wc_2shw_vtx_doca.append(np.nan)
            continue

        shower_indices.sort(key=lambda j: reco_startMomentum[i][j][3], reverse=True)
        j0, j1 = shower_indices[0], shower_indices[1]

        p1 = np.array([reco_startXYZT[i][j0][0], reco_startXYZT[i][j0][1], reco_startXYZT[i][j0][2]])
        p2 = np.array([reco_startXYZT[i][j1][0], reco_startXYZT[i][j1][1], reco_startXYZT[i][j1][2]])
        m1 = np.array([reco_startMomentum[i][j0][0], reco_startMomentum[i][j0][1], reco_startMomentum[i][j0][2]])
        m2 = np.array([reco_startMomentum[i][j1][0], reco_startMomentum[i][j1][1], reco_startMomentum[i][j1][2]])

        mag1 = np.linalg.norm(m1); mag2 = np.linalg.norm(m2)
        if mag1 < 1e-10 or mag2 < 1e-10:
            wc_2shw_vtx_xs.append(np.nan); wc_2shw_vtx_ys.append(np.nan); wc_2shw_vtx_zs.append(np.nan); wc_2shw_vtx_doca.append(np.nan)
            continue
        d1 = m1 / mag1; d2 = m2 / mag2

        # Closest approach between line (p1, d1) and line (p2, d2)
        w0 = p1 - p2
        b = np.dot(d1, d2)
        d = np.dot(d1, w0)
        e = np.dot(d2, w0)
        denom = 1.0 - b * b   # = sin²θ, zero when lines are parallel

        if abs(denom) < 1e-10: # parallel lines, no intersection
            wc_2shw_vtx_xs.append(np.nan); wc_2shw_vtx_ys.append(np.nan); wc_2shw_vtx_zs.append(np.nan); wc_2shw_vtx_doca.append(np.nan)
            continue

        t = (b * e - d) / denom
        s = (e - b * d) / denom

        closest1 = p1 + t * d1
        closest2 = p2 + s * d2
        midpoint = 0.5 * (closest1 + closest2)

        wc_2shw_vtx_xs.append(float(midpoint[0]))
        wc_2shw_vtx_ys.append(float(midpoint[1]))
        wc_2shw_vtx_zs.append(float(midpoint[2]))
        wc_2shw_vtx_doca.append(float(np.linalg.norm(closest2 - closest1)))

    df["wc_2shw_vtx_x"]    = wc_2shw_vtx_xs
    df["wc_2shw_vtx_y"]    = wc_2shw_vtx_ys
    df["wc_2shw_vtx_z"]    = wc_2shw_vtx_zs
    df["wc_2shw_vtx_doca"] = wc_2shw_vtx_doca

    proton_vtx_defs = {
        "wcvtx":     (nuvtx_x,     nuvtx_y,     nuvtx_z),
        "wcshwrvtx":   (showervtx_x, showervtx_y, showervtx_z),
        "wc2shwvtx":   (df["wc_2shw_vtx_x"].to_numpy(), df["wc_2shw_vtx_y"].to_numpy(), df["wc_2shw_vtx_z"].to_numpy()),
        "lanternvtx":  (df["lantern_vtxX"].to_numpy(), df["lantern_vtxY"].to_numpy(), df["lantern_vtxZ"].to_numpy()),
        "gleevtx":     (df["glee_reco_vertex_x"].to_numpy(), df["glee_reco_vertex_y"].to_numpy(), df["glee_reco_vertex_z"].to_numpy()),
        "pandoravtx":  (df["pandora_reco_nu_vtx_sce_x"].to_numpy(), df["pandora_reco_nu_vtx_sce_y"].to_numpy(), df["pandora_reco_nu_vtx_sce_z"].to_numpy()),
    }
    num_protons_35_MeV_75cm_lists = {k: [] for k in proton_vtx_defs}

    for i in tqdm(range(df.shape[0]), desc="Adding WC reco prim proton and other track variables", mininterval=10):
        if isinstance(reco_pdg[i], float) and np.isnan(reco_pdg[i]):
            max_prim_proton_energies.append(np.nan)
            max_prim_proton_costhetas.append(np.nan)
            max_prim_proton_phis.append(np.nan)
            max_prim_other_track_energies.append(np.nan)
            max_prim_other_track_costhetas.append(np.nan)
            max_prim_other_track_phis.append(np.nan)
            for k in proton_vtx_defs:
                num_protons_35_MeV_75cm_lists[k].append(np.nan)
            continue

        num_particles = len(reco_pdg[i])
        max_prim_proton_energy = -1.
        max_prim_proton_costheta = -2.
        max_prim_proton_phi = -999.
        max_prim_other_track_energy = -1.
        max_prim_other_track_costheta = -2.
        max_prim_other_track_phi = -999.
        num_protons_35_MeV_75cm = {k: 0 for k in proton_vtx_defs}
        vtx_valid = {k: not (isinstance(vx[i], float) and np.isnan(vx[i]))
                     for k, (vx, vy, vz) in proton_vtx_defs.items()}
        for j in range(num_particles):
            if reco_pdg[i][j] == 2212 and reco_startMomentum[i][j][3] > 35: # proton (any, not just primary)
                for k, (vx, vy, vz) in proton_vtx_defs.items():
                    if vtx_valid[k]:
                        dist = np.sqrt((reco_startXYZT[i][j][0] - vx[i])**2 +
                                       (reco_startXYZT[i][j][1] - vy[i])**2 +
                                       (reco_startXYZT[i][j][2] - vz[i])**2)
                        if dist < 75:
                            num_protons_35_MeV_75cm[k] += 1
            if reco_mother[i][j] == 0: # primary
                if reco_pdg[i][j] == 2212: # proton
                    if reco_startMomentum[i][j][3] > max_prim_proton_energy:
                        max_prim_proton_energy = reco_startMomentum[i][j][3]
                        max_prim_proton_costheta = reco_startMomentum[i][j][2] / reco_startMomentum[i][j][3] # should be basically z / (x**2 + y**2 + z**2)**0.5
                        max_prim_proton_phi = np.arctan2(reco_startMomentum[i][j][0], reco_startMomentum[i][j][1]) * 180. / np.pi
                elif reco_pdg[i][j] == 13: # other track (I think 13 is the only one)
                    if reco_startMomentum[i][j][3] > max_prim_other_track_energy:
                        max_prim_other_track_energy = reco_startMomentum[i][j][3]
                        max_prim_other_track_costheta = reco_startMomentum[i][j][2] / reco_startMomentum[i][j][3] # should be basically z / (x**2 + y**2 + z**2)**0.5
                        max_prim_other_track_phi = np.arctan2(reco_startMomentum[i][j][0], reco_startMomentum[i][j][1]) * 180. / np.pi

        max_prim_proton_energies.append(max_prim_proton_energy)
        max_prim_proton_costhetas.append(max_prim_proton_costheta)
        max_prim_proton_phis.append(max_prim_proton_phi)
        max_prim_other_track_energies.append(max_prim_other_track_energy)
        max_prim_other_track_costhetas.append(max_prim_other_track_costheta)
        max_prim_other_track_phis.append(max_prim_other_track_phi)
        for k in proton_vtx_defs:
            num_protons_35_MeV_75cm_lists[k].append(num_protons_35_MeV_75cm[k])

    df["wc_reco_max_prim_proton_energy"] = max_prim_proton_energies
    df["wc_reco_max_prim_proton_costheta"] = max_prim_proton_costhetas
    df["wc_reco_max_prim_proton_phi"] = max_prim_proton_phis
    df["wc_reco_max_prim_other_track_energy"] = max_prim_other_track_energies
    df["wc_reco_max_prim_other_track_costheta"] = max_prim_other_track_costhetas
    df["wc_reco_max_prim_other_track_phi"] = max_prim_other_track_phis
    for k in proton_vtx_defs:
        df[f"wc_reco_num_protons_35_MeV_75cm_from_{k}"] = num_protons_35_MeV_75cm_lists[k]

    return df


def add_extra_true_photon_variables(df):

    if not "wc_truth_pdg" in df.columns: # don't add these variables if the whole df is EXT, for example
        return df

    true_num_gamma = []
    true_gamma_energies = []
    true_gamma_pairconversion_xs = []
    true_gamma_pairconversion_ys = []
    true_gamma_pairconversion_zs = []
    true_num_gamma_pairconvert = []
    true_num_gamma_pairconvert_in_FV = []
    true_num_gamma_pairconvert_in_FV_20_MeV = []
    true_num_prim_gamma = []

    truth_pdg_arr = df["wc_truth_pdg"].to_numpy()
    truth_id_arr = df["wc_truth_id"].to_numpy()
    truth_mother_arr = df["wc_truth_mother"].to_numpy()
    truth_startMomentum_arr = df["wc_truth_startMomentum"].to_numpy()
    truth_startXYZT_arr = df["wc_truth_startXYZT"].to_numpy()

    num_infinite_loops_broken = 0

    for event_i in tqdm(range(df.shape[0]), desc="Adding true photon variables", mininterval=10):

        if isinstance(truth_id_arr[event_i], float) and np.isnan(truth_id_arr[event_i]):
            true_num_gamma.append(np.nan)
            true_gamma_energies.append(np.nan)
            true_gamma_pairconversion_xs.append(np.nan)
            true_gamma_pairconversion_ys.append(np.nan)
            true_gamma_pairconversion_zs.append(np.nan)
            true_num_gamma_pairconvert.append(np.nan)
            true_num_gamma_pairconvert_in_FV.append(np.nan)
            true_num_gamma_pairconvert_in_FV_20_MeV.append(np.nan)
            true_num_prim_gamma.append(np.nan)
            continue

        num_particles = len(truth_id_arr[event_i])
        
        curr_true_num_gamma = 0
        curr_true_gamma_energies = []
        curr_true_gamma_pairconversion_xs = []
        curr_true_gamma_pairconversion_ys = []
        curr_true_gamma_pairconversion_zs = []
        curr_true_num_gamma_pairconvert = 0
        curr_true_num_gamma_pairconvert_in_FV = 0
        curr_true_num_gamma_pairconvert_in_FV_20_MeV = 0
        curr_true_num_prim_gamma = 0

        pi0_ids = []
        for i in range(num_particles):
            if truth_pdg_arr[event_i][i] == 111:
                pi0_ids.append(truth_id_arr[event_i][i])

        primary_or_pi0_gamma_ids = []
        for i in range(num_particles):

            if truth_mother_arr[event_i][i] in pi0_ids or truth_mother_arr[event_i][i] == 0: # this is a daughter of a pi0 or a primary particle
                if truth_pdg_arr[event_i][i] == 22: # this is a photon from a pi0 or a primary photon (most likely from an eta or Delta radiative)

                    curr_true_num_gamma += 1
                    curr_true_gamma_energies.append(truth_startMomentum_arr[event_i][i][3])
                    primary_or_pi0_gamma_ids.append(truth_id_arr[event_i][i])

                    if truth_mother_arr[event_i][i] == 0:
                        curr_true_num_prim_gamma += 1


        # looking for the first point where the photon transfers more than half its energy to daughter charged particles
        # should be 100% for pair production, but compton scatters can also effectively cause the start of a shower
        # daughter particles could disappear from the geant tree even if it pair converts, that type of photon won't be included here

        # Create a mapping from photon ID to its position in the energy list
        photon_id_to_position = {photon_id: pos for pos, photon_id in enumerate(primary_or_pi0_gamma_ids)}
        
        # Initialize conversion point lists with None values
        curr_true_gamma_pairconversion_xs = [None] * len(primary_or_pi0_gamma_ids)
        curr_true_gamma_pairconversion_ys = [None] * len(primary_or_pi0_gamma_ids)
        curr_true_gamma_pairconversion_zs = [None] * len(primary_or_pi0_gamma_ids)
        
        # looking for pair conversion points, allowing for the possibility of Compton scattering
        for i in range(num_particles):
            if truth_id_arr[event_i][i] in primary_or_pi0_gamma_ids: # pi0/primary -> gamma, this won't include the manually deleted photon

                original_gamma_energy = truth_startMomentum_arr[event_i][i][3]
                cumulative_deposited_energy = 0

                visited_ids = set()
                iteration_count = 0
                
                while True:
                    curr_id = truth_id_arr[event_i][i]
                    
                    if curr_id in visited_ids:
                        num_infinite_loops_broken += 1
                        break

                    visited_ids.add(curr_id)
                    iteration_count += 1
                    
                    descendants_ids = []
                    descendants_indices = []
                    descendants_pdgs = []
                    for j in range(num_particles):
                        if truth_mother_arr[event_i][j] == curr_id: # pi0/primary -> gamma -> this particle
                            descendants_ids.append(truth_id_arr[event_i][j])
                            descendants_indices.append(j)
                            descendants_pdgs.append(truth_pdg_arr[event_i][j])

                    for descendant_i in range(len(descendants_indices)):
                        if abs(descendants_pdgs[descendant_i]) == 11: # electron/positron daughter
                            cumulative_deposited_energy += truth_startMomentum_arr[event_i][descendants_indices[descendant_i]][3]

                    if cumulative_deposited_energy > original_gamma_energy / 2: # it has deposited enough energy to effectively count as a pair conversion
                        break

                    if 22 in descendants_pdgs: # found a compton scatter, hasn't deposited enough energy yet, loop to consider that next photon
                        curr_id = descendants_ids[descendants_pdgs.index(22)]
                        #print("doing a compton scatter")
                    else: # no compton scatter, we're done, it's either a pair conversion or photoelectric absorption or a Geant tree deletion
                        break

                if cumulative_deposited_energy < original_gamma_energy / 2: # weird event, didn't deposit enough energy to count as a pair conversion
                    #print(f"weird event, no daughter photon, but also deposited less than half the energy: {cumulative_deposited_energy} / {original_gamma_energy}")
                    pass
                else:
                    # Store conversion point in the correct position based on photon ID
                    photon_id = truth_id_arr[event_i][i]
                    position = photon_id_to_position[photon_id]
                    
                    curr_true_gamma_pairconversion_xs[position] = truth_startXYZT_arr[event_i][descendants_indices[0]][0]
                    curr_true_gamma_pairconversion_ys[position] = truth_startXYZT_arr[event_i][descendants_indices[0]][1]
                    curr_true_gamma_pairconversion_zs[position] = truth_startXYZT_arr[event_i][descendants_indices[0]][2]
                    curr_true_num_gamma_pairconvert += 1

                    if -1 < curr_true_gamma_pairconversion_xs[position] <= 254.3 and -115.0 < curr_true_gamma_pairconversion_ys[position] <= 117.0 and 0.6 < curr_true_gamma_pairconversion_zs[position] <= 1036.4:
                        curr_true_num_gamma_pairconvert_in_FV += 1

                        if original_gamma_energy > 0.02:
                            curr_true_num_gamma_pairconvert_in_FV_20_MeV += 1

        # Filter out None values from conversion point lists
        curr_true_gamma_pairconversion_xs = [x for x in curr_true_gamma_pairconversion_xs if x is not None]
        curr_true_gamma_pairconversion_ys = [y for y in curr_true_gamma_pairconversion_ys if y is not None]
        curr_true_gamma_pairconversion_zs = [z for z in curr_true_gamma_pairconversion_zs if z is not None]
        
        true_num_gamma.append(curr_true_num_gamma)
        true_gamma_energies.append(curr_true_gamma_energies)
        true_gamma_pairconversion_xs.append(curr_true_gamma_pairconversion_xs)
        true_gamma_pairconversion_ys.append(curr_true_gamma_pairconversion_ys)
        true_gamma_pairconversion_zs.append(curr_true_gamma_pairconversion_zs)
        true_num_gamma_pairconvert.append(curr_true_num_gamma_pairconvert)
        true_num_gamma_pairconvert_in_FV.append(curr_true_num_gamma_pairconvert_in_FV)
        true_num_gamma_pairconvert_in_FV_20_MeV.append(curr_true_num_gamma_pairconvert_in_FV_20_MeV)
        true_num_prim_gamma.append(curr_true_num_prim_gamma)

    if num_infinite_loops_broken > 0:
        print(f"Broke infinite loops in the true gamma daughter search {num_infinite_loops_broken} / {df.shape[0]} times")

    df["true_num_gamma"] = true_num_gamma
    df["true_gamma_energies"] = true_gamma_energies
    df["true_gamma_pairconversion_xs"] = true_gamma_pairconversion_xs
    df["true_gamma_pairconversion_ys"] = true_gamma_pairconversion_ys
    df["true_gamma_pairconversion_zs"] = true_gamma_pairconversion_zs
    df["true_num_gamma_pairconvert"] = true_num_gamma_pairconvert
    df["true_num_gamma_pairconvert_in_FV"] = true_num_gamma_pairconvert_in_FV
    df["true_num_gamma_pairconvert_in_FV_20_MeV"] = true_num_gamma_pairconvert_in_FV_20_MeV
    df["true_one_pairconvert_in_FV_20_MeV"] = true_num_gamma_pairconvert_in_FV_20_MeV == 1
    df["true_num_prim_gamma"] = true_num_prim_gamma

    return df

def do_spacepoint_postprocessing(df):

    if not "wc_truth_pdg" in df.columns: # don't add these variables if the whole df is EXT, for example
        return df

    # min distance from true pair conversion vertex to any WC reco spacepoint
    true_gamma_pairconversion_xs_events = df["true_gamma_pairconversion_xs"].to_numpy()
    true_gamma_pairconversion_ys_events = df["true_gamma_pairconversion_ys"].to_numpy()
    true_gamma_pairconversion_zs_events = df["true_gamma_pairconversion_zs"].to_numpy()
    spacepoint_xs_events = df["wc_Trecchargeblob_spacepoints_x"].to_numpy()
    spacepoint_ys_events = df["wc_Trecchargeblob_spacepoints_y"].to_numpy()
    spacepoint_zs_events = df["wc_Trecchargeblob_spacepoints_z"].to_numpy()
    min_distances_events = []
    for event_i in tqdm(range(len(true_gamma_pairconversion_xs_events)), desc="Adding spacepoint postprocessing variables", mininterval=10):
        true_gamma_pairconversion_xs = true_gamma_pairconversion_xs_events[event_i]
        true_gamma_pairconversion_ys = true_gamma_pairconversion_ys_events[event_i]
        true_gamma_pairconversion_zs = true_gamma_pairconversion_zs_events[event_i]
        spacepoint_xs = spacepoint_xs_events[event_i]
        spacepoint_ys = spacepoint_ys_events[event_i]
        spacepoint_zs = spacepoint_zs_events[event_i]
        min_distances = []
        # Handle events with no true pair conversion points recorded
        if isinstance(true_gamma_pairconversion_xs, float) or len(true_gamma_pairconversion_xs) == 0:
            min_distances_events.append(np.nan)
            continue
        if isinstance(spacepoint_xs, float) or len(spacepoint_xs) == 0:
            min_distances_events.append(np.nan)
            continue
        for true_photon_i in range(len(true_gamma_pairconversion_xs)):
            min_distance = 999999
            pairconvert_x = true_gamma_pairconversion_xs[true_photon_i]
            pairconvert_y = true_gamma_pairconversion_ys[true_photon_i]
            pairconvert_z = true_gamma_pairconversion_zs[true_photon_i]
            for spacepoint_i in range(len(spacepoint_xs)):
                spacepoint_x = spacepoint_xs[spacepoint_i]
                spacepoint_y = spacepoint_ys[spacepoint_i]
                spacepoint_z = spacepoint_zs[spacepoint_i]
                distance = np.sqrt((pairconvert_x - spacepoint_x)**2 + (pairconvert_y - spacepoint_y)**2 + (pairconvert_z - spacepoint_z)**2)
                if distance < min_distance:
                    min_distance = distance
            min_distances.append(min_distance)
        min_distances_events.append(min_distances)
    df["wc_true_gamma_pairconversion_spacepoint_min_distances"] = min_distances_events
    df["wc_true_gamma_pairconversion_spacepoint_max_min_distance"] = [np.max(x) if isinstance(x, list) and len(x) > 0 else np.nan for x in min_distances_events]

    return df


def add_signal_categories(all_df):

    print("Adding extra columns for truth categories...")

    # Add overlay type columns using Polars expressions
    all_df = all_df.with_columns([
        (
            (pl.col("filetype") == "nu_overlay") | 
            (pl.col("filetype") == "nue_overlay") | 
            (pl.col("filetype") == "nc_pi0_overlay") | 
            (pl.col("filetype") == "numucc_pi0_overlay")
        ).alias("normal_overlay"),
        (pl.col("filetype") == "delete_one_gamma_overlay").alias("del1g_overlay"),
        (pl.col("filetype") == "isotropic_one_gamma_overlay").alias("iso1g_overlay"),
        
        # Truth variables
        pl.col("wc_truth_vtxInside").cast(pl.Boolean).alias("wc_truth_inFV"),
        
        # Gamma counts
        (pl.col("true_num_gamma_pairconvert_in_FV") == 0).alias("wc_truth_0g"),
        (pl.col("true_num_gamma_pairconvert_in_FV") == 1).alias("wc_truth_1g"),
        (pl.col("true_num_gamma_pairconvert_in_FV") == 2).alias("wc_truth_2g"),
        (pl.col("true_num_gamma_pairconvert_in_FV") >= 3).alias("wc_truth_3plusg"),
        
        # Proton energy
        (pl.col("wc_true_max_prim_proton_energy") >= 35).alias("wc_truth_Np"),
        (pl.col("wc_true_max_prim_proton_energy") < 35).alias("wc_truth_0p"),

        # neutron energy
        (pl.col("wc_true_max_prim_neutron_energy") >= 35).alias("wc_truth_Nn"),
        (pl.col("wc_true_max_prim_neutron_energy") < 35).alias("wc_truth_0n"),
        
        # CC/NC types
        (pl.col("wc_truth_isCC").cast(pl.Boolean) & (pl.col("wc_truth_nuPdg").abs() == 14)).alias("wc_truth_numuCC"),
        (pl.col("wc_truth_isCC").cast(pl.Boolean) & (pl.col("wc_truth_nuPdg").abs() == 12)).alias("wc_truth_nueCC"),
        pl.col("wc_truth_isCC").cast(pl.Boolean).alias("wc_truth_isCC"),
        
        # Pi0 counts
        (pl.col("wc_truth_NprimPio") == 0).alias("wc_truth_0pi0"),
        (pl.col("wc_truth_NprimPio") == 1).alias("wc_truth_1pi0"),
        (pl.col("wc_truth_NprimPio") > 1).alias("wc_truth_multi_pi0"),
    ])
    
    # Add derived boolean columns
    all_df = all_df.with_columns([
        (~pl.col("wc_truth_numuCC")).alias("wc_truth_notnumuCC"),
        (~pl.col("wc_truth_nueCC")).alias("wc_truth_notnueCC"),
        (~pl.col("wc_truth_isCC")).alias("wc_truth_isNC"),
        (pl.col("wc_truth_numuCC") & pl.col("wc_truth_inFV")).alias("wc_truth_1mu"),
        (~(pl.col("wc_truth_numuCC") & pl.col("wc_truth_inFV"))).alias("wc_truth_0mu"),
        (pl.col("wc_truth_nueCC") & pl.col("wc_truth_inFV")).alias("wc_truth_1e"),
        (~(pl.col("wc_truth_nueCC") & pl.col("wc_truth_inFV"))).alias("wc_truth_0e"),
    ])
    
    # wc_truth_NCDelta actually means true NC Delta radiative or true CC Delta radiative
    # see https://microboone.slack.com/archives/C08LHGZSXC4/p1757450436839739
    all_df = all_df.with_columns([
        ((~pl.col("wc_truth_isCC")) & pl.col("wc_truth_NCDelta").cast(pl.Boolean)).alias("wc_truth_NCDeltaRad"),
        (pl.col("wc_truth_numuCC") & pl.col("wc_truth_NCDelta").cast(pl.Boolean)).alias("wc_truth_numuCCDeltaRad"),
        (pl.col("wc_truth_nueCC") & pl.col("wc_truth_NCDelta").cast(pl.Boolean)).alias("wc_truth_nueCCDeltaRad"),
    ])

    all_df = all_df.with_columns([
        (
            (pl.col("wc_truth_single_photon") == 1) &
            (
                (pl.col("wc_truth_isCC") == 0) |
                ((pl.col("wc_truth_isCC") == 1) & (pl.col("wc_truth_nuPdg").abs() == 14) & ((pl.col("wc_truth_muonMomentum_3") - 0.105658).abs() < 0.1))
            )
        ).cast(pl.Int32).alias("erin_inclusive_1g_true_sig")
    ])


    """
    df = df.with_columns([
        (
            ((pl.col("wc_reco_nuvtxX") - pl.col("pandora_reco_nu_vtx_x"))**2 +
             (pl.col("wc_reco_nuvtxY") - pl.col("pandora_reco_nu_vtx_y"))**2 +
             (pl.col("wc_reco_nuvtxZ") - pl.col("pandora_reco_nu_vtx_z"))**2)
            .sqrt()
            .alias("wc_pandora_dist")
        ),
        (
            ((pl.col("wc_reco_nuvtxX") - pl.col("lantern_vtxX"))**2 +
             (pl.col("wc_reco_nuvtxY") - pl.col("lantern_vtxY"))**2 +
             (pl.col("wc_reco_nuvtxZ") - pl.col("lantern_vtxZ"))**2)
            .sqrt()
            .alias("wc_lantern_dist")
        ),
        (
            ((pl.col("lantern_vtxX") - pl.col("pandora_reco_nu_vtx_x"))**2 +
             (pl.col("lantern_vtxY") - pl.col("pandora_reco_nu_vtx_y"))**2 +
             (pl.col("lantern_vtxZ") - pl.col("pandora_reco_nu_vtx_z"))**2)
            .sqrt()
            .alias("lantern_pandora_dist")
        ),
    ])
    """

    # For each vertex type and radius, build combined proton + blip counts.
    # Two proton cut variants: with 75cm distance cut (vtx-specific), and without.
    _vtx_keys = ["wcvtx", "wcshwrvtx", "wc2shwvtx", "lanternvtx", "gleevtx", "pandoravtx"]
    _blip_2shwr_radii = [1, 3, 10]
    _combined_exprs = [
        (pl.col("blip_backtrack_cones_n") + pl.col("wc_reco_num_protons_35_MeV"))
        .alias("wc_reco_num_protons_35_MeV_plus_backtrack_blips"),
    ]
    for _vtx in _vtx_keys:
        _combined_exprs.append(
            (pl.col("blip_backtrack_cones_n") + pl.col(f"wc_reco_num_protons_35_MeV_75cm_from_{_vtx}"))
            .alias(f"wc_reco_num_protons_35_MeV_75cm_from_{_vtx}_plus_backtrack_blips")
        )
        for _R in _blip_2shwr_radii:
            _combined_exprs.append(
                (pl.col(f"blip_2shwr_{_vtx}_nWithin_{_R}cm") + pl.col("wc_reco_num_protons_35_MeV"))
                .alias(f"wc_reco_num_protons_35_MeV_plus_{_vtx}_{_R}cm_vertex_blips")
            )
            _combined_exprs.append(
                (pl.col(f"blip_2shwr_{_vtx}_nWithin_{_R}cm") + pl.col(f"wc_reco_num_protons_35_MeV_75cm_from_{_vtx}"))
                .alias(f"wc_reco_num_protons_35_MeV_75cm_from_{_vtx}_plus_{_vtx}_{_R}cm_vertex_blips")
            )
    all_df = all_df.with_columns(_combined_exprs)

    topological_conditions = []
    print("Adding topological signal categories...")
    for query_text in topological_category_queries:
        topological_conditions.append(eval(query_text, {'pl': pl, '__builtins__': {}}))
    topological_category_expr = None
    for i, condition in enumerate(topological_conditions):
        if topological_category_expr is None:
            topological_category_expr = pl.when(condition).then(pl.lit(i))
        else:
            topological_category_expr = topological_category_expr.when(condition).then(pl.lit(i))
    topological_category_expr = topological_category_expr.otherwise(pl.lit(-1))
    all_df = all_df.with_columns([
        topological_category_expr.alias('topological_signal_category')
    ])
    print("\ntopological signal categories:")
    category_counts_unweighted = []
    category_counts_weighted = []
    for topological_signal_category_i, topological_signal_category in enumerate(topological_category_labels):
        curr_df = all_df.filter(pl.col('topological_signal_category') == topological_signal_category_i)
        unweighted_num = curr_df.height
        weighted_num = curr_df['wc_net_weight'].sum()
        category_counts_unweighted.append(unweighted_num)
        category_counts_weighted.append(weighted_num)
        print(f"    {topological_signal_category}: {weighted_num:.2f} ({unweighted_num})")
    total_events = all_df.height
    if sum(category_counts_unweighted) != total_events:
        print(f"Error: Sum of topological category counts ({sum(category_counts_unweighted)}) != total events ({total_events}), missing or overlapping categories?")
        condition_match_count = pl.sum_horizontal([condition.cast(pl.Int8) for condition in topological_conditions])
        num_no_match, num_multi_match = all_df.select([
            (condition_match_count == 0).sum().alias("num_no_match"),
            (condition_match_count > 1).sum().alias("num_multi_match"),
        ]).row(0)
        print(f"Debug topological category assignment: no_match={num_no_match}, multi_match={num_multi_match}")
        # Also check for category=-1 directly; events where ALL conditions are null
        # won't be caught by the no_match check (since null == 0 returns null, not True).
        cat_neg1_count = all_df.filter(pl.col('topological_signal_category') == -1).height
        cat_null_count = all_df.filter(pl.col('topological_signal_category').is_null()).height
        print(f"  topological_signal_category == -1: {cat_neg1_count}, null: {cat_null_count}")
        if cat_neg1_count > 0:
            cat_neg1_df = all_df.filter(pl.col('topological_signal_category') == -1)
            print("Category=-1 by filetype (top 10):")
            print(cat_neg1_df.group_by("filetype").len().sort("len", descending=True).head(10))
            print("Category=-1 null truth variable counts:")
            print(cat_neg1_df.select([
                pl.col("wc_truth_inFV").is_null().sum().alias("null_inFV"),
                pl.col("wc_truth_0e").is_null().sum().alias("null_0e"),
                pl.col("wc_truth_0g").is_null().sum().alias("null_0g"),
                pl.col("wc_truth_3plusg").is_null().sum().alias("null_3plusg"),
                pl.col("normal_overlay").is_null().sum().alias("null_normal_overlay"),
            ]))
            for row in cat_neg1_df.select([
                "filename", "filetype", "run", "subrun", "event",
                "normal_overlay", "del1g_overlay", "iso1g_overlay",
                "wc_truth_inFV", "wc_truth_0e", "wc_truth_0g",
                "wc_truth_1g", "wc_truth_2g", "wc_truth_3plusg",
                "wc_truth_0mu", "wc_truth_1mu", "wc_truth_Np", "wc_truth_0p",
                "wc_truth_nuPdg", "wc_truth_isCC",
            ]).head(5).iter_rows(named=True):
                print(f"Category=-1 example: {row}")
        if num_no_match > 0:
            no_match_df = all_df.filter(condition_match_count == 0)
            print("No-match by filetype (top 10):")
            print(no_match_df.group_by("filetype").len().sort("len", descending=True).head(10))
            print("No-match overlay flags:")
            print(no_match_df.select([
                pl.col("normal_overlay").cast(pl.Int64).sum().alias("normal_overlay"),
                pl.col("del1g_overlay").cast(pl.Int64).sum().alias("del1g_overlay"),
                pl.col("iso1g_overlay").cast(pl.Int64).sum().alias("iso1g_overlay"),
            ]))
            print("No-match truth signature counts (top 10):")
            print(no_match_df.group_by([
                "wc_truth_inFV",
                "wc_truth_0e",
                "wc_truth_1e",
                "wc_truth_0g",
                "wc_truth_1g",
                "wc_truth_2g",
                "wc_truth_3plusg",
                "wc_truth_0mu",
                "wc_truth_1mu",
                "wc_truth_Np",
                "wc_truth_0p",
            ]).len().sort("len", descending=True).head(10))
            for row in no_match_df.select([
                "filename",
                "filetype",
                "run",
                "subrun",
                "event",
                "normal_overlay",
                "del1g_overlay",
                "iso1g_overlay",
                "wc_truth_inFV",
                "wc_truth_0e",
                "wc_truth_1e",
                "wc_truth_0g",
                "wc_truth_1g",
                "wc_truth_2g",
                "wc_truth_3plusg",
                "wc_truth_0mu",
                "wc_truth_1mu",
                "wc_truth_Np",
                "wc_truth_0p",
                "wc_truth_nuPdg",
                "wc_truth_isCC",
            ]).head(5).iter_rows(named=True):
                print(f"No-match example: {row}")
        if num_multi_match > 0:
            multi_match_df = all_df.filter(condition_match_count > 1).with_columns([
                condition_match_count.alias("n_topological_matches")
            ])
            print("Multi-match by filetype (top 10):")
            print(multi_match_df.group_by("filetype").len().sort("len", descending=True).head(10))
            for row in multi_match_df.select([
                "filename",
                "filetype",
                "run",
                "subrun",
                "event",
                "n_topological_matches",
            ]).head(5).iter_rows(named=True):
                print(f"Multi-match example: {row}")
        raise AssertionError
    uncategorized_count = all_df.filter(pl.col('topological_signal_category') == -1).height
    if uncategorized_count > 0:
        print(f"Uncategorized topological signal categories ({uncategorized_count} events)!")
        row = all_df.filter(pl.col('topological_signal_category') == -1).row(0, named=True)
        print(f"Example: {row['filename']=}, {row['filetype']=}, {row['run']=}, {row['subrun']=}, {row['event']=}, {row['true_num_gamma_pairconvert_in_FV']=}, {row['wc_truth_isCC']=}, {row['wc_truth_nuPdg']=}, {row['wc_truth_NprimPio']=}, {row['wc_truth_0e']=}, {row['wc_truth_0g']=}, {row['wc_truth_1g']=}, {row['wc_truth_2g']=}")
        raise AssertionError

    print("Adding del1g detailed signal categories...")
    del1g_detailed_conditions = []
    for query_text in del1g_detailed_category_queries:
        del1g_detailed_conditions.append(eval(query_text, {'pl': pl, '__builtins__': {}}))
    # assign integers to categories
    del1g_detailed_category_expr = None
    for i, condition in enumerate(del1g_detailed_conditions):
        if del1g_detailed_category_expr is None:
            del1g_detailed_category_expr = pl.when(condition).then(pl.lit(i))
        else:
            del1g_detailed_category_expr = del1g_detailed_category_expr.when(condition).then(pl.lit(i))
    del1g_detailed_category_expr = del1g_detailed_category_expr.otherwise(pl.lit(-1))
    all_df = all_df.with_columns([
        del1g_detailed_category_expr.alias("del1g_detailed_signal_category")
    ])
    print("\ndel1g detailed signal categories:")
    category_counts_unweighted = []
    category_counts_weighted = []
    for del1g_detailed_signal_category_i, del1g_detailed_signal_category in enumerate(del1g_detailed_category_labels):
        curr_df = all_df.filter(pl.col('del1g_detailed_signal_category') == del1g_detailed_signal_category_i)
        unweighted_num = curr_df.height
        weighted_num = curr_df['wc_net_weight'].sum()
        category_counts_unweighted.append(unweighted_num)
        category_counts_weighted.append(weighted_num)
        print(f"    {del1g_detailed_signal_category}: {weighted_num:.2f} ({unweighted_num})")
    total_events = all_df.height
    if sum(category_counts_unweighted) > total_events:
        print(f"Error: Sum of del1g detailed category counts ({sum(category_counts_unweighted)}) > total events ({total_events}), overlapping categories?")
        raise AssertionError
    uncategorized_count = all_df.filter(pl.col('del1g_detailed_signal_category') == -1).height
    if uncategorized_count > 0:
        print(f"Uncategorized detailed del1g signal categories ({uncategorized_count} events)!")
        row = all_df.filter(pl.col('del1g_detailed_signal_category') == -1).row(0, named=True)
        print(f"Example: {row['filename']=}, {row['filetype']=}, {row['run']=}, {row['subrun']=}, {row['event']=}, {row['del1g_overlay']=}, {row['iso1g_overlay']=}, {row['wc_truth_inFV']=}, {row['wc_truth_Np']=}, {row['wc_truth_NCDelta']=}, {row['true_num_prim_gamma']=}, {row['wc_truth_0pi0']=}, {row['wc_truth_1pi0']=}, {row['wc_truth_multi_pi0']=}")
        print(f"Additional fields: {row.get('wc_truth_isNC', 'N/A')=}, {row.get('wc_truth_nueCC', 'N/A')=}, {row.get('wc_truth_numuCC', 'N/A')=}, {row.get('wc_truth_NCDeltaRad', 'N/A')=}, {row.get('normal_overlay', 'N/A')=}")
        print(f"Additional fields: {row.get('wc_true_has_photonuclear_absorption', 'N/A')=}, {row.get('true_num_gamma_pairconvert_in_FV', 'N/A')=}, {row.get('true_num_gamma_pairconvert_in_FV_20_MeV', 'N/A')=}, {row.get('wc_true_gamma_pairconversion_spacepoint_max_min_distance', 'N/A')=}")
        print(f"Additional fields: {row.get('wc_true_has_pi0_dalitz_decay', 'N/A')=}, {row.get('wc_truth_notnueCC', 'N/A')=}")
        raise AssertionError

    print("Adding del1g simple signal categories...")
    del1g_simple_conditions = []
    for query_text in del1g_simple_category_queries:
        del1g_simple_conditions.append(eval(query_text, {'pl': pl, '__builtins__': {}}))
    # assign integers to categories
    del1g_simple_category_expr = None
    for i, condition in enumerate(del1g_simple_conditions):
        if del1g_simple_category_expr is None:
            del1g_simple_category_expr = pl.when(condition).then(pl.lit(i))
        else:
            del1g_simple_category_expr = del1g_simple_category_expr.when(condition).then(pl.lit(i))
    del1g_simple_category_expr = del1g_simple_category_expr.otherwise(pl.lit(-1))
    all_df = all_df.with_columns([
        del1g_simple_category_expr.alias("del1g_simple_signal_category")
    ])
    print("\ndel1g simple signal categories:")
    category_counts_unweighted = []
    category_counts_weighted = []
    for del1g_simple_signal_category_i, del1g_simple_signal_category in enumerate(del1g_simple_category_labels):
        curr_df = all_df.filter(pl.col('del1g_simple_signal_category') == del1g_simple_signal_category_i)
        unweighted_num = curr_df.height
        weighted_num = curr_df['wc_net_weight'].sum()
        category_counts_unweighted.append(unweighted_num)
        category_counts_weighted.append(weighted_num)
        print(f"    {del1g_simple_signal_category}: {weighted_num:.2f} ({unweighted_num})")
    total_events = all_df.height
    if sum(category_counts_unweighted) != total_events:
        print(f"Error: Sum of del1g simple category counts ({sum(category_counts_unweighted)}) != total events ({total_events}), missing or overlapping categories?")
        raise AssertionError
    uncategorized_count = all_df.filter(pl.col('del1g_simple_signal_category') == -1).height
    if uncategorized_count > 0:
        print(f"Uncategorized simple del1g signal categories ({uncategorized_count} events)!")
        row = all_df.filter(pl.col('del1g_simple_signal_category') == -1).row(0, named=True)
        print(f"Example: {row['filename']=}, {row['filetype']=}, {row['run']=}, {row['subrun']=}, {row['event']=}, {row['del1g_overlay']=}, {row['iso1g_overlay']=}, {row['wc_truth_inFV']=}, {row['wc_truth_Np']=}, {row['wc_truth_0mu']=}, {row['wc_truth_numuCC']=}, {row['wc_truth_nueCC']=}, {row['wc_truth_0pi0']=}, {row['wc_truth_1pi0']=}, {row['wc_truth_NCDelta']=}, {row['wc_truth_numuCCDeltaRad']=}, {row['wc_truth_NCDeltaRad']=}")
        raise AssertionError

    print("Adding filetype signal categories...")
    filetype_conditions = []
    for query_text in filetype_category_queries:
        filetype_conditions.append(eval(query_text, {'pl': pl, '__builtins__': {}}))
    # assign integers to categories
    filetype_category_expr = None
    for i, condition in enumerate(filetype_conditions):
        if filetype_category_expr is None:
            filetype_category_expr = pl.when(condition).then(pl.lit(i))
        else:
            filetype_category_expr = filetype_category_expr.when(condition).then(pl.lit(i))
    filetype_category_expr = filetype_category_expr.otherwise(pl.lit(-1))
    all_df = all_df.with_columns([
        filetype_category_expr.alias("filetype_signal_category")
    ])
    print("\nfiletype signal categories:")
    category_counts_unweighted = []
    category_counts_weighted = []
    for filetype_signal_category_i, filetype_signal_category in enumerate(filetype_category_labels):
        curr_df = all_df.filter(pl.col('filetype_signal_category') == filetype_signal_category_i)
        unweighted_num = curr_df.height
        weighted_num = curr_df['wc_net_weight'].sum()
        category_counts_unweighted.append(unweighted_num)
        category_counts_weighted.append(weighted_num)
        print(f"    {filetype_signal_category}: {weighted_num:.2f} ({unweighted_num})")
    total_events = all_df.height
    uncategorized_count = all_df.filter(pl.col('filetype_signal_category') == -1).height
    if uncategorized_count > 0:
        print(f"\nUncategorized filetype signal categories ({uncategorized_count} events)!")
        uncategorized_df = all_df.filter(pl.col('filetype_signal_category') == -1)
        # Print unique filetypes in uncategorized events
        unique_filetypes = uncategorized_df['filetype'].unique().to_list()
        print(f"Unique filetypes in uncategorized events: {unique_filetypes}")
        # Print counts per filetype
        for filetype in unique_filetypes:
            count = uncategorized_df.filter(pl.col('filetype') == filetype).height
            print(f"  {filetype}: {count} events")
        row = uncategorized_df.row(0, named=True)
        print(f"Example uncategorized event: {row['filename']=}, {row['filetype']=}, {row['run']=}, {row['subrun']=}, {row['event']=}, {row['wc_truth_inFV']=}, {row['wc_truth_Np']=}, {row['wc_truth_0mu']=}")
        raise AssertionError
    if sum(category_counts_unweighted) != total_events:
        print(f"Error: Sum of filetype category counts ({sum(category_counts_unweighted)}) != total events ({total_events}), missing or overlapping categories?")
        raise AssertionError

    print("Adding erin signal categories...")
    erin_conditions = []
    for query_text in erin_category_queries:
        erin_conditions.append(eval(query_text, {'pl': pl, '__builtins__': {}}))
    # assign integers to categories
    erin_category_expr = None
    for i, condition in enumerate(erin_conditions):
        if erin_category_expr is None:
            erin_category_expr = pl.when(condition).then(pl.lit(i))
        else:
            erin_category_expr = erin_category_expr.when(condition).then(pl.lit(i))
    erin_category_expr = erin_category_expr.otherwise(pl.lit(-1))
    all_df = all_df.with_columns([
        erin_category_expr.alias("erin_signal_category")
    ])
    print("\nerin signal categories:")
    category_counts_unweighted = []
    category_counts_weighted = []
    for erin_signal_category_i, erin_signal_category in enumerate(erin_category_labels):
        curr_df = all_df.filter(pl.col('erin_signal_category') == erin_signal_category_i)
        unweighted_num = curr_df.height
        weighted_num = curr_df['wc_net_weight'].sum()
        category_counts_unweighted.append(unweighted_num)
        category_counts_weighted.append(weighted_num)
        print(f"    {erin_signal_category}: {weighted_num:.2f} ({unweighted_num})")
    total_events = all_df.height
    uncategorized_count = all_df.filter(pl.col('erin_signal_category') == -1).height
    if uncategorized_count > 0:
        print(f"\nUncategorized erin signal categories ({uncategorized_count} events)!")
        uncategorized_df = all_df.filter(pl.col('erin_signal_category') == -1)
        # Print unique filetypes in uncategorized events
        unique_filetypes = uncategorized_df['filetype'].unique().to_list()
        print(f"Unique filetypes in uncategorized events: {unique_filetypes}")
        # Print counts per filetype
        for filetype in unique_filetypes:
            count = uncategorized_df.filter(pl.col('filetype') == filetype).height
            print(f"  {filetype}: {count} events")
        row = uncategorized_df.row(0, named=True)
        print(f"Example uncategorized event: {row['filename']=}, {row['filetype']=}, {row['run']=}, {row['subrun']=}, {row['event']=}, {row['wc_truth_inFV']=}, {row['wc_truth_Np']=}, {row['wc_truth_0mu']=}")
        raise AssertionError
    if sum(category_counts_unweighted) != total_events:
        print(f"Error: Sum of erin category counts ({sum(category_counts_unweighted)}) != total events ({total_events}), missing or overlapping categories?")
        raise AssertionError

    print("finished adding signal categories")

    return all_df


def do_pandora_postprocessing(df):

    vector_var_dic = {}
    for var in pandora_vector_vars_with_prefix:
        vector_var_dic[var] = df[var].to_numpy()
    
    processed_var_dic = defaultdict(list)

    num_events = len(vector_var_dic[pandora_vector_vars_with_prefix[0]])
    for event_i in tqdm(range(num_events), desc="Analyzing pandora pfps", mininterval=10):
        curr_event_vector_var_dic = {}
        for var in pandora_vector_vars_with_prefix:
            curr_event_vector_var_dic[var] = vector_var_dic[var][event_i]

        curr_event_processed_var_dic = {}

        for var in pandora_vector_vars:
            curr_event_processed_var_dic[f"pandora_max3_len_trk_{var}"] = np.nan
            curr_event_processed_var_dic[f"pandora_max2_len_trk_{var}"] = np.nan
            curr_event_processed_var_dic[f"pandora_max_len_trk_{var}"] = np.nan
        
        max_pfp_len = 0
        max2_pfp_len = -1
        max3_pfp_len = -2
        for pfp_i in range(len(curr_event_vector_var_dic[pandora_vector_vars_with_prefix[0]])):
            curr_pfp_len = curr_event_vector_var_dic["pandora_trk_len_v"][pfp_i]
            if curr_pfp_len > max_pfp_len:
                for var in pandora_vector_vars:
                    curr_event_processed_var_dic[f"pandora_max3_len_trk_{var}"] = curr_event_processed_var_dic[f"pandora_max2_len_trk_{var}"]
                    curr_event_processed_var_dic[f"pandora_max2_len_trk_{var}"] = curr_event_processed_var_dic[f"pandora_max_len_trk_{var}"]
                    curr_event_processed_var_dic[f"pandora_max_len_trk_{var}"] = curr_event_vector_var_dic[f"pandora_{var}"][pfp_i]
            elif curr_pfp_len > max2_pfp_len:
                for var in pandora_vector_vars:
                    curr_event_processed_var_dic[f"pandora_max3_len_trk_{var}"] = curr_event_processed_var_dic[f"pandora_max2_len_trk_{var}"]
                    curr_event_processed_var_dic[f"pandora_max2_len_trk_{var}"] = curr_event_vector_var_dic[f"pandora_{var}"][pfp_i]
            elif curr_pfp_len > max3_pfp_len:
                for var in pandora_vector_vars:
                    curr_event_processed_var_dic[f"pandora_max_len_trk_{var}"] = curr_event_vector_var_dic[f"pandora_{var}"][pfp_i]
        
        for processed_var in curr_event_processed_var_dic.keys():
            processed_var_dic[processed_var].append(curr_event_processed_var_dic[processed_var])

    return pd.concat([df, pd.DataFrame(processed_var_dic)], axis=1)


def do_glee_postprocessing(df):

    isolation_min_dist_trk_unassoc = df["glee_isolation_min_dist_trk_unassoc"].to_numpy()
    isolation_min_dist_trk_shr = df["glee_isolation_min_dist_trk_shr"].to_numpy()
    isolation_nearest_shr_hit_to_trk_time = df["glee_isolation_nearest_shr_hit_to_trk_time"].to_numpy()
    isolation_nearest_shr_hit_to_trk_wire = df["glee_isolation_nearest_shr_hit_to_trk_wire"].to_numpy()
    isolation_nearest_unassoc_hit_to_trk_time = df["glee_isolation_nearest_unassoc_hit_to_trk_time"].to_numpy()
    isolation_nearest_unassoc_hit_to_trk_wire = df["glee_isolation_nearest_unassoc_hit_to_trk_wire"].to_numpy()
    isolation_num_shr_hits_win_10cm_trk = df["glee_isolation_num_shr_hits_win_10cm_trk"].to_numpy()
    isolation_num_shr_hits_win_1cm_trk = df["glee_isolation_num_shr_hits_win_1cm_trk"].to_numpy()
    isolation_num_shr_hits_win_2cm_trk = df["glee_isolation_num_shr_hits_win_2cm_trk"].to_numpy()
    isolation_num_shr_hits_win_5cm_trk = df["glee_isolation_num_shr_hits_win_5cm_trk"].to_numpy()
    isolation_num_unassoc_hits_win_10cm_trk = df["glee_isolation_num_unassoc_hits_win_10cm_trk"].to_numpy()
    isolation_num_unassoc_hits_win_1cm_trk = df["glee_isolation_num_unassoc_hits_win_1cm_trk"].to_numpy()
    isolation_num_unassoc_hits_win_2cm_trk = df["glee_isolation_num_unassoc_hits_win_2cm_trk"].to_numpy()
    isolation_num_unassoc_hits_win_5cm_trk = df["glee_isolation_num_unassoc_hits_win_5cm_trk"].to_numpy()
    min_isolation_min_dist_trk_unassoc = []
    min_isolation_min_dist_trk_shr = []
    min_isolation_nearest_shr_hit_to_trk_time = []
    min_isolation_nearest_shr_hit_to_trk_wire = []
    min_isolation_nearest_unassoc_hit_to_trk_time = []
    min_isolation_nearest_unassoc_hit_to_trk_wire = []
    sum_isolation_num_shr_hits_win_10cm_trk = []
    sum_isolation_num_shr_hits_win_1cm_trk = []
    sum_isolation_num_shr_hits_win_2cm_trk = []
    sum_isolation_num_shr_hits_win_5cm_trk = []
    sum_isolation_num_unassoc_hits_win_10cm_trk = []
    sum_isolation_num_unassoc_hits_win_1cm_trk = []
    sum_isolation_num_unassoc_hits_win_2cm_trk = []
    sum_isolation_num_unassoc_hits_win_5cm_trk = []
    for event_i in tqdm(range(len(isolation_min_dist_trk_shr)), desc="Analyzing gLEE isolation variables", mininterval=10):
        min_dist = 1e9
        curr_min_isolation_min_dist_trk_unassoc = np.nan
        curr_min_isolation_min_dist_trk_shr = np.nan
        curr_min_isolation_nearest_shr_hit_to_trk_time = np.nan
        curr_min_isolation_nearest_shr_hit_to_trk_wire = np.nan
        curr_min_isolation_nearest_unassoc_hit_to_trk_time = np.nan
        curr_min_isolation_nearest_unassoc_hit_to_trk_wire = np.nan
        curr_sum_isolation_num_shr_hits_win_10cm_trk = np.nan
        curr_sum_isolation_num_shr_hits_win_1cm_trk = np.nan
        curr_sum_isolation_num_shr_hits_win_2cm_trk = np.nan
        curr_sum_isolation_num_shr_hits_win_5cm_trk = np.nan
        curr_sum_isolation_num_unassoc_hits_win_10cm_trk = np.nan
        curr_sum_isolation_num_unassoc_hits_win_1cm_trk = np.nan
        curr_sum_isolation_num_unassoc_hits_win_2cm_trk = np.nan
        curr_sum_isolation_num_unassoc_hits_win_5cm_trk = np.nan
        # ensuring we have a non-empty list of unassoc hits that we can loop over
        temp_list = isolation_min_dist_trk_unassoc[event_i]
        if not (isinstance(temp_list, (float, np.floating)) or not hasattr(temp_list, '__len__') or len(temp_list) == 0): 
            # looping over all unassoc hits
            for unassoc_hit_j in range(len(temp_list)):
                # saving values for the closest unassoc hit to the track
                if isolation_min_dist_trk_unassoc[event_i][unassoc_hit_j] < min_dist:
                    min_dist = isolation_min_dist_trk_unassoc[event_i][unassoc_hit_j]
                    curr_min_isolation_min_dist_trk_unassoc = isolation_min_dist_trk_unassoc[event_i][unassoc_hit_j]
                    curr_min_isolation_min_dist_trk_shr = isolation_min_dist_trk_shr[event_i][unassoc_hit_j]
                    curr_min_isolation_nearest_shr_hit_to_trk_time = isolation_nearest_shr_hit_to_trk_time[event_i][unassoc_hit_j]
                    curr_min_isolation_nearest_shr_hit_to_trk_wire = isolation_nearest_shr_hit_to_trk_wire[event_i][unassoc_hit_j]
                    curr_min_isolation_nearest_unassoc_hit_to_trk_time = isolation_nearest_unassoc_hit_to_trk_time[event_i][unassoc_hit_j]
                    curr_min_isolation_nearest_unassoc_hit_to_trk_wire = isolation_nearest_unassoc_hit_to_trk_wire[event_i][unassoc_hit_j]
                    curr_sum_isolation_num_shr_hits_win_10cm_trk = isolation_num_shr_hits_win_10cm_trk[event_i][unassoc_hit_j]
                    curr_sum_isolation_num_shr_hits_win_1cm_trk = isolation_num_shr_hits_win_1cm_trk[event_i][unassoc_hit_j]
                    curr_sum_isolation_num_shr_hits_win_2cm_trk = isolation_num_shr_hits_win_2cm_trk[event_i][unassoc_hit_j]
                    curr_sum_isolation_num_shr_hits_win_5cm_trk = isolation_num_shr_hits_win_5cm_trk[event_i][unassoc_hit_j]
                    curr_sum_isolation_num_unassoc_hits_win_10cm_trk = isolation_num_unassoc_hits_win_10cm_trk[event_i][unassoc_hit_j]
                    curr_sum_isolation_num_unassoc_hits_win_1cm_trk = isolation_num_unassoc_hits_win_1cm_trk[event_i][unassoc_hit_j]
                    curr_sum_isolation_num_unassoc_hits_win_2cm_trk = isolation_num_unassoc_hits_win_2cm_trk[event_i][unassoc_hit_j]
                    curr_sum_isolation_num_unassoc_hits_win_5cm_trk = isolation_num_unassoc_hits_win_5cm_trk[event_i][unassoc_hit_j]
        min_isolation_min_dist_trk_unassoc.append(curr_min_isolation_min_dist_trk_unassoc)
        min_isolation_min_dist_trk_shr.append(curr_min_isolation_min_dist_trk_shr)
        min_isolation_nearest_shr_hit_to_trk_time.append(curr_min_isolation_nearest_shr_hit_to_trk_time)
        min_isolation_nearest_shr_hit_to_trk_wire.append(curr_min_isolation_nearest_shr_hit_to_trk_wire)
        min_isolation_nearest_unassoc_hit_to_trk_time.append(curr_min_isolation_nearest_unassoc_hit_to_trk_time)
        min_isolation_nearest_unassoc_hit_to_trk_wire.append(curr_min_isolation_nearest_unassoc_hit_to_trk_wire)
        sum_isolation_num_shr_hits_win_10cm_trk.append(curr_sum_isolation_num_shr_hits_win_10cm_trk)
        sum_isolation_num_shr_hits_win_1cm_trk.append(curr_sum_isolation_num_shr_hits_win_1cm_trk)
        sum_isolation_num_shr_hits_win_2cm_trk.append(curr_sum_isolation_num_shr_hits_win_2cm_trk)
        sum_isolation_num_shr_hits_win_5cm_trk.append(curr_sum_isolation_num_shr_hits_win_5cm_trk)
        sum_isolation_num_unassoc_hits_win_10cm_trk.append(curr_sum_isolation_num_unassoc_hits_win_10cm_trk)
        sum_isolation_num_unassoc_hits_win_1cm_trk.append(curr_sum_isolation_num_unassoc_hits_win_1cm_trk)
        sum_isolation_num_unassoc_hits_win_2cm_trk.append(curr_sum_isolation_num_unassoc_hits_win_2cm_trk)
        sum_isolation_num_unassoc_hits_win_5cm_trk.append(curr_sum_isolation_num_unassoc_hits_win_5cm_trk)
    glee_isolation_df = pd.DataFrame({
        "glee_dist_ranked_isolation_min_dist_trk_unassoc": min_isolation_min_dist_trk_unassoc,
        "glee_dist_ranked_isolation_min_dist_trk_shr": min_isolation_min_dist_trk_shr,
        "glee_dist_ranked_isolation_nearest_shr_hit_to_trk_time": min_isolation_nearest_shr_hit_to_trk_time,
        "glee_dist_ranked_isolation_nearest_shr_hit_to_trk_wire": min_isolation_nearest_shr_hit_to_trk_wire,
        "glee_dist_ranked_isolation_nearest_unassoc_hit_to_trk_time": min_isolation_nearest_unassoc_hit_to_trk_time,
        "glee_dist_ranked_isolation_nearest_unassoc_hit_to_trk_wire": min_isolation_nearest_unassoc_hit_to_trk_wire,
        "glee_dist_ranked_isolation_num_shr_hits_win_10cm_trk": sum_isolation_num_shr_hits_win_10cm_trk,
        "glee_dist_ranked_isolation_num_shr_hits_win_1cm_trk": sum_isolation_num_shr_hits_win_1cm_trk,
        "glee_dist_ranked_isolation_num_shr_hits_win_2cm_trk": sum_isolation_num_shr_hits_win_2cm_trk,
        "glee_dist_ranked_isolation_num_shr_hits_win_5cm_trk": sum_isolation_num_shr_hits_win_5cm_trk,
        "glee_dist_ranked_isolation_num_unassoc_hits_win_10cm_trk": sum_isolation_num_unassoc_hits_win_10cm_trk,
        "glee_dist_ranked_isolation_num_unassoc_hits_win_1cm_trk": sum_isolation_num_unassoc_hits_win_1cm_trk,
        "glee_dist_ranked_isolation_num_unassoc_hits_win_2cm_trk": sum_isolation_num_unassoc_hits_win_2cm_trk,
        "glee_dist_ranked_isolation_num_unassoc_hits_win_5cm_trk": sum_isolation_num_unassoc_hits_win_5cm_trk,
    }, index=df.index)

    trackstub_candidate_in_nu_slice = df["glee_trackstub_candidate_in_nu_slice"].to_numpy()
    trackstub_candidate_num_hits = df["glee_trackstub_candidate_num_hits"].to_numpy()
    trackstub_candidate_num_wires = df["glee_trackstub_candidate_num_wires"].to_numpy()
    trackstub_candidate_num_ticks = df["glee_trackstub_candidate_num_ticks"].to_numpy()
    trackstub_candidate_plane = df["glee_trackstub_candidate_plane"].to_numpy()
    trackstub_candidate_PCA = df["glee_trackstub_candidate_PCA"].to_numpy()
    trackstub_candidate_mean_ADC = df["glee_trackstub_candidate_mean_ADC"].to_numpy()
    trackstub_candidate_ADC_RMS = df["glee_trackstub_candidate_ADC_RMS"].to_numpy()
    trackstub_candidate_min_dist = df["glee_trackstub_candidate_min_dist"].to_numpy()
    trackstub_candidate_min_impact_parameter_to_shower = df["glee_trackstub_candidate_min_impact_parameter_to_shower"].to_numpy()
    trackstub_candidate_min_conversion_dist_to_shower_start = df["glee_trackstub_candidate_min_conversion_dist_to_shower_start"].to_numpy()
    trackstub_candidate_min_ioc_to_shower_start = df["glee_trackstub_candidate_min_ioc_to_shower_start"].to_numpy()
    trackstub_candidate_ioc_based_length = df["glee_trackstub_candidate_ioc_based_length"].to_numpy()
    trackstub_candidate_wire_tick_based_length = df["glee_trackstub_candidate_wire_tick_based_length"].to_numpy()
    trackstub_candidate_mean_ADC_first_half = df["glee_trackstub_candidate_mean_ADC_first_half"].to_numpy()
    trackstub_candidate_mean_ADC_second_half = df["glee_trackstub_candidate_mean_ADC_second_half"].to_numpy()
    trackstub_candidate_mean_ADC_first_to_second_ratio = df["glee_trackstub_candidate_mean_ADC_first_to_second_ratio"].to_numpy()
    trackstub_candidate_track_angle_wrt_shower_direction = df["glee_trackstub_candidate_track_angle_wrt_shower_direction"].to_numpy()
    trackstub_candidate_linear_fit_chi2 = df["glee_trackstub_candidate_linear_fit_chi2"].to_numpy()
    trackstub_candidate_energy = df["glee_trackstub_candidate_energy"].to_numpy()

    dist_ranked_trackstub_candidate_in_nu_slice = []
    dist_ranked_trackstub_candidate_num_hits = []
    dist_ranked_trackstub_candidate_num_wires = []
    dist_ranked_trackstub_candidate_num_ticks = []
    dist_ranked_trackstub_candidate_plane = []
    dist_ranked_trackstub_candidate_PCA = []
    dist_ranked_trackstub_candidate_mean_ADC = []
    dist_ranked_trackstub_candidate_ADC_RMS = []
    dist_ranked_trackstub_candidate_min_dist = []
    dist_ranked_trackstub_candidate_min_impact_parameter_to_shower = []
    dist_ranked_trackstub_candidate_min_conversion_dist_to_shower_start = []
    dist_ranked_trackstub_candidate_min_ioc_to_shower_start = []
    dist_ranked_trackstub_candidate_ioc_based_length = []
    dist_ranked_trackstub_candidate_wire_tick_based_length = []
    dist_ranked_trackstub_candidate_mean_ADC_first_half = []
    dist_ranked_trackstub_candidate_mean_ADC_second_half = []
    dist_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio = []
    dist_ranked_trackstub_candidate_track_angle_wrt_shower_direction = []
    dist_ranked_trackstub_candidate_linear_fit_chi2 = []
    dist_ranked_trackstub_candidate_energy = []
    energy_ranked_trackstub_candidate_in_nu_slice = []
    energy_ranked_trackstub_candidate_num_hits = []
    energy_ranked_trackstub_candidate_num_wires = []
    energy_ranked_trackstub_candidate_num_ticks = []
    energy_ranked_trackstub_candidate_plane = []
    energy_ranked_trackstub_candidate_PCA = []
    energy_ranked_trackstub_candidate_mean_ADC = []
    energy_ranked_trackstub_candidate_ADC_RMS = []
    energy_ranked_trackstub_candidate_min_dist = []
    energy_ranked_trackstub_candidate_min_impact_parameter_to_shower = []
    energy_ranked_trackstub_candidate_min_conversion_dist_to_shower_start = []
    energy_ranked_trackstub_candidate_min_ioc_to_shower_start = []
    energy_ranked_trackstub_candidate_ioc_based_length = []
    energy_ranked_trackstub_candidate_wire_tick_based_length = []
    energy_ranked_trackstub_candidate_mean_ADC_first_half = []
    energy_ranked_trackstub_candidate_mean_ADC_second_half = []
    energy_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio = []
    energy_ranked_trackstub_candidate_track_angle_wrt_shower_direction = []
    energy_ranked_trackstub_candidate_linear_fit_chi2 = []
    energy_ranked_trackstub_candidate_energy = []
    conv_ranked_trackstub_candidate_in_nu_slice = []
    conv_ranked_trackstub_candidate_num_hits = []
    conv_ranked_trackstub_candidate_num_wires = []
    conv_ranked_trackstub_candidate_num_ticks = []
    conv_ranked_trackstub_candidate_plane = []
    conv_ranked_trackstub_candidate_PCA = []
    conv_ranked_trackstub_candidate_mean_ADC = []
    conv_ranked_trackstub_candidate_ADC_RMS = []
    conv_ranked_trackstub_candidate_min_dist = []
    conv_ranked_trackstub_candidate_min_impact_parameter_to_shower = []
    conv_ranked_trackstub_candidate_min_conversion_dist_to_shower_start = []
    conv_ranked_trackstub_candidate_min_ioc_to_shower_start = []
    conv_ranked_trackstub_candidate_ioc_based_length = []
    conv_ranked_trackstub_candidate_wire_tick_based_length = []
    conv_ranked_trackstub_candidate_mean_ADC_first_half = []
    conv_ranked_trackstub_candidate_mean_ADC_second_half = []
    conv_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio = []
    conv_ranked_trackstub_candidate_track_angle_wrt_shower_direction = []
    conv_ranked_trackstub_candidate_linear_fit_chi2 = []
    conv_ranked_trackstub_candidate_energy = []
    impact_param_ranked_trackstub_candidate_in_nu_slice = []
    impact_param_ranked_trackstub_candidate_num_hits = []
    impact_param_ranked_trackstub_candidate_num_wires = []
    impact_param_ranked_trackstub_candidate_num_ticks = []
    impact_param_ranked_trackstub_candidate_plane = []
    impact_param_ranked_trackstub_candidate_PCA = []
    impact_param_ranked_trackstub_candidate_mean_ADC = []
    impact_param_ranked_trackstub_candidate_ADC_RMS = []
    impact_param_ranked_trackstub_candidate_min_dist = []
    impact_param_ranked_trackstub_candidate_min_impact_parameter_to_shower = []
    impact_param_ranked_trackstub_candidate_min_conversion_dist_to_shower_start = []
    impact_param_ranked_trackstub_candidate_min_ioc_to_shower_start = []
    impact_param_ranked_trackstub_candidate_ioc_based_length = []
    impact_param_ranked_trackstub_candidate_wire_tick_based_length = []
    impact_param_ranked_trackstub_candidate_mean_ADC_first_half = []
    impact_param_ranked_trackstub_candidate_mean_ADC_second_half = []
    impact_param_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio = []
    impact_param_ranked_trackstub_candidate_track_angle_wrt_shower_direction = []
    impact_param_ranked_trackstub_candidate_linear_fit_chi2 = []
    impact_param_ranked_trackstub_candidate_energy = []
    for event_i in tqdm(range(len(trackstub_candidate_in_nu_slice)), desc="Analyzing gLEE trackstub variables", mininterval=10):
        min_dist = 1e9
        max_energy = 0
        min_conv = 1e9
        min_impact_param = 1e9
        curr_dist_ranked_trackstub_candidate_in_nu_slice = np.nan
        curr_dist_ranked_trackstub_candidate_num_hits = np.nan
        curr_dist_ranked_trackstub_candidate_num_wires = np.nan
        curr_dist_ranked_trackstub_candidate_num_ticks = np.nan
        curr_dist_ranked_trackstub_candidate_plane = np.nan
        curr_dist_ranked_trackstub_candidate_PCA = np.nan
        curr_dist_ranked_trackstub_candidate_mean_ADC = np.nan
        curr_dist_ranked_trackstub_candidate_ADC_RMS = np.nan
        curr_dist_ranked_trackstub_candidate_min_dist = np.nan
        curr_dist_ranked_trackstub_candidate_min_impact_parameter_to_shower = np.nan
        curr_dist_ranked_trackstub_candidate_min_conversion_dist_to_shower_start = np.nan
        curr_dist_ranked_trackstub_candidate_min_ioc_to_shower_start = np.nan
        curr_dist_ranked_trackstub_candidate_ioc_based_length = np.nan
        curr_dist_ranked_trackstub_candidate_wire_tick_based_length = np.nan
        curr_dist_ranked_trackstub_candidate_mean_ADC_first_half = np.nan
        curr_dist_ranked_trackstub_candidate_mean_ADC_second_half = np.nan
        curr_dist_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio = np.nan
        curr_dist_ranked_trackstub_candidate_track_angle_wrt_shower_direction = np.nan
        curr_dist_ranked_trackstub_candidate_linear_fit_chi2 = np.nan
        curr_dist_ranked_trackstub_candidate_energy = np.nan
        curr_energy_ranked_trackstub_candidate_in_nu_slice = np.nan
        curr_energy_ranked_trackstub_candidate_num_hits = np.nan
        curr_energy_ranked_trackstub_candidate_num_wires = np.nan
        curr_energy_ranked_trackstub_candidate_num_ticks = np.nan
        curr_energy_ranked_trackstub_candidate_plane = np.nan
        curr_energy_ranked_trackstub_candidate_PCA = np.nan
        curr_energy_ranked_trackstub_candidate_mean_ADC = np.nan
        curr_energy_ranked_trackstub_candidate_ADC_RMS = np.nan
        curr_energy_ranked_trackstub_candidate_min_dist = np.nan
        curr_energy_ranked_trackstub_candidate_min_impact_parameter_to_shower = np.nan
        curr_energy_ranked_trackstub_candidate_min_conversion_dist_to_shower_start = np.nan
        curr_energy_ranked_trackstub_candidate_min_ioc_to_shower_start = np.nan
        curr_energy_ranked_trackstub_candidate_ioc_based_length = np.nan
        curr_energy_ranked_trackstub_candidate_wire_tick_based_length = np.nan
        curr_energy_ranked_trackstub_candidate_mean_ADC_first_half = np.nan
        curr_energy_ranked_trackstub_candidate_mean_ADC_second_half = np.nan
        curr_energy_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio = np.nan
        curr_energy_ranked_trackstub_candidate_track_angle_wrt_shower_direction = np.nan
        curr_energy_ranked_trackstub_candidate_linear_fit_chi2 = np.nan
        curr_energy_ranked_trackstub_candidate_energy = np.nan
        curr_conv_ranked_trackstub_candidate_in_nu_slice = np.nan
        curr_conv_ranked_trackstub_candidate_num_hits = np.nan
        curr_conv_ranked_trackstub_candidate_num_wires = np.nan
        curr_conv_ranked_trackstub_candidate_num_ticks = np.nan
        curr_conv_ranked_trackstub_candidate_plane = np.nan
        curr_conv_ranked_trackstub_candidate_PCA = np.nan
        curr_conv_ranked_trackstub_candidate_mean_ADC = np.nan
        curr_conv_ranked_trackstub_candidate_ADC_RMS = np.nan
        curr_conv_ranked_trackstub_candidate_min_dist = np.nan
        curr_conv_ranked_trackstub_candidate_min_impact_parameter_to_shower = np.nan
        curr_conv_ranked_trackstub_candidate_min_conversion_dist_to_shower_start = np.nan
        curr_conv_ranked_trackstub_candidate_min_ioc_to_shower_start = np.nan
        curr_conv_ranked_trackstub_candidate_ioc_based_length = np.nan
        curr_conv_ranked_trackstub_candidate_wire_tick_based_length = np.nan
        curr_conv_ranked_trackstub_candidate_mean_ADC_first_half = np.nan
        curr_conv_ranked_trackstub_candidate_mean_ADC_second_half = np.nan
        curr_conv_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio = np.nan
        curr_conv_ranked_trackstub_candidate_track_angle_wrt_shower_direction = np.nan
        curr_conv_ranked_trackstub_candidate_linear_fit_chi2 = np.nan
        curr_conv_ranked_trackstub_candidate_energy = np.nan
        curr_impact_param_ranked_trackstub_candidate_in_nu_slice = np.nan
        curr_impact_param_ranked_trackstub_candidate_num_hits = np.nan
        curr_impact_param_ranked_trackstub_candidate_num_wires = np.nan
        curr_impact_param_ranked_trackstub_candidate_num_ticks = np.nan
        curr_impact_param_ranked_trackstub_candidate_plane = np.nan
        curr_impact_param_ranked_trackstub_candidate_PCA = np.nan
        curr_impact_param_ranked_trackstub_candidate_mean_ADC = np.nan
        curr_impact_param_ranked_trackstub_candidate_ADC_RMS = np.nan
        curr_impact_param_ranked_trackstub_candidate_min_dist = np.nan
        curr_impact_param_ranked_trackstub_candidate_min_impact_parameter_to_shower = np.nan
        curr_impact_param_ranked_trackstub_candidate_min_conversion_dist_to_shower_start = np.nan
        curr_impact_param_ranked_trackstub_candidate_min_ioc_to_shower_start = np.nan
        curr_impact_param_ranked_trackstub_candidate_ioc_based_length = np.nan
        curr_impact_param_ranked_trackstub_candidate_wire_tick_based_length = np.nan
        curr_impact_param_ranked_trackstub_candidate_mean_ADC_first_half = np.nan
        curr_impact_param_ranked_trackstub_candidate_mean_ADC_second_half = np.nan
        curr_impact_param_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio = np.nan
        curr_impact_param_ranked_trackstub_candidate_track_angle_wrt_shower_direction = np.nan
        curr_impact_param_ranked_trackstub_candidate_linear_fit_chi2 = np.nan
        curr_impact_param_ranked_trackstub_candidate_energy = np.nan
        # ensuring we have a non-empty list of unassoc hits that we can loop over
        temp_list = trackstub_candidate_in_nu_slice[event_i]
        if not (isinstance(temp_list, (float, np.floating)) or not hasattr(temp_list, '__len__') or len(temp_list) == 0): 
            # looping over all unassoc hits
            for unassoc_hit_j in range(len(temp_list)):
                # saving values for the closest candidate
                if trackstub_candidate_min_dist[event_i][unassoc_hit_j] < min_dist:
                    min_dist = trackstub_candidate_min_dist[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_in_nu_slice = trackstub_candidate_in_nu_slice[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_num_hits = trackstub_candidate_num_hits[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_num_wires = trackstub_candidate_num_wires[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_num_ticks = trackstub_candidate_num_ticks[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_plane = trackstub_candidate_plane[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_PCA = trackstub_candidate_PCA[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_mean_ADC = trackstub_candidate_mean_ADC[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_ADC_RMS = trackstub_candidate_ADC_RMS[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_min_dist = trackstub_candidate_min_dist[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_min_impact_parameter_to_shower = trackstub_candidate_min_impact_parameter_to_shower[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_min_conversion_dist_to_shower_start = trackstub_candidate_min_conversion_dist_to_shower_start[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_min_ioc_to_shower_start = trackstub_candidate_min_ioc_to_shower_start[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_ioc_based_length = trackstub_candidate_ioc_based_length[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_wire_tick_based_length = trackstub_candidate_wire_tick_based_length[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_mean_ADC_first_half = trackstub_candidate_mean_ADC_first_half[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_mean_ADC_second_half = trackstub_candidate_mean_ADC_second_half[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio = trackstub_candidate_mean_ADC_first_to_second_ratio[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_track_angle_wrt_shower_direction = trackstub_candidate_track_angle_wrt_shower_direction[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_linear_fit_chi2 = trackstub_candidate_linear_fit_chi2[event_i][unassoc_hit_j]
                    curr_dist_ranked_trackstub_candidate_energy = trackstub_candidate_energy[event_i][unassoc_hit_j]
                # saving values for the highest energy candidate
                if trackstub_candidate_energy[event_i][unassoc_hit_j] > max_energy:
                    max_energy = trackstub_candidate_energy[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_in_nu_slice = trackstub_candidate_in_nu_slice[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_num_hits = trackstub_candidate_num_hits[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_num_wires = trackstub_candidate_num_wires[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_num_ticks = trackstub_candidate_num_ticks[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_plane = trackstub_candidate_plane[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_PCA = trackstub_candidate_PCA[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_mean_ADC = trackstub_candidate_mean_ADC[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_ADC_RMS = trackstub_candidate_ADC_RMS[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_min_dist = trackstub_candidate_min_dist[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_min_impact_parameter_to_shower = trackstub_candidate_min_impact_parameter_to_shower[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_min_conversion_dist_to_shower_start = trackstub_candidate_min_conversion_dist_to_shower_start[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_min_ioc_to_shower_start = trackstub_candidate_min_ioc_to_shower_start[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_ioc_based_length = trackstub_candidate_ioc_based_length[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_wire_tick_based_length = trackstub_candidate_wire_tick_based_length[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_mean_ADC_first_half = trackstub_candidate_mean_ADC_first_half[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_mean_ADC_second_half = trackstub_candidate_mean_ADC_second_half[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio = trackstub_candidate_mean_ADC_first_to_second_ratio[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_track_angle_wrt_shower_direction = trackstub_candidate_track_angle_wrt_shower_direction[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_linear_fit_chi2 = trackstub_candidate_linear_fit_chi2[event_i][unassoc_hit_j]
                    curr_energy_ranked_trackstub_candidate_energy = trackstub_candidate_energy[event_i][unassoc_hit_j]
                # saving values for the min conversion distance candidate
                if trackstub_candidate_min_conversion_dist_to_shower_start[event_i][unassoc_hit_j] < min_conv:
                    min_conv = trackstub_candidate_min_conversion_dist_to_shower_start[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_in_nu_slice = trackstub_candidate_in_nu_slice[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_num_hits = trackstub_candidate_num_hits[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_num_wires = trackstub_candidate_num_wires[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_num_ticks = trackstub_candidate_num_ticks[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_plane = trackstub_candidate_plane[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_PCA = trackstub_candidate_PCA[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_mean_ADC = trackstub_candidate_mean_ADC[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_ADC_RMS = trackstub_candidate_ADC_RMS[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_min_dist = trackstub_candidate_min_dist[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_min_impact_parameter_to_shower = trackstub_candidate_min_impact_parameter_to_shower[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_min_conversion_dist_to_shower_start = trackstub_candidate_min_conversion_dist_to_shower_start[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_min_ioc_to_shower_start = trackstub_candidate_min_ioc_to_shower_start[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_ioc_based_length = trackstub_candidate_ioc_based_length[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_wire_tick_based_length = trackstub_candidate_wire_tick_based_length[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_mean_ADC_first_half = trackstub_candidate_mean_ADC_first_half[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_mean_ADC_second_half = trackstub_candidate_mean_ADC_second_half[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio = trackstub_candidate_mean_ADC_first_to_second_ratio[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_track_angle_wrt_shower_direction = trackstub_candidate_track_angle_wrt_shower_direction[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_linear_fit_chi2 = trackstub_candidate_linear_fit_chi2[event_i][unassoc_hit_j]
                    curr_conv_ranked_trackstub_candidate_energy = trackstub_candidate_energy[event_i][unassoc_hit_j]
                # saving values for the min impact parameter candidate
                if trackstub_candidate_min_impact_parameter_to_shower[event_i][unassoc_hit_j] < min_impact_param:
                    min_impact_param = trackstub_candidate_min_impact_parameter_to_shower[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_in_nu_slice = trackstub_candidate_in_nu_slice[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_num_hits = trackstub_candidate_num_hits[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_num_wires = trackstub_candidate_num_wires[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_num_ticks = trackstub_candidate_num_ticks[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_plane = trackstub_candidate_plane[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_PCA = trackstub_candidate_PCA[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_mean_ADC = trackstub_candidate_mean_ADC[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_ADC_RMS = trackstub_candidate_ADC_RMS[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_min_dist = trackstub_candidate_min_dist[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_min_impact_parameter_to_shower = trackstub_candidate_min_impact_parameter_to_shower[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_min_conversion_dist_to_shower_start = trackstub_candidate_min_conversion_dist_to_shower_start[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_min_ioc_to_shower_start = trackstub_candidate_min_ioc_to_shower_start[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_ioc_based_length = trackstub_candidate_ioc_based_length[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_wire_tick_based_length = trackstub_candidate_wire_tick_based_length[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_mean_ADC_first_half = trackstub_candidate_mean_ADC_first_half[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_mean_ADC_second_half = trackstub_candidate_mean_ADC_second_half[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio = trackstub_candidate_mean_ADC_first_to_second_ratio[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_track_angle_wrt_shower_direction = trackstub_candidate_track_angle_wrt_shower_direction[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_linear_fit_chi2 = trackstub_candidate_linear_fit_chi2[event_i][unassoc_hit_j]
                    curr_impact_param_ranked_trackstub_candidate_energy = trackstub_candidate_energy[event_i][unassoc_hit_j]

        dist_ranked_trackstub_candidate_in_nu_slice.append(curr_dist_ranked_trackstub_candidate_in_nu_slice)
        dist_ranked_trackstub_candidate_num_hits.append(curr_dist_ranked_trackstub_candidate_num_hits)
        dist_ranked_trackstub_candidate_num_wires.append(curr_dist_ranked_trackstub_candidate_num_wires)
        dist_ranked_trackstub_candidate_num_ticks.append(curr_dist_ranked_trackstub_candidate_num_ticks)
        dist_ranked_trackstub_candidate_plane.append(curr_dist_ranked_trackstub_candidate_plane)
        dist_ranked_trackstub_candidate_PCA.append(curr_dist_ranked_trackstub_candidate_PCA)
        dist_ranked_trackstub_candidate_mean_ADC.append(curr_dist_ranked_trackstub_candidate_mean_ADC)
        dist_ranked_trackstub_candidate_ADC_RMS.append(curr_dist_ranked_trackstub_candidate_ADC_RMS)
        dist_ranked_trackstub_candidate_min_dist.append(curr_dist_ranked_trackstub_candidate_min_dist)
        dist_ranked_trackstub_candidate_min_impact_parameter_to_shower.append(curr_dist_ranked_trackstub_candidate_min_impact_parameter_to_shower)
        dist_ranked_trackstub_candidate_min_conversion_dist_to_shower_start.append(curr_dist_ranked_trackstub_candidate_min_conversion_dist_to_shower_start)
        dist_ranked_trackstub_candidate_min_ioc_to_shower_start.append(curr_dist_ranked_trackstub_candidate_min_ioc_to_shower_start)
        dist_ranked_trackstub_candidate_ioc_based_length.append(curr_dist_ranked_trackstub_candidate_ioc_based_length)
        dist_ranked_trackstub_candidate_wire_tick_based_length.append(curr_dist_ranked_trackstub_candidate_wire_tick_based_length)
        dist_ranked_trackstub_candidate_mean_ADC_first_half.append(curr_dist_ranked_trackstub_candidate_mean_ADC_first_half)
        dist_ranked_trackstub_candidate_mean_ADC_second_half.append(curr_dist_ranked_trackstub_candidate_mean_ADC_second_half)
        dist_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio.append(curr_dist_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio)
        dist_ranked_trackstub_candidate_track_angle_wrt_shower_direction.append(curr_dist_ranked_trackstub_candidate_track_angle_wrt_shower_direction)
        dist_ranked_trackstub_candidate_linear_fit_chi2.append(curr_dist_ranked_trackstub_candidate_linear_fit_chi2)
        dist_ranked_trackstub_candidate_energy.append(curr_dist_ranked_trackstub_candidate_energy)
        energy_ranked_trackstub_candidate_in_nu_slice.append(curr_energy_ranked_trackstub_candidate_in_nu_slice)
        energy_ranked_trackstub_candidate_num_hits.append(curr_energy_ranked_trackstub_candidate_num_hits)
        energy_ranked_trackstub_candidate_num_wires.append(curr_energy_ranked_trackstub_candidate_num_wires)
        energy_ranked_trackstub_candidate_num_ticks.append(curr_energy_ranked_trackstub_candidate_num_ticks)
        energy_ranked_trackstub_candidate_plane.append(curr_energy_ranked_trackstub_candidate_plane)
        energy_ranked_trackstub_candidate_PCA.append(curr_energy_ranked_trackstub_candidate_PCA)
        energy_ranked_trackstub_candidate_mean_ADC.append(curr_energy_ranked_trackstub_candidate_mean_ADC)
        energy_ranked_trackstub_candidate_ADC_RMS.append(curr_energy_ranked_trackstub_candidate_ADC_RMS)
        energy_ranked_trackstub_candidate_min_dist.append(curr_energy_ranked_trackstub_candidate_min_dist)
        energy_ranked_trackstub_candidate_min_impact_parameter_to_shower.append(curr_energy_ranked_trackstub_candidate_min_impact_parameter_to_shower)
        energy_ranked_trackstub_candidate_min_conversion_dist_to_shower_start.append(curr_energy_ranked_trackstub_candidate_min_conversion_dist_to_shower_start)
        energy_ranked_trackstub_candidate_min_ioc_to_shower_start.append(curr_energy_ranked_trackstub_candidate_min_ioc_to_shower_start)
        energy_ranked_trackstub_candidate_ioc_based_length.append(curr_energy_ranked_trackstub_candidate_ioc_based_length)
        energy_ranked_trackstub_candidate_wire_tick_based_length.append(curr_energy_ranked_trackstub_candidate_wire_tick_based_length)
        energy_ranked_trackstub_candidate_mean_ADC_first_half.append(curr_energy_ranked_trackstub_candidate_mean_ADC_first_half)
        energy_ranked_trackstub_candidate_mean_ADC_second_half.append(curr_energy_ranked_trackstub_candidate_mean_ADC_second_half)
        energy_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio.append(curr_energy_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio)
        energy_ranked_trackstub_candidate_track_angle_wrt_shower_direction.append(curr_energy_ranked_trackstub_candidate_track_angle_wrt_shower_direction)
        energy_ranked_trackstub_candidate_linear_fit_chi2.append(curr_energy_ranked_trackstub_candidate_linear_fit_chi2)
        energy_ranked_trackstub_candidate_energy.append(curr_energy_ranked_trackstub_candidate_energy)
        conv_ranked_trackstub_candidate_in_nu_slice.append(curr_conv_ranked_trackstub_candidate_in_nu_slice)
        conv_ranked_trackstub_candidate_num_hits.append(curr_conv_ranked_trackstub_candidate_num_hits)
        conv_ranked_trackstub_candidate_num_wires.append(curr_conv_ranked_trackstub_candidate_num_wires)
        conv_ranked_trackstub_candidate_num_ticks.append(curr_conv_ranked_trackstub_candidate_num_ticks)
        conv_ranked_trackstub_candidate_plane.append(curr_conv_ranked_trackstub_candidate_plane)
        conv_ranked_trackstub_candidate_PCA.append(curr_conv_ranked_trackstub_candidate_PCA)
        conv_ranked_trackstub_candidate_mean_ADC.append(curr_conv_ranked_trackstub_candidate_mean_ADC)
        conv_ranked_trackstub_candidate_ADC_RMS.append(curr_conv_ranked_trackstub_candidate_ADC_RMS)
        conv_ranked_trackstub_candidate_min_dist.append(curr_conv_ranked_trackstub_candidate_min_dist)
        conv_ranked_trackstub_candidate_min_impact_parameter_to_shower.append(curr_conv_ranked_trackstub_candidate_min_impact_parameter_to_shower)
        conv_ranked_trackstub_candidate_min_conversion_dist_to_shower_start.append(curr_conv_ranked_trackstub_candidate_min_conversion_dist_to_shower_start)
        conv_ranked_trackstub_candidate_min_ioc_to_shower_start.append(curr_conv_ranked_trackstub_candidate_min_ioc_to_shower_start)
        conv_ranked_trackstub_candidate_ioc_based_length.append(curr_conv_ranked_trackstub_candidate_ioc_based_length)
        conv_ranked_trackstub_candidate_wire_tick_based_length.append(curr_conv_ranked_trackstub_candidate_wire_tick_based_length)
        conv_ranked_trackstub_candidate_mean_ADC_first_half.append(curr_conv_ranked_trackstub_candidate_mean_ADC_first_half)
        conv_ranked_trackstub_candidate_mean_ADC_second_half.append(curr_conv_ranked_trackstub_candidate_mean_ADC_second_half)
        conv_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio.append(curr_conv_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio)
        conv_ranked_trackstub_candidate_track_angle_wrt_shower_direction.append(curr_conv_ranked_trackstub_candidate_track_angle_wrt_shower_direction)
        conv_ranked_trackstub_candidate_linear_fit_chi2.append(curr_conv_ranked_trackstub_candidate_linear_fit_chi2)
        conv_ranked_trackstub_candidate_energy.append(curr_conv_ranked_trackstub_candidate_energy)
        impact_param_ranked_trackstub_candidate_in_nu_slice.append(curr_impact_param_ranked_trackstub_candidate_in_nu_slice)
        impact_param_ranked_trackstub_candidate_num_hits.append(curr_impact_param_ranked_trackstub_candidate_num_hits)
        impact_param_ranked_trackstub_candidate_num_wires.append(curr_impact_param_ranked_trackstub_candidate_num_wires)
        impact_param_ranked_trackstub_candidate_num_ticks.append(curr_impact_param_ranked_trackstub_candidate_num_ticks)
        impact_param_ranked_trackstub_candidate_plane.append(curr_impact_param_ranked_trackstub_candidate_plane)
        impact_param_ranked_trackstub_candidate_PCA.append(curr_impact_param_ranked_trackstub_candidate_PCA)
        impact_param_ranked_trackstub_candidate_mean_ADC.append(curr_impact_param_ranked_trackstub_candidate_mean_ADC)
        impact_param_ranked_trackstub_candidate_ADC_RMS.append(curr_impact_param_ranked_trackstub_candidate_ADC_RMS)
        impact_param_ranked_trackstub_candidate_min_dist.append(curr_impact_param_ranked_trackstub_candidate_min_dist)
        impact_param_ranked_trackstub_candidate_min_impact_parameter_to_shower.append(curr_impact_param_ranked_trackstub_candidate_min_impact_parameter_to_shower)
        impact_param_ranked_trackstub_candidate_min_conversion_dist_to_shower_start.append(curr_impact_param_ranked_trackstub_candidate_min_conversion_dist_to_shower_start)
        impact_param_ranked_trackstub_candidate_min_ioc_to_shower_start.append(curr_impact_param_ranked_trackstub_candidate_min_ioc_to_shower_start)
        impact_param_ranked_trackstub_candidate_ioc_based_length.append(curr_impact_param_ranked_trackstub_candidate_ioc_based_length)
        impact_param_ranked_trackstub_candidate_wire_tick_based_length.append(curr_impact_param_ranked_trackstub_candidate_wire_tick_based_length)
        impact_param_ranked_trackstub_candidate_mean_ADC_first_half.append(curr_impact_param_ranked_trackstub_candidate_mean_ADC_first_half)
        impact_param_ranked_trackstub_candidate_mean_ADC_second_half.append(curr_impact_param_ranked_trackstub_candidate_mean_ADC_second_half)
        impact_param_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio.append(curr_impact_param_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio)
        impact_param_ranked_trackstub_candidate_track_angle_wrt_shower_direction.append(curr_impact_param_ranked_trackstub_candidate_track_angle_wrt_shower_direction)
        impact_param_ranked_trackstub_candidate_linear_fit_chi2.append(curr_impact_param_ranked_trackstub_candidate_linear_fit_chi2)
        impact_param_ranked_trackstub_candidate_energy.append(curr_impact_param_ranked_trackstub_candidate_energy)

    glee_trackstub_candidate_df = pd.DataFrame({
        "glee_dist_ranked_trackstub_candidate_in_nu_slice": dist_ranked_trackstub_candidate_in_nu_slice,
        "glee_dist_ranked_trackstub_candidate_num_hits": dist_ranked_trackstub_candidate_num_hits,
        "glee_dist_ranked_trackstub_candidate_num_wires": dist_ranked_trackstub_candidate_num_wires,
        "glee_dist_ranked_trackstub_candidate_num_ticks": dist_ranked_trackstub_candidate_num_ticks,
        "glee_dist_ranked_trackstub_candidate_plane": dist_ranked_trackstub_candidate_plane,
        "glee_dist_ranked_trackstub_candidate_PCA": dist_ranked_trackstub_candidate_PCA,
        "glee_dist_ranked_trackstub_candidate_mean_ADC": dist_ranked_trackstub_candidate_mean_ADC,
        "glee_dist_ranked_trackstub_candidate_ADC_RMS": dist_ranked_trackstub_candidate_ADC_RMS,
        "glee_dist_ranked_trackstub_candidate_min_dist": dist_ranked_trackstub_candidate_min_dist,
        "glee_dist_ranked_trackstub_candidate_min_impact_parameter_to_shower": dist_ranked_trackstub_candidate_min_impact_parameter_to_shower,
        "glee_dist_ranked_trackstub_candidate_min_conversion_dist_to_shower_start": dist_ranked_trackstub_candidate_min_conversion_dist_to_shower_start,
        "glee_dist_ranked_trackstub_candidate_min_ioc_to_shower_start": dist_ranked_trackstub_candidate_min_ioc_to_shower_start,
        "glee_dist_ranked_trackstub_candidate_ioc_based_length": dist_ranked_trackstub_candidate_ioc_based_length,
        "glee_dist_ranked_trackstub_candidate_wire_tick_based_length": dist_ranked_trackstub_candidate_wire_tick_based_length,
        "glee_dist_ranked_trackstub_candidate_mean_ADC_first_half": dist_ranked_trackstub_candidate_mean_ADC_first_half,
        "glee_dist_ranked_trackstub_candidate_mean_ADC_second_half": dist_ranked_trackstub_candidate_mean_ADC_second_half,
        "glee_dist_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio": dist_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio,
        "glee_dist_ranked_trackstub_candidate_track_angle_wrt_shower_direction": dist_ranked_trackstub_candidate_track_angle_wrt_shower_direction,
        "glee_dist_ranked_trackstub_candidate_linear_fit_chi2": dist_ranked_trackstub_candidate_linear_fit_chi2,
        "glee_dist_ranked_trackstub_candidate_energy": dist_ranked_trackstub_candidate_energy,
        "glee_energy_ranked_trackstub_candidate_in_nu_slice": energy_ranked_trackstub_candidate_in_nu_slice,
        "glee_energy_ranked_trackstub_candidate_num_hits": energy_ranked_trackstub_candidate_num_hits,
        "glee_energy_ranked_trackstub_candidate_num_wires": energy_ranked_trackstub_candidate_num_wires,
        "glee_energy_ranked_trackstub_candidate_num_ticks": energy_ranked_trackstub_candidate_num_ticks,
        "glee_energy_ranked_trackstub_candidate_plane": energy_ranked_trackstub_candidate_plane,
        "glee_energy_ranked_trackstub_candidate_PCA": energy_ranked_trackstub_candidate_PCA,
        "glee_energy_ranked_trackstub_candidate_mean_ADC": energy_ranked_trackstub_candidate_mean_ADC,
        "glee_energy_ranked_trackstub_candidate_ADC_RMS": energy_ranked_trackstub_candidate_ADC_RMS,
        "glee_energy_ranked_trackstub_candidate_min_dist": energy_ranked_trackstub_candidate_min_dist,
        "glee_energy_ranked_trackstub_candidate_min_impact_parameter_to_shower": energy_ranked_trackstub_candidate_min_impact_parameter_to_shower,
        "glee_energy_ranked_trackstub_candidate_min_conversion_dist_to_shower_start": energy_ranked_trackstub_candidate_min_conversion_dist_to_shower_start,
        "glee_energy_ranked_trackstub_candidate_min_ioc_to_shower_start": energy_ranked_trackstub_candidate_min_ioc_to_shower_start,
        "glee_energy_ranked_trackstub_candidate_ioc_based_length": energy_ranked_trackstub_candidate_ioc_based_length,
        "glee_energy_ranked_trackstub_candidate_wire_tick_based_length": energy_ranked_trackstub_candidate_wire_tick_based_length,
        "glee_energy_ranked_trackstub_candidate_mean_ADC_first_half": energy_ranked_trackstub_candidate_mean_ADC_first_half,
        "glee_energy_ranked_trackstub_candidate_mean_ADC_second_half": energy_ranked_trackstub_candidate_mean_ADC_second_half,
        "glee_energy_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio": energy_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio,
        "glee_energy_ranked_trackstub_candidate_track_angle_wrt_shower_direction": energy_ranked_trackstub_candidate_track_angle_wrt_shower_direction,
        "glee_energy_ranked_trackstub_candidate_linear_fit_chi2": energy_ranked_trackstub_candidate_linear_fit_chi2,
        "glee_energy_ranked_trackstub_candidate_energy": energy_ranked_trackstub_candidate_energy,
        "glee_conv_ranked_trackstub_candidate_in_nu_slice": conv_ranked_trackstub_candidate_in_nu_slice,
        "glee_conv_ranked_trackstub_candidate_num_hits": conv_ranked_trackstub_candidate_num_hits,
        "glee_conv_ranked_trackstub_candidate_num_wires": conv_ranked_trackstub_candidate_num_wires,
        "glee_conv_ranked_trackstub_candidate_num_ticks": conv_ranked_trackstub_candidate_num_ticks,
        "glee_conv_ranked_trackstub_candidate_plane": conv_ranked_trackstub_candidate_plane,
        "glee_conv_ranked_trackstub_candidate_PCA": conv_ranked_trackstub_candidate_PCA,
        "glee_conv_ranked_trackstub_candidate_mean_ADC": conv_ranked_trackstub_candidate_mean_ADC,
        "glee_conv_ranked_trackstub_candidate_ADC_RMS": conv_ranked_trackstub_candidate_ADC_RMS,
        "glee_conv_ranked_trackstub_candidate_min_dist": conv_ranked_trackstub_candidate_min_dist,
        "glee_conv_ranked_trackstub_candidate_min_impact_parameter_to_shower": conv_ranked_trackstub_candidate_min_impact_parameter_to_shower,
        "glee_conv_ranked_trackstub_candidate_min_conversion_dist_to_shower_start": conv_ranked_trackstub_candidate_min_conversion_dist_to_shower_start,
        "glee_conv_ranked_trackstub_candidate_min_ioc_to_shower_start": conv_ranked_trackstub_candidate_min_ioc_to_shower_start,
        "glee_conv_ranked_trackstub_candidate_ioc_based_length": conv_ranked_trackstub_candidate_ioc_based_length,
        "glee_conv_ranked_trackstub_candidate_wire_tick_based_length": conv_ranked_trackstub_candidate_wire_tick_based_length,
        "glee_conv_ranked_trackstub_candidate_mean_ADC_first_half": conv_ranked_trackstub_candidate_mean_ADC_first_half,
        "glee_conv_ranked_trackstub_candidate_mean_ADC_second_half": conv_ranked_trackstub_candidate_mean_ADC_second_half,
        "glee_conv_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio": conv_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio,
        "glee_conv_ranked_trackstub_candidate_track_angle_wrt_shower_direction": conv_ranked_trackstub_candidate_track_angle_wrt_shower_direction,
        "glee_conv_ranked_trackstub_candidate_linear_fit_chi2": conv_ranked_trackstub_candidate_linear_fit_chi2,
        "glee_conv_ranked_trackstub_candidate_energy": conv_ranked_trackstub_candidate_energy,
        "glee_impact_param_ranked_trackstub_candidate_in_nu_slice": impact_param_ranked_trackstub_candidate_in_nu_slice,
        "glee_impact_param_ranked_trackstub_candidate_num_hits": impact_param_ranked_trackstub_candidate_num_hits,
        "glee_impact_param_ranked_trackstub_candidate_num_wires": impact_param_ranked_trackstub_candidate_num_wires,
        "glee_impact_param_ranked_trackstub_candidate_num_ticks": impact_param_ranked_trackstub_candidate_num_ticks,
        "glee_impact_param_ranked_trackstub_candidate_plane": impact_param_ranked_trackstub_candidate_plane,
        "glee_impact_param_ranked_trackstub_candidate_PCA": impact_param_ranked_trackstub_candidate_PCA,
        "glee_impact_param_ranked_trackstub_candidate_mean_ADC": impact_param_ranked_trackstub_candidate_mean_ADC,
        "glee_impact_param_ranked_trackstub_candidate_ADC_RMS": impact_param_ranked_trackstub_candidate_ADC_RMS,
        "glee_impact_param_ranked_trackstub_candidate_min_dist": impact_param_ranked_trackstub_candidate_min_dist,
        "glee_impact_param_ranked_trackstub_candidate_min_impact_parameter_to_shower": impact_param_ranked_trackstub_candidate_min_impact_parameter_to_shower,
        "glee_impact_param_ranked_trackstub_candidate_min_conversion_dist_to_shower_start": impact_param_ranked_trackstub_candidate_min_conversion_dist_to_shower_start,
        "glee_impact_param_ranked_trackstub_candidate_min_ioc_to_shower_start": impact_param_ranked_trackstub_candidate_min_ioc_to_shower_start,
        "glee_impact_param_ranked_trackstub_candidate_ioc_based_length": impact_param_ranked_trackstub_candidate_ioc_based_length,
        "glee_impact_param_ranked_trackstub_candidate_wire_tick_based_length": impact_param_ranked_trackstub_candidate_wire_tick_based_length,
        "glee_impact_param_ranked_trackstub_candidate_mean_ADC_first_half": impact_param_ranked_trackstub_candidate_mean_ADC_first_half,
        "glee_impact_param_ranked_trackstub_candidate_mean_ADC_second_half": impact_param_ranked_trackstub_candidate_mean_ADC_second_half,
        "glee_impact_param_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio": impact_param_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio,
        "glee_impact_param_ranked_trackstub_candidate_track_angle_wrt_shower_direction": impact_param_ranked_trackstub_candidate_track_angle_wrt_shower_direction,
        "glee_impact_param_ranked_trackstub_candidate_linear_fit_chi2": impact_param_ranked_trackstub_candidate_linear_fit_chi2,
        "glee_impact_param_ranked_trackstub_candidate_energy": impact_param_ranked_trackstub_candidate_energy,
    }, index=df.index)

    df = pd.concat([df, glee_isolation_df, glee_trackstub_candidate_df], axis=1)

    return df


def do_lantern_postprocessing(df):

    # References of variables that the LANTERN team has found useful:
    # LANTERN 1e selection reco variables, from https://github.com/NuTufts/lantern_ana/blob/main/studies/nue_cc_inclusive/plot_mmr_nue_selection_vars_run4b.py
    """
    vars = [('eventweight_weight',200,0,10,'event weight',0), # don't include event weight as a training variable
            ('recoElectron_emax_econfidence', 40, 0, 20,'electron confidence score',0),
            ('recoElectron_emax_primary_score',40,0,1.0,'primary score',0),
            ('recoElectron_emax_fromneutral_score',40,0,1.0,'from neutral parent score',0),
            ('recoElectron_emax_fromcharged_score',40,0,1.0,'from charged parent score',0),
            ('recoElectron_emax_el_normedscore',40,0,1.0,'electron-like score (normalized)',0),
            ('recoMuonTrack_nMuTracks',20,0,20,'num of muon-like tracks',0),
            ('recoMuonTrack_max_muscore',50,-20,0.5,'max mu-like score',0),
            ('vertex_properties_score',60,0.4,1.0,'keypoint score',0),
            ('vertex_properties_cosmicfrac',50,0,1.01,'fraction of cosmic pixel near vertex',0),
            ('visible_energy',15,0,3000,'visible energy; MeV',0)
    """
    # LANTERN 1g selection reco variables, from https://github.com/NuTufts/lantern_ana/blob/main/studies/1gXp_nc/plot_1gX_vars.py
    """
    selection_cut = "vertex_properties_found==1"
    selection_cut += " && nphotons==1"
    selection_cut += " && nProtons==0"
    selection_cut += " && nPions==0"
    selection_cut += " && nElectrons==0"
    selection_cut += " && nMuons==0"
    selection_cut += " && TMath::IsNaN(vertex_properties_score)==0"
    """

    # see function here: https://github.com/NuTufts/lantern_ana/blob/94f31b83e8a170230ca7b948c701671ab3099bbd/lantern_ana/utils/get_primary_electron_candidates.py
    # this code is similar, but adds information about up to two electrons and two photons, and extra pi0-relevant information for the two photon case, and counts all particles

    max_shower_RecoEs = []

    max_electron_shower_RecoEs = []
    max_electron_shower_PhScores = []
    max_electron_shower_ElScores = []
    max_electron_shower_MuScores = []
    max_electron_shower_PiScores = []
    max_electron_shower_PrScores = []
    max_electron_shower_el_normedscores = []
    max_electron_shower_ph_normedscores = []
    max_electron_shower_electron_confidences = []
    max_electron_shower_PrimaryScores = []
    max_electron_shower_FromNeutralScores = []
    max_electron_shower_FromChargedScores = []
    max_electron_shower_CosThetas = []
    max_electron_shower_CosThetaYs = []
    max_electron_shower_DistToVtxs = []

    second_max_electron_shower_RecoEs = []
    second_max_electron_shower_PhScores = []
    second_max_electron_shower_ElScores = []
    second_max_electron_shower_MuScores = []
    second_max_electron_shower_PiScores = []
    second_max_electron_shower_PrScores = []
    second_max_electron_shower_el_normedscores = []
    second_max_electron_shower_ph_normedscores = []
    second_max_electron_shower_electron_confidences = []
    second_max_electron_shower_PrimaryScores = []
    second_max_electron_shower_FromNeutralScores = []
    second_max_electron_shower_FromChargedScores = []
    second_max_electron_shower_CosThetas = []
    second_max_electron_shower_CosThetaYs = []
    second_max_electron_shower_DistToVtxs = []
    
    max_photon_shower_RecoEs = []
    max_photon_shower_PhScores = []
    max_photon_shower_ElScores = []
    max_photon_shower_MuScores = []
    max_photon_shower_PiScores = []
    max_photon_shower_PrScores = []
    max_photon_shower_el_normedscores = []
    max_photon_shower_ph_normedscores = []
    max_photon_shower_electron_confidences = []
    max_photon_shower_PrimaryScores = []
    max_photon_shower_FromNeutralScores = []
    max_photon_shower_FromChargedScores = []
    max_photon_shower_CosThetas = []
    max_photon_shower_CosThetaYs = []
    max_photon_shower_DistToVtxs = []

    second_max_photon_shower_RecoEs = []
    second_max_photon_shower_PhScores = []
    second_max_photon_shower_ElScores = []
    second_max_photon_shower_MuScores = []
    second_max_photon_shower_PiScores = []
    second_max_photon_shower_PrScores = []
    second_max_photon_shower_el_normedscores = []
    second_max_photon_shower_ph_normedscores = []
    second_max_photon_shower_electron_confidences = []
    second_max_photon_shower_PrimaryScores = []
    second_max_photon_shower_FromNeutralScores = []
    second_max_photon_shower_FromChargedScores = []
    second_max_photon_shower_CosThetas = []
    second_max_photon_shower_CosThetaYs = []
    second_max_photon_shower_DistToVtxs = []

    max_nonprimary_shower_RecoEs = []
    max_nonprimary_shower_PhScores = []
    max_nonprimary_shower_ElScores = []
    max_nonprimary_shower_MuScores = []
    max_nonprimary_shower_PiScores = []
    max_nonprimary_shower_PrScores = []
    max_nonprimary_shower_electron_confidences = []
    max_nonprimary_shower_ph_normedscores = []
    max_nonprimary_shower_el_normedscores = []
    max_nonprimary_shower_PrimaryScores = []
    max_nonprimary_shower_FromNeutralScores = []
    max_nonprimary_shower_FromChargedScores = []
    max_nonprimary_shower_CosThetas = []
    max_nonprimary_shower_CosThetaYs = []
    max_nonprimary_shower_DistToVtxs = []
    second_max_nonprimary_shower_RecoEs = []
    second_max_nonprimary_shower_PhScores = []
    second_max_nonprimary_shower_ElScores = []
    second_max_nonprimary_shower_MuScores = []
    second_max_nonprimary_shower_PiScores = []
    second_max_nonprimary_shower_PrScores = []
    second_max_nonprimary_shower_electron_confidences = []
    second_max_nonprimary_shower_ph_normedscores = []
    second_max_nonprimary_shower_el_normedscores = []
    second_max_nonprimary_shower_PrimaryScores = []
    second_max_nonprimary_shower_FromNeutralScores = []
    second_max_nonprimary_shower_FromChargedScores = []
    second_max_nonprimary_shower_CosThetas = []
    second_max_nonprimary_shower_CosThetaYs = []
    second_max_nonprimary_shower_DistToVtxs = []
    
    diphoton_opening_angles = []
    diphoton_energies = []
    diphoton_costhetas = []
    diphoton_masses = []

    prim_shower_photon_nums = []
    prim_shower_electron_nums = []
    prim_shower_muon_nums = []
    prim_shower_charged_pion_nums = []
    prim_shower_proton_nums = []

    nShowers = df["lantern_nShowers"].to_numpy()
    showerIsSecondary = df["lantern_showerIsSecondary"].to_numpy()
    showerPID = df["lantern_showerPID"].to_numpy()
    showerPhScore = df["lantern_showerPhScore"].to_numpy()
    showerElScore = df["lantern_showerElScore"].to_numpy()
    showerMuScore = df["lantern_showerMuScore"].to_numpy()
    showerPiScore = df["lantern_showerPiScore"].to_numpy()
    showerPrScore = df["lantern_showerPrScore"].to_numpy()
    shower_charge = df["lantern_showerCharge"].to_numpy()
    showerRecoE = df["lantern_showerRecoE"].to_numpy()
    showerPurity = df["lantern_showerPurity"].to_numpy()
    showerComp = df["lantern_showerComp"].to_numpy()
    showerPrimaryScore = df["lantern_showerPrimaryScore"].to_numpy()
    showerFromNeutralScore = df["lantern_showerFromNeutralScore"].to_numpy()
    showerFromChargedScore = df["lantern_showerFromChargedScore"].to_numpy()
    showerCosTheta = df["lantern_showerCosTheta"].to_numpy()
    showerCosThetaY = df["lantern_showerCosThetaY"].to_numpy()
    showerDistToVtx = df["lantern_showerDistToVtx"].to_numpy()
    showerStartDirX = df["lantern_showerStartDirX"].to_numpy()
    showerStartDirY = df["lantern_showerStartDirY"].to_numpy()
    showerStartDirZ = df["lantern_showerStartDirZ"].to_numpy()
    for event_i in tqdm(range(len(df)), desc="Analyzing LANTERN showers", mininterval=10):
        curr_nShowers_val = nShowers[event_i]
        if np.isnan(curr_nShowers_val):
            curr_nShowers = 0
        else:
            curr_nShowers = int(curr_nShowers_val)
        curr_showerIsSecondary = showerIsSecondary[event_i]
        curr_showerPID = showerPID[event_i]
        curr_showerPhScore = showerPhScore[event_i]
        curr_showerElScore = showerElScore[event_i]
        curr_showerMuScore = showerMuScore[event_i]
        curr_showerPiScore = showerPiScore[event_i]
        curr_showerPrScore = showerPrScore[event_i]
        curr_showerRecoE = showerRecoE[event_i]
        curr_shower_charge = shower_charge[event_i]
        curr_showerPurity = showerPurity[event_i]
        curr_showerComp = showerComp[event_i]
        curr_showerPrimaryScore = showerPrimaryScore[event_i]
        curr_showerFromNeutralScore = showerFromNeutralScore[event_i]
        curr_showerFromChargedScore = showerFromChargedScore[event_i]
        curr_showerCosTheta = showerCosTheta[event_i]
        curr_showerCosThetaY = showerCosThetaY[event_i]
        curr_showerDistToVtx = showerDistToVtx[event_i]
        curr_showerStartDirX = showerStartDirX[event_i]
        curr_showerStartDirY = showerStartDirY[event_i]
        curr_showerStartDirZ = showerStartDirZ[event_i]

        max_shower_RecoE = 0

        max_photon_shower_RecoE = 0.
        max_photon_shower_PhScore = np.nan
        max_photon_shower_ElScore = np.nan
        max_photon_shower_MuScore = np.nan
        max_photon_shower_PiScore = np.nan
        max_photon_shower_PrScore = np.nan
        max_photon_shower_electron_confidence = np.nan
        max_photon_shower_ph_normedscore = np.nan
        max_photon_shower_el_normedscore = np.nan
        max_photon_shower_PrimaryScore = np.nan
        max_photon_shower_FromNeutralScore = np.nan
        max_photon_shower_FromChargedScore = np.nan
        max_photon_shower_CosTheta = np.nan
        max_photon_shower_CosThetaY = np.nan
        max_photon_shower_DistToVtx = np.nan
        max_photon_shower_StartDirX = np.nan
        max_photon_shower_StartDirY = np.nan
        max_photon_shower_StartDirZ = np.nan
        second_max_photon_shower_RecoE = -1e-6
        second_max_photon_shower_PhScore = np.nan
        second_max_photon_shower_ElScore = np.nan
        second_max_photon_shower_MuScore = np.nan
        second_max_photon_shower_PiScore = np.nan
        second_max_photon_shower_PrScore = np.nan
        second_max_photon_shower_electron_confidence = np.nan
        second_max_photon_shower_ph_normedscore = np.nan
        second_max_photon_shower_el_normedscore = np.nan
        second_max_photon_shower_PrimaryScore = np.nan
        second_max_photon_shower_FromNeutralScore = np.nan
        second_max_photon_shower_FromChargedScore = np.nan
        second_max_photon_shower_CosTheta = np.nan
        second_max_photon_shower_CosThetaY = np.nan
        second_max_photon_shower_DistToVtx = np.nan
        second_max_photon_shower_StartDirX = np.nan
        second_max_photon_shower_StartDirY = np.nan
        second_max_photon_shower_StartDirZ = np.nan

        max_electron_shower_RecoE = 0.
        max_electron_shower_PhScore = np.nan
        max_electron_shower_ElScore = np.nan
        max_electron_shower_MuScore = np.nan
        max_electron_shower_PiScore = np.nan
        max_electron_shower_PrScore = np.nan
        max_electron_shower_electron_confidence = np.nan
        max_electron_shower_ph_normedscore = np.nan
        max_electron_shower_el_normedscore = np.nan
        max_electron_shower_PrimaryScore = np.nan
        max_electron_shower_FromNeutralScore = np.nan
        max_electron_shower_FromChargedScore = np.nan
        max_electron_shower_CosTheta = np.nan
        max_electron_shower_CosThetaY = np.nan
        max_electron_shower_DistToVtx = np.nan
        second_max_electron_shower_RecoE = -1e-6
        second_max_electron_shower_PhScore = np.nan
        second_max_electron_shower_ElScore = np.nan
        second_max_electron_shower_MuScore = np.nan
        second_max_electron_shower_PiScore = np.nan
        second_max_electron_shower_PrScore = np.nan
        second_max_electron_shower_electron_confidence = np.nan
        second_max_electron_shower_ph_normedscore = np.nan
        second_max_electron_shower_el_normedscore = np.nan
        second_max_electron_shower_PrimaryScore = np.nan
        second_max_electron_shower_FromNeutralScore = np.nan
        second_max_electron_shower_FromChargedScore = np.nan
        second_max_electron_shower_CosTheta = np.nan
        second_max_electron_shower_CosThetaY = np.nan
        second_max_electron_shower_DistToVtx = np.nan

        max_nonprimary_shower_RecoE = 0.
        max_nonprimary_shower_PhScore = np.nan
        max_nonprimary_shower_ElScore = np.nan
        max_nonprimary_shower_MuScore = np.nan
        max_nonprimary_shower_PiScore = np.nan
        max_nonprimary_shower_PrScore = np.nan
        max_nonprimary_shower_electron_confidence = np.nan
        max_nonprimary_shower_ph_normedscore = np.nan
        max_nonprimary_shower_el_normedscore = np.nan
        max_nonprimary_shower_PrimaryScore = np.nan
        max_nonprimary_shower_FromNeutralScore = np.nan
        max_nonprimary_shower_FromChargedScore = np.nan
        max_nonprimary_shower_CosTheta = np.nan
        max_nonprimary_shower_CosThetaY = np.nan
        max_nonprimary_shower_DistToVtx = np.nan
        second_max_nonprimary_shower_RecoE = -1e-6
        second_max_nonprimary_shower_PhScore = np.nan
        second_max_nonprimary_shower_ElScore = np.nan
        second_max_nonprimary_shower_MuScore = np.nan
        second_max_nonprimary_shower_PiScore = np.nan
        second_max_nonprimary_shower_PrScore = np.nan
        second_max_nonprimary_shower_electron_confidence = np.nan
        second_max_nonprimary_shower_ph_normedscore = np.nan
        second_max_nonprimary_shower_el_normedscore = np.nan
        second_max_nonprimary_shower_PrimaryScore = np.nan
        second_max_nonprimary_shower_FromNeutralScore = np.nan
        second_max_nonprimary_shower_FromChargedScore = np.nan
        second_max_nonprimary_shower_CosTheta = np.nan
        second_max_nonprimary_shower_CosThetaY = np.nan
        second_max_nonprimary_shower_DistToVtx = np.nan

        diphoton_opening_angle = np.nan
        diphoton_energy = np.nan
        diphoton_costheta = np.nan
        diphoton_mass = np.nan

        curr_num_photons = 0
        curr_num_electrons = 0
        curr_num_muons = 0
        curr_num_charged_pions = 0
        curr_num_protons = 0

        for shower_i in range(curr_nShowers):

            min_charge, min_completeness, min_purity = 0, 0, 0
            if curr_shower_charge[shower_i] < min_charge or curr_showerComp[shower_i] < min_completeness or curr_showerPurity[shower_i] < min_purity:
                continue

            is_electron = curr_showerElScore[shower_i] > curr_showerPhScore[shower_i]
            electron_confidence = curr_showerElScore[shower_i] - (curr_showerPhScore[shower_i] + curr_showerPiScore[shower_i]) / 2
            el_normedscore = np.exp(curr_showerElScore[shower_i]) / (np.exp(curr_showerPhScore[shower_i])
                                                                   + np.exp(curr_showerElScore[shower_i])
                                                                   + np.exp(curr_showerMuScore[shower_i])
                                                                   + np.exp(curr_showerPiScore[shower_i])
                                                                   + np.exp(curr_showerPrScore[shower_i]))
            ph_normedscore = np.exp(curr_showerPhScore[shower_i]) / (np.exp(curr_showerElScore[shower_i])
                                                                   + np.exp(curr_showerPhScore[shower_i])
                                                                   + np.exp(curr_showerMuScore[shower_i])
                                                                   + np.exp(curr_showerPiScore[shower_i])
                                                                   + np.exp(curr_showerPrScore[shower_i]))

            if curr_showerIsSecondary[shower_i] != 0: # predicted secondary shower, so we're not using for most variables, but are filling some
                if curr_showerRecoE[shower_i] > max_nonprimary_shower_RecoE: # new largest nonprimary shower
                    second_max_nonprimary_shower_RecoE = max_nonprimary_shower_RecoE
                    second_max_nonprimary_shower_PhScore = max_nonprimary_shower_PhScore
                    second_max_nonprimary_shower_ElScore = max_nonprimary_shower_ElScore
                    second_max_nonprimary_shower_MuScore = max_nonprimary_shower_MuScore
                    second_max_nonprimary_shower_PiScore = max_nonprimary_shower_PiScore
                    second_max_nonprimary_shower_PrScore = max_nonprimary_shower_PrScore
                    second_max_nonprimary_shower_electron_confidence = max_nonprimary_shower_electron_confidence
                    second_max_nonprimary_shower_ph_normedscore = max_nonprimary_shower_ph_normedscore
                    second_max_nonprimary_shower_el_normedscore = max_nonprimary_shower_el_normedscore
                    second_max_nonprimary_shower_PrimaryScore = max_nonprimary_shower_PrimaryScore
                    second_max_nonprimary_shower_FromNeutralScore = max_nonprimary_shower_FromNeutralScore
                    second_max_nonprimary_shower_FromChargedScore = max_nonprimary_shower_FromChargedScore
                    second_max_nonprimary_shower_CosTheta = max_nonprimary_shower_CosTheta
                    second_max_nonprimary_shower_CosThetaY = max_nonprimary_shower_CosThetaY
                    second_max_nonprimary_shower_DistToVtx = max_nonprimary_shower_DistToVtx
                    max_nonprimary_shower_RecoE = curr_showerRecoE[shower_i]
                    max_nonprimary_shower_PhScore = curr_showerPhScore[shower_i]
                    max_nonprimary_shower_ElScore = curr_showerElScore[shower_i]
                    max_nonprimary_shower_MuScore = curr_showerMuScore[shower_i]
                    max_nonprimary_shower_PiScore = curr_showerPiScore[shower_i]
                    max_nonprimary_shower_PrScore = curr_showerPrScore[shower_i]
                    max_nonprimary_shower_electron_confidence = electron_confidence
                    max_nonprimary_shower_ph_normedscore = ph_normedscore
                    max_nonprimary_shower_el_normedscore = el_normedscore
                    max_nonprimary_shower_PrimaryScore = curr_showerPrimaryScore[shower_i]
                    max_nonprimary_shower_FromNeutralScore = curr_showerFromNeutralScore[shower_i]
                    max_nonprimary_shower_FromChargedScore = curr_showerFromChargedScore[shower_i]
                    max_nonprimary_shower_CosTheta = curr_showerCosTheta[shower_i]
                    max_nonprimary_shower_CosThetaY = curr_showerCosThetaY[shower_i]
                    max_nonprimary_shower_DistToVtx = curr_showerDistToVtx[shower_i]
                elif curr_showerRecoE[shower_i] > second_max_nonprimary_shower_RecoE: # new second largest nonprimary shower
                    second_max_nonprimary_shower_RecoE = curr_showerRecoE[shower_i]
                    second_max_nonprimary_shower_PhScore = curr_showerPhScore[shower_i]
                    second_max_nonprimary_shower_ElScore = curr_showerElScore[shower_i]
                    second_max_nonprimary_shower_MuScore = curr_showerMuScore[shower_i]
                    second_max_nonprimary_shower_PiScore = curr_showerPiScore[shower_i]
                    second_max_nonprimary_shower_PrScore = curr_showerPrScore[shower_i]
                    second_max_nonprimary_shower_electron_confidence = electron_confidence
                    second_max_nonprimary_shower_ph_normedscore = ph_normedscore
                    second_max_nonprimary_shower_el_normedscore = el_normedscore
                    second_max_nonprimary_shower_PrimaryScore = curr_showerPrimaryScore[shower_i]
                    second_max_nonprimary_shower_FromNeutralScore = curr_showerFromNeutralScore[shower_i]
                    second_max_nonprimary_shower_FromChargedScore = curr_showerFromChargedScore[shower_i]
                    second_max_nonprimary_shower_CosTheta = curr_showerCosTheta[shower_i]
                    second_max_nonprimary_shower_CosThetaY = curr_showerCosThetaY[shower_i]
                    second_max_nonprimary_shower_DistToVtx = curr_showerDistToVtx[shower_i]


            if curr_showerPID[shower_i] == 22:
                curr_num_photons += 1
            elif curr_showerPID[shower_i] == 11:
                curr_num_electrons += 1
            elif curr_showerPID[shower_i] == 13:
                curr_num_muons += 1
            elif curr_showerPID[shower_i] == 211:
                curr_num_charged_pions += 1
            elif curr_showerPID[shower_i] == 2212:
                curr_num_protons += 1
            else:
                assert False, "Invalid shower PID"
            
            if curr_showerRecoE[shower_i] > max_shower_RecoE:
                max_shower_RecoE = curr_showerRecoE[shower_i]

            if is_electron:
                if curr_showerRecoE[shower_i] > max_electron_shower_RecoE: # new largest electron shower

                    second_max_electron_shower_RecoE = max_electron_shower_RecoE
                    second_max_electron_shower_PhScore = max_electron_shower_PhScore
                    second_max_electron_shower_ElScore = max_electron_shower_ElScore
                    second_max_electron_shower_MuScore = max_electron_shower_MuScore
                    second_max_electron_shower_PiScore = max_electron_shower_PiScore
                    second_max_electron_shower_PrScore = max_electron_shower_PrScore
                    second_max_electron_shower_ph_normedscore = max_electron_shower_ph_normedscore
                    second_max_electron_shower_el_normedscore = max_electron_shower_el_normedscore
                    second_max_electron_shower_electron_confidence = max_electron_shower_electron_confidence
                    second_max_electron_shower_PrimaryScore = max_electron_shower_PrimaryScore
                    second_max_electron_shower_FromNeutralScore = max_electron_shower_FromNeutralScore
                    second_max_electron_shower_FromChargedScore = max_electron_shower_FromChargedScore
                    second_max_electron_shower_CosTheta = max_electron_shower_CosTheta
                    second_max_electron_shower_CosThetaY = max_electron_shower_CosThetaY
                    second_max_electron_shower_DistToVtx = max_electron_shower_DistToVtx
                    max_electron_shower_RecoE = curr_showerRecoE[shower_i]
                    max_electron_shower_PhScore = curr_showerPhScore[shower_i]
                    max_electron_shower_ElScore = curr_showerElScore[shower_i]
                    max_electron_shower_MuScore = curr_showerMuScore[shower_i]
                    max_electron_shower_PiScore = curr_showerPiScore[shower_i]
                    max_electron_shower_PrScore = curr_showerPrScore[shower_i]
                    max_electron_shower_ph_normedscore = ph_normedscore
                    max_electron_shower_el_normedscore = el_normedscore
                    max_electron_shower_electron_confidence = electron_confidence
                    max_electron_shower_PrimaryScore = curr_showerPrimaryScore[shower_i]
                    max_electron_shower_FromNeutralScore = curr_showerFromNeutralScore[shower_i]
                    max_electron_shower_FromChargedScore = curr_showerFromChargedScore[shower_i]
                    max_electron_shower_CosTheta = curr_showerCosTheta[shower_i]
                    max_electron_shower_CosThetaY = curr_showerCosThetaY[shower_i]
                    max_electron_shower_DistToVtx = curr_showerDistToVtx[shower_i]
                elif curr_showerRecoE[shower_i] > second_max_electron_shower_RecoE: # new second largest electron shower
                    second_max_electron_shower_RecoE = curr_showerRecoE[shower_i]
                    second_max_electron_shower_PhScore = curr_showerPhScore[shower_i]
                    second_max_electron_shower_ElScore = curr_showerElScore[shower_i]
                    second_max_electron_shower_MuScore = curr_showerMuScore[shower_i]
                    second_max_electron_shower_PiScore = curr_showerPiScore[shower_i]
                    second_max_electron_shower_PrScore = curr_showerPrScore[shower_i]
                    second_max_electron_shower_ph_normedscore = ph_normedscore
                    second_max_electron_shower_el_normedscore = el_normedscore
                    second_max_electron_shower_electron_confidence = electron_confidence
                    second_max_electron_shower_PrimaryScore = curr_showerPrimaryScore[shower_i]
                    second_max_electron_shower_FromNeutralScore = curr_showerFromNeutralScore[shower_i]
                    second_max_electron_shower_FromChargedScore = curr_showerFromChargedScore[shower_i]
                    second_max_electron_shower_CosTheta = curr_showerCosTheta[shower_i]
                    second_max_electron_shower_CosThetaY = curr_showerCosThetaY[shower_i]
                    second_max_electron_shower_DistToVtx = curr_showerDistToVtx[shower_i]

            else: # photon shower
                if curr_showerRecoE[shower_i] > max_photon_shower_RecoE: # new largest photon shower
                    second_max_photon_shower_RecoE = max_photon_shower_RecoE
                    second_max_photon_shower_PhScore = max_photon_shower_PhScore
                    second_max_photon_shower_ElScore = max_photon_shower_ElScore
                    second_max_photon_shower_MuScore = max_photon_shower_MuScore
                    second_max_photon_shower_PiScore = max_photon_shower_PiScore
                    second_max_photon_shower_PrScore = max_photon_shower_PrScore
                    second_max_photon_shower_ph_normedscore = max_photon_shower_ph_normedscore
                    second_max_photon_shower_el_normedscore = max_photon_shower_el_normedscore
                    second_max_photon_shower_electron_confidence = max_photon_shower_electron_confidence
                    second_max_photon_shower_PrimaryScore = max_photon_shower_PrimaryScore
                    second_max_photon_shower_FromNeutralScore = max_photon_shower_FromNeutralScore
                    second_max_photon_shower_FromChargedScore = max_photon_shower_FromChargedScore
                    second_max_photon_shower_CosTheta = max_photon_shower_CosTheta
                    second_max_photon_shower_CosThetaY = max_photon_shower_CosThetaY
                    second_max_photon_shower_DistToVtx = max_photon_shower_DistToVtx
                    second_max_photon_shower_StartDirX = max_photon_shower_StartDirX
                    second_max_photon_shower_StartDirY = max_photon_shower_StartDirY
                    second_max_photon_shower_StartDirZ = max_photon_shower_StartDirZ

                    max_photon_shower_RecoE = curr_showerRecoE[shower_i]
                    max_photon_shower_PhScore = curr_showerPhScore[shower_i]
                    max_photon_shower_ElScore = curr_showerElScore[shower_i]
                    max_photon_shower_MuScore = curr_showerMuScore[shower_i]
                    max_photon_shower_PiScore = curr_showerPiScore[shower_i]
                    max_photon_shower_PrScore = curr_showerPrScore[shower_i]
                    max_photon_shower_ph_normedscore = ph_normedscore
                    max_photon_shower_el_normedscore = el_normedscore
                    max_photon_shower_electron_confidence = electron_confidence
                    max_photon_shower_PrimaryScore = curr_showerPrimaryScore[shower_i]
                    max_photon_shower_FromNeutralScore = curr_showerFromNeutralScore[shower_i]
                    max_photon_shower_FromChargedScore = curr_showerFromChargedScore[shower_i]
                    max_photon_shower_CosTheta = curr_showerCosTheta[shower_i]
                    max_photon_shower_CosThetaY = curr_showerCosThetaY[shower_i]
                    max_photon_shower_DistToVtx = curr_showerDistToVtx[shower_i]
                    max_photon_shower_StartDirX = curr_showerStartDirX[shower_i]
                    max_photon_shower_StartDirY = curr_showerStartDirY[shower_i]
                    max_photon_shower_StartDirZ = curr_showerStartDirZ[shower_i]

                elif curr_showerRecoE[shower_i] > second_max_photon_shower_RecoE: # new second largest photon shower
                    second_max_photon_shower_RecoE = curr_showerRecoE[shower_i]
                    second_max_photon_shower_PhScore = curr_showerPhScore[shower_i]
                    second_max_photon_shower_ElScore = curr_showerElScore[shower_i]
                    second_max_photon_shower_MuScore = curr_showerMuScore[shower_i]
                    second_max_photon_shower_PiScore = curr_showerPiScore[shower_i]
                    second_max_photon_shower_PrScore = curr_showerPrScore[shower_i]
                    second_max_photon_shower_ph_normedscore = ph_normedscore
                    second_max_photon_shower_el_normedscore = el_normedscore
                    second_max_photon_shower_electron_confidence = electron_confidence
                    second_max_photon_shower_PrimaryScore = curr_showerPrimaryScore[shower_i]
                    second_max_photon_shower_FromNeutralScore = curr_showerFromNeutralScore[shower_i]
                    second_max_photon_shower_FromChargedScore = curr_showerFromChargedScore[shower_i]
                    second_max_photon_shower_CosTheta = curr_showerCosTheta[shower_i]
                    second_max_photon_shower_CosThetaY = curr_showerCosThetaY[shower_i]
                    second_max_photon_shower_DistToVtx = curr_showerDistToVtx[shower_i]
                    second_max_photon_shower_StartDirX = curr_showerStartDirX[shower_i]
                    second_max_photon_shower_StartDirY = curr_showerStartDirY[shower_i]
                    second_max_photon_shower_StartDirZ = curr_showerStartDirZ[shower_i]

        if max_photon_shower_RecoE > 0 and second_max_photon_shower_RecoE > 0: # LANTERN sees two photons

            photon_1_energy = max_photon_shower_RecoE
            photon_2_energy = second_max_photon_shower_RecoE

            photon_1_dir = np.array([max_photon_shower_StartDirX, max_photon_shower_StartDirY, max_photon_shower_StartDirZ])
            photon_2_dir = np.array([second_max_photon_shower_StartDirX, second_max_photon_shower_StartDirY, second_max_photon_shower_StartDirZ])
            photon_1_dir = photon_1_dir / np.linalg.norm(photon_1_dir)
            photon_2_dir = photon_2_dir / np.linalg.norm(photon_2_dir)

            diphoton_energy = photon_1_energy + photon_2_energy
            diphoton_momentum = photon_1_energy * photon_1_dir + photon_2_energy * photon_2_dir

            dot_product = np.dot(photon_1_dir, photon_2_dir)
            dot_product = np.clip(dot_product, -1.0, 1.0) # accounting for floating point errors near 1 or -1
            diphoton_opening_angle = np.arccos(dot_product) * 180 / np.pi
            diphoton_costheta = diphoton_momentum[2] / diphoton_energy
            diphoton_mass_squared = diphoton_energy**2 - diphoton_momentum[0]**2 - diphoton_momentum[1]**2 - diphoton_momentum[2]**2
            diphoton_mass_squared = np.clip(diphoton_mass_squared, 0.0, None) # accounting for floating point errors near 0
            diphoton_mass = np.sqrt(diphoton_mass_squared)

        max_shower_RecoEs.append(max_shower_RecoE)
        max_electron_shower_RecoEs.append(max_electron_shower_RecoE)
        max_electron_shower_PhScores.append(max_electron_shower_PhScore)
        max_electron_shower_ElScores.append(max_electron_shower_ElScore)
        max_electron_shower_MuScores.append(max_electron_shower_MuScore)
        max_electron_shower_PiScores.append(max_electron_shower_PiScore)
        max_electron_shower_PrScores.append(max_electron_shower_PrScore)
        max_electron_shower_electron_confidences.append(max_electron_shower_electron_confidence)
        max_electron_shower_ph_normedscores.append(max_electron_shower_ph_normedscore)
        max_electron_shower_el_normedscores.append(max_electron_shower_el_normedscore)
        max_electron_shower_PrimaryScores.append(max_electron_shower_PrimaryScore)
        max_electron_shower_FromNeutralScores.append(max_electron_shower_FromNeutralScore)
        max_electron_shower_FromChargedScores.append(max_electron_shower_FromChargedScore)
        max_electron_shower_CosThetas.append(max_electron_shower_CosTheta)
        max_electron_shower_CosThetaYs.append(max_electron_shower_CosThetaY)
        max_electron_shower_DistToVtxs.append(max_electron_shower_DistToVtx)
        second_max_electron_shower_RecoEs.append(second_max_electron_shower_RecoE)
        second_max_electron_shower_PhScores.append(second_max_electron_shower_PhScore)
        second_max_electron_shower_ElScores.append(second_max_electron_shower_ElScore)
        second_max_electron_shower_MuScores.append(second_max_electron_shower_MuScore)
        second_max_electron_shower_PiScores.append(second_max_electron_shower_PiScore)
        second_max_electron_shower_PrScores.append(second_max_electron_shower_PrScore)
        second_max_electron_shower_electron_confidences.append(second_max_electron_shower_electron_confidence)
        second_max_electron_shower_ph_normedscores.append(second_max_electron_shower_ph_normedscore)
        second_max_electron_shower_el_normedscores.append(second_max_electron_shower_el_normedscore)
        second_max_electron_shower_PrimaryScores.append(second_max_electron_shower_PrimaryScore)
        second_max_electron_shower_FromNeutralScores.append(second_max_electron_shower_FromNeutralScore)
        second_max_electron_shower_FromChargedScores.append(second_max_electron_shower_FromChargedScore)
        second_max_electron_shower_CosThetas.append(second_max_electron_shower_CosTheta)
        second_max_electron_shower_CosThetaYs.append(second_max_electron_shower_CosThetaY)
        second_max_electron_shower_DistToVtxs.append(second_max_electron_shower_DistToVtx)
        max_photon_shower_RecoEs.append(max_photon_shower_RecoE)
        max_photon_shower_PhScores.append(max_photon_shower_PhScore)
        max_photon_shower_ElScores.append(max_photon_shower_ElScore)
        max_photon_shower_MuScores.append(max_photon_shower_MuScore)
        max_photon_shower_PiScores.append(max_photon_shower_PiScore)
        max_photon_shower_PrScores.append(max_photon_shower_PrScore)
        max_photon_shower_electron_confidences.append(max_photon_shower_electron_confidence)
        max_photon_shower_ph_normedscores.append(max_photon_shower_ph_normedscore)
        max_photon_shower_el_normedscores.append(max_photon_shower_el_normedscore)
        max_photon_shower_PrimaryScores.append(max_photon_shower_PrimaryScore)
        max_photon_shower_FromNeutralScores.append(max_photon_shower_FromNeutralScore)
        max_photon_shower_FromChargedScores.append(max_photon_shower_FromChargedScore)
        max_photon_shower_CosThetas.append(max_photon_shower_CosTheta)
        max_photon_shower_CosThetaYs.append(max_photon_shower_CosThetaY)
        max_photon_shower_DistToVtxs.append(max_photon_shower_DistToVtx)
        second_max_photon_shower_RecoEs.append(second_max_photon_shower_RecoE)
        second_max_photon_shower_PhScores.append(second_max_photon_shower_PhScore)
        second_max_photon_shower_ElScores.append(second_max_photon_shower_ElScore)
        second_max_photon_shower_MuScores.append(second_max_photon_shower_MuScore)
        second_max_photon_shower_PiScores.append(second_max_photon_shower_PiScore)
        second_max_photon_shower_PrScores.append(second_max_photon_shower_PrScore)
        second_max_photon_shower_electron_confidences.append(second_max_photon_shower_electron_confidence)
        second_max_photon_shower_ph_normedscores.append(second_max_photon_shower_ph_normedscore)
        second_max_photon_shower_el_normedscores.append(second_max_photon_shower_el_normedscore)
        second_max_photon_shower_PrimaryScores.append(second_max_photon_shower_PrimaryScore)
        second_max_photon_shower_FromNeutralScores.append(second_max_photon_shower_FromNeutralScore)
        second_max_photon_shower_FromChargedScores.append(second_max_photon_shower_FromChargedScore)
        second_max_photon_shower_CosThetas.append(second_max_photon_shower_CosTheta)
        second_max_photon_shower_CosThetaYs.append(second_max_photon_shower_CosThetaY)
        second_max_photon_shower_DistToVtxs.append(second_max_photon_shower_DistToVtx)
        max_nonprimary_shower_RecoEs.append(max_nonprimary_shower_RecoE)
        max_nonprimary_shower_PhScores.append(max_nonprimary_shower_PhScore)
        max_nonprimary_shower_ElScores.append(max_nonprimary_shower_ElScore)
        max_nonprimary_shower_MuScores.append(max_nonprimary_shower_MuScore)
        max_nonprimary_shower_PiScores.append(max_nonprimary_shower_PiScore)
        max_nonprimary_shower_PrScores.append(max_nonprimary_shower_PrScore)
        max_nonprimary_shower_electron_confidences.append(max_nonprimary_shower_electron_confidence)
        max_nonprimary_shower_ph_normedscores.append(max_nonprimary_shower_ph_normedscore)
        max_nonprimary_shower_el_normedscores.append(max_nonprimary_shower_el_normedscore)
        max_nonprimary_shower_PrimaryScores.append(max_nonprimary_shower_PrimaryScore)
        max_nonprimary_shower_FromNeutralScores.append(max_nonprimary_shower_FromNeutralScore)
        max_nonprimary_shower_FromChargedScores.append(max_nonprimary_shower_FromChargedScore)
        max_nonprimary_shower_CosThetas.append(max_nonprimary_shower_CosTheta)
        max_nonprimary_shower_CosThetaYs.append(max_nonprimary_shower_CosThetaY)
        max_nonprimary_shower_DistToVtxs.append(max_nonprimary_shower_DistToVtx)
        second_max_nonprimary_shower_RecoEs.append(second_max_nonprimary_shower_RecoE)
        second_max_nonprimary_shower_PhScores.append(second_max_nonprimary_shower_PhScore)
        second_max_nonprimary_shower_ElScores.append(second_max_nonprimary_shower_ElScore)
        second_max_nonprimary_shower_MuScores.append(second_max_nonprimary_shower_MuScore)
        second_max_nonprimary_shower_PiScores.append(second_max_nonprimary_shower_PiScore)
        second_max_nonprimary_shower_PrScores.append(second_max_nonprimary_shower_PrScore)
        second_max_nonprimary_shower_electron_confidences.append(second_max_nonprimary_shower_electron_confidence)
        second_max_nonprimary_shower_ph_normedscores.append(second_max_nonprimary_shower_ph_normedscore)
        second_max_nonprimary_shower_el_normedscores.append(second_max_nonprimary_shower_el_normedscore)
        second_max_nonprimary_shower_PrimaryScores.append(second_max_nonprimary_shower_PrimaryScore)
        second_max_nonprimary_shower_FromNeutralScores.append(second_max_nonprimary_shower_FromNeutralScore)
        second_max_nonprimary_shower_FromChargedScores.append(second_max_nonprimary_shower_FromChargedScore)
        second_max_nonprimary_shower_CosThetas.append(second_max_nonprimary_shower_CosTheta)
        second_max_nonprimary_shower_CosThetaYs.append(second_max_nonprimary_shower_CosThetaY)
        second_max_nonprimary_shower_DistToVtxs.append(second_max_nonprimary_shower_DistToVtx)
        diphoton_opening_angles.append(diphoton_opening_angle)
        diphoton_energies.append(diphoton_energy)
        diphoton_costhetas.append(diphoton_costheta)
        diphoton_masses.append(diphoton_mass)
        prim_shower_photon_nums.append(curr_num_photons)
        prim_shower_electron_nums.append(curr_num_electrons)
        prim_shower_muon_nums.append(curr_num_muons)
        prim_shower_charged_pion_nums.append(curr_num_charged_pions)
        prim_shower_proton_nums.append(curr_num_protons)

    new_lantern_cols_dic = {}

    new_lantern_cols_dic.update({
        "lantern_max_shower_RecoE": max_shower_RecoEs,

        "lantern_max_electron_shower_RecoE": max_electron_shower_RecoEs,
        "lantern_max_electron_shower_ElScore": max_electron_shower_ElScores,
        "lantern_max_electron_shower_PhScore": max_electron_shower_PhScores,
        "lantern_max_electron_shower_MuScore": max_electron_shower_MuScores,
        "lantern_max_electron_shower_PiScore": max_electron_shower_PiScores,
        "lantern_max_electron_shower_PrScore": max_electron_shower_PrScores,
        "lantern_max_electron_shower_electron_confidence": max_electron_shower_electron_confidences,
        "lantern_max_electron_shower_ph_normedscore": max_electron_shower_ph_normedscores,
        "lantern_max_electron_shower_el_normedscore": max_electron_shower_el_normedscores,
        "lantern_max_electron_shower_PrimaryScore": max_electron_shower_PrimaryScores,
        "lantern_max_electron_shower_FromNeutralScore": max_electron_shower_FromNeutralScores,
        "lantern_max_electron_shower_FromChargedScore": max_electron_shower_FromChargedScores,
        "lantern_max_electron_shower_CosTheta": max_electron_shower_CosThetas,
        "lantern_max_electron_shower_CosThetaY": max_electron_shower_CosThetaYs,
        "lantern_max_electron_shower_DistToVtx": max_electron_shower_DistToVtxs,
        "lantern_second_max_electron_shower_RecoE": second_max_electron_shower_RecoEs,
        "lantern_second_max_electron_shower_PhScore": second_max_electron_shower_PhScores,
        "lantern_second_max_electron_shower_ElScore": second_max_electron_shower_ElScores,
        "lantern_second_max_electron_shower_MuScore": second_max_electron_shower_MuScores,
        "lantern_second_max_electron_shower_PiScore": second_max_electron_shower_PiScores,
        "lantern_second_max_electron_shower_PrScore": second_max_electron_shower_PrScores,
        "lantern_second_max_electron_shower_electron_confidence": second_max_electron_shower_electron_confidences,
        "lantern_second_max_electron_shower_ph_normedscore": second_max_electron_shower_ph_normedscores,
        "lantern_second_max_electron_shower_el_normedscore": second_max_electron_shower_el_normedscores,
        "lantern_second_max_electron_shower_PrimaryScore": second_max_electron_shower_PrimaryScores,
        "lantern_second_max_electron_shower_FromNeutralScore": second_max_electron_shower_FromNeutralScores,
        "lantern_second_max_electron_shower_FromChargedScore": second_max_electron_shower_FromChargedScores,
        "lantern_second_max_electron_shower_CosTheta": second_max_electron_shower_CosThetas,
        "lantern_second_max_electron_shower_CosThetaY": second_max_electron_shower_CosThetaYs,
        "lantern_second_max_electron_shower_DistToVtx": second_max_electron_shower_DistToVtxs,

        "lantern_max_photon_shower_RecoE": max_photon_shower_RecoEs,
        "lantern_max_photon_shower_PhScore": max_photon_shower_PhScores,
        "lantern_max_photon_shower_ElScore": max_photon_shower_ElScores,
        "lantern_max_photon_shower_MuScore": max_photon_shower_MuScores,
        "lantern_max_photon_shower_PiScore": max_photon_shower_PiScores,
        "lantern_max_photon_shower_PrScore": max_photon_shower_PrScores,
        "lantern_max_photon_shower_electron_confidence": max_photon_shower_electron_confidences,
        "lantern_max_photon_shower_ph_normedscore": max_photon_shower_ph_normedscores,
        "lantern_max_photon_shower_el_normedscore": max_photon_shower_el_normedscores,
        "lantern_max_photon_shower_PrimaryScore": max_photon_shower_PrimaryScores,
        "lantern_max_photon_shower_FromNeutralScore": max_photon_shower_FromNeutralScores,
        "lantern_max_photon_shower_FromChargedScore": max_photon_shower_FromChargedScores,
        "lantern_max_photon_shower_CosTheta": max_photon_shower_CosThetas,
        "lantern_max_photon_shower_CosThetaY": max_photon_shower_CosThetaYs,
        "lantern_max_photon_shower_DistToVtx": max_photon_shower_DistToVtxs,
        "lantern_second_max_photon_shower_RecoE": second_max_photon_shower_RecoEs,
        "lantern_second_max_photon_shower_PhScore": second_max_photon_shower_PhScores,
        "lantern_second_max_photon_shower_ElScore": second_max_photon_shower_ElScores,
        "lantern_second_max_photon_shower_MuScore": second_max_photon_shower_MuScores,
        "lantern_second_max_photon_shower_PiScore": second_max_photon_shower_PiScores,
        "lantern_second_max_photon_shower_PrScore": second_max_photon_shower_PrScores,
        "lantern_second_max_photon_shower_electron_confidence": second_max_photon_shower_electron_confidences,
        "lantern_second_max_photon_shower_ph_normedscore": second_max_photon_shower_ph_normedscores,
        "lantern_second_max_photon_shower_el_normedscore": second_max_photon_shower_el_normedscores,
        "lantern_second_max_photon_shower_PrimaryScore": second_max_photon_shower_PrimaryScores,
        "lantern_second_max_photon_shower_FromNeutralScore": second_max_photon_shower_FromNeutralScores,
        "lantern_second_max_photon_shower_FromChargedScore": second_max_photon_shower_FromChargedScores,
        "lantern_second_max_photon_shower_CosTheta": second_max_photon_shower_CosThetas,
        "lantern_second_max_photon_shower_CosThetaY": second_max_photon_shower_CosThetaYs,
        "lantern_second_max_photon_shower_DistToVtx": second_max_photon_shower_DistToVtxs,

        "lantern_max_nonprimary_shower_RecoE": max_nonprimary_shower_RecoEs,
        "lantern_max_nonprimary_shower_PhScore": max_nonprimary_shower_PhScores,
        "lantern_max_nonprimary_shower_ElScore": max_nonprimary_shower_ElScores,
        "lantern_max_nonprimary_shower_MuScore": max_nonprimary_shower_MuScores,
        "lantern_max_nonprimary_shower_PiScore": max_nonprimary_shower_PiScores,
        "lantern_max_nonprimary_shower_PrScore": max_nonprimary_shower_PrScores,
        "lantern_max_nonprimary_shower_electron_confidence": max_nonprimary_shower_electron_confidences,
        "lantern_max_nonprimary_shower_ph_normedscore": max_nonprimary_shower_ph_normedscores,
        "lantern_max_nonprimary_shower_el_normedscore": max_nonprimary_shower_el_normedscores,
        "lantern_max_nonprimary_shower_PrimaryScore": max_nonprimary_shower_PrimaryScores,
        "lantern_max_nonprimary_shower_FromNeutralScore": max_nonprimary_shower_FromNeutralScores,
        "lantern_max_nonprimary_shower_FromChargedScore": max_nonprimary_shower_FromChargedScores,
        "lantern_max_nonprimary_shower_CosTheta": max_nonprimary_shower_CosThetas,
        "lantern_max_nonprimary_shower_CosThetaY": max_nonprimary_shower_CosThetaYs,
        "lantern_max_nonprimary_shower_DistToVtx": max_nonprimary_shower_DistToVtxs,
        "lantern_second_max_nonprimary_shower_RecoE": second_max_nonprimary_shower_RecoEs,
        "lantern_second_max_nonprimary_shower_PhScore": second_max_nonprimary_shower_PhScores,
        "lantern_second_max_nonprimary_shower_ElScore": second_max_nonprimary_shower_ElScores,
        "lantern_second_max_nonprimary_shower_MuScore": second_max_nonprimary_shower_MuScores,
        "lantern_second_max_nonprimary_shower_PiScore": second_max_nonprimary_shower_PiScores,
        "lantern_second_max_nonprimary_shower_PrScore": second_max_nonprimary_shower_PrScores,
        "lantern_second_max_nonprimary_shower_electron_confidence": second_max_nonprimary_shower_electron_confidences,
        "lantern_second_max_nonprimary_shower_ph_normedscore": second_max_nonprimary_shower_ph_normedscores,
        "lantern_second_max_nonprimary_shower_el_normedscore": second_max_nonprimary_shower_el_normedscores,
        "lantern_second_max_nonprimary_shower_PrimaryScore": second_max_nonprimary_shower_PrimaryScores,
        "lantern_second_max_nonprimary_shower_FromNeutralScore": second_max_nonprimary_shower_FromNeutralScores,
        "lantern_second_max_nonprimary_shower_FromChargedScore": second_max_nonprimary_shower_FromChargedScores,
        "lantern_second_max_nonprimary_shower_CosTheta": second_max_nonprimary_shower_CosThetas,
        "lantern_second_max_nonprimary_shower_CosThetaY": second_max_nonprimary_shower_CosThetaYs,
        "lantern_second_max_nonprimary_shower_DistToVtx": second_max_nonprimary_shower_DistToVtxs,

        "lantern_diphoton_opening_angle": diphoton_opening_angles,
        "lantern_diphoton_energy": diphoton_energies,
        "lantern_diphoton_costheta": diphoton_costhetas,
        "lantern_diphoton_mass": diphoton_masses,

        "lantern_prim_shower_photon_num": prim_shower_photon_nums,
        "lantern_prim_shower_electron_num": prim_shower_electron_nums,
        "lantern_prim_shower_muon_num": prim_shower_muon_nums,
        "lantern_prim_shower_charged_pion_num": prim_shower_charged_pion_nums,
        "lantern_prim_shower_proton_num": prim_shower_proton_nums,

    })


    # see function here: https://github.com/NuTufts/lantern_ana/blob/94f31b83e8a170230ca7b948c701671ab3099bbd/lantern_ana/utils/get_primary_muon_candidates.py
    # this code is similar, but considers all types of tracks, and counts all types of particles

    prim_muon_track_max_muscores = []
    prim_proton_track_max_prscores = []
    prim_charged_pion_track_max_piscores = []

    prim_track_photon_nums = []
    prim_track_electron_nums = []
    prim_track_muon_nums = []
    prim_track_charged_pion_nums = []
    prim_track_proton_nums = []

    prim_track_photon_nums_5MeV = []
    prim_track_electron_nums_5MeV = []
    prim_track_muon_nums_5MeV = []
    prim_track_charged_pion_nums_5MeV = []
    prim_track_proton_nums_5MeV = []

    prim_track_photon_nums_10MeV = []
    prim_track_electron_nums_10MeV = []
    prim_track_muon_nums_10MeV = []
    prim_track_charged_pion_nums_10MeV = []
    prim_track_proton_nums_10MeV = []

    prim_track_photon_nums_15MeV = []
    prim_track_electron_nums_15MeV = []
    prim_track_muon_nums_15MeV = []
    prim_track_charged_pion_nums_15MeV = []
    prim_track_proton_nums_15MeV = []

    prim_track_photon_nums_20MeV = []
    prim_track_electron_nums_20MeV = []
    prim_track_muon_nums_20MeV = []
    prim_track_charged_pion_nums_20MeV = []
    prim_track_proton_nums_20MeV = []

    prim_track_photon_nums_25MeV = []
    prim_track_electron_nums_25MeV = []
    prim_track_muon_nums_25MeV = []
    prim_track_charged_pion_nums_25MeV = []
    prim_track_proton_nums_25MeV = []

    prim_track_photon_nums_30MeV = []
    prim_track_electron_nums_30MeV = []
    prim_track_muon_nums_30MeV = []
    prim_track_charged_pion_nums_30MeV = []
    prim_track_proton_nums_30MeV = []

    prim_track_photon_nums_35MeV = []
    prim_track_electron_nums_35MeV = []
    prim_track_muon_nums_35MeV = []
    prim_track_charged_pion_nums_35MeV = []
    prim_track_proton_nums_35MeV = []

    prim_track_photon_nums_40MeV = []
    prim_track_electron_nums_40MeV = []
    prim_track_muon_nums_40MeV = []
    prim_track_charged_pion_nums_40MeV = []
    prim_track_proton_nums_40MeV = []

    prim_track_photon_nums_45MeV = []
    prim_track_electron_nums_45MeV = []
    prim_track_muon_nums_45MeV = []
    prim_track_charged_pion_nums_45MeV = []
    prim_track_proton_nums_45MeV = []

    prim_track_photon_nums_50MeV = []
    prim_track_electron_nums_50MeV = []
    prim_track_muon_nums_50MeV = []
    prim_track_charged_pion_nums_50MeV = []
    prim_track_proton_nums_50MeV = []

    nTracks = df["lantern_nTracks"].to_numpy()
    trackIsSecondary = df["lantern_trackIsSecondary"].to_numpy()
    trackClassified = df["lantern_trackClassified"].to_numpy()
    trackCharge = df["lantern_trackCharge"].to_numpy()
    trackCosTheta = df["lantern_trackCosTheta"].to_numpy()
    trackCosThetaY = df["lantern_trackCosThetaY"].to_numpy()
    trackDistToVtx = df["lantern_trackDistToVtx"].to_numpy()
    trackComp = df["lantern_trackComp"].to_numpy()
    trackPurity = df["lantern_trackPurity"].to_numpy()
    trackPrimaryScore = df["lantern_trackPrimaryScore"].to_numpy()
    trackFromNeutralScore = df["lantern_trackFromNeutralScore"].to_numpy()
    trackFromChargedScore = df["lantern_trackFromChargedScore"].to_numpy()
    trackPID = df["lantern_trackPID"].to_numpy()
    trackElScore = df["lantern_trackElScore"].to_numpy()
    trackPhScore = df["lantern_trackPhScore"].to_numpy()
    trackMuScore = df["lantern_trackMuScore"].to_numpy()
    trackPiScore = df["lantern_trackPiScore"].to_numpy()
    trackPrScore = df["lantern_trackPrScore"].to_numpy()
    trackRecoE = df["lantern_trackRecoE"].to_numpy()

    for event_i in tqdm(range(len(df)), desc="Analyzing LANTERN tracks", mininterval=10):
        curr_nTracks_val = nTracks[event_i]
        if np.isnan(curr_nTracks_val):
            curr_nTracks = 0
        else:
            curr_nTracks = int(curr_nTracks_val)
        curr_trackIsSecondary = trackIsSecondary[event_i]
        curr_trackClassified = trackClassified[event_i]
        curr_trackCharge = trackCharge[event_i]
        curr_trackCosTheta = trackCosTheta[event_i]
        curr_trackCosThetaY = trackCosThetaY[event_i]
        curr_trackDistToVtx = trackDistToVtx[event_i]
        curr_trackComp = trackComp[event_i]
        curr_trackPurity = trackPurity[event_i]
        curr_trackPrimaryScore = trackPrimaryScore[event_i]
        curr_trackFromNeutralScore = trackFromNeutralScore[event_i]
        curr_trackFromChargedScore = trackFromChargedScore[event_i]
        curr_trackPID = trackPID[event_i]
        curr_trackElScore = trackElScore[event_i]
        curr_trackPhScore = trackPhScore[event_i]
        curr_trackMuScore = trackMuScore[event_i]
        curr_trackPiScore = trackPiScore[event_i]
        curr_trackPrScore = trackPrScore[event_i]
        curr_trackRecoE = trackRecoE[event_i]
        curr_trackCosTheta = trackCosTheta[event_i]
        curr_trackCosThetaY = trackCosThetaY[event_i]
        curr_trackDistToVtx = trackDistToVtx[event_i]
        curr_num_photons = 0
        curr_num_electrons = 0
        curr_num_muons = 0
        curr_num_protons = 0
        curr_num_charged_pions = 0

        curr_num_photons_5MeV = 0
        curr_num_electrons_5MeV = 0
        curr_num_muons_5MeV = 0
        curr_num_charged_pions_5MeV = 0
        curr_num_protons_5MeV = 0

        curr_num_photons_10MeV = 0
        curr_num_electrons_10MeV = 0
        curr_num_muons_10MeV = 0
        curr_num_charged_pions_10MeV = 0
        curr_num_protons_10MeV = 0

        curr_num_photons_15MeV = 0
        curr_num_electrons_15MeV = 0
        curr_num_muons_15MeV = 0
        curr_num_charged_pions_15MeV = 0
        curr_num_protons_15MeV = 0

        curr_num_photons_20MeV = 0
        curr_num_electrons_20MeV = 0
        curr_num_muons_20MeV = 0
        curr_num_charged_pions_20MeV = 0
        curr_num_protons_20MeV = 0

        curr_num_photons_25MeV = 0
        curr_num_electrons_25MeV = 0
        curr_num_muons_25MeV = 0
        curr_num_charged_pions_25MeV = 0
        curr_num_protons_25MeV = 0

        curr_num_photons_30MeV = 0
        curr_num_electrons_30MeV = 0
        curr_num_muons_30MeV = 0
        curr_num_charged_pions_30MeV = 0
        curr_num_protons_30MeV = 0

        curr_num_photons_35MeV = 0
        curr_num_electrons_35MeV = 0
        curr_num_muons_35MeV = 0
        curr_num_charged_pions_35MeV = 0
        curr_num_protons_35MeV = 0

        curr_num_photons_40MeV = 0
        curr_num_electrons_40MeV = 0
        curr_num_muons_40MeV = 0
        curr_num_charged_pions_40MeV = 0
        curr_num_protons_40MeV = 0

        curr_num_photons_45MeV = 0
        curr_num_electrons_45MeV = 0
        curr_num_muons_45MeV = 0
        curr_num_charged_pions_45MeV = 0
        curr_num_protons_45MeV = 0

        curr_num_photons_50MeV = 0
        curr_num_electrons_50MeV = 0
        curr_num_muons_50MeV = 0
        curr_num_charged_pions_50MeV = 0
        curr_num_protons_50MeV = 0

        curr_max_muscore = np.nan   
        curr_max_prscore = np.nan
        curr_max_piscore = np.nan
        
        for track_i in range(curr_nTracks):

            if curr_trackIsSecondary[track_i] != 0: # only considering primary tracks
                continue

            if curr_trackClassified[track_i] != 1: # only considering classified tracks
                continue

            min_charge, min_completeness, min_purity = 0, 0, 0
            if curr_trackCharge[track_i] < min_charge or curr_trackComp[track_i] < min_completeness or curr_trackPurity[track_i] < min_purity:
                continue

            curr_trackRecoE_curr_track = curr_trackRecoE[track_i]

            if curr_trackPID[track_i] == 13: # muon
                curr_num_muons += 1
                if curr_trackRecoE_curr_track > 5:
                    curr_num_muons_5MeV += 1
                if curr_trackRecoE_curr_track > 10:
                    curr_num_muons_10MeV += 1
                if curr_trackRecoE_curr_track > 15:
                    curr_num_muons_15MeV += 1
                if curr_trackRecoE_curr_track > 20:
                    curr_num_muons_20MeV += 1
                if curr_trackRecoE_curr_track > 25:
                    curr_num_muons_25MeV += 1
                if curr_trackRecoE_curr_track > 30:
                    curr_num_muons_30MeV += 1
                if curr_trackRecoE_curr_track > 35:
                    curr_num_muons_35MeV += 1
                if curr_trackRecoE_curr_track > 40:
                    curr_num_muons_40MeV += 1
                if curr_trackRecoE_curr_track > 45:
                    curr_num_muons_45MeV += 1
                if curr_trackRecoE_curr_track > 50:
                    curr_num_muons_50MeV += 1
                if curr_trackMuScore[track_i] > curr_max_muscore or curr_max_muscore is np.nan:
                    curr_max_muscore = curr_trackMuScore[track_i]
            elif curr_trackPID[track_i] == 2212: # proton
                curr_num_protons += 1
                if curr_trackRecoE_curr_track > 5:
                    curr_num_protons_5MeV += 1
                if curr_trackRecoE_curr_track > 10:
                    curr_num_protons_10MeV += 1
                if curr_trackRecoE_curr_track > 15:
                    curr_num_protons_15MeV += 1
                if curr_trackRecoE_curr_track > 20:
                    curr_num_protons_20MeV += 1
                if curr_trackRecoE_curr_track > 25:
                    curr_num_protons_25MeV += 1
                if curr_trackRecoE_curr_track > 30:
                    curr_num_protons_30MeV += 1
                if curr_trackRecoE_curr_track > 35:
                    curr_num_protons_35MeV += 1
                if curr_trackRecoE_curr_track > 40:
                    curr_num_protons_40MeV += 1
                if curr_trackRecoE_curr_track > 45:
                    curr_num_protons_45MeV += 1
                if curr_trackRecoE_curr_track > 50:
                    curr_num_protons_50MeV += 1
                if curr_trackPrScore[track_i] > curr_max_prscore or curr_max_prscore is np.nan:
                    curr_max_prscore = curr_trackPrScore[track_i]
            elif curr_trackPID[track_i] == 211: # charged pion
                curr_num_charged_pions += 1
                if curr_trackRecoE_curr_track > 5:
                    curr_num_charged_pions_5MeV += 1
                if curr_trackRecoE_curr_track > 10:
                    curr_num_charged_pions_10MeV += 1
                if curr_trackRecoE_curr_track > 15:
                    curr_num_charged_pions_15MeV += 1
                if curr_trackRecoE_curr_track > 20:
                    curr_num_charged_pions_20MeV += 1
                if curr_trackRecoE_curr_track > 25:
                    curr_num_charged_pions_25MeV += 1
                if curr_trackRecoE_curr_track > 30:
                    curr_num_charged_pions_30MeV += 1
                if curr_trackRecoE_curr_track > 35:
                    curr_num_charged_pions_35MeV += 1
                if curr_trackRecoE_curr_track > 40:
                    curr_num_charged_pions_40MeV += 1
                if curr_trackRecoE_curr_track > 45:
                    curr_num_charged_pions_45MeV += 1
                if curr_trackRecoE_curr_track > 50:
                    curr_num_charged_pions_50MeV += 1
                if curr_trackPiScore[track_i] > curr_max_piscore or curr_max_piscore is np.nan:
                    curr_max_piscore = curr_trackPiScore[track_i]
            elif curr_trackPID[track_i] == 22: # photon
                curr_num_photons += 1
                if curr_trackRecoE_curr_track > 5:
                    curr_num_photons_5MeV += 1
                if curr_trackRecoE_curr_track > 10:
                    curr_num_photons_10MeV += 1
                if curr_trackRecoE_curr_track > 15:
                    curr_num_photons_15MeV += 1
                if curr_trackRecoE_curr_track > 20:
                    curr_num_photons_20MeV += 1
                if curr_trackRecoE_curr_track > 25:
                    curr_num_photons_25MeV += 1
                if curr_trackRecoE_curr_track > 30:
                    curr_num_photons_30MeV += 1
                if curr_trackRecoE_curr_track > 35:
                    curr_num_photons_35MeV += 1
                if curr_trackRecoE_curr_track > 40:
                    curr_num_photons_40MeV += 1
                if curr_trackRecoE_curr_track > 45:
                    curr_num_photons_45MeV += 1
                if curr_trackRecoE_curr_track > 50:
                    curr_num_photons_50MeV += 1
            elif curr_trackPID[track_i] == 11: # electron
                curr_num_electrons += 1
                if curr_trackRecoE_curr_track > 5:
                    curr_num_electrons_5MeV += 1
                if curr_trackRecoE_curr_track > 10:
                    curr_num_electrons_10MeV += 1
                if curr_trackRecoE_curr_track > 15:
                    curr_num_electrons_15MeV += 1
                if curr_trackRecoE_curr_track > 20:
                    curr_num_electrons_20MeV += 1
                if curr_trackRecoE_curr_track > 25:
                    curr_num_electrons_25MeV += 1
                if curr_trackRecoE_curr_track > 30:
                    curr_num_electrons_30MeV += 1
                if curr_trackRecoE_curr_track > 35:
                    curr_num_electrons_35MeV += 1
                if curr_trackRecoE_curr_track > 40:
                    curr_num_electrons_40MeV += 1
                if curr_trackRecoE_curr_track > 45:
                    curr_num_electrons_45MeV += 1
                if curr_trackRecoE_curr_track > 50:
                    curr_num_electrons_50MeV += 1
            else:
                assert False, "Invalid track PID"

        prim_track_photon_nums.append(curr_num_photons)
        prim_track_electron_nums.append(curr_num_electrons)
        prim_track_muon_nums.append(curr_num_muons)
        prim_track_charged_pion_nums.append(curr_num_charged_pions)
        prim_track_proton_nums.append(curr_num_protons)

        prim_track_photon_nums_5MeV.append(curr_num_photons_5MeV)
        prim_track_electron_nums_5MeV.append(curr_num_electrons_5MeV)
        prim_track_muon_nums_5MeV.append(curr_num_muons_5MeV)
        prim_track_charged_pion_nums_5MeV.append(curr_num_charged_pions_5MeV)
        prim_track_proton_nums_5MeV.append(curr_num_protons_5MeV)

        prim_track_photon_nums_10MeV.append(curr_num_photons_10MeV)
        prim_track_electron_nums_10MeV.append(curr_num_electrons_10MeV)
        prim_track_muon_nums_10MeV.append(curr_num_muons_10MeV)
        prim_track_charged_pion_nums_10MeV.append(curr_num_charged_pions_10MeV)
        prim_track_proton_nums_10MeV.append(curr_num_protons_10MeV)

        prim_track_photon_nums_15MeV.append(curr_num_photons_15MeV)
        prim_track_electron_nums_15MeV.append(curr_num_electrons_15MeV)
        prim_track_muon_nums_15MeV.append(curr_num_muons_15MeV)
        prim_track_charged_pion_nums_15MeV.append(curr_num_charged_pions_15MeV)
        prim_track_proton_nums_15MeV.append(curr_num_protons_15MeV)

        prim_track_photon_nums_20MeV.append(curr_num_photons_20MeV)
        prim_track_electron_nums_20MeV.append(curr_num_electrons_20MeV)
        prim_track_muon_nums_20MeV.append(curr_num_muons_20MeV)
        prim_track_charged_pion_nums_20MeV.append(curr_num_charged_pions_20MeV)
        prim_track_proton_nums_20MeV.append(curr_num_protons_20MeV)

        prim_track_photon_nums_25MeV.append(curr_num_photons_25MeV)
        prim_track_electron_nums_25MeV.append(curr_num_electrons_25MeV)
        prim_track_muon_nums_25MeV.append(curr_num_muons_25MeV)
        prim_track_charged_pion_nums_25MeV.append(curr_num_charged_pions_25MeV)
        prim_track_proton_nums_25MeV.append(curr_num_protons_25MeV)

        prim_track_photon_nums_30MeV.append(curr_num_photons_30MeV)
        prim_track_electron_nums_30MeV.append(curr_num_electrons_30MeV)
        prim_track_muon_nums_30MeV.append(curr_num_muons_30MeV)
        prim_track_charged_pion_nums_30MeV.append(curr_num_charged_pions_30MeV)
        prim_track_proton_nums_30MeV.append(curr_num_protons_30MeV)

        prim_track_photon_nums_35MeV.append(curr_num_photons_35MeV)
        prim_track_electron_nums_35MeV.append(curr_num_electrons_35MeV)
        prim_track_muon_nums_35MeV.append(curr_num_muons_35MeV)
        prim_track_charged_pion_nums_35MeV.append(curr_num_charged_pions_35MeV)
        prim_track_proton_nums_35MeV.append(curr_num_protons_35MeV)

        prim_track_photon_nums_40MeV.append(curr_num_photons_40MeV)
        prim_track_electron_nums_40MeV.append(curr_num_electrons_40MeV)
        prim_track_muon_nums_40MeV.append(curr_num_muons_40MeV)
        prim_track_charged_pion_nums_40MeV.append(curr_num_charged_pions_40MeV)
        prim_track_proton_nums_40MeV.append(curr_num_protons_40MeV)

        prim_track_photon_nums_45MeV.append(curr_num_photons_45MeV)
        prim_track_electron_nums_45MeV.append(curr_num_electrons_45MeV)
        prim_track_muon_nums_45MeV.append(curr_num_muons_45MeV)
        prim_track_charged_pion_nums_45MeV.append(curr_num_charged_pions_45MeV)
        prim_track_proton_nums_45MeV.append(curr_num_protons_45MeV)

        prim_track_photon_nums_50MeV.append(curr_num_photons_50MeV)
        prim_track_electron_nums_50MeV.append(curr_num_electrons_50MeV)
        prim_track_muon_nums_50MeV.append(curr_num_muons_50MeV)
        prim_track_charged_pion_nums_50MeV.append(curr_num_charged_pions_50MeV)
        prim_track_proton_nums_50MeV.append(curr_num_protons_50MeV)

        prim_muon_track_max_muscores.append(curr_max_muscore)
        prim_proton_track_max_prscores.append(curr_max_prscore)
        prim_charged_pion_track_max_piscores.append(curr_max_piscore)

    new_lantern_cols_dic.update({
        "lantern_prim_track_photon_num": prim_track_photon_nums,
        "lantern_prim_track_electron_num": prim_track_electron_nums,
        "lantern_prim_track_muon_num": prim_track_muon_nums,
        "lantern_prim_track_charged_pion_num": prim_track_charged_pion_nums,
        "lantern_prim_track_proton_num": prim_track_proton_nums,

        "lantern_prim_photon_num": np.array(prim_shower_photon_nums) + np.array(prim_track_photon_nums),
        "lantern_prim_electron_num": np.array(prim_shower_electron_nums) + np.array(prim_track_electron_nums),
        "lantern_prim_muon_num": np.array(prim_shower_muon_nums) + np.array(prim_track_muon_nums),
        "lantern_prim_charged_pion_num": np.array(prim_shower_charged_pion_nums) + np.array(prim_track_charged_pion_nums),
        "lantern_prim_proton_num": np.array(prim_shower_proton_nums) + np.array(prim_track_proton_nums),

        "lantern_prim_track_photon_num_5MeV": prim_track_photon_nums_5MeV,
        "lantern_prim_track_electron_num_5MeV": prim_track_electron_nums_5MeV,
        "lantern_prim_track_muon_num_5MeV": prim_track_muon_nums_5MeV,
        "lantern_prim_track_charged_pion_num_5MeV": prim_track_charged_pion_nums_5MeV,
        "lantern_prim_track_proton_num_5MeV": prim_track_proton_nums_5MeV,

        "lantern_prim_track_photon_num_10MeV": prim_track_photon_nums_10MeV,
        "lantern_prim_track_electron_num_10MeV": prim_track_electron_nums_10MeV,
        "lantern_prim_track_muon_num_10MeV": prim_track_muon_nums_10MeV,
        "lantern_prim_track_charged_pion_num_10MeV": prim_track_charged_pion_nums_10MeV,
        "lantern_prim_track_proton_num_10MeV": prim_track_proton_nums_10MeV,

        "lantern_prim_track_photon_num_15MeV": prim_track_photon_nums_15MeV,
        "lantern_prim_track_electron_num_15MeV": prim_track_electron_nums_15MeV,
        "lantern_prim_track_muon_num_15MeV": prim_track_muon_nums_15MeV,
        "lantern_prim_track_charged_pion_num_15MeV": prim_track_charged_pion_nums_15MeV,
        "lantern_prim_track_proton_num_15MeV": prim_track_proton_nums_15MeV,

        "lantern_prim_track_photon_num_20MeV": prim_track_photon_nums_20MeV,
        "lantern_prim_track_electron_num_20MeV": prim_track_electron_nums_20MeV,
        "lantern_prim_track_muon_num_20MeV": prim_track_muon_nums_20MeV,
        "lantern_prim_track_charged_pion_num_20MeV": prim_track_charged_pion_nums_20MeV,
        "lantern_prim_track_proton_num_20MeV": prim_track_proton_nums_20MeV,

        "lantern_prim_track_photon_num_25MeV": prim_track_photon_nums_25MeV,
        "lantern_prim_track_electron_num_25MeV": prim_track_electron_nums_25MeV,
        "lantern_prim_track_muon_num_25MeV": prim_track_muon_nums_25MeV,
        "lantern_prim_track_charged_pion_num_25MeV": prim_track_charged_pion_nums_25MeV,
        "lantern_prim_track_proton_num_25MeV": prim_track_proton_nums_25MeV,

        "lantern_prim_track_photon_num_30MeV": prim_track_photon_nums_30MeV,
        "lantern_prim_track_electron_num_30MeV": prim_track_electron_nums_30MeV,
        "lantern_prim_track_muon_num_30MeV": prim_track_muon_nums_30MeV,
        "lantern_prim_track_charged_pion_num_30MeV": prim_track_charged_pion_nums_30MeV,
        "lantern_prim_track_proton_num_30MeV": prim_track_proton_nums_30MeV,

        "lantern_prim_track_photon_num_35MeV": prim_track_photon_nums_35MeV,
        "lantern_prim_track_electron_num_35MeV": prim_track_electron_nums_35MeV,
        "lantern_prim_track_muon_num_35MeV": prim_track_muon_nums_35MeV,
        "lantern_prim_track_charged_pion_num_35MeV": prim_track_charged_pion_nums_35MeV,
        "lantern_prim_track_proton_num_35MeV": prim_track_proton_nums_35MeV,

        "lantern_prim_track_photon_num_40MeV": prim_track_photon_nums_40MeV,
        "lantern_prim_track_electron_num_40MeV": prim_track_electron_nums_40MeV,
        "lantern_prim_track_muon_num_40MeV": prim_track_muon_nums_40MeV,
        "lantern_prim_track_charged_pion_num_40MeV": prim_track_charged_pion_nums_40MeV,
        "lantern_prim_track_proton_num_40MeV": prim_track_proton_nums_40MeV,

        "lantern_prim_track_photon_num_45MeV": prim_track_photon_nums_45MeV,
        "lantern_prim_track_electron_num_45MeV": prim_track_electron_nums_45MeV,
        "lantern_prim_track_muon_num_45MeV": prim_track_muon_nums_45MeV,
        "lantern_prim_track_charged_pion_num_45MeV": prim_track_charged_pion_nums_45MeV,
        "lantern_prim_track_proton_num_45MeV": prim_track_proton_nums_45MeV,

        "lantern_prim_track_photon_num_50MeV": prim_track_photon_nums_50MeV,
        "lantern_prim_track_electron_num_50MeV": prim_track_electron_nums_50MeV,
        "lantern_prim_track_muon_num_50MeV": prim_track_muon_nums_50MeV,
        "lantern_prim_track_charged_pion_num_50MeV": prim_track_charged_pion_nums_50MeV,
        "lantern_prim_track_proton_num_50MeV": prim_track_proton_nums_50MeV,

        "lantern_prim_muon_track_max_muscore": prim_muon_track_max_muscores,
        "lantern_prim_proton_track_max_prscore": prim_proton_track_max_prscores,
        "lantern_prim_charged_pion_track_max_piscore": prim_charged_pion_track_max_piscores,
    })

    df = pd.concat([df, pd.DataFrame(new_lantern_cols_dic)], axis=1)

    return df

def do_combined_postprocessing(df):

    # checked that using the Pandora SCE vertex is further from the WC and Lantern vertices

    # compute distances using sqrt-of-squares
    df = df.with_columns([
        (
            ((pl.col("wc_reco_nuvtxX") - pl.col("pandora_reco_nu_vtx_x"))**2 +
             (pl.col("wc_reco_nuvtxY") - pl.col("pandora_reco_nu_vtx_y"))**2 +
             (pl.col("wc_reco_nuvtxZ") - pl.col("pandora_reco_nu_vtx_z"))**2)
            .sqrt()
            .alias("wc_pandora_dist")
        ),
        (
            ((pl.col("wc_reco_nuvtxX") - pl.col("lantern_vtxX"))**2 +
             (pl.col("wc_reco_nuvtxY") - pl.col("lantern_vtxY"))**2 +
             (pl.col("wc_reco_nuvtxZ") - pl.col("lantern_vtxZ"))**2)
            .sqrt()
            .alias("wc_lantern_dist")
        ),
        (
            ((pl.col("lantern_vtxX") - pl.col("pandora_reco_nu_vtx_x"))**2 +
             (pl.col("lantern_vtxY") - pl.col("pandora_reco_nu_vtx_y"))**2 +
             (pl.col("lantern_vtxZ") - pl.col("pandora_reco_nu_vtx_z"))**2)
            .sqrt()
            .alias("lantern_pandora_dist")
        ),
    ])

    # changing to nan when lantern vtx is the default (-999, -999, -999)
    # the WC vtx will always be valid after generic neutrino selection
    # the Pandora vtx will always be valid since a slice is always chosen
    df = df.with_columns([
        pl.when((pl.col("lantern_vtxX") - (-999)).abs() < 1e-2).then(np.nan).otherwise(pl.col("wc_lantern_dist")).alias("wc_lantern_dist"),
        pl.when((pl.col("lantern_vtxX") - (-999)).abs() < 1e-2).then(np.nan).otherwise(pl.col("lantern_pandora_dist")).alias("lantern_pandora_dist"),
    ])

    return df


def remove_vector_variables(df):

    print("removing vector variables...")

    def get_vector_component(x, index):
        """Extract component from vector, handling float/NaN values."""
        if isinstance(x, (float, np.floating)) or not hasattr(x, '__getitem__'):
            return np.nan
        return x[index]

    df["wc_reco_muonMomentum_0"] = df["wc_reco_muonMomentum"].apply(lambda x: get_vector_component(x, 0))
    df["wc_reco_muonMomentum_1"] = df["wc_reco_muonMomentum"].apply(lambda x: get_vector_component(x, 1))
    df["wc_reco_muonMomentum_2"] = df["wc_reco_muonMomentum"].apply(lambda x: get_vector_component(x, 2))
    df["wc_reco_muonMomentum_3"] = df["wc_reco_muonMomentum"].apply(lambda x: get_vector_component(x, 3))

    df["wc_reco_showerMomentum_0"] = df["wc_reco_showerMomentum"].apply(lambda x: get_vector_component(x, 0))
    df["wc_reco_showerMomentum_1"] = df["wc_reco_showerMomentum"].apply(lambda x: get_vector_component(x, 1))
    df["wc_reco_showerMomentum_2"] = df["wc_reco_showerMomentum"].apply(lambda x: get_vector_component(x, 2))
    df["wc_reco_showerMomentum_3"] = df["wc_reco_showerMomentum"].apply(lambda x: get_vector_component(x, 3))

    if "wc_truth_muonMomentum" in df.columns:
        df["wc_truth_muonMomentum_0"] = df["wc_truth_muonMomentum"].apply(lambda x: get_vector_component(x, 0))
        df["wc_truth_muonMomentum_1"] = df["wc_truth_muonMomentum"].apply(lambda x: get_vector_component(x, 1))
        df["wc_truth_muonMomentum_2"] = df["wc_truth_muonMomentum"].apply(lambda x: get_vector_component(x, 2))
        df["wc_truth_muonMomentum_3"] = df["wc_truth_muonMomentum"].apply(lambda x: get_vector_component(x, 3))

    save_columns = [col for col in df.columns if col not in vector_columns]
    return df[save_columns]


def add_1g1mu_rad_corr_events(df):
    """Append numuCC_rad_corrected events derived from delete_one_gamma_overlay.

    Accepts either an eager DataFrame or a LazyFrame.

    - Eager DataFrame: returns the full df with rad-corrected rows appended.
    - LazyFrame: uses predicate pushdown to collect *only* the new rad-corrected
      rows (small, cheap) and returns them as an eager DataFrame.  The caller is
      responsible for concatenating these rows with the full df after calling
      pl.read_parquet separately.
    """
    print("adding 1g1mu rad correction events to the dataframe")

    is_lazy = isinstance(df, pl.LazyFrame)

    # Weights are pre-computed in rad_corrections_reweighting.ipynb and saved to parquet.
    # Columns: run, subrun, event, fix_del1g_weight, x_eta_uniform_weight, rad_frac_x_eta,
    #          wc_muon_gamma_opening_angle
    # Only events with all weights > 0 are present in the parquet.
    # Use scan_parquet so the weights are read lazily as part of the same query plan.
    rad_weights_lf = pl.scan_parquet(
        f"{intermediate_files_location}/del1g_rad_correction_weights.parquet"
    )
    print("  rad_weights parquet queued for lazy scan")

    lf = df if is_lazy else df.lazy()

    # Drop any stale weight columns that may already exist on the main df (e.g.
    # from a previous run whose temp parquet was not cleaned up) to avoid column
    # naming collisions when the join suffix is applied.
    # Note: wc_muon_gamma_opening_angle comes from the weights parquet and should
    # NOT be in the main df before this step (we no longer add it as a null column
    # to the full df since that caused a ~44 GB memory spike).
    stale = [c for c in ["fix_del1g_weight", "x_eta_uniform_weight",
                          "rad_frac_x_eta", "wc_muon_gamma_opening_angle"]
             if c in lf.collect_schema().names()]
    if stale:
        print(f"  WARNING: dropping stale weight columns before join: {stale}")
        lf = lf.drop(stale)

    rad_corrected_lf = (
        lf.filter(
            (pl.col("filetype") == "delete_one_gamma_overlay") &
            (pl.col("wc_truth_muonMomentum_3") > 0.0)
        )
        # inner join: events not in the weights parquet (invalid weights) are dropped.
        # No suffix needed: wc_muon_gamma_opening_angle only exists in the weights parquet
        # (we dropped it from the main df above if it was stale).
        .join(rad_weights_lf, on=["run", "subrun", "event"], how="inner")
        .with_columns([
            (pl.col("wc_net_weight")
             * pl.col("fix_del1g_weight")
             * pl.col("x_eta_uniform_weight")
             * pl.col("rad_frac_x_eta")
            ).cast(pl.Float32).alias("wc_net_weight"),
            pl.lit("numuCC_rad_corrected").alias("filetype"),
            pl.col("wc_muon_gamma_opening_angle").cast(pl.Float32),
            pl.lit(False).alias("normal_overlay"),
            pl.lit(False).alias("del1g_overlay"),
            pl.lit(False).alias("iso1g_overlay"),
        ])
        .drop(["fix_del1g_weight", "x_eta_uniform_weight", "rad_frac_x_eta"])
    )

    if is_lazy:
        # Collect only the small filtered sub-df (predicate pushdown reads only
        # delete_one_gamma_overlay rows from the parquet scan).  Caller concats
        # these new rows with the full df after reading it separately.
        print("  collecting rad_corrected rows (lazy mode)...")
        rad_corrected_df = rad_corrected_lf.collect()
        print(f"  collected {rad_corrected_df.height} rad-corrected rows")
        return rad_corrected_df

    # Eager path: concat immediately and return full df.
    print("  starting concat...")
    rad_corrected_df = rad_corrected_lf.collect()
    df = pl.concat([df, rad_corrected_df], how="diagonal_relaxed")
    del rad_corrected_df
    gc.collect()
    return df

def add_nc_coh_1g_reweighted_events(df):
    """Append NC_coherent_1g_reweighted events derived from isotropic_one_gamma_overlay.

    Accepts either an eager DataFrame or a LazyFrame.

    - Eager DataFrame: returns the full df with coherent rows appended.
    - LazyFrame: uses predicate pushdown to collect *only* the new coherent rows
      (small, cheap) and returns them as an eager DataFrame.  The caller is
      responsible for concatenating these rows with the full df after calling
      pl.read_parquet separately.
    """
    print("adding NC Coherent 1g reweighted events to the dataframe")

    is_lazy = isinstance(df, pl.LazyFrame)

    # Weights are pre-computed in coherent_1g_reweighting.ipynb and saved to parquet.
    # Columns: run, subrun, event, coherent_1g_weight, coherent_1g_keep
    # Only events with coherent_1g_weight > 0 are present in the parquet.
    # Use scan_parquet so the weights are read lazily as part of the same query plan.
    coherent_weights_lf = pl.scan_parquet(
        f"{intermediate_files_location}/coherent_1g_reweighting_weights.parquet"
    )
    print("  coherent_weights parquet queued for lazy scan")

    lf = df if is_lazy else df.lazy()

    coherent_1g_lf = (
        lf.filter(pl.col("filetype") == "isotropic_one_gamma_overlay")
        # inner join: events in empty coherent bins are dropped
        .join(coherent_weights_lf, on=["run", "subrun", "event"], how="inner")
        .with_columns([
            (pl.col("coherent_1g_weight"))
              .cast(pl.Float32).alias("wc_net_weight"),
            pl.lit("NC_coherent_1g_reweighted").alias("filetype"),
            pl.lit(False).alias("normal_overlay"),
            pl.lit(False).alias("del1g_overlay"),
            pl.lit(False).alias("iso1g_overlay"),
        ])
        .drop("coherent_1g_weight")
    )

    if is_lazy:
        # Collect only the small filtered sub-df (predicate pushdown reads only
        # isotropic_one_gamma_overlay rows from the parquet scan).  Caller concats
        # these new rows with the full df after reading it separately.
        # coherent_1g_keep is preserved from the weights parquet via the join above.
        print("  collecting coherent_1g rows (lazy mode)...")
        coherent_1g_df = coherent_1g_lf.collect()
        print(f"  collected {coherent_1g_df.height} coherent-1g rows")
        return coherent_1g_df

    # Eager path: concat immediately and return full df.
    # Append reweighted events; original iso1g rows remain but are excluded
    # downstream by the filter on "isotropic_one_gamma_overlay".
    # coherent_1g_keep comes from the weights parquet via the join above.
    # Use diagonal_relaxed so that the coherent_1g_keep column (only present in
    # coherent_1g_df) is filled with null for all original df rows, avoiding a
    # full df copy to add that column upfront.
    print("  starting concat...")
    coherent_1g_df = coherent_1g_lf.collect()
    df = pl.concat([df, coherent_1g_df], how="diagonal_relaxed")
    del coherent_1g_df
    gc.collect()
    return df
