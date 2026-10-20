import polars as pl
import numpy as np


def format_duration(seconds):
    """Format a duration in seconds as e.g. '2h 13m 05s', '3m 12s', or '42.1s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, sec = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


# DetVar CV groups.  The run 3b detector variations come in two separately generated
# sets, each with its own run 3b CV sample (whose run/subrun/event numbers mostly overlap
# with the other's):
#     main group:        LY, WireMod variations   <-> 13a + run 3b 1mil + run 4d/5 CVs
#     SCE/Recomb2 group: SCE, Recomb2 variations  <-> 13a + run 3b 500k + run 4d/5 CVs
# create_detvar_df.py writes one POT-normalized weight column per group, each normalized
# to only that group's CV POT and null on rows outside the group (the other group's run 3b
# CV, and the other group's variations).  CV rows shared by both groups carry both weights.
# Within one group the CV (filetype, run, subrun, event) keys are unique, so CV-variation
# matching only needs those keys plus the group's weight column.
DETVAR_SCE_RECOMB2_VARTYPES = ("SCE", "Recomb2")
DETVAR_WEIGHT_COL = "wc_net_weight"
DETVAR_SCE_RECOMB2_WEIGHT_COL = "wc_net_weight_sce_recomb2"
DETVAR_MATCH_KEYS = ["filetype", "run", "subrun", "event"]


def detvar_weight_col(vartype):
    """The DetVar weight column of the CV group that a variation type is matched within."""
    return DETVAR_SCE_RECOMB2_WEIGHT_COL if vartype in DETVAR_SCE_RECOMB2_VARTYPES else DETVAR_WEIGHT_COL


def lazy_height(lf):
    return lf.select(pl.len()).collect().item()


# Filetypes never used for BDT training (src/bdt_training_setup.py NON_TRAINING_FILETYPES is
# the union of these two lists):
#   data roles: real data and the NuWro fake data
#   whole-sample prediction filetypes: nothing was trained on them, so they need no test-half
#     selection and are kept whole (all events, no 1/frac_test upweighting)
BDT_DATA_ROLE_FILETYPES = ["data", "nuwro_fake_data"]
BDT_WHOLE_SAMPLE_FILETYPES = ["fullosc_overlay", "numucc_rad_corr_sim"]


def select_bdt_test_events(pred_df, weight_cols):
    """The prediction events usable with BDT scores: the BDT test-half events, with
    `weight_cols` scaled up by 1/frac_test so the total normalization is preserved, plus every
    BDT_WHOLE_SAMPLE_FILETYPES event, kept whole and unscaled.

    Whole-sample filetypes are recognized by filetype rather than by their split flags, so
    this is also correct for predictions.parquet files from trainings that still gave them
    train/test flags. Works on a DataFrame or a LazyFrame. Returns (selected_df, frac_test).
    """
    in_split = ~pl.col("filetype").is_in(BDT_WHOLE_SAMPLE_FILETYPES)
    counts = pred_df.filter(in_split).select(
        pl.col("used_for_training").sum().alias("n_train"),
        pl.col("used_for_testing").sum().alias("n_test"),
    )
    if isinstance(counts, pl.LazyFrame):
        counts = counts.collect()
    num_train, num_test = counts["n_train"][0], counts["n_test"][0]
    frac_test = num_test / (num_train + num_test)
    print(f"BDT train={num_train}, test={num_test} -> scaling test weights by 1/{frac_test:.4f}; "
          f"keeping {BDT_WHOLE_SAMPLE_FILETYPES} whole")

    is_test = in_split & pl.col("used_for_testing")
    selected = pred_df.with_columns([
        pl.when(is_test).then(pl.col(c) / frac_test).otherwise(pl.col(c)).alias(c)
        for c in weight_cols
    ]).filter(is_test | ~in_split)
    return selected, frac_test

def get_vals(df, var):
    if isinstance(df, pl.LazyFrame):
        if var == "(wc_flash_measPe - wc_flash_predPe) / wc_flash_predPe":
            collected = df.select(["wc_flash_measPe", "wc_flash_predPe"]).collect()
            vals = (collected.get_column("wc_flash_measPe") - collected.get_column("wc_flash_predPe")) / collected.get_column("wc_flash_predPe")
            return vals.to_numpy()
        elif var == "wc_WCPMTInfoChi2 / wc_WCPMTInfoNDF":
            collected = df.select(["wc_WCPMTInfoChi2", "wc_WCPMTInfoNDF"]).collect()
            vals = collected.get_column("wc_WCPMTInfoChi2") / collected.get_column("wc_WCPMTInfoNDF")
            return np.nan_to_num(vals.to_numpy(), nan=-1, posinf=-1, neginf=-1)
        else:
            return df.select(var).collect().get_column(var).to_numpy()
    if var == "(wc_flash_measPe - wc_flash_predPe) / wc_flash_predPe":
        vals = (df.get_column("wc_flash_measPe") - df.get_column("wc_flash_predPe")) / df.get_column("wc_flash_predPe")
        vals = vals.to_numpy()
    elif var == "wc_WCPMTInfoChi2 / wc_WCPMTInfoNDF":
        vals = df.get_column("wc_WCPMTInfoChi2") / df.get_column("wc_WCPMTInfoNDF")
        vals = np.nan_to_num(vals.to_numpy(), nan=-1, posinf=-1, neginf=-1)
    else:
        vals = df.get_column(var)
        vals = vals.to_numpy()
    return vals

def align_columns_for_concat(dfs):
    # Find all columns across all DataFrames
    all_cols = {col for df in dfs for col in df.columns}

    # Build a mapping of column -> dtype (prefer from first df that has it)
    dtype_map = {}
    for df in dfs:
        for col, dtype in zip(df.columns, df.dtypes):
            dtype_map.setdefault(col, dtype)

    aligned = []
    for i, df in enumerate(dfs):
        # Store original filetype values before any operations
        original_filetype = None
        if "filetype" in df.columns:
            original_filetype = df["filetype"].clone()
            # Validate filetype before processing
            empty_or_null_count_before = df.filter(
                (pl.col("filetype") == '') | pl.col("filetype").is_null()
            ).height
            if empty_or_null_count_before > 0:
                raise ValueError(f"DataFrame {i}: filetype has {empty_or_null_count_before} empty/null values BEFORE alignment!")
        
        missing = all_cols - set(df.columns)
        if missing:
            # Add missing columns as nulls, cast to the desired dtype
            # For string columns, use empty string "" as a placeholder that we'll check for
            defaults = []
            for c in missing:
                defaults.append(pl.lit(None).cast(dtype_map[c]).alias(c))
            df = df.with_columns(defaults)

        # Cast any existing columns whose dtype differs from the reference
        mismatched = [c for c in df.columns if df.schema[c] != dtype_map[c]]
        if mismatched:
            df = df.with_columns([pl.col(c).cast(dtype_map[c]) for c in mismatched])

        # Ensure consistent column order
        df = df.select(sorted(all_cols))
        
        # Restore and validate filetype column after all transformations
        if original_filetype is not None:
            # Check if filetype was corrupted during select operation
            empty_or_null_count_after = df.filter(
                (pl.col("filetype") == '') | pl.col("filetype").is_null()
            ).height
            if empty_or_null_count_after > 0:
                # Debug: print the first problematic row
                problem_row = df.filter(
                    (pl.col("filetype") == '') | pl.col("filetype").is_null()
                ).head(1)
                print(f"ERROR in align_columns_for_concat DataFrame {i}: Found {empty_or_null_count_after} rows with empty/null filetype AFTER column operations")
                print(f"Example row columns: {problem_row.columns[:10]}")
                if "filename" in problem_row.columns:
                    print(f"Example filename: {problem_row['filename'][0]}")
                print(f"Original filetype unique values: {original_filetype.unique().to_list()}")
                print(f"After transform filetype unique values: {df['filetype'].unique().to_list()}")
                raise ValueError(f"align_columns_for_concat DataFrame {i}: filetype column was corrupted during column alignment!")
        
        aligned.append(df)

    return aligned
