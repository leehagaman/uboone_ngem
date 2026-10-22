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


# BDT bookkeeping columns, set in src/bdt_training_setup.py (add_bdt_role_columns, the one place
# that decides how each event is used) and written to predictions.parquet by train.py:
#   bdt_role: "train", "validation", "test", or "not_trained" (see bdt_training_setup)
#   bdt_prediction_weight_factor: what to multiply an event's weights by when it enters a
#     prediction that uses BDT scores: 0 for train/validation events (their scores are biased),
#     1/(test fraction) for test events, and 1 for not_trained events (kept whole)


def select_bdt_test_events(pred_df, weight_cols):
    """The prediction events usable with BDT scores, with `weight_cols` multiplied by
    bdt_prediction_weight_factor: the test events, scaled up so the total normalization is
    preserved, and the not_trained events, kept whole.

    Every row needs a BDT prediction (pred_df joined with a predictions.parquet from a training
    that writes bdt_prediction_weight_factor); raises otherwise. Works on a DataFrame or a
    LazyFrame.
    """
    schema = pred_df.collect_schema()
    if "bdt_prediction_weight_factor" not in schema.names():
        raise ValueError("no bdt_prediction_weight_factor column: join a predictions.parquet from a "
                         "training made after the bdt_role refactor (2026-10-22)")
    factor = pl.col("bdt_prediction_weight_factor")

    summary = pred_df.group_by("bdt_role", factor).agg(pl.len()).sort("bdt_role", nulls_last=True)
    if isinstance(summary, pl.LazyFrame):
        summary = summary.collect()
    if summary["bdt_prediction_weight_factor"].is_null().any():
        n_null = summary.filter(factor.is_null())["len"].sum()
        raise ValueError(f"{n_null} rows have no BDT prediction (bdt_prediction_weight_factor is null)")
    print("BDT events: " + ", ".join(
        f"{row['bdt_role']}={row['len']} (x{row['bdt_prediction_weight_factor']:g})"
        for row in summary.iter_rows(named=True)))

    # cast back so e.g. Float32 weight columns stay Float32
    return pred_df.filter(factor > 0).with_columns([(pl.col(c) * factor).cast(schema[c]).alias(c) for c in weight_cols])

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
