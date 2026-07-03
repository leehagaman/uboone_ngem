#!/usr/bin/env python
"""Write the PROfit input ROOT files from the processed dataframes.

This is the command-line, plot-free version of ipynb_notebooks/save_PROfit_rootfiles.ipynb.
It produces two kinds of output (both written with the PyROOT std::vector<double>
writer that PROfit's SetBranchAddress pattern expects):

  * nominal MC + data with GENIE spline weights ->  minimal_withspline_df.root
  * one detector-variation file per vartype     ->  minimal_detvar_<vartype>_df.root

The nominal prediction uses the runs 1-5 open-data POT weighting
(wc_net_weight_open_data) across all run periods.  (The notebook was hard-coded to
Run 4b because that was the only sample with spline weights; splines now exist for
every prediction file, so we use all runs.)

Usage:
    python src/save_PROfit_rootfiles.py                    # splines + detvar
    python src/save_PROfit_rootfiles.py --no-detvar        # nominal only
    python src/save_PROfit_rootfiles.py --training all_vars
"""

import argparse
import os
import time

import numpy as np
import polars as pl
import xgboost as xgb
from tqdm import tqdm

from file_locations import intermediate_files_location
from signal_categories import train_category_labels
from ntuple_variables.variables import combined_training_vars

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_TRAINING = "all_vars_r15"

# Runs 1-5 open-data POT weighting (all run periods) for the nominal prediction.
NET_WEIGHT_COL = "wc_net_weight_open_data"

# DetVar has its own single weighting config (create_detvar_df.py) whose column is
# "wc_net_weight"; the detvar covariance is a fractional (CV-var)/CV difference so the
# absolute normalization cancels.
DETVAR_NET_WEIGHT_COL = "wc_net_weight"

# The detector-variation samples PROfit expects (CV + the 7 variations used by the
# covariance).  Only these get a ROOT file; any other vartype value (e.g. the empty
# "" that create_detvar_df.py mislabels some events with) is skipped with a warning.
DETVAR_VARTYPES = ["CV", "LYAtt", "LYDown", "LYRayleigh", "WireModX", "WireModYZ", "Recomb2", "SCE"]

# The reco categories (and therefore the prob_<category> BDT-score columns) come from
# the training definition.
RECO_CATEGORIES = train_category_labels
TRAINING_VARS = combined_training_vars

# ---------------------------------------------------------------------------
# Variables saved to the output ROOT trees.
#
# OUTPUT_SCALAR_COLUMNS lists the non-spline scalar branches.  In addition the pipeline
# always appends, for the NOMINAL file:
#     prob_<category>                (one BDT score per reco category)
#     isdata, isext, isdirt          (filetype flags)
#     <NET_WEIGHT_COL>               (raw open-data net weight)
#     has_spline_weights, fraction_with_spline_weights, spline_processed_fraction_weight
#     net_weight                     (final weight = open-data weight x spline-fraction weight)
#     weightsReint + every GENIE spline-knob column (from spline_weights_df)
# and for each DETVAR file: filetype, vartype, run, subrun, event, isdata/isext/isdirt,
#     reco_category, wc_kine_reco_Enu, net_weight, prob_<category>.
#
# Edit this list to change which non-spline variables are written.
# ---------------------------------------------------------------------------
OUTPUT_SCALAR_COLUMNS = [
    "filetype",
    "run",
    "subrun",
    "event",
    "reco_category",
    "wc_kine_reco_Enu",
]

# reco_category selection.  For each category:  (prob threshold on prob_<category>,
# priority).  A threshold of None means the category is selected by the argmax of the
# prob_ columns instead of a fixed cut.  Lower priority number wins; each category's
# mask additionally excludes every higher-priority category (orthogonalization).
RECO_CATEGORY_THRESHOLDS = {
    "1gNp": (0.3, 1),
    "1g0p": (0.9, 2),
    "1gNp1mu": (0.5, 3),
    "1g0p1mu": (0.2, 4),
    "1g_outFV": (0.5, 5),
    "pi0_dalitz_decay": (0.1, 5.5),   # high priority for the rare Dalitz topology
    "NC1pi0_Np": (None, 6),
    "NC1pi0_0p": (None, 7),
    "numuCC1pi0_0p": (0.15, 8),       # 0p takes priority over Np for orthogonality
    "numuCC1pi0_Np": (0.1, 9),
    "1pi0_outFV": (0.1, 10),
    "nueCC_0p": (0.05, 11),           # 0p takes priority over Np
    "nueCC_Np": (0.05, 12),
    "numuCC_Np": (0.5, 13),
    "numuCC_0p": (0.5, 14),
    "other_outFV_dirt": (None, 15),
    "multi_pi0": (0.02, 16),
    "eta_other": (0.01, 17),
    "NC_no_gamma": (None, 19),
    "ext": (None, 20),
}

# Batch size for the DetVar BDT inference.
DETVAR_INFERENCE_BATCH_SIZE = 100_000


# ============================================================================
# Helpers
# ============================================================================

def _prob_cols():
    return [f"prob_{c}" for c in RECO_CATEGORIES]


def build_reco_category_exprs():
    """Return one polars mask expression per reco category (in RECO_CATEGORIES order).

    Reproduces the notebook's priority/orthogonalization logic: build each category's
    raw mask (a prob cut, or the argmax of the prob_ columns when the threshold is
    None), sort by priority, then AND each mask with the negation of every
    higher-priority raw mask so the categories are mutually exclusive.
    """
    missing = set(RECO_CATEGORIES) - set(RECO_CATEGORY_THRESHOLDS)
    if missing:
        raise ValueError(f"RECO_CATEGORY_THRESHOLDS is missing categories: {sorted(missing)}")

    argmax_query = {cat: (pl.col("reco_category_argmax_index") == i)
                    for i, cat in enumerate(RECO_CATEGORIES)}

    # (name, raw_mask, priority)
    triples = []
    for cat in RECO_CATEGORIES:
        threshold, priority = RECO_CATEGORY_THRESHOLDS[cat]
        raw = argmax_query[cat] if threshold is None else (pl.col(f"prob_{cat}") > threshold)
        triples.append((cat, raw, priority))
    triples.sort(key=lambda t: t[2])

    ortho = {}
    for i, (name, raw, _priority) in enumerate(triples):
        expr = raw
        for j in range(i):
            expr = expr & ~triples[j][1]   # exclude every higher-priority raw mask
        ortho[name] = expr

    return [ortho[cat] for cat in RECO_CATEGORIES]


def _reco_category_expr():
    """when/then chain assigning each event its reco category index (else null)."""
    exprs = build_reco_category_exprs()
    expr = pl.when(exprs[0]).then(0)
    for i in range(1, len(exprs)):
        expr = expr.when(exprs[i]).then(i)
    return expr.otherwise(None).cast(pl.Int32)


# ROOT branch format (kept identical to the notebook's proven PyROOT writer): scalar
# columns become POD branches; List[float] columns become std::vector<double> object
# branches (what PROfit's SetBranchAddress expects); integer columns that contain nulls
# fall through to float (NaN), matching the earlier uproot-based writer.
_INT_DTYPES = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}


def _column_kind(dtype, has_nulls):
    if isinstance(dtype, pl.List):
        return "list"
    if dtype in (pl.String, pl.Utf8):
        return "str"
    if dtype == pl.Boolean:
        return "bool"
    if dtype in _INT_DTYPES and not has_nulls:
        return "int"
    return "float"


def _bind_branches(tree, kinds, ROOT):
    buffers = {}
    for col, kind in kinds.items():
        if kind == "list":
            buf = ROOT.std.vector("double")()
            tree.Branch(col, buf)
        elif kind == "str":
            buf = ROOT.std.string()
            tree.Branch(col, buf)
        elif kind == "bool":
            buf = np.zeros(1, dtype=np.bool_)
            tree.Branch(col, buf, f"{col}/O")
        elif kind == "int":
            buf = np.zeros(1, dtype=np.int32)
            tree.Branch(col, buf, f"{col}/I")
        else:
            buf = np.zeros(1, dtype=np.float64)
            tree.Branch(col, buf, f"{col}/D")
        buffers[col] = buf
    return buffers


def _extract_columns(df, kinds):
    columns = {}
    for col, kind in kinds.items():
        s = df[col]
        if kind == "list":
            columns[col] = s.fill_null([]).to_list()
        elif kind == "str":
            columns[col] = s.fill_null("").to_list()
        elif kind == "bool":
            columns[col] = s.fill_null(False).to_numpy()
        elif kind == "int":
            columns[col] = s.to_numpy().astype(np.int32)
        else:
            columns[col] = s.to_numpy().astype(np.float64)
    return columns


def _fill_rows(tree, n, columns, buffers, kinds, desc):
    for i in tqdm(range(n), desc=desc):
        for col, kind in kinds.items():
            buf = buffers[col]
            v = columns[col][i]
            if kind == "list":
                buf.clear()
                if v is not None:
                    for x in v:
                        buf.push_back(float(x))
            elif kind == "str":
                buf.assign(str(v) if v is not None else "")
            elif kind == "bool":
                buf[0] = bool(v)
            else:
                buf[0] = v
        tree.Fill()


def _import_root():
    """Import ROOT and remove its signal handlers.  ROOT installs SIGSEGV/SIGBUS handlers
    that intercept benign signals from polars' Rust threads/deallocation and turn them
    into spurious 'crashes'; resetting them lets polars and ROOT coexist."""
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gSystem.ResetSignals()
    return ROOT


def write_df_to_root(df_to_save, output_path, desc="writing"):
    """Write an eager polars DataFrame to a ROOT TTree named 'tree' (used for the small
    per-vartype detvar files, which have no huge list columns)."""
    ROOT = _import_root()  # imported lazily: heavy, and only needed when actually writing
    kinds = {c: _column_kind(df_to_save.schema[c], df_to_save[c].null_count() > 0)
             for c in df_to_save.columns}
    f = ROOT.TFile.Open(output_path, "RECREATE")
    tree = ROOT.TTree("tree", "tree")
    buffers = _bind_branches(tree, kinds, ROOT)
    columns = _extract_columns(df_to_save, kinds)
    _fill_rows(tree, df_to_save.height, columns, buffers, kinds, desc)
    tree.Write()
    f.Close()
    print(f"  wrote {df_to_save.height} events to {output_path}")


def _spline_list_lengths(spline_path):
    """Return (fixed_len, head_len) for every list column of the spline parquet.

    fixed_len[c] is the common length if every row agrees (enables the memcpy fast
    path), else None.  head_len[c] is the first row's length, used to build the
    data/ext unit spline lists (same source of truth as the old lit([1.0]*len))."""
    schema = pl.scan_parquet(spline_path).collect_schema()
    list_cols = [c for c, dt in schema.items() if isinstance(dt, pl.List)]
    if not list_cols:
        return {}, {}
    fixed_len, head_len = {}, {}
    heads = pl.scan_parquet(spline_path).select(
        [pl.col(c).list.len().alias(c) for c in list_cols]).head(1).collect()
    # One small query per column.  Do NOT fold these into a single giant select: with
    # ~144 list.len() aggregations in one streaming select, polars 1.34 silently returns
    # garbage min/max (observed weightsReint min=0 max=39696000 when every row is
    # exactly 1000), which demotes fixed-length columns to the slow push_back path.
    for c in list_cols:
        agg = pl.scan_parquet(spline_path).select(
            mn=pl.col(c).list.len().min(), mx=pl.col(c).list.len().max()
        ).collect(engine="streaming")
        mn, mx = agg["mn"][0], agg["mx"][0]
        fixed_len[c] = int(mn) if (mn is not None and mn == mx and mn > 0) else None
        head_len[c] = int(heads[c][0])
    n_fixed = sum(v is not None for v in fixed_len.values())
    print(f"  {n_fixed}/{len(list_cols)} list columns use the fixed-length memcpy fast path", flush=True)
    return fixed_len, head_len


def _bind_branches_fast(tree, out_cols, kinds, fixed_len, ROOT):
    """Bind one branch per output column.  Fixed-length list columns get a fixed-size
    std::vector plus a numpy view onto its buffer, so each row is filled with a single
    memcpy (view[:] = row) instead of ~1000 Python<->C++ push_back calls (the original
    per-element writer was billions of calls for the 1000-universe weightsReint).
    Variable-length list columns fall back to push_back."""
    buffers, views = {}, {}
    for col in out_cols:
        kind = kinds[col]
        if kind == "list":
            L = fixed_len.get(col)
            if L is not None:
                buf = ROOT.std.vector("double")(L)
                tree.Branch(col, buf)
                views[col] = np.frombuffer(buf.data(), dtype=np.float64, count=L)
            else:
                buf = ROOT.std.vector("double")()
                tree.Branch(col, buf)
        elif kind == "str":
            buf = ROOT.std.string()
            tree.Branch(col, buf)
        elif kind == "bool":
            buf = np.zeros(1, dtype=np.bool_)
            tree.Branch(col, buf, f"{col}/O")
        elif kind == "int":
            buf = np.zeros(1, dtype=np.int32)
            tree.Branch(col, buf, f"{col}/I")
        else:
            buf = np.zeros(1, dtype=np.float64)
            tree.Branch(col, buf, f"{col}/D")
        buffers[col] = buf
    return buffers, views


def _fill_root_batch(tree, batch, out_cols, kinds, fixed_len, buffers, views, desc):
    """Fill `tree` from an eager polars DataFrame holding every column in out_cols."""
    # pre-extract each column once for the whole batch (tag drives the per-row fill)
    data = {}
    for col in out_cols:
        kind = kinds[col]
        s = batch[col]
        if kind == "list":
            L = fixed_len.get(col)
            if L is not None:
                # (n, L) contiguous float64 -- one Arrow->numpy conversion for the batch
                data[col] = ("list_fixed", s.list.to_array(L).to_numpy())
            else:
                data[col] = ("list_var", s.fill_null([]).to_list())
        elif kind == "str":
            data[col] = ("str", s.fill_null("").to_list())
        elif kind == "bool":
            data[col] = ("bool", s.fill_null(False).to_numpy())
        elif kind == "int":
            data[col] = ("int", s.to_numpy().astype(np.int32))
        else:
            data[col] = ("float", s.to_numpy().astype(np.float64))

    for i in tqdm(range(batch.height), desc=desc):
        for col in out_cols:
            tag, arr = data[col]
            if tag == "list_fixed":
                views[col][:] = arr[i]            # single memcpy into the vector buffer
            elif tag == "list_var":
                buf = buffers[col]
                buf.clear()
                v = arr[i]
                if v is not None:
                    for x in v:
                        buf.push_back(float(x))
            elif tag == "str":
                buffers[col].assign(str(arr[i]))
            elif tag == "bool":
                buffers[col][0] = bool(arr[i])
            else:
                buffers[col][0] = arr[i]
        tree.Fill()


def write_withspline_root(mc_df, data_df, spline_path, output_path, batch_size=131_072):
    """Write the nominal minimal_withspline tree with bounded memory.

    Instead of sinking the whole MC<->spline join to one intermediate parquet (whose
    1000-wide weightsReint/piplus/piminus list columns are ~40 GB and drove polars'
    streaming sink past 100 GB RSS on runs 1-5), stream the 19 GB spline parquet in
    row batches: join each batch against the small in-memory MC table (scalars only),
    compute net_weight, and fill the ROOT tree directly.  Peak memory is one spline
    batch plus the MC table, independent of the total spline size.  data/ext rows are
    appended afterwards with constant unit spline branches.

    mc_df must already carry fraction_with_spline_weights / spline_processed_fraction_weight
    (from the per-filetype fractions join); data_df is the untouched data/ext minimal df.
    """
    import pyarrow.parquet as pq

    ROOT = _import_root()
    keys = ["filetype", "run", "subrun", "event"]

    spline_schema = pl.scan_parquet(spline_path).collect_schema()
    fixed_len, head_len = _spline_list_lengths(spline_path)

    # output columns: full spline schema, then the MC-side scalars, then the flag/weight
    mc_extra_cols = [c for c in mc_df.columns if c not in keys]
    out_cols = list(spline_schema.names()) + ["has_spline_weights"] + mc_extra_cols + ["net_weight"]

    # branch kinds; integer columns that contain nulls anywhere fall through to float
    # (NaN), matching write_df_to_root / the earlier uproot-based writer.
    kinds = {}
    spline_int_cols = [c for c, dt in spline_schema.items()
                       if not isinstance(dt, pl.List) and dt in _INT_DTYPES]
    # one query per column (see _spline_list_lengths for why we avoid wide multi-agg
    # selects on streaming scans: polars 1.34 can return silently wrong values under load)
    spline_nulls = {}
    for c in spline_int_cols:
        n = pl.scan_parquet(spline_path).select(pl.col(c).null_count()).collect(engine="streaming").item()
        spline_nulls[c] = n > 0
    for c, dt in spline_schema.items():
        kinds[c] = _column_kind(dt, spline_nulls.get(c, False))
    kinds["has_spline_weights"] = "bool"
    for c in mc_extra_cols:
        has_nulls = (mc_df[c].null_count() > 0) or (c in data_df.columns and data_df[c].null_count() > 0)
        kinds[c] = _column_kind(mc_df.schema[c], has_nulls)
    kinds["net_weight"] = "float"

    pf = pq.ParquetFile(spline_path)
    n_spline = pf.metadata.num_rows
    n_batches = -(-n_spline // batch_size)
    print(f"  streaming {n_spline} spline rows in {n_batches} batches "
          f"(joining {mc_df.height} MC events)...", flush=True)

    f = ROOT.TFile.Open(output_path, "RECREATE")
    tree = ROOT.TTree("tree", "tree")
    buffers, views = _bind_branches_fast(tree, out_cols, kinds, fixed_len, ROOT)

    # ---- MC: spline batch (probe) x in-memory MC scalars (hash side) ----
    n_mc_written = 0
    for b, rb in enumerate(pf.iter_batches(batch_size=batch_size)):
        batch = (
            pl.from_arrow(rb)
            .join(mc_df, on=keys, how="inner")
            .with_columns([
                pl.lit(True).alias("has_spline_weights"),
                (pl.col(NET_WEIGHT_COL) * pl.col("spline_processed_fraction_weight")).alias("net_weight"),
            ])
            .select(out_cols)
        )
        n_mc_written += batch.height
        _fill_root_batch(tree, batch, out_cols, kinds, fixed_len, buffers, views,
                         f"nominal MC batch {b + 1}/{n_batches}")

    # ---- data/ext: constant spline branches, per-row scalars ----
    # constants (same values the old writer produced from the lit() columns): unit
    # spline lists, empty strings for the spline-only string columns, has_spline_weights
    # True, unit fraction weights.
    data_cols = set(data_df.columns)
    for c in out_cols:
        if c in data_cols or c == "net_weight":
            continue
        kind = kinds[c]
        if kind == "list":
            L = fixed_len.get(c)
            if L is not None:
                views[c][:] = 1.0
            else:
                buffers[c].clear()
                for _ in range(head_len[c]):
                    buffers[c].push_back(1.0)
        elif kind == "str":
            buffers[c].assign("")
        elif kind == "bool":
            buffers[c][0] = (c == "has_spline_weights")
        elif kind == "int":
            buffers[c][0] = 0
        else:
            buffers[c][0] = 1.0 if c in ("fraction_with_spline_weights",
                                         "spline_processed_fraction_weight") else np.nan

    update_cols = [c for c in out_cols if c in data_cols]
    data_extract = {}
    for c in update_cols:
        kind = kinds[c]
        s = data_df[c]
        if kind == "str":
            data_extract[c] = ("str", s.fill_null("").to_list())
        elif kind == "bool":
            data_extract[c] = ("bool", s.fill_null(False).to_numpy())
        elif kind == "int":
            data_extract[c] = ("int", s.to_numpy().astype(np.int32))
        else:
            data_extract[c] = ("float", s.to_numpy().astype(np.float64))
    net = data_df[NET_WEIGHT_COL].to_numpy().astype(np.float64)

    for i in tqdm(range(data_df.height), desc="nominal data/ext"):
        for c in update_cols:
            tag, arr = data_extract[c]
            if tag == "str":
                buffers[c].assign(str(arr[i]))
            elif tag == "bool":
                buffers[c][0] = bool(arr[i])
            else:
                buffers[c][0] = arr[i]
        buffers["net_weight"][0] = net[i]   # data/ext net_weight = the open-data weight
        tree.Fill()

    tree.Write()
    f.Close()
    total = n_mc_written + data_df.height
    print(f"  wrote {total} events ({n_mc_written} MC + {data_df.height} data/ext) to {output_path}")


# ============================================================================
# Nominal MC + data (with spline weights)
# ============================================================================

def build_minimal_df(training):
    """Return the lazy minimal nominal df (MC test events + data), scored and with a
    reco_category, before spline merging."""
    prob_cols = _prob_cols()

    all_df = pl.scan_parquet(f"{intermediate_files_location}/all_df.parquet")
    preds = pl.scan_parquet(f"{PROJECT_ROOT}/training_outputs/{training}/predictions.parquet")
    merged = all_df.join(preds, on=["filetype", "run", "subrun", "event"], how="left")

    # BDT scores: fill missing with -1, then the per-event argmax over the prob columns
    merged = merged.with_columns([pl.col(p).fill_null(-1) for p in prob_cols])
    merged = merged.with_columns(
        pl.concat_list(prob_cols).list.arg_max().alias("reco_category_argmax_index")
    )

    # prediction (drop raw 1g overlays, kept as their reweighted filetypes) vs real data
    pred = merged.filter(~pl.col("filetype").is_in(
        ["data", "isotropic_one_gamma_overlay", "delete_one_gamma_overlay"]))
    data = merged.filter(pl.col("filetype") == "data")

    # generic preselection + only events with a valid open-data weight
    pred = pred.filter((pl.col("wc_kine_reco_Enu") > 0) & pl.col(NET_WEIGHT_COL).is_not_null())

    # Use only test events (the BDT trained on the train half), weighted up by
    # 1/frac_test so the total normalization is preserved.  Both counts in one pass.
    counts = pred.select([
        pl.col("used_for_training").sum().alias("n_train"),
        pl.col("used_for_testing").sum().alias("n_test"),
    ]).collect()
    num_train, num_test = counts["n_train"][0], counts["n_test"][0]
    frac_test = num_test / (num_train + num_test)
    print(f"  train={num_train}, test={num_test} -> scaling test weights by 1/{frac_test:.4f}")
    pred = pred.with_columns(
        pl.when(pl.col("used_for_testing"))
        .then(pl.col(NET_WEIGHT_COL) / frac_test)
        .otherwise(pl.col(NET_WEIGHT_COL))
        .alias(NET_WEIGHT_COL)
    ).filter(pl.col("used_for_testing"))

    data = data.filter(pl.col("wc_kine_reco_Enu") > 0)

    combined = pl.concat([pred, data], how="vertical")
    combined = combined.with_columns(_reco_category_expr().alias("reco_category"))

    minimal = combined.select(OUTPUT_SCALAR_COLUMNS + [NET_WEIGHT_COL] + prob_cols).with_columns([
        (pl.col("filetype") == "data").alias("isdata"),
        (pl.col("filetype") == "ext").alias("isext"),
        (pl.col("filetype") == "dirt_overlay").alias("isdirt"),
    ])
    return minimal


def compute_spline_fractions(mc_df, spline_path):
    """Per-filetype fraction of MC events that carry spline weights, and its inverse
    (the weight that scales the surviving events back up to the full normalization).
    Keys-only left join against the spline parquet, so no list columns are read."""
    keys = ["filetype", "run", "subrun", "event"]
    spline_keys = pl.scan_parquet(spline_path).select(keys).with_columns(pl.lit(True).alias("_matched"))
    return (
        mc_df.lazy().select(keys)
        .join(spline_keys, on=keys, how="left")
        .group_by("filetype")
        .agg([pl.len().alias("num_events"), pl.col("_matched").sum().alias("num_with_spline_weights")])
        .with_columns((pl.col("num_with_spline_weights") / pl.col("num_events")).alias("fraction_with_spline_weights"))
        .with_columns((1.0 / pl.col("fraction_with_spline_weights")).alias("spline_processed_fraction_weight"))
        .select(["filetype", "fraction_with_spline_weights", "spline_processed_fraction_weight"])
        .collect(engine="streaming")
    )


def save_nominal(training, output_dir):
    """Nominal MC + data with spline weights -> minimal_withspline_df.root.

    The minimal df (30 scalar columns) is small enough to hold in memory; only the
    spline parquet with its 1000-wide list columns is big, and write_withspline_root
    streams that in bounded batches.  MC events without spline weights are dropped
    (inner join) and each filetype is weighted up by 1/fraction_with_spline_weights,
    exactly as the old lazy merge_splines/sink pipeline did -- but without ever
    materializing the ~40 GB joined output, which drove polars' streaming sink past
    100 GB RSS on runs 1-5."""
    spline_path = f"{intermediate_files_location}/spline_weights_df.parquet"

    print("Building nominal minimal df (runs 1-5 open-data weighting)...")
    minimal_df = build_minimal_df(training)
    mc_df = minimal_df.filter(~pl.col("filetype").is_in(["data", "ext"])).collect(engine="streaming")
    data_df = minimal_df.filter(pl.col("filetype").is_in(["data", "ext"])).collect(engine="streaming")
    print(f"  {mc_df.height} MC events, {data_df.height} data/ext events")

    print("Merging spline weights...")
    fractions = compute_spline_fractions(mc_df, spline_path)
    for row in fractions.sort("filetype").iter_rows(named=True):
        frac = row["fraction_with_spline_weights"]
        print(f"    {row['filetype']}: fraction_with_spline_weights = "
              f"{frac if frac is None else f'{frac:.4f}'}")
    mc_df = mc_df.join(fractions, on="filetype", how="left")

    output_path = f"{output_dir}/minimal_withspline_df.root"
    write_withspline_root(mc_df, data_df, spline_path, output_path)


# ============================================================================
# Detector variations
# ============================================================================

def save_detvar(training, output_dir):
    print("Building DetVar minimal dfs...")
    prob_cols = _prob_cols()

    # Collect only the columns we need (ids + weight + inference vars), so the BDT
    # scores can be attached by position and we never hold the full detvar df.
    # dict.fromkeys dedups columns that are both an explicit id/weight and a training var
    # (e.g. wc_kine_reco_Enu is in TRAINING_VARS), which .select would reject as duplicate.
    keep = list(dict.fromkeys(
        ["filetype", "vartype", "run", "subrun", "event", "wc_kine_reco_Enu", DETVAR_NET_WEIGHT_COL]
        + TRAINING_VARS))
    presel = (
        pl.scan_parquet(f"{intermediate_files_location}/detvar_presel_df_train_vars.parquet")
        .select(keep)
        .collect()
    )
    print(f"  {presel.height} detvar events")

    model = xgb.XGBClassifier()
    model.load_model(f"{PROJECT_ROOT}/training_outputs/{training}/bdt.json")

    # batched inference over the training variables
    probs = []
    for start in tqdm(range(0, presel.height, DETVAR_INFERENCE_BATCH_SIZE), desc="detvar inference"):
        x = presel.select(TRAINING_VARS).slice(start, DETVAR_INFERENCE_BATCH_SIZE).to_numpy().astype(np.float64)
        x[np.isinf(x)] = np.nan
        probs.append(model.predict_proba(x))
    probs = np.vstack(probs)

    presel = presel.with_columns([pl.Series(prob_cols[i], probs[:, i]) for i in range(len(prob_cols))])
    presel = presel.with_columns([pl.col(p).fill_null(-1) for p in prob_cols])
    presel = presel.with_columns(
        pl.concat_list(prob_cols).list.arg_max().alias("reco_category_argmax_index")
    )
    presel = presel.with_columns(_reco_category_expr().alias("reco_category"))
    presel = presel.with_columns([
        (pl.col("filetype") == "data").alias("isdata"),
        (pl.col("filetype") == "ext").alias("isext"),
        (pl.col("filetype") == "dirt_overlay").alias("isdirt"),
    ]).rename({DETVAR_NET_WEIGHT_COL: "net_weight"})

    detvar_minimal = presel.select(
        ["filetype", "vartype", "run", "subrun", "event", "isdata", "isext", "isdirt",
         "reco_category", "wc_kine_reco_Enu", "net_weight"] + prob_cols
    )

    present = detvar_minimal["vartype"].unique().to_list()
    unexpected = [v for v in present if v not in DETVAR_VARTYPES]
    if unexpected:
        counts = detvar_minimal.filter(pl.col("vartype").is_in(unexpected)).group_by("vartype").agg(pl.len().alias("n"))
        print(f"  WARNING: skipping {counts.select(pl.col('n').sum()).item()} events with unexpected vartype(s) "
              f"{counts.to_dicts()} -- not writing ROOT files for them (likely mislabeled in create_detvar_df.py)")

    for vartype in DETVAR_VARTYPES:
        df_to_save = detvar_minimal.filter(pl.col("vartype") == vartype)
        if df_to_save.height == 0:
            print(f"  WARNING: no events for detvar vartype '{vartype}'; skipping")
            continue
        output_path = f"{output_dir}/minimal_detvar_{vartype}_df.root"
        write_df_to_root(df_to_save, output_path, desc=f"detvar {vartype}")


# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Write PROfit input ROOT files from the processed dataframes.")
    parser.add_argument("--training", default=DEFAULT_TRAINING,
                        help=f"training_outputs/<name> to read predictions + BDT from (default: {DEFAULT_TRAINING})")
    parser.add_argument("--output-dir", default=intermediate_files_location,
                        help="directory to write the ROOT files into (default: intermediate_files_location)")
    parser.add_argument("--no-splines", action="store_true", help="skip the nominal MC+data spline ROOT file")
    parser.add_argument("--no-detvar", action="store_true", help="skip the per-vartype detvar ROOT files")
    args = parser.parse_args()

    start = time.time()
    if not args.no_splines:
        save_nominal(args.training, args.output_dir)
    if not args.no_detvar:
        save_detvar(args.training, args.output_dir)
    print(f"Done in {time.time() - start:.1f} s", flush=True)
    # All ROOT files are written and closed; skip Python/polars teardown, which can
    # segfault while ROOT is loaded (ROOT's signal handlers vs polars' Rust threads).
    os._exit(0)


if __name__ == "__main__":
    main()
