#!/usr/bin/env python
"""Write the PROfit input ROOT files from the processed dataframes.

This is the command-line, plot-free version of ipynb_notebooks/save_PROfit_rootfiles.ipynb.
It produces two kinds of output (both written with the PyROOT std::vector<double>
writer that PROfit's SetBranchAddress pattern expects):

  * nominal MC + data with GENIE spline weights ->  minimal_withspline[_nuwro]_df.root
  * one detector-variation file per vartype     ->  minimal_detvar[_nuwro]_<vartype>_df.root

Two "studies" share every writer and differ only in what plays prediction and data
(see STUDIES):

  * default:  runs 1-5 open-data weighting (wc_net_weight_open_data).  Prediction =
              overlays + EXT + dirt (+ the reweighted rad-corr / coherent-1g samples),
              data = real data.
  * --nuwro:  NuWro fake-data study (wc_net_weight_nuwro).  Prediction = overlays (+
              the reweighted samples) normalized per run period to the NuWro POT with the
              run 4b/4d/4bcd overlays folded into the 4c group; EXT and dirt have no
              weight in that config and are left out.  Data = the NuWro fake data,
              written with isdata == 1 (and isnuwro == 1) so an open-data PROfit XML
              applies unchanged; it keeps its own NuWro-config weight, gets unit spline
              branches like real data, and is kept whole (never in the train/test
              split).  The DetVar weights are rescaled per run period from the
              expected-full-dataset POT to the NuWro POT so the CV / variation mix
              matches the prediction.  The total NuWro POT (for the XML pot attributes)
              is printed.

Usage:
    python src/save_PROfit_rootfiles.py                    # open data: splines + detvar
    python src/save_PROfit_rootfiles.py --nuwro            # NuWro fake-data study
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
from df_helpers import format_duration
from signal_categories import train_category_labels
from ntuple_variables.variables import combined_training_vars

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_TRAINING = "all_vars_r15_2026_09_01"

# Filetypes that are never in the BDT train/test split but are usable as prediction
# (nothing was trained on them): kept whole, left out of the frac_test counts, and not
# test-half upweighted, if a study lists them among its prediction filetypes.
WHOLE_SAMPLE_FILETYPES = ["fullosc_overlay"]

# Filetypes with no GENIE spline weights that are written with constant unit spline
# branches (the data role of each study, and EXT when it is part of the prediction).
NO_SPLINE_FILETYPES = ["data", "ext", "nuwro_fake_data"]

# DetVar detailed_run_period -> NuWro normalizing run period, for the --nuwro DetVar
# rescale.  The DetVar config groups {3, 3a, 3b1, 3b2} -> "3" and {4b, 4c, 4d, 4bcd} ->
# "4nota"; the NuWro config groups the same detailed periods as "3" and "4c".  Because
# both configs pool identical detailed periods, the per-(filetype, group) denominator
# POTs are the same and only the goal POT differs, so
# net_weight_nuwro = net_weight * goal_nuwro / goal_detvar exactly.
DETVAR_TO_NUWRO_RUN_PERIOD = {
    "1": "1", "2": "2",
    "3": "3", "3a": "3", "3b1": "3", "3b2": "3",
    "4a": "4a",
    "4b": "4c", "4c": "4c", "4d": "4c", "4bcd": "4c",
    "5": "5",
}

# The two studies.  Prediction / excluded filetypes are explicit allowlists: every
# filetype found in all_df must be in one of prediction_filetypes, excluded_filetypes
# or be the data_filetype, otherwise the script raises -- so a new sample has to be
# classified on purpose.  The raw 1g overlays are excluded everywhere (kept only through
# their reweighted filetypes).  To include fullosc, move it into a prediction list.
STUDIES = {
    "open_data": dict(
        label="runs 1-5 open-data weighting",
        net_weight_col="wc_net_weight_open_data",
        weight_config_name="open_data",     # suffix of the normalizing_run_period_* / norm_goal_pot_* helper columns
        data_filetype="data",
        prediction_filetypes=[
            "nu_overlay", "nue_overlay", "nc_pi0_overlay", "numucc_pi0_overlay",
            "dirt_overlay", "ext",
            "numuCC_rad_corrected", "NC_coherent_1g_reweighted",
        ],
        excluded_filetypes=[
            "nuwro_fake_data",               # only in the --nuwro study
            "isotropic_one_gamma_overlay",   # raw 1g overlay, kept as NC_coherent_1g_reweighted
            "delete_one_gamma_overlay",      # raw 1g overlay, kept as numuCC_rad_corrected
            "fullosc_overlay",               # evaluation-only sample
        ],
        nominal_output="minimal_withspline_df.root",
        detvar_prefix="minimal_detvar_",
        detvar_run_period_map=None,          # DetVar weights used as written by create_detvar_df.py
    ),
    "nuwro": dict(
        label="NuWro fake-data weighting",
        net_weight_col="wc_net_weight_nuwro",
        weight_config_name="nuwro",
        data_filetype="nuwro_fake_data",
        prediction_filetypes=[
            "nu_overlay", "nue_overlay", "nc_pi0_overlay", "numucc_pi0_overlay",
            "numuCC_rad_corrected", "NC_coherent_1g_reweighted",
        ],
        excluded_filetypes=[
            "data",                          # real data: not part of the fake-data study
            "ext",                           # no weight in the NuWro config (no beam-off in the fake data)
            "dirt_overlay",                  # no weight in the NuWro config (no dirt in the fake data)
            "isotropic_one_gamma_overlay",
            "delete_one_gamma_overlay",
            "fullosc_overlay",
        ],
        nominal_output="minimal_withspline_nuwro_df.root",
        detvar_prefix="minimal_detvar_nuwro_",
        detvar_run_period_map=DETVAR_TO_NUWRO_RUN_PERIOD,   # rescale DetVar weights to the NuWro POT per group
    ),
}

# DetVar has its own single weighting config (create_detvar_df.py) whose column is
# "wc_net_weight" (each run period scaled to the expected full-dataset data POT); the
# detvar covariance is a fractional (CV-var)/CV difference so the absolute
# normalization cancels.  norm_goal_pot_detvar is that config's per-group goal POT.
DETVAR_NET_WEIGHT_COL = "wc_net_weight"
DETVAR_GOAL_POT_COL = "norm_goal_pot_detvar"

# Integer code for the filetype string, written as the `filetype_code` branch so PROfit
# can use it in cv_variation_matching_vars (TTreeFormula can't match on a string
# branch).  Needed since the DetVar files mix nu_overlay and nue_overlay samples, whose
# (detvar_sample, run, subrun, event) keys are not guaranteed to be distinct.  Append
# only -- existing codes are baked into written ROOT files.
FILETYPE_CODES = {
    "nu_overlay": 1,
    "nue_overlay": 2,
    "nc_pi0_overlay": 3,
    "numucc_pi0_overlay": 4,
    "dirt_overlay": 5,
    "ext": 6,
    "data": 7,
    "nuwro_fake_data": 8,
    "delete_one_gamma_overlay": 9,
    "isotropic_one_gamma_overlay": 10,
    "fullosc_overlay": 11,
    "numuCC_rad_corrected": 12,
    "NC_coherent_1g_reweighted": 13,
}


def _filetype_code_expr():
    """Polars expression mapping the filetype string column to its FILETYPE_CODES int
    (Int32, 0 for anything not in the table)."""
    return (
        pl.col("filetype")
        .replace_strict(FILETYPE_CODES, default=0, return_dtype=pl.Int32)
        .alias("filetype_code")
    )

# The detector-variation samples PROfit expects (CV + the 7 variations used by the
# covariance).  Only these get a ROOT file; any other vartype value (e.g. the empty
# "" that create_detvar_df.py mislabels some events with) is skipped with a warning.
DETVAR_VARTYPES = ["CV", "LYAtt", "LYDown", "LYRayleigh", "WireModX", "WireModYZ", "WireModThetaXZ", "WireModThetaYZ", "Recomb2", "SCE"]

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
#     <study net_weight_col>         (raw POT net weight of the study, e.g. wc_net_weight_open_data)
#     filetype_code, isnuwro         (see FILETYPE_CODES; isnuwro flags the NuWro fake data)
#     has_spline_weights, fraction_with_spline_weights, spline_processed_fraction_weight
#     net_weight                     (final weight = open-data weight x spline-fraction weight)
#     weightsReint + every GENIE spline-knob column (from spline_weights_df)
# and for each DETVAR file: filetype, filetype_code, vartype, detvar_sample, run,
#     subrun, event, isdata/isext/isdirt, reco_category, wc_kine_reco_Enu, net_weight,
#     prob_<category>.
#     (detvar_sample distinguishes the two overlapping run 3b CV samples: 0 = 1mil,
#     matched by all run 3b variations except SCE/Recomb2; 1 = 500k, matched by
#     SCE/Recomb2.  filetype_code (FILETYPE_CODES) separates the nu_overlay and
#     nue_overlay DetVar samples; PROfit's cv_variation_matching_vars should be
#     "filetype_code,detvar_sample,run,subrun,event".)
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
    "afro_1mu1p_sel",
    "afro_1mu1p_true",
    "afro_1mu1p_PMiss",
    "afro_1mu1p_Pn",
    "afro_1mu1p_Pt",
    "afro_1mu1p_Q2"
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


def write_withspline_root(mc_df, data_df, spline_path, output_path, net_weight_col, batch_size=131_072):
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
    net_weight_col names the POT-weight column (in both dfs) that becomes net_weight
    (the study's STUDIES[...]["net_weight_col"]).
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
                (pl.col(net_weight_col) * pl.col("spline_processed_fraction_weight")).alias("net_weight"),
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
    net = data_df[net_weight_col].to_numpy().astype(np.float64)

    for i in tqdm(range(data_df.height), desc="nominal data/ext"):
        for c in update_cols:
            tag, arr = data_extract[c]
            if tag == "str":
                buffers[c].assign(str(arr[i]))
            elif tag == "bool":
                buffers[c][0] = bool(arr[i])
            else:
                buffers[c][0] = arr[i]
        buffers["net_weight"][0] = net[i]   # data/ext net_weight = the POT weight itself
        tree.Fill()

    tree.Write()
    f.Close()
    total = n_mc_written + data_df.height
    print(f"  wrote {total} events ({n_mc_written} MC + {data_df.height} data/ext) to {output_path}")



# ============================================================================
# Nominal MC + data (with spline weights)
# ============================================================================

def _check_filetypes(present, study):
    """Every filetype in all_df must be classified by the study (prediction / excluded /
    data); warn about listed prediction filetypes that are absent."""
    known = set(study["prediction_filetypes"]) | set(study["excluded_filetypes"]) | {study["data_filetype"]}
    unknown = sorted(set(present) - known)
    if unknown:
        raise ValueError(
            f"filetypes {unknown} are in all_df but not in this study's prediction_filetypes / "
            f"excluded_filetypes / data_filetype -- decide where they belong and add them to STUDIES")
    missing = sorted(set(study["prediction_filetypes"]) - set(present))
    if missing:
        print(f"  WARNING: prediction filetypes {missing} are not present in all_df")
    if study["data_filetype"] not in present:
        raise ValueError(f"no {study['data_filetype']} rows in all_df")


def build_minimal_df(training, study):
    """Return the lazy minimal nominal df (prediction test events + the data-role
    sample), scored and with a reco_category, before spline merging."""
    prob_cols = _prob_cols()
    net_weight_col = study["net_weight_col"]

    all_df = pl.scan_parquet(f"{intermediate_files_location}/all_df.parquet")
    preds = pl.scan_parquet(f"{PROJECT_ROOT}/training_outputs/{training}/predictions.parquet")
    merged = all_df.join(preds, on=["filetype", "run", "subrun", "event"], how="left")

    present = merged.select(pl.col("filetype").unique()).collect(engine="streaming")["filetype"].to_list()
    _check_filetypes(present, study)

    # BDT scores: fill missing with -1, then the per-event argmax over the prob columns
    merged = merged.with_columns([pl.col(p).fill_null(-1) for p in prob_cols])
    merged = merged.with_columns(
        pl.concat_list(prob_cols).list.arg_max().alias("reco_category_argmax_index")
    )

    pred = merged.filter(pl.col("filetype").is_in(study["prediction_filetypes"]))
    data = merged.filter(pl.col("filetype") == study["data_filetype"])

    # generic preselection; every prediction row must carry a weight in this config
    pred = pred.filter(pl.col("wc_kine_reco_Enu") > 0)
    n_unweighted = pred.filter(pl.col(net_weight_col).is_null()).select(pl.len()).collect(engine="streaming").item()
    if n_unweighted:
        raise ValueError(f"{n_unweighted} prediction rows have a null {net_weight_col}; the weighting config "
                         f"does not cover one of {study['prediction_filetypes']}")

    # Use only test events (the BDT trained on the train half), weighted up by
    # 1/frac_test so the total normalization is preserved.  WHOLE_SAMPLE_FILETYPES are
    # never in the split: kept whole, left out of the counts, not upweighted.
    in_split = ~pl.col("filetype").is_in(WHOLE_SAMPLE_FILETYPES)
    counts = pred.filter(in_split).select([
        pl.col("used_for_training").sum().alias("n_train"),
        pl.col("used_for_testing").sum().alias("n_test"),
    ]).collect(engine="streaming")
    num_train, num_test = counts["n_train"][0], counts["n_test"][0]
    frac_test = num_test / (num_train + num_test)
    print(f"  train={num_train}, test={num_test} -> scaling test weights by 1/{frac_test:.4f}")
    pred = pred.with_columns(
        pl.when(pl.col("used_for_testing") & in_split)
        .then(pl.col(net_weight_col) / frac_test)
        .otherwise(pl.col(net_weight_col))
        .alias(net_weight_col)
    ).filter(pl.col("used_for_testing") | ~in_split)

    # the data role is kept whole (real data or NuWro fake data: never in the split)
    data = data.filter(pl.col("wc_kine_reco_Enu") > 0)

    combined = pl.concat([pred, data], how="vertical")
    combined = combined.with_columns(_reco_category_expr().alias("reco_category"))

    # isdata flags the study's data role (real data, or the NuWro fake data in --nuwro)
    # so the same PROfit XML (isdata == 1 data section; isdata==0 && isext==0 &&
    # isdirt==0 overlays) applies to both studies.
    minimal = combined.select(OUTPUT_SCALAR_COLUMNS + [net_weight_col] + prob_cols).with_columns([
        _filetype_code_expr(),
        (pl.col("filetype") == study["data_filetype"]).alias("isdata"),
        (pl.col("filetype") == "ext").alias("isext"),
        (pl.col("filetype") == "dirt_overlay").alias("isdirt"),
        (pl.col("filetype") == "nuwro_fake_data").alias("isnuwro"),
    ])
    # all_df.parquet files made before postprocessing.py zero-filled the fullosc-only
    # branches: null here becomes NaN in ROOT, where wc_fullosc==0 is always false.
    fullosc_cols = [c for c in ("wc_fullosc", "wc_fullosc_cv_weight")
                    if c in minimal.collect_schema().names()]
    if fullosc_cols:
        minimal = minimal.with_columns([pl.col(c).fill_null(0.0) for c in fullosc_cols])
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


def get_goal_pot(study):
    """{normalizing_run_period: goal POT} of the study's weighting config, read from
    all_df's norm_goal_pot_<config> helper column on the data-role rows (one value per
    group), so nothing is hardcoded.  Raises if a group carries more than one value."""
    name = study["weight_config_name"]
    nrp_col, goal_col = f"normalizing_run_period_{name}", f"norm_goal_pot_{name}"
    table = (
        pl.scan_parquet(f"{intermediate_files_location}/all_df.parquet")
        .filter(pl.col("filetype") == study["data_filetype"])
        .group_by(nrp_col)
        .agg([pl.col(goal_col).min().alias("mn"), pl.col(goal_col).max().alias("mx")])
        .sort(nrp_col)
        .collect(engine="streaming")
    )
    goal = {}
    for row in table.iter_rows(named=True):
        nrp, mn, mx = row[nrp_col], row["mn"], row["mx"]
        if nrp is None or mn is None or mn <= 0 or mn != mx:
            raise ValueError(f"inconsistent {goal_col} for normalizing run period {nrp!r}: min={mn}, max={mx}")
        goal[nrp] = float(mn)
    if not goal:
        raise ValueError(f"no {study['data_filetype']} rows with a {nrp_col} found in all_df")
    return goal


def print_goal_pot(goal, study):
    print(f"  {study['data_filetype']} POT per normalizing run period:")
    for nrp, pot in goal.items():
        print(f"    {nrp:>4}: {pot:.4e}")
    print(f"  total POT (for the PROfit XML pot attributes): {sum(goal.values()):.4e}")


def save_nominal(training, output_dir, study):
    """Prediction + data role with spline weights -> study['nominal_output'].

    The minimal df (~30 scalar columns) is small enough to hold in memory; only the
    spline parquet with its 1000-wide list columns is big, and write_withspline_root
    streams that in bounded batches.  MC events without spline weights are dropped
    (inner join) and each filetype is weighted up by 1/fraction_with_spline_weights;
    the data role and EXT (no spline weights) are written with unit spline branches."""
    spline_path = f"{intermediate_files_location}/spline_weights_df.parquet"
    net_weight_col = study["net_weight_col"]

    print(f"Building nominal minimal df ({study['label']})...")
    minimal_df = build_minimal_df(training, study)
    unit_spline = [ft for ft in NO_SPLINE_FILETYPES
                   if ft == study["data_filetype"] or ft in study["prediction_filetypes"]]
    mc_df = minimal_df.filter(~pl.col("filetype").is_in(unit_spline)).collect(engine="streaming")
    data_df = minimal_df.filter(pl.col("filetype").is_in(unit_spline)).collect(engine="streaming")
    print(f"  {mc_df.height} MC events, {data_df.height} events written with unit spline branches ({unit_spline})")
    print(f"  preselected prediction total (before spline-fraction weighting): "
          f"{mc_df[net_weight_col].sum() + data_df.filter(pl.col('filetype') != study['data_filetype'])[net_weight_col].sum():.1f}")
    print(f"  preselected {study['data_filetype']} total: "
          f"{data_df.filter(pl.col('filetype') == study['data_filetype'])[net_weight_col].sum():.1f}")

    print("Merging spline weights...")
    fractions = compute_spline_fractions(mc_df, spline_path)
    for row in fractions.sort("filetype").iter_rows(named=True):
        frac = row["fraction_with_spline_weights"]
        print(f"    {row['filetype']}: fraction_with_spline_weights = "
              f"{frac if frac is None else f'{frac:.4f}'}")
    mc_df = mc_df.join(fractions, on="filetype", how="left")

    output_path = f"{output_dir}/{study['nominal_output']}"
    write_withspline_root(mc_df, data_df, spline_path, output_path, net_weight_col=net_weight_col)

    print_goal_pot(get_goal_pot(study), study)


# ============================================================================
# Detector variations
# ============================================================================

def score_detvar_df(training, extra_cols=()):
    """Return the scored DetVar minimal df (all vartypes): ids, flags, reco_category,
    wc_kine_reco_Enu, net_weight (= DETVAR_NET_WEIGHT_COL) and the prob_ columns, plus any
    extra_cols carried through untouched (the --nuwro DetVar rescale uses these to
    rescale the run-period mix)."""
    print("Building DetVar minimal dfs...")
    prob_cols = _prob_cols()

    model = xgb.XGBClassifier()
    model.load_model(f"{PROJECT_ROOT}/training_outputs/{training}/bdt.json")

    # The BDT was trained on the combined_training_vars of its time; if the variable
    # lists in src/ntuple_variables/ have changed since, positional inference would be
    # silently wrong (numpy input carries no feature names), so check the count first.
    n_model_features = model.get_booster().num_features()
    if len(TRAINING_VARS) != n_model_features:
        raise ValueError(
            f"Training variable mismatch for training '{training}': the BDT was trained with "
            f"{n_model_features} features but the current combined_training_vars has "
            f"{len(TRAINING_VARS)}. The variable lists in src/ntuple_variables/ have changed "
            f"since this BDT was trained -- retrain with src/train.py (and rerun so "
            f"predictions.parquet matches too)."
        )

    # Collect only the columns we need (ids + weight + inference vars), so the BDT
    # scores can be attached by position and we never hold the full detvar df.
    # dict.fromkeys dedups columns that are both an explicit id/weight and a training var
    # (e.g. wc_kine_reco_Enu is in TRAINING_VARS), which .select would reject as duplicate.
    keep = list(dict.fromkeys(
        ["filetype", "vartype", "detvar_sample", "run", "subrun", "event", "wc_kine_reco_Enu", DETVAR_NET_WEIGHT_COL]
        + TRAINING_VARS + list(extra_cols)))
    presel = (
        pl.scan_parquet(f"{intermediate_files_location}/detvar_presel_df_train_vars.parquet")
        .select(keep)
        .collect()
    )
    print(f"  {presel.height} detvar events")

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
        _filetype_code_expr(),
        (pl.col("filetype") == "data").alias("isdata"),
        (pl.col("filetype") == "ext").alias("isext"),
        (pl.col("filetype") == "dirt_overlay").alias("isdirt"),
    ]).rename({DETVAR_NET_WEIGHT_COL: "net_weight"})

    unknown_filetypes = presel.filter(pl.col("filetype_code") == 0)["filetype"].unique().to_list()
    if unknown_filetypes:
        raise ValueError(f"filetypes {unknown_filetypes} are missing from FILETYPE_CODES -- add them (append only)")

    detvar_minimal = presel.select(
        ["filetype", "filetype_code", "vartype", "detvar_sample", "run", "subrun", "event", "isdata", "isext", "isdirt",
         "reco_category", "wc_kine_reco_Enu", "net_weight"] + prob_cols + list(extra_cols)
    )
    return detvar_minimal


def write_detvar_files(detvar_minimal, output_dir, file_prefix="minimal_detvar_"):
    """Write one <file_prefix><vartype>_df.root per DETVAR_VARTYPES from the scored df."""
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
        output_path = f"{output_dir}/{file_prefix}{vartype}_df.root"
        write_df_to_root(df_to_save, output_path, desc=f"detvar {vartype}")



def rescale_detvar_weights(detvar_minimal, run_period_map, goal):
    """Multiply each DetVar event's net_weight by (goal POT of its mapped group) /
    (the DetVar config's goal POT of its own group), see DETVAR_TO_NUWRO_RUN_PERIOD."""
    present = detvar_minimal["detailed_run_period"].unique().to_list()
    unmapped = sorted(p for p in present if p not in run_period_map)
    if unmapped:
        raise ValueError(f"DetVar detailed_run_period(s) {unmapped} are missing from the run-period map")
    missing_goal = sorted({run_period_map[p] for p in present} - set(goal))
    if missing_goal:
        raise ValueError(f"DetVar events map to run period(s) {missing_goal} that have no goal POT")
    if detvar_minimal.filter(pl.col(DETVAR_GOAL_POT_COL).is_null() | (pl.col(DETVAR_GOAL_POT_COL) <= 0)).height:
        raise ValueError(f"DetVar rows with a null/zero {DETVAR_GOAL_POT_COL}")

    mapped = pl.col("detailed_run_period").replace_strict(run_period_map, return_dtype=pl.String)
    scale = mapped.replace_strict(goal, return_dtype=pl.Float64) / pl.col(DETVAR_GOAL_POT_COL).cast(pl.Float64)

    summary = (
        detvar_minimal.with_columns(scale.alias("_scale"))
        .group_by(["filetype", "detailed_run_period"])
        .agg([pl.len().alias("n"), pl.col("_scale").min().alias("scale_min"), pl.col("_scale").max().alias("scale_max")])
        .sort(["filetype", "detailed_run_period"])
    )
    print("  DetVar weight rescale (expected-full-dataset POT -> study POT), per filetype / detailed_run_period:")
    for row in summary.iter_rows(named=True):
        print(f"    {row['filetype']:<12} {row['detailed_run_period']:<5} -> group "
              f"{run_period_map[row['detailed_run_period']]:<3} x{row['scale_min']:.4f}  ({row['n']} events)")
        if abs(row["scale_min"] - row["scale_max"]) > 1e-6 * abs(row["scale_min"]):
            raise ValueError(f"non-constant rescale within {row['filetype']} {row['detailed_run_period']}: "
                             f"{row['scale_min']} .. {row['scale_max']}")

    return (
        detvar_minimal
        .with_columns((pl.col("net_weight") * scale).alias("net_weight"))
        .drop(["detailed_run_period", DETVAR_GOAL_POT_COL])
    )


def save_detvar(training, output_dir, study):
    run_period_map = study["detvar_run_period_map"]
    if run_period_map is None:
        detvar_minimal = score_detvar_df(training)
    else:
        detvar_minimal = score_detvar_df(training, extra_cols=["detailed_run_period", DETVAR_GOAL_POT_COL])
        goal = get_goal_pot(study)
        print_goal_pot(goal, study)
        detvar_minimal = rescale_detvar_weights(detvar_minimal, run_period_map, goal)
    write_detvar_files(detvar_minimal, output_dir, file_prefix=study["detvar_prefix"])


# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Write PROfit input ROOT files from the processed dataframes.")
    parser.add_argument("--training", default=DEFAULT_TRAINING,
                        help=f"training_outputs/<name> to read predictions + BDT from (default: {DEFAULT_TRAINING})")
    parser.add_argument("--output-dir", default=intermediate_files_location,
                        help="directory to write the ROOT files into (default: intermediate_files_location)")
    parser.add_argument("--nuwro", action="store_true",
                        help="NuWro fake-data study: NuWro-POT weighting, NuWro fake data as the data role "
                             "(isdata == 1), DetVar weights rescaled to the NuWro POT; writes the *_nuwro_* files")
    parser.add_argument("--no-splines", action="store_true", help="skip the nominal MC+data spline ROOT file")
    parser.add_argument("--no-detvar", action="store_true", help="skip the per-vartype detvar ROOT files")
    args = parser.parse_args()
    study = STUDIES["nuwro" if args.nuwro else "open_data"]
    print(f"Study: {study['label']} ({study['net_weight_col']}, data role = {study['data_filetype']})")

    start = time.time()
    if not args.no_splines:
        save_nominal(args.training, args.output_dir, study)
    if not args.no_detvar:
        save_detvar(args.training, args.output_dir, study)
    print(f"Done in {format_duration(time.time() - start)}", flush=True)
    # All ROOT files are written and closed; skip Python/polars teardown, which can
    # segfault while ROOT is loaded (ROOT's signal handlers vs polars' Rust threads).
    os._exit(0)


if __name__ == "__main__":
    main()
