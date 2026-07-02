#!/usr/bin/env python
"""EXPERIMENTAL faster variant of save_PROfit_rootfiles.py.

Same pipeline and same ROOT output format (std::vector<double> object branches, which is
what PROfit's SetBranchAddress needs -- this is why the original moved off uproot, whose
jagged-array layout PROfit could not read).  The only change is HOW the list branches are
filled:

  * original: for each event, push_back() every element -> for the 1000-universe
    weightsReint that is ~1000 Python<->C++ calls per event, billions total.
  * here: every list in a column has a FIXED length (weightsReint=1000, each knob=7), so
    we bind a fixed-size std::vector once, take a numpy view onto its buffer
    (np.frombuffer(vec.data())), and copy each row in with a single memcpy
    (view[:] = row).  Verified to round-trip and to produce a vector<double> branch.

Columns whose list lengths vary fall back to push_back automatically.

IMPORTANT: verify this produces identical output to save_PROfit_rootfiles.py before
trusting it (a comparison is easy -- open both trees and diff a few branches).

Usage:  python src/save_PROfit_rootfiles_fast.py --training all_vars_r15
"""

import argparse
import os
import time

import numpy as np
import polars as pl
from tqdm import tqdm

import save_PROfit_rootfiles as base
from file_locations import intermediate_files_location


def write_lazy_to_root_fast(lazy_df, output_path, desc="writing", batch_size=200_000):
    """Stream a lazy frame to a ROOT TTree, filling fixed-length list branches with one
    numpy memcpy per row instead of per-element push_back."""
    ROOT = base._import_root()

    tmp = f"{output_path}.tmp.parquet"
    print(f"  streaming plan to {tmp} ...", flush=True)
    lazy_df.sink_parquet(tmp)

    schema = pl.scan_parquet(tmp).collect_schema()
    int_cols = [c for c, dt in schema.items() if dt in base._INT_DTYPES]
    has_nulls = {}
    if int_cols:
        nc = pl.scan_parquet(tmp).select([pl.col(c).null_count().alias(c) for c in int_cols]).collect()
        has_nulls = {c: nc[c][0] > 0 for c in int_cols}
    kinds = {c: base._column_kind(dt, has_nulls.get(c, False)) for c, dt in schema.items()}

    # fixed list length per list column (min == max) enables the memcpy fast path
    list_cols = [c for c, k in kinds.items() if k == "list"]
    fixed_len = {}
    if list_cols:
        agg = pl.scan_parquet(tmp).select(
            [pl.col(c).list.len().min().alias(f"{c}__mn") for c in list_cols]
            + [pl.col(c).list.len().max().alias(f"{c}__mx") for c in list_cols]
        ).collect()
        for c in list_cols:
            mn, mx = agg[f"{c}__mn"][0], agg[f"{c}__mx"][0]
            fixed_len[c] = int(mn) if (mn is not None and mn == mx and mn > 0) else None
        n_fixed = sum(v is not None for v in fixed_len.values())
        print(f"  {n_fixed}/{len(list_cols)} list columns use the fixed-length memcpy fast path", flush=True)

    total = pl.scan_parquet(tmp).select(pl.len()).collect().item()
    n_batches = -(-total // batch_size)
    print(f"  writing {total} events to {output_path} in {n_batches} batches...", flush=True)

    f = ROOT.TFile.Open(output_path, "RECREATE")
    tree = ROOT.TTree("tree", "tree")

    buffers = {}
    views = {}   # col -> numpy view onto a fixed-size std::vector's buffer
    for col, kind in kinds.items():
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

    for b, start in enumerate(range(0, total, batch_size)):
        batch = pl.scan_parquet(tmp).slice(start, batch_size).collect()

        # pre-extract each column for this batch (tag drives the per-row fill)
        data = {}
        for col, kind in kinds.items():
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

        n = batch.height
        for i in tqdm(range(n), desc=f"{desc} batch {b + 1}/{n_batches}"):
            for col, kind in kinds.items():
                buf = buffers[col]
                tag, arr = data[col]
                if tag == "list_fixed":
                    views[col][:] = arr[i]            # single memcpy into the vector buffer
                elif tag == "list_var":
                    buf.clear()
                    v = arr[i]
                    if v is not None:
                        for x in v:
                            buf.push_back(float(x))
                elif tag == "str":
                    buf.assign(str(arr[i]))
                elif tag == "bool":
                    buf[0] = bool(arr[i])
                else:
                    buf[0] = arr[i]
            tree.Fill()

    tree.Write()
    f.Close()
    os.remove(tmp)
    print(f"  wrote {total} events to {output_path}")


def save_nominal_fast(training, output_dir):
    print("Building nominal minimal df (runs 1-5 open-data weighting)...")
    minimal_df = base.build_minimal_df(training)
    print("Merging spline weights...")
    minimal_withspline_df, fractions = base.merge_splines(minimal_df)
    for row in fractions.collect().sort("filetype").iter_rows(named=True):
        frac = row["fraction_with_spline_weights"]
        print(f"    {row['filetype']}: fraction_with_spline_weights = "
              f"{frac if frac is None else f'{frac:.4f}'}")
    write_lazy_to_root_fast(minimal_withspline_df, f"{output_dir}/minimal_withspline_df.root", desc="nominal")


def main():
    parser = argparse.ArgumentParser(description="Faster variant of save_PROfit_rootfiles.py (fixed-length memcpy list fill).")
    parser.add_argument("--training", default=base.DEFAULT_TRAINING)
    parser.add_argument("--output-dir", default=intermediate_files_location)
    parser.add_argument("--no-splines", action="store_true")
    parser.add_argument("--no-detvar", action="store_true")
    args = parser.parse_args()

    start = time.time()
    if not args.no_splines:
        save_nominal_fast(args.training, args.output_dir)
    if not args.no_detvar:
        base.save_detvar(args.training, args.output_dir)   # detvar has no big list columns; reuse
    print(f"Done in {time.time() - start:.1f} s", flush=True)
    # ROOT files written; skip teardown, which can segfault while ROOT is loaded.
    os._exit(0)


if __name__ == "__main__":
    main()
