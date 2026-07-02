#!/usr/bin/env python
"""Reproduce the polars 1.34 streaming wide-multi-agg race against the REAL spline file.

This is the harness that reliably reproduces the bug (~1 wrong query in 12) on this
machine, 2026-07-02.  It runs one background "load generator" process doing heavy
streaming polars work, then repeatedly executes a single select with 144 aggregation
expressions (list.len().min()/.max() for all 72 list columns) over the 19 GB
spline_weights_df.parquet, checking each result.

Every row of every list column in that file has a fixed known length (weightsReint =
1000, knobs = 7 etc.; verified by per-column queries and by a group_by length
value-count), so any min != max is a wrong result.  Observed corruption:
    weightsReint: min=0, max=39696000   (= 39696  * 1000)
    weightsReint: min=0, max=221824000  (= 221824 * 1000)
i.e. the "max" is a whole morsel's flattened element count (chunk_rows x list_width),
not a per-row length -- the aggregation appears to race on the list offsets under load.

Never reproduces on an idle machine.  A small fully-cached synthetic file (zeros,
716 KB compressed) also does not reproduce; see polars_streaming_race_repro.py for the
self-contained synthetic attempt.

Usage:  python debug/polars_streaming_race_realfile.py [n_trials]
"""
import multiprocessing as mp
import sys

import polars as pl

SPLINE = "/nevis/riverside/data/leehagaman/ngem/intermediate_files/spline_weights_df.parquet"


def load_gen():
    """Continuous heavy streaming work, mimicking a concurrent pipeline run."""
    while True:
        pl.scan_parquet(SPLINE).select(
            pl.col("weightsReint").list.sum().sum()).collect(engine="streaming")


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 12

    schema = pl.scan_parquet(SPLINE).collect_schema()
    lc = [c for c, dt in schema.items() if isinstance(dt, pl.List)]
    print(f"{len(lc)} list columns -> {2 * len(lc)} aggregations per query")

    load = mp.Process(target=load_gen, daemon=True)
    load.start()

    n_bad = 0
    for t in range(n_trials):
        agg = pl.scan_parquet(SPLINE).select(
            [pl.col(c).list.len().min().alias(f"{c}__mn") for c in lc]
            + [pl.col(c).list.len().max().alias(f"{c}__mx") for c in lc]
        ).collect(engine="streaming")
        bad = [(c, int(agg[f"{c}__mn"][0]), int(agg[f"{c}__mx"][0]))
               for c in lc if agg[f"{c}__mn"][0] != agg[f"{c}__mx"][0]]
        if bad:
            n_bad += 1
            print(f"trial {t}: WRONG {bad[:4]}", flush=True)
        else:
            print(f"trial {t}: clean", flush=True)

    load.terminate()
    print(f"\n{n_bad}/{n_trials} queries returned wrong results")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
