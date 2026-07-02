"""Minimal reproduction: wide multi-aggregation select over a streaming parquet scan
returns silently wrong list.len() min/max under concurrent load.

Shape mimics a real physics file (event weight vectors): 2.6M rows, 72 List(Float64)
columns -- 3 of width 1000, 69 of width 7.  Every row of every column has the SAME
length, so every min/max must equal that width.  Several worker processes run the
same 144-aggregation query concurrently; each verifies its own results.

Observed on polars 1.34.0 / linux x86_64 / 56 cores: workers intermittently report
e.g. min=0 max=221824000 for a width-1000 column (221824000 = 221824 * 1000 -- i.e.
a whole chunk's flattened element count, not a per-row length).
"""
import multiprocessing as mp
import os
import sys

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repro_lists.parquet")
N_ROWS = 2_600_000
WIDTHS = {f"c{i}": (1000 if i < 3 else 7) for i in range(72)}
CHUNK = 100_000
N_WORKERS = 4
TRIALS_PER_WORKER = 8


def make_file():
    if os.path.exists(PATH):
        return
    print(f"writing {PATH} ...")
    fields = [pa.field(c, pa.list_(pa.float64())) for c in WIDTHS]
    schema = pa.schema(fields)
    with pq.ParquetWriter(PATH, schema, compression="zstd") as w:
        for start in range(0, N_ROWS, CHUNK):
            n = min(CHUNK, N_ROWS - start)
            arrays = []
            for c, width in WIDTHS.items():
                offsets = pa.array(np.arange(n + 1, dtype=np.int32) * width)
                values = pa.array(np.zeros(n * width, dtype=np.float64))
                arrays.append(pa.ListArray.from_arrays(offsets, values))
            w.write_table(pa.Table.from_arrays(arrays, schema=schema))
    print(f"  done ({os.path.getsize(PATH)/1e6:.0f} MB)")


def worker(wid, q):
    cols = list(WIDTHS)
    n_bad = 0
    for t in range(TRIALS_PER_WORKER):
        agg = pl.scan_parquet(PATH).select(
            [pl.col(c).list.len().min().alias(f"{c}__mn") for c in cols]
            + [pl.col(c).list.len().max().alias(f"{c}__mx") for c in cols]
        ).collect(engine="streaming")
        bad = [(c, int(agg[f"{c}__mn"][0]), int(agg[f"{c}__mx"][0]))
               for c in cols
               if agg[f"{c}__mn"][0] != WIDTHS[c] or agg[f"{c}__mx"][0] != WIDTHS[c]]
        if bad:
            n_bad += 1
            print(f"[worker {wid} trial {t}] WRONG: {bad[:3]}", flush=True)
        else:
            print(f"[worker {wid} trial {t}] ok", flush=True)
    q.put(n_bad)


if __name__ == "__main__":
    make_file()
    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(i, q)) for i in range(N_WORKERS)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    total_bad = sum(q.get() for _ in procs)
    total = N_WORKERS * TRIALS_PER_WORKER
    print(f"\n{total_bad}/{total} queries returned WRONG results")
    sys.exit(1 if total_bad else 0)
