"""DEPRECATED: spline-weight loading has been unified into create_rw_syst_df.py.

Every overlay ROOT file now carries a ``spline_weights`` tree, so create_rw_syst_df.py
reads the per-knob spline weights (plus weightsReint) alongside the GENIE/flux/reint
multisim systematic weights in a single pass over each file, and writes
``spline_weights_df.parquet`` itself (covering all run periods rather than only the
old run-4b spline files).  Run ``python src/create_rw_syst_df.py`` instead.
"""
import sys

if __name__ == "__main__":
    sys.exit(
        "create_splines_df.py is deprecated: spline weights are now produced by "
        "create_rw_syst_df.py, which writes spline_weights_df.parquet in the same pass. "
        "Run `python src/create_rw_syst_df.py` instead."
    )
