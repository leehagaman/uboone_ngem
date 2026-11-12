import sys
from pathlib import Path

# Add project root to path to allow imports with src. prefix
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.2
import numpy as np
import pandas as pd
import pickle
from tqdm.notebook import tqdm
import polars as pl

import argparse
import os
import time
import shutil
from pypdf import PdfWriter

from src.signal_categories import topological_category_labels, topological_category_colors, topological_category_labels_latex, topological_category_hatches, topological_categories_dic
from src.signal_categories import filetype_category_labels, filetype_category_colors, filetype_category_hatches
from src.signal_categories import del1g_detailed_category_labels, del1g_detailed_category_colors, del1g_detailed_category_labels_latex, del1g_detailed_category_hatches, del1g_detailed_categories_dic
from src.signal_categories import del1g_detailed_category_queries
from src.signal_categories import del1g_simple_category_labels, del1g_simple_category_colors, del1g_simple_category_labels_latex, del1g_simple_category_hatches, del1g_simple_categories_dic
from src.signal_categories import train_category_labels, train_category_labels_latex

from src.file_locations import intermediate_files_location
from src.df_helpers import format_duration
from src.plot_helpers import make_histogram_plot
from src.ntuple_variables.variables import combined_training_vars
from src.systematics import get_significance_from_p_value, SuspectedPolarsCorruption

import signal
import subprocess

MAX_RESTARTS = 5
CHILD_ENV_VAR = "PLOT_MANY_VARIABLES_CHILD"
CORRUPTION_EXIT_CODE = 3
# a crashed child (polars has segfaulted / aborted under heavy machine load) is restarted; a child
# stopped by the user (Ctrl-C, kill) or by the OOM killer (SIGKILL) is not
RESTARTABLE_SIGNALS = {signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGILL, signal.SIGFPE}


def run_supervisor():
    """Run the plotting in a child process, restarting it after a crash or suspected corruption.

    Under heavy machine load polars 1.44 has corrupted its own memory inside this long-running
    process: events silently dropped from the weights join, "capacity overflow" panics, and
    segfaults.  None of that is recoverable inside the process, so the work runs in a child, and a
    fresh child resumes at the failed variable (finished pages are skipped).  --clear-prev only
    applies to the first child, so restarts keep the finished pages."""
    argv = sys.argv[1:]
    env = dict(os.environ, **{CHILD_ENV_VAR: "1"})
    for attempt in range(MAX_RESTARTS + 1):
        returncode = subprocess.run([sys.executable, "-X", "faulthandler", __file__] + argv, env=env).returncode
        if returncode == 0:
            return 0
        crashed = returncode < 0 and -returncode in RESTARTABLE_SIGNALS
        if not (crashed or returncode == CORRUPTION_EXIT_CODE):
            return returncode if returncode > 0 else 128 - returncode
        reason = f"crashed with {signal.Signals(-returncode).name}" if crashed else "reported suspected polars corruption"
        if attempt == MAX_RESTARTS:
            print(f"ERROR: plotting child {reason}; giving up after {MAX_RESTARTS} restarts")
            return 1
        print(f"\nWARNING: plotting child {reason}; restarting in a fresh process "
              f"(restart {attempt + 1}/{MAX_RESTARTS}, finished pages are kept)...", flush=True)
        argv = [a for a in argv if a != "--clear-prev"]

plt.rcParams.update({'font.size': 12})


if __name__ == "__main__" and os.environ.get(CHILD_ENV_VAR) != "1":
    sys.exit(run_supervisor())

if __name__ == "__main__":
    main_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_plots", type=int, default=None)
    parser.add_argument("--vars", type=str, default=None)
    parser.add_argument("--clear-prev", action="store_true", help="Delete all files in plots/all_bdt_vars before starting")
    parser.add_argument("--no-systematics", action="store_true", help="Do not use systematics")
    args = parser.parse_args()

    if args.vars is None:
        vars = combined_training_vars
    else:
        raise ValueError(f"Invalid vars argument: {args.vars}")
    
    vars = sorted(vars)
    
    if args.num_plots is None:
        args.num_plots = len(vars)

    vars = vars[:args.num_plots]

    print(f"{len(vars)=}")

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Only the columns a plot needs are held in memory: the shared "base" columns (event keys,
    # filetype, weights, POT normalization, and the truth columns behind the del1g_detailed
    # categories) are loaded once, and each variable's own column is read per plot.  Reading the
    # full files (~115 GB DetVar + ~40 GB presel + ~30 GB weights, ~200 GB peak RSS) exhausted
    # the shared machine's memory, and under that pressure the runs were killed or polars
    # crashed / silently corrupted joins.
    presel_path = f"{intermediate_files_location}/presel_df_train_vars.parquet"
    detvar_path = f"{intermediate_files_location}/detvar_presel_df_train_vars.parquet"
    presel_schema = pl.read_parquet_schema(presel_path)
    detvar_schema = pl.read_parquet_schema(detvar_path)

    category_columns = set()
    for query in del1g_detailed_category_queries:
        category_columns |= set(eval(query, {"pl": pl, "__builtins__": {}}).meta.root_names())
    base_columns = (
        {"filename", "run", "subrun", "event", "filetype", "vartype",
         "wc_net_weight_open_data", "wc_net_weight", "wc_net_weight_sce_recomb2", "pandora_weightTune",
         "normal_overlay", "del1g_overlay", "iso1g_overlay"}
        | {c for c in list(presel_schema) + list(detvar_schema) if c.startswith(("norm_goal_pot_", "normalizing_run_period_"))}
        | category_columns
    )

    def read_base_columns(path, schema):
        missing_vars = [v for v in vars if v not in schema]
        if missing_vars:
            raise ValueError(f"{path} is missing {len(missing_vars)} plotted variables, e.g. {missing_vars[:5]}")
        return pl.read_parquet(path, columns=[c for c in schema if c in base_columns])

    def with_var_column(base_df, path, var):
        # parquet row order is fixed, so the single-column read lines up with the base columns row by row
        var_df = pl.read_parquet(path, columns=[var])
        if var_df.height != base_df.height:
            raise SuspectedPolarsCorruption(f"read {var_df.height} rows of {var} from {path}, expected {base_df.height}")
        return base_df.with_columns(var_df.get_column(var)) if var not in base_df.columns else base_df

    print("loading base columns of detvar_presel_df_train_vars.parquet...")
    presel_detvar_base_df = read_base_columns(detvar_path, detvar_schema)
    print(f"{presel_detvar_base_df.shape=}")

    print("loading base columns of presel_df_train_vars.parquet...")
    all_base_df = read_base_columns(presel_path, presel_schema)
    print(f"{all_base_df.shape=}")

    if not args.no_systematics:
        print("loading presel_weights_df.parquet...")
        presel_weights_df = pl.read_parquet(f"{intermediate_files_location}/presel_weights_df.parquet")
        print(f"{presel_weights_df.shape=}")

    # explicit allowlist of the open-data prediction samples; fake-data samples (nuwro_fake_data,
    # gibuu_fake_data) have null wc_net_weight_open_data, and raw iso1g/del1g are replaced by
    # NC_coherent_1g_reweighted
    pred_filter = pl.col("filetype").is_in([
        "nu_overlay", "nue_overlay", "dirt_overlay", "ext",
        "nc_pi0_overlay", "numucc_pi0_overlay", "ncdeltarad_overlay",
        "NC_coherent_1g_reweighted", "numucc_rad_corr_sim",
    ])
    data_filter = pl.col("filetype") == "data"

    # Create directory for individual pages
    pages_dir = PROJECT_ROOT / "plots" / "all_bdt_vars"
    
    # Clear previous files if requested
    if args.clear_prev:
        if pages_dir.exists():
            print(f"Clearing all files in {pages_dir}...")
            for file_path in pages_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            print("Cleared previous files.")
        else:
            print(f"Directory {pages_dir} does not exist, nothing to clear.")
    
    pages_dir.mkdir(parents=True, exist_ok=True)
    
    all_p_value_info = []
    individual_pdf_paths = []
    
    for i, var in enumerate(vars):
        # Create filename for individual page and CSV
        page_filename = f"{var}.pdf"
        csv_filename = f"{var}.csv"
        page_path = pages_dir / page_filename
        csv_path = pages_dir / csv_filename
        
        # Skip if page already exists
        if page_path.exists():
            print(f"\nskipping {var} ({i+1}/{len(vars)}) - already exists")
            individual_pdf_paths.append(page_path)
            # Try to load p_value_info from CSV if it exists
            if csv_path.exists():
                try:
                    p_value_info_df = pd.read_csv(csv_path)
                    p_value_info_dic = p_value_info_df.iloc[0].to_dict()
                    all_p_value_info.append(p_value_info_dic)
                except Exception as e:
                    print(f"  Warning: Could not load CSV for {var}: {e}")
                    all_p_value_info.append({"var": var, "skipped": True})
            else:
                all_p_value_info.append({"var": var, "skipped": True})
            continue
        
        print("\nplotting", var, f"({i+1}/{len(vars)})")

        all_df = with_var_column(all_base_df, presel_path, var)
        pred_df = all_df.filter(pred_filter)
        data_df = all_df.filter(data_filter)
        del all_df

        if args.no_systematics:
            use_rw_systematics = False
            use_detvar_systematics = False
            detvar_df = None
            include_decomposition = False
            weights_df = None
            return_p_value_info = False
        else:
            use_rw_systematics = True
            use_detvar_systematics = True
            detvar_df = with_var_column(presel_detvar_base_df, detvar_path, var)
            include_decomposition = True
            weights_df = presel_weights_df
            return_p_value_info = True

        try:
            p_value_info_dic = make_histogram_plot(pred_sel_df=pred_df, data_sel_df=data_df,
                include_overflow=False, include_underflow=False, log_y=True, include_legend=False,
                var=var, title="Preselection", selname="wc_generic_sel",
                include_ratio=True, include_decomposition=include_decomposition,
                use_rw_systematics=use_rw_systematics, use_detvar_systematics=use_detvar_systematics, detvar_df=detvar_df,
                page_num=i+1, weights_df=weights_df, require_all_rw_events=True,
                show=False, return_p_value_info=return_p_value_info,
                )
        except BaseException as exc:
            # pyo3 PanicException derives from BaseException, not Exception
            if isinstance(exc, SuspectedPolarsCorruption) or type(exc).__name__ == "PanicException":
                print(f"\nWARNING: suspected polars data corruption while plotting {var}: {type(exc).__name__}: {exc}", flush=True)
                sys.exit(CORRUPTION_EXIT_CODE)
            raise

        all_p_value_info.append(p_value_info_dic)
        
        # Save individual page
        plt.savefig(page_path)
        plt.close()
        individual_pdf_paths.append(page_path)
        
        # Save p-value info as CSV
        p_value_info_df = pd.DataFrame([p_value_info_dic])
        p_value_info_df.to_csv(csv_path, index=False)
    
    # Concatenate all pages into one PDF
    print("\nConcatenating all pages into final PDF...")
    final_pdf_path = PROJECT_ROOT / "plots" / "all_bdt_vars.pdf"
    pdf_writer = PdfWriter()
    
    for page_path in individual_pdf_paths:
        if page_path.exists():
            pdf_writer.append(str(page_path))
    
    with open(final_pdf_path, 'wb') as output_pdf:
        pdf_writer.write(output_pdf)
    
    print(f"Final PDF saved to {final_pdf_path}")

    print("saving all_p_value_info to a pickle file...")
    with open(PROJECT_ROOT / "plots" / "all_p_value_info.pkl", "wb") as f:
        pickle.dump(all_p_value_info, f)

    main_end_time = time.time()
    print(f"Total time to create plots: {format_duration(main_end_time - main_start_time)}")
