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
import time
import shutil
from pypdf import PdfWriter

from src.signal_categories import topological_category_labels, topological_category_colors, topological_category_labels_latex, topological_category_hatches, topological_categories_dic
from src.signal_categories import filetype_category_labels, filetype_category_colors, filetype_category_hatches
from src.signal_categories import del1g_detailed_category_labels, del1g_detailed_category_colors, del1g_detailed_category_labels_latex, del1g_detailed_category_hatches, del1g_detailed_categories_dic
from src.signal_categories import del1g_simple_category_labels, del1g_simple_category_colors, del1g_simple_category_labels_latex, del1g_simple_category_hatches, del1g_simple_categories_dic
from src.signal_categories import train_category_labels, train_category_labels_latex

from src.ntuple_variables.pandora_variables import pandora_scalar_second_half_training_vars
from src.file_locations import intermediate_files_location
from src.plot_helpers import make_histogram_plot
from src.ntuple_variables.variables import combined_training_vars
from src.systematics import get_significance_from_p_value

plt.rcParams.update({'font.size': 12})


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

    print("loading presel_detvar_df_train_vars.parquet...")
    presel_detvar_df = pl.read_parquet(f"{intermediate_files_location}/detvar_presel_df_train_vars.parquet")
    print(f"{presel_detvar_df.shape=}")

    print("loading all_df.parquet...")
    all_df = pl.read_parquet(f"{intermediate_files_location}/presel_df_train_vars.parquet")
    print(f"{all_df.shape=}")

    if not args.no_systematics:
        print("loading presel_weights_df.parquet...")
        presel_weights_df = pl.read_parquet(f"{intermediate_files_location}/presel_weights_df.parquet")
        print(f"{presel_weights_df.shape=}")

    pred_df = all_df.filter(
        ~pl.col("filetype").is_in(["data", "isotropic_one_gamma_overlay", "delete_one_gamma_overlay"])
    )
    data_df = all_df.filter(
        pl.col("filetype") == "data"
    )

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
            detvar_df = presel_detvar_df
            include_decomposition = True
            weights_df = presel_weights_df
            return_p_value_info = True

        p_value_info_dic = make_histogram_plot(pred_sel_df=pred_df, data_sel_df=data_df, 
            include_overflow=False, include_underflow=False, log_y=True, include_legend=False,
            var=var, title="Preselection", selname="wc_generic_sel",
            include_ratio=True, include_decomposition=include_decomposition,
            use_rw_systematics=use_rw_systematics, use_detvar_systematics=use_detvar_systematics, detvar_df=detvar_df,
            page_num=i+1, weights_df=weights_df,
            show=False, return_p_value_info=return_p_value_info,
            )

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
    print(f"Total time to create plots: {main_end_time - main_start_time:.2f} seconds ({((main_end_time - main_start_time) / 60):.2f} minutes, {((main_end_time - main_start_time) / 3600):.2f} hours)")
