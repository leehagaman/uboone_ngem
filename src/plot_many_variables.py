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

    print("loading presel_weights_df.parquet...")
    presel_weights_df = pl.read_parquet(f"{intermediate_files_location}/presel_weights_df.parquet")
    print(f"{presel_weights_df.shape=}")

    pred_df = all_df.filter(
        ~pl.col("filetype").is_in(["data", "isotropic_one_gamma_overlay", "delete_one_gamma_overlay"])
    )
    data_df = all_df.filter(
        pl.col("filetype") == "data"
    )

    all_p_value_info = []
    with PdfPages(PROJECT_ROOT / "plots" / "all_bdt_vars_open_data.pdf") as pdf:
        for i, var in tqdm(enumerate(vars), total=len(vars)):
            print("\nplotting", var)

            p_value_info_dic = make_histogram_plot(pred_sel_df=pred_df, data_sel_df=data_df, 
                include_overflow=False, include_underflow=False, log_y=True, include_legend=False,
                var=var, title="Preselection", selname="wc_generic_sel",
                include_ratio=True, include_decomposition=True,
                use_rw_systematics=True, use_detvar_systematics=True, detvar_df=presel_detvar_df,
                page_num=i+1, weights_df=presel_weights_df,
                show=False, return_p_value_info=True,
                )

            all_p_value_info.append(p_value_info_dic)
            pdf.savefig()
            plt.close()

    print("saving all_p_value_info to a pickle file...")
    with open(f"{intermediate_files_location}/all_p_value_info.pkl", "wb") as f:
        pickle.dump(all_p_value_info, f)

    main_end_time = time.time()
    print(f"Total time to create plots: {main_end_time - main_start_time:.2f} seconds ({((main_end_time - main_start_time) / 60):.2f} minutes, {((main_end_time - main_start_time) / 3600):.2f} hours)")
