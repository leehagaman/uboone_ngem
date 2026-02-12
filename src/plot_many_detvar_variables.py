import sys
from pathlib import Path

# Add project root to path to allow imports with src. prefix
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.2
import numpy as np
import polars as pl

import argparse
import time
import shutil
from pypdf import PdfWriter

from src.file_locations import intermediate_files_location
from src.plot_helpers import make_det_variation_histogram, auto_binning
from src.ntuple_variables.variables import combined_training_vars
from src.df_helpers import get_vals

from src.ntuple_variables.pandora_variables import blip_postprocessing_vars

plt.rcParams.update({'font.size': 12})


if __name__ == "__main__":
    main_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_plots", type=int, default=None)
    parser.add_argument("--vars", type=str, default=None)
    parser.add_argument("--clear-prev", action="store_true", help="Delete all files in plots/all_bdt_vars before starting")
    args = parser.parse_args()

    vertex_vars = [
        "wc_reco_nuvtxX",
        "wc_reco_nuvtxY",
        "wc_reco_nuvtxZ",
        "lantern_vtxX",
        "lantern_vtxY",
        "lantern_vtxZ",
        "pandora_reco_nu_vtx_x",
        "pandora_reco_nu_vtx_y",
        "pandora_reco_nu_vtx_z",
        "glee_reco_vertex_x",
        "glee_reco_vertex_y",
        "glee_reco_vertex_z",
        "wc_pandora_dist",
        "wc_lantern_dist",
        "lantern_pandora_dist",
    ]

    print(f"{args.vars=}")

    if args.vars is None:
        vars = combined_training_vars

        vars += vertex_vars

        vars += ["wc_kine_reco_Enu"]

        vars += [
            "wc_flash_measPe",
            "wc_flash_predPe",
            "wc_WCPMTInfoChi2",
            "wc_WCPMTInfoNDF",
            "(wc_flash_measPe - wc_flash_predPe) / wc_flash_predPe", 
            "wc_WCPMTInfoChi2 / wc_WCPMTInfoNDF",
        ]

        vars += blip_postprocessing_vars


    elif args.vars == "vertex_vars":
        vars = vertex_vars
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
    cv_df = presel_detvar_df.filter(pl.col("vartype") == "CV")

    print(f"{presel_detvar_df.shape=}, {cv_df.shape=}")

    # Create directory for individual pages
    pages_dir = PROJECT_ROOT / "plots" / "detvar_plots"
    
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
    
    individual_pdf_paths = []

    for i, var in enumerate(vars):
        # Create filename for individual page
        page_filename = f"{var}.pdf"
        page_filename = page_filename.replace("/", "_").replace(" ", "_")

        page_path = pages_dir / page_filename

        # Skip if page already exists
        if page_path.exists():
            print(f"\nskipping {var} ({i+1}/{len(vars)}) - already exists")
            individual_pdf_paths.append(page_path)
            continue

        print("\nplotting", var, f"({i+1}/{len(vars)})")

        # Auto-determine bins from the CV detvar data
        all_vals = get_vals(cv_df, var)
        bins, display_bins, display_bin_centers, log_x = auto_binning(all_vals)

        make_det_variation_histogram(
            var=var, display_var=var, bins=bins, display_bins=display_bins,
            display_bin_centers=display_bin_centers,
            log_x=log_x, log_y=True,
            page_num=i+1, show=False, detvar_df=presel_detvar_df,
        )

        # Save individual page
        plt.savefig(page_path)
        plt.close()
        individual_pdf_paths.append(page_path)
    
    # Concatenate all pages into one PDF
    print("\nConcatenating all pages into final PDF...")
    final_pdf_path = PROJECT_ROOT / "plots" / "all_detvar_plots.pdf"
    pdf_writer = PdfWriter()
    
    for page_path in individual_pdf_paths:
        if page_path.exists():
            pdf_writer.append(str(page_path))
    
    with open(final_pdf_path, 'wb') as output_pdf:
        pdf_writer.write(output_pdf)
    
    print(f"Final PDF saved to {final_pdf_path}")

    main_end_time = time.time()
    print(f"Total time to create plots: {main_end_time - main_start_time:.2f} seconds ({((main_end_time - main_start_time) / 60):.2f} minutes, {((main_end_time - main_start_time) / 3600):.2f} hours)")
