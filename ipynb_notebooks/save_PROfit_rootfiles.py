#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.2
import numpy as np
import polars as pl
import xgboost as xgb
from tqdm.notebook import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from src.signal_categories import del1g_detailed_category_labels, del1g_detailed_category_colors, del1g_detailed_category_labels_latex
from src.signal_categories import del1g_detailed_category_hatches, del1g_detailed_categories_dic, del1g_detailed_category_queries
from src.signal_categories import train_category_labels, train_category_labels_latex

from src.file_locations import intermediate_files_location

from src.plot_helpers import make_histogram_plot

from src.ntuple_variables.variables import combined_training_vars

from src.df_helpers import lazy_height, get_vals


# In[2]:


save_splines_now = True
save_detvar_now = True


# In[3]:


training = "all_vars"
training_vars = combined_training_vars
reco_categories = train_category_labels
reco_category_labels_latex = train_category_labels_latex


# # File Loading

# In[4]:


print("Loading all_df.parquet...")
all_df = pl.scan_parquet(f"{intermediate_files_location}/all_df.parquet")
print(f"Total events: {lazy_height(all_df)}")

print("loading weights_df.parquet...")
weights_df = pl.scan_parquet(f"{intermediate_files_location}/presel_weights_df.parquet")
print(f"num events in weights_df: {lazy_height(weights_df)}")

print("Loading predictions.parquet...")
pred_bdt_scores_df = pl.scan_parquet(f"../training_outputs/{training}/predictions.parquet")
print(f"Total events: {lazy_height(pred_bdt_scores_df)}")


# In[5]:


print("merging all_df and predictions.pkl...")
merged_df_no_data_drop = all_df.join(
    pred_bdt_scores_df, 
    on=["filetype", "run", "subrun", "event"], 
    how="left"
)

del all_df
del pred_bdt_scores_df


# In[6]:


# adding empty score columns
prob_categories = ["prob_" + cat for cat in reco_categories]
for prob in prob_categories:
    merged_df_no_data_drop = merged_df_no_data_drop.with_columns(pl.col(prob).fill_null(-1))
    
    # Get the probabilities and find argmax index for each row
merged_df_no_data_drop = merged_df_no_data_drop.with_columns(
    pl.concat_list(prob_categories).list.arg_max().alias("reco_category_argmax_index")
)

# removing data and training-only events that aren't part of the prediction
full_pred = merged_df_no_data_drop.filter(
    ~pl.col("filetype").is_in(["data", "isotropic_one_gamma_overlay", "delete_one_gamma_overlay"])
)
full_data = merged_df_no_data_drop.filter(pl.col("filetype") == "data")

# Build list of query strings
reco_category_argmax_queries = []
for i, signal_category in enumerate(reco_categories):
    reco_category_argmax_queries.append(pl.col("reco_category_argmax_index") == i)

generic_pred_df = full_pred.filter(pl.col("wc_kine_reco_Enu") > 0)
del full_pred

num_train_events = lazy_height(generic_pred_df.filter(pl.col("used_for_training") == True))
num_test_events = lazy_height(generic_pred_df.filter(pl.col("used_for_testing") == True))
print(f"num_train_events: {num_train_events}, num_test_events: {num_test_events}")

frac_test = num_test_events / (num_train_events + num_test_events)
print(f"weighting up preselected prediction events by the fraction of test/train events: {frac_test:.3f}")

# Modify weights using polars expressions
generic_pred_df = generic_pred_df.with_columns(
    pl.when(pl.col("used_for_testing"))
    .then(pl.col("wc_net_weight") / frac_test)
    .otherwise(pl.col("wc_net_weight"))
    .alias("wc_net_weight"),
    
    pl.when(pl.col("used_for_testing"))
    .then(pl.col("run4b_only_wc_net_weight") / frac_test)
    .otherwise(pl.col("run4b_only_wc_net_weight"))
    .alias("run4b_only_wc_net_weight")
)

generic_test_pred_df = generic_pred_df.filter(pl.col("used_for_testing") == True)
del generic_pred_df

generic_data_df = full_data.filter(pl.col("wc_kine_reco_Enu") > 0)
del full_data

generic_test_pred_data_df = pl.concat([generic_test_pred_df, generic_data_df])
del generic_test_pred_df


# In[7]:


generic_test_pred_data_df.select(["filetype", "detailed_run_period", "run", "subrun", "event", "wc_kine_reco_Enu"]).collect()


# # Generic All Runs

# In[8]:


make_histogram_plot(
    pred_and_data_sel_df=generic_test_pred_data_df,
    weights_df=None, # streamed lazily at plot generation
    use_rw_systematics=True,
    bins=np.linspace(0, 2000, 21),
    var="wc_kine_reco_Enu",
    display_var=r"WC Reconstructed $E_\nu$ (GeV)",
    data_type="4a+4b open data",
    title="Run 4a+4b WC Generic Neutrino Selection",
    selname="generic_presel"
)


# # Generic Run 4b

# In[9]:


run4b_generic_test_pred_data_lazy_df = generic_test_pred_data_df.filter(pl.col("detailed_run_period")=="4b")

generic_test_pred_data_df.select(["filetype", "detailed_run_period", "run", "subrun", "event", "wc_kine_reco_Enu"]).collect()


# In[10]:


# putting the dataframe into memory, sacrificing memory usage for speed

plot_cols = ["filetype", "run", "subrun", "event",
             "wc_kine_reco_Enu", "wc_net_weight", "run4b_only_wc_net_weight",
             # breakdown query columns:
             "normal_overlay", "wc_truth_inFV", "wc_truth_NCDeltaRad", "wc_truth_0pi0",
             "wc_truth_Np", "wc_truth_0p", "wc_truth_numuCCDeltaRad", "wc_truth_isNC",
             "wc_truth_nueCC", "wc_truth_notnueCC",
             "wc_truth_numuCC", "wc_truth_notnumuCC", "wc_truth_1mu", "wc_truth_0mu",
             "wc_truth_1pi0", "wc_truth_multi_pi0", "wc_true_has_pi0_dalitz_decay",
             "true_num_prim_gamma",
             "wc_true_has_photonuclear_absorption", "true_num_gamma_pairconvert_in_FV",
             "true_num_gamma_pairconvert_in_FV_20_MeV",
             "wc_true_gamma_pairconversion_spacepoint_max_min_distance",
             "del1g_overlay", "iso1g_overlay",
             "wc_truth_nuEnergy",
             #"wc_mcflux_dk2gen", "wc_mcflux_gen2vtx",
             # reco category query columns:
             "reco_category_argmax_index"] + [f"prob_{cat}" for cat in reco_categories]

run4b_generic_test_pred_data_df = run4b_generic_test_pred_data_lazy_df.select(plot_cols).collect()


# In[11]:


make_histogram_plot(
    pred_and_data_sel_df=run4b_generic_test_pred_data_df,
    weights_df=weights_df, # streamed lazily at plot generation
    use_rw_systematics=True,
    bins=np.linspace(0, 2000, 21),
    var="wc_kine_reco_Enu",
    display_var=r"WC Reconstructed $E_\nu$ (GeV)",
    data_type="4b open data",
    title="Run 4b WC Generic Neutrino Selection",
    selname="generic_presel",
    net_weight_var="run4b_only_wc_net_weight"
)


# # Multi-Class Selections

# In[12]:


name_expr_priority_vals_w_None = [
    ("1gNp", pl.col("prob_1gNp") > 0.3, 1),
    ("1g0p", pl.col("prob_1g0p") > 0.9, 2),
    ("1gNp1mu", pl.col("prob_1gNp1mu") > 0.5, 3),
    ("1g0p1mu", pl.col("prob_1g0p1mu") > 0.2, 4),
    ("1g_outFV", pl.col("prob_1g_outFV") > 0.5, 5),
    ("NC1pi0_Np", None, 6),
    ("NC1pi0_0p", None, 7),
    ("numuCC1pi0_Np", pl.col("prob_numuCC1pi0_Np") > 0.1, 9),
    ("numuCC1pi0_0p", pl.col("prob_numuCC1pi0_0p") > 0.15, 8), # 0p takes priority over Np in orthogonality
    ("1pi0_outFV", pl.col("prob_1pi0_outFV") > 0.1, 10),
    ("nueCC_Np", pl.col("prob_nueCC_Np") > 0.05, 12),
    ("nueCC_0p", pl.col("prob_nueCC_0p") > 0.05, 11), # 0p takes priority over Np in orthogonality
    ("numuCC_Np", pl.col("prob_numuCC_Np") > 0.5, 13),
    ("numuCC_0p", pl.col("prob_numuCC_0p") > 0.5, 14),
    ("other_outFV_dirt", None, 15),
    ("multi_pi0", pl.col("prob_multi_pi0") > 0.02, 16),
    ("eta_other", pl.col("prob_eta_other") > 0.01, 17),
    ("pi0_dalitz_decay", pl.col("prob_pi0_dalitz_decay") > 0.1, 5.5), # high priority for dalitz, rare topology
    ("NC_no_gamma", None, 19),
    ("ext", None, 20),
]

name_expr_priority_vals_possible_overlap = []
for i, name_expr_priority_val_w_None in enumerate(name_expr_priority_vals_w_None):
    name, expr, priority = name_expr_priority_val_w_None
    if expr is None:
        expr = reco_category_argmax_queries[i]
    name_expr_priority_vals_possible_overlap.append((name, expr, priority))

name_expr_priority_vals_possible_overlap.sort(key=lambda x: x[2])

name_expr_priority_vals = []
for i, name_expr_priority_val_possible_overlap in enumerate(name_expr_priority_vals_possible_overlap):
    name, expr, priority = name_expr_priority_val_possible_overlap
    for j in range(i):
        expr = expr & ~name_expr_priority_vals_possible_overlap[j][1]
    name_expr_priority_vals.append((name, expr, priority))

reco_category_query_dic = {}
for name, expr, priority in name_expr_priority_vals:
    reco_category_query_dic[name] = expr

reco_category_queries = []
for reco_category in reco_categories:
    reco_category_queries.append(reco_category_query_dic[reco_category])


# In[13]:


additional_scaling_factor = 1

breakdown_queries = del1g_detailed_category_queries
breakdown_labels = del1g_detailed_category_labels
breakdown_labels_latex = del1g_detailed_category_labels_latex
breakdown_colors = del1g_detailed_category_colors
breakdown_hatches = del1g_detailed_category_hatches

fig, axs = plt.subplots(7, 5, figsize=(20, 20))
axs = axs.flatten()

bins = np.linspace(0, 2000, 21)
bin_centers = (bins[:-1] + bins[1:]) / 2

for i in tqdm(range(len(reco_categories))):

    signal_category = reco_categories[i]
    signal_category_latex = reco_category_labels_latex[i]

    sel_df = run4b_generic_test_pred_data_df.filter(reco_category_queries[i])

    pred_sel_df = sel_df.filter(pl.col("filetype") != "data")
    data_sel_df = sel_df.filter(pl.col("filetype") == "data")

    breakdown_counts = []
    for breakdown_i, breakdown_label in enumerate(breakdown_labels):
        curr_df = pred_sel_df.filter(eval(breakdown_queries[breakdown_i], {'pl': pl, '__builtins__': {}}))
        breakdown_counts.append(np.histogram(get_vals(curr_df, "wc_kine_reco_Enu"),
                                            weights=get_vals(curr_df, "run4b_only_wc_net_weight")*additional_scaling_factor,
                                            bins=bins)[0])
    data_counts = np.histogram(get_vals(data_sel_df, "wc_kine_reco_Enu"), bins=bins)[0]

    axnum = i

    bottom = np.zeros(len(bins)-1)
    for breakdown_i, (breakdown_label, breakdown_count, breakdown_color, breakdown_hatch, breakdown_label_latex) in enumerate(zip(breakdown_labels, breakdown_counts, breakdown_colors, breakdown_hatches, breakdown_labels_latex)):
        if "data" in breakdown_label:
            continue

        n, _, _ = axs[axnum].hist(bin_centers, weights=breakdown_count, bins=bins, bottom=bottom if breakdown_i > 0 else None, color=breakdown_color, hatch=breakdown_hatch, label=breakdown_label_latex)
        axs[axnum].hist(bin_centers, weights=breakdown_count, bins=bins, bottom=bottom if breakdown_i > 0 else None, histtype="step", color="k", lw=0.5)

        if breakdown_i == 0:
            bottom = n
        else:
            bottom += n

    axs[axnum].errorbar(bin_centers, data_counts, yerr=np.sqrt(data_counts), fmt="o", color="k", lw=0.5, capsize=2, capthick=1, markersize=2, 
                        label=f"{4.038e19:.2e} POT Run 4b Open Data")

    max_pred = np.max(bottom)
    max_data = np.max(data_counts)

    axs[axnum].set_ylim(0, max(max_pred, max_data) * 1.1)

    if axnum == 19:
        axs[axnum].legend(ncol=4, loc='upper right', bbox_to_anchor=(-1, -0.5))

    if axnum in [15, 16, 17, 18, 19]:
        axs[axnum].set_xlabel(r"WC Reconstructed $E_\nu$ (MeV)")
    if axnum % 5 == 0: # Only show y-label for leftmost column
        if additional_scaling_factor != 1.0:
            axs[axnum].set_ylabel(f"Counts (weighted\nto {additional_scaling_factor*4.038e19:.2e)} POT)")
        else:
            axs[axnum].set_ylabel("Counts (weighted\nto "f"{4.038e19:.2e} POT)")
    axs[axnum].set_title(f"{signal_category_latex} Selection")
    axs[axnum].set_xlim(0, 2000)

for axnum in range(len(axs)):
    if axnum > 19:
        axs[axnum].remove()

fig.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.15)

plt.show()


# # Minimal df

# In[ ]:


expr = pl.when(reco_category_queries[0]).then(0)
for i in range(1, len(reco_categories)):
    expr = expr.when(reco_category_queries[i]).then(i)
expr = expr.otherwise(None)

run4b_generic_test_pred_data_df = run4b_generic_test_pred_data_df.with_columns(
    expr.cast(pl.Int32).alias("reco_category")
)

minimal_df = run4b_generic_test_pred_data_df.select([
    "filetype", "run", "subrun", "event", "reco_category", "wc_kine_reco_Enu", "run4b_only_wc_net_weight"
    ] + prob_categories)

minimal_df = (
    minimal_df.with_columns([
        (pl.col("filetype") == "data")
        .alias("isdata"),
        (pl.col("filetype") == "ext")
        .alias("isext"),
        (pl.col("filetype") == "dirt_overlay")
        .alias("isdirt"),
    ])
)

display(minimal_df.select("filetype")["filetype"].value_counts())

minimal_df


# In[15]:


fig, axs = plt.subplots(7, 5, figsize=(20, 20))
axs = axs.flatten()

bins = np.linspace(0, 2000, 21)
bin_centers = (bins[:-1] + bins[1:]) / 2

for i in tqdm(range(len(reco_categories))):

    signal_category_latex = reco_category_labels_latex[i]

    minimal_reco_df = minimal_df.filter(pl.col("reco_category") == i)
    minimal_reco_overlay_df = minimal_reco_df.filter((pl.col("filetype") != "data") & ~pl.col("isext") & ~pl.col("isdirt"))
    minimal_reco_dirt_df = minimal_reco_df.filter(pl.col("isdirt"))
    minimal_reco_ext_df = minimal_reco_df.filter(pl.col("isext"))
    minimal_reco_data_df = minimal_reco_df.filter(pl.col("filetype") == "data")

    overlay_counts = np.histogram(minimal_reco_overlay_df.get_column("wc_kine_reco_Enu").to_numpy(),
                                  weights=minimal_reco_overlay_df.get_column("run4b_only_wc_net_weight").to_numpy(), bins=bins)[0]
    dirt_counts = np.histogram(minimal_reco_dirt_df.get_column("wc_kine_reco_Enu").to_numpy(),
                               weights=minimal_reco_dirt_df.get_column("run4b_only_wc_net_weight").to_numpy(), bins=bins)[0]
    ext_counts = np.histogram(minimal_reco_ext_df.get_column("wc_kine_reco_Enu").to_numpy(),
                              weights=minimal_reco_ext_df.get_column("run4b_only_wc_net_weight").to_numpy(), bins=bins)[0]
    pred_counts = overlay_counts + dirt_counts + ext_counts

    data_counts = np.histogram(minimal_reco_data_df.get_column("wc_kine_reco_Enu").to_numpy(), bins=bins)[0]

    axnum = i

    axs[axnum].hist(bin_centers, weights=overlay_counts, bins=bins, color="blue", label="overlay")
    axs[axnum].hist(bin_centers, weights=dirt_counts, bins=bins, bottom=overlay_counts, color="brown", label="dirt")
    axs[axnum].hist(bin_centers, weights=ext_counts, bins=bins, bottom=overlay_counts + dirt_counts, color="green", label="ext")
    axs[axnum].errorbar(bin_centers, data_counts, yerr=np.sqrt(data_counts), fmt="o", color="k", lw=0.5, capsize=2, capthick=1, markersize=2,
                        label=f"{4.038e19:.2e} POT Run 4b Open Data")

    max_pred = np.max(pred_counts)
    max_data = np.max(data_counts)

    axs[axnum].set_ylim(0, max(max_pred, max_data) * 1.1)

    if axnum == 19:
        axs[axnum].legend(ncol=4, loc='upper right', bbox_to_anchor=(-1, -0.5))

    if axnum in [15, 16, 17, 18, 19]:
        axs[axnum].set_xlabel(r"WC Reconstructed $E_\nu$ (MeV)")
    if axnum % 5 == 0: # Only show y-label for leftmost column
        if additional_scaling_factor != 1.0:
            axs[axnum].set_ylabel(f"Counts (weighted\nto {additional_scaling_factor*4.038e19:.2e} POT)")
        else:
            axs[axnum].set_ylabel("Counts (weighted\nto "f"{4.038e19:.2e} POT)")
    axs[axnum].set_title(f"{signal_category_latex} Selection")
    axs[axnum].set_xlim(0, 2000)

for axnum in range(len(axs)):
    if axnum > 19:
        axs[axnum].remove()

fig.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.15)

plt.show()


# # Merging Splines

# In[16]:


print("Loading spline_weights_df.parquet...")
spline_weights_df = pl.scan_parquet(f"{intermediate_files_location}/spline_weights_df.parquet")
print(f"Total events: {lazy_height(spline_weights_df)}")

spline_weights_df = spline_weights_df.with_columns(
    pl.lit(True).alias("has_spline_weights")
)

spline_weights_df.select(["filetype", "run", "subrun", "event", "MaCCQE_UBGenie"]).collect()


# In[17]:


data_minimal_df = minimal_df.filter((pl.col("filetype") == "data") | (pl.col("filetype") == "ext"))
mc_minimal_df = minimal_df.filter((pl.col("filetype") != "data") & (pl.col("filetype") != "ext"))

merged_mc_minimal_spline_df = mc_minimal_df.lazy().join(spline_weights_df, on=["filetype", "run", "subrun", "event"], how="left")

merged_mc_minimal_spline_df.select(["filetype", "run", "subrun", "event", 
                                    "run4b_only_wc_net_weight", "has_spline_weights", "MaCCQE_UBGenie"]).collect()


# In[18]:


"""for filetype in merged_mc_minimal_spline_df.select("filetype").collect()["filetype"].unique():
    if filetype in ["data", "ext"]:
        continue

    num_events = (
        merged_mc_minimal_spline_df
        .filter(pl.col("filetype") == filetype)
        .select("has_spline_weights")
        .collect()
        .height
    )

    num_with_spline_weights = (
        merged_mc_minimal_spline_df
        .filter(pl.col("filetype") == filetype)
        .select(pl.col("has_spline_weights").sum())
        .collect()
        .item()
    )

    print(f"{filetype} fraction with spline weights included: {num_with_spline_weights} / {num_events} = {num_with_spline_weights / num_events}")
"""

fractions = (
    merged_mc_minimal_spline_df
    .group_by("filetype")
    .agg([
        pl.len().alias("num_events"),
        pl.col("has_spline_weights").sum().alias("num_with_spline_weights"),
    ])
    .with_columns([
        (pl.col("num_with_spline_weights") / pl.col("num_events"))
        .alias("fraction_with_spline_weights")
    ])
    .with_columns([
        # inverse weight; override for data/ext
        pl.when(pl.col("filetype").is_in(["data", "ext"]))
        .then(1.0)
        .otherwise(1.0 / pl.col("fraction_with_spline_weights"))
        .alias("spline_processed_fraction_weight")
    ])
    .select(["filetype", "fraction_with_spline_weights", "spline_processed_fraction_weight"])
)

# join back onto original dataframe
merged_mc_minimal_spline_df = (
    merged_mc_minimal_spline_df
    .join(fractions, on="filetype", how="left")
)

# cut out non-spline weight events (the weighting just accounted for this)
merged_mc_minimal_withspline_df = (
    merged_mc_minimal_spline_df
    .filter(pl.col("has_spline_weights"))
)

# create net_weight column
merged_mc_minimal_withspline_df = (
    merged_mc_minimal_withspline_df.with_columns([
        (pl.col("run4b_only_wc_net_weight") * pl.col("spline_processed_fraction_weight"))
        .alias("net_weight")
    ])
)

merged_mc_minimal_withspline_df.select(["filetype", "run", "subrun", "event", 
                                    "run4b_only_wc_net_weight", "spline_processed_fraction_weight", "net_weight", 
                                    "has_spline_weights", "MaCCQE_UBGenie"]).collect()


# In[ ]:


# re-merge data_minimal_df with merged_mc_minimal_withspline_df:
# add the same spline-weight column structure to data with all-unity list values,
# set net_weight = run4b_only_wc_net_weight, then concat both sides.

mc_schema = merged_mc_minimal_withspline_df.collect_schema()
mc_cols = mc_schema.names()
missing_cols = [c for c in mc_cols if c not in data_minimal_df.columns]

list_cols_missing = [c for c in missing_cols if isinstance(mc_schema[c], pl.List)]

# per-column list length, pulled from the first row of the MC side
list_lens_row = (
    merged_mc_minimal_withspline_df
    .select([pl.col(c).list.len().alias(c) for c in list_cols_missing])
    .head(1)
    .collect()
)
list_col_lengths = {c: int(list_lens_row[c][0]) for c in list_cols_missing}

extra_exprs = []
for c in missing_cols:
    dtype = mc_schema[c]
    if isinstance(dtype, pl.List):
        extra_exprs.append(pl.lit([1.0] * list_col_lengths[c], dtype=dtype).alias(c))
    elif c == "has_spline_weights":
        extra_exprs.append(pl.lit(True).alias(c))
    elif c == "net_weight":
        extra_exprs.append(pl.col("run4b_only_wc_net_weight").cast(dtype).alias(c))
    elif c in ("fraction_with_spline_weights", "spline_processed_fraction_weight"):
        extra_exprs.append(pl.lit(1.0).cast(dtype).alias(c))
    else:
        extra_exprs.append(pl.lit(None).cast(dtype).alias(c))

data_minimal_withspline_df = (
    data_minimal_df.lazy()
    .with_columns(extra_exprs)
    .select(mc_cols)
)

minimal_withspline_df = pl.concat([merged_mc_minimal_withspline_df, data_minimal_withspline_df])

minimal_withspline_df.select([
    "filetype", "isdata", "isext", "isdirt", "run", "subrun", "event",
    "run4b_only_wc_net_weight", "net_weight", "has_spline_weights", "MaCCQE_UBGenie",
]).collect()


# # Save to ROOT

# In[20]:


output_path = f"{intermediate_files_location}/minimal_withspline_df.root"


# In[21]:


import ROOT
import numpy as np

if save_splines_now:

    df_to_save = minimal_withspline_df.collect()

    # Pre-extract every column as Python-native data tagged with its kind.
    # Integer columns containing nulls fall through to float (NaN) — matching
    # the dtype-promotion behavior of the previous uproot-based writer.
    columns = {}
    for col in df_to_save.columns:
        s = df_to_save[col]
        if isinstance(s.dtype, pl.List):
            columns[col] = ("list", s.fill_null([]).to_list())
        elif s.dtype in (pl.String, pl.Utf8):
            columns[col] = ("str", s.fill_null("").to_list())
        elif s.dtype == pl.Boolean:
            columns[col] = ("bool", s.fill_null(False).to_numpy())
        else:
            arr = s.to_numpy()
            if arr.dtype.kind in ("i", "u"):
                columns[col] = ("int", arr.astype(np.int32))
            else:
                columns[col] = ("float", arr.astype(np.float64))

    f = ROOT.TFile.Open(output_path, "RECREATE")
    tree = ROOT.TTree("tree", "tree")

    # Allocate per-branch buffers and bind them. Jagged float columns become
    # std::vector<double> object branches — what PROfit's SetBranchAddress
    # pattern expects.
    buffers = {}
    for col, (kind, _) in columns.items():
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

    # Fill rows.
    n = df_to_save.height
    for i in tqdm(range(n)):
        for col, (kind, data) in columns.items():
            buf = buffers[col]
            v = data[i]
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

    tree.Write()
    f.Close()
    print(f"wrote {n} events to {output_path}")


# # Load From ROOT

# In[23]:


# re-make the 20-panel plot using only info loaded from the ROOT file

import uproot

with uproot.open(output_path) as f:
    t = f["tree"]
    arrs = t.arrays(
        ["filetype", "isdata", "isext", "isdirt", "reco_category", "wc_kine_reco_Enu", "net_weight"],
        library="np",
    )

filetype = arrs["filetype"]
isdata = arrs["isdata"]
isext = arrs["isext"]
isdirt = arrs["isdirt"]
print(isdata)
reco_category = arrs["reco_category"]
energies = arrs["wc_kine_reco_Enu"]
net_weight = arrs["net_weight"]

print(f"{np.unique(filetype)=}")

is_data = filetype == "data"

fig, axs = plt.subplots(7, 5, figsize=(20, 20))
axs = axs.flatten()

bins = np.linspace(0, 2000, 21)
bin_centers = (bins[:-1] + bins[1:]) / 2

for i in tqdm(range(len(reco_categories))):

    signal_category_latex = reco_category_labels_latex[i]

    sel = reco_category == i
    overlay_sel = sel & ~is_data & ~isext & ~isdirt
    dirt_sel = sel & isdirt
    ext_sel = sel & isext
    data_sel = sel & is_data

    overlay_counts = np.histogram(energies[overlay_sel], weights=net_weight[overlay_sel], bins=bins)[0]
    dirt_counts = np.histogram(energies[dirt_sel], weights=net_weight[dirt_sel], bins=bins)[0]
    ext_counts = np.histogram(energies[ext_sel], weights=net_weight[ext_sel], bins=bins)[0]
    pred_counts = overlay_counts + dirt_counts + ext_counts
    data_counts = np.histogram(energies[data_sel], bins=bins)[0]

    axnum = i

    axs[axnum].hist(bin_centers, weights=overlay_counts, bins=bins, color="blue", label="overlay")
    axs[axnum].hist(bin_centers, weights=dirt_counts, bins=bins, bottom=overlay_counts, color="brown", label="dirt")
    axs[axnum].hist(bin_centers, weights=ext_counts, bins=bins, bottom=overlay_counts + dirt_counts, color="green", label="ext")
    axs[axnum].errorbar(bin_centers, data_counts, yerr=np.sqrt(data_counts), fmt="o", color="k", lw=0.5, capsize=2, capthick=1, markersize=2,
                        label=f"{4.038e19:.2e} POT Run 4b Open Data")

    max_pred = float(np.max(pred_counts)) if pred_counts.size else 0.0
    max_data = float(np.max(data_counts)) if data_counts.size else 0.0
    ymax = max(max_pred, max_data)
    axs[axnum].set_ylim(0, ymax * 1.1 if ymax > 0 else 1)

    if axnum == 19:
        axs[axnum].legend(ncol=4, loc='upper right', bbox_to_anchor=(-1, -0.5))
    if axnum in [15, 16, 17, 18, 19]:
        axs[axnum].set_xlabel(r"WC Reconstructed $E_\nu$ (MeV)")
    if axnum % 5 == 0:
        axs[axnum].set_ylabel("Counts (weighted\nto "f"{4.038e19:.2e} POT)")
    axs[axnum].set_title(f"{signal_category_latex} Selection (from ROOT)")
    axs[axnum].set_xlim(0, 2000)

for axnum in range(len(axs)):
    if axnum > 19:
        axs[axnum].remove()

fig.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.15)
plt.show()


# In[24]:


# fractional uncertainty per (reco channel, spline knob)
# for each knob k with N universes, compute per-universe selection sum
#   S_u = sum_events(net_weight * spline_weight_k[u])
# then frac_unc = std_u(S_u) / nominal_sum, where nominal_sum = sum_events(net_weight)

minimal_withspline_df = minimal_withspline_df.collect()

spline_cols = [c for c, t in minimal_withspline_df.schema.items() if isinstance(t, pl.List)]

reco_cat_arr = minimal_withspline_df["reco_category"].to_numpy()
nw = minimal_withspline_df["net_weight"].to_numpy()

# loop knobs outer / channels inner so peak memory is one knob's 2D array
results = {i: [] for i in range(len(reco_categories))}
for col in tqdm(spline_cols):
    sw = np.array(minimal_withspline_df[col].to_list())  # shape (E, N_universes)
    for i in range(len(reco_categories)):
        mask = reco_cat_arr == i
        nw_sel = nw[mask]
        nominal_total = float(nw_sel.sum())
        if nominal_total == 0:
            results[i].append((col, float("nan")))
            continue
        contrib = nw_sel[:, None] * sw[mask]
        # drop non-finite per-event contributions (a single nan/inf would
        # otherwise contaminate that universe's sum)
        contrib = np.where(np.isfinite(contrib), contrib, 0.0)
        per_univ_sum = contrib.sum(axis=0)
        finite_universes = per_univ_sum[np.isfinite(per_univ_sum)]
        if finite_universes.size == 0:
            results[i].append((col, float("nan")))
            continue
        results[i].append((col, float(finite_universes.std() / nominal_total)))
    del sw

for i, cat_name in enumerate(reco_categories):
    mask = reco_cat_arr == i
    n_events = int(mask.sum())
    nominal_total = float(nw[mask].sum())
    print(f"\n=== reco_category={i} ({cat_name}): n_events={n_events}, nominal_total={nominal_total:.3f} ===")
    rows = sorted(results[i], key=lambda r: (-r[1]) if r[1] == r[1] else 1)
    for col, frac in rows:
        if frac != frac:
            print(f"  {col:50s}  frac unc =    nan")
        else:
            print(f"  {col:50s}  frac unc = {frac*100:6.2f}%")


# # Minimal DetVar df

# In[25]:


print("Loading presel_detvar_df.parquet...")
presel_detvar_df = pl.scan_parquet(f"{intermediate_files_location}/detvar_presel_df_train_vars.parquet")
print(f"Total events: {lazy_height(presel_detvar_df)}")


# In[26]:


presel_detvar_df = (
    presel_detvar_df.with_columns([
        (pl.col("filetype") == "data")
        .alias("isdata"),
        (pl.col("filetype") == "ext")
        .alias("isext"),
        (pl.col("filetype") == "dirt_overlay")
        .alias("isdirt"),
    ])
)


# ## Doing Inference

# In[27]:


model = xgb.XGBClassifier()
model.load_model(f"../training_outputs/{training}/bdt.json")


# In[28]:


batch_size = 100_000

# Get total rows for progress bar
total_rows = presel_detvar_df.select(pl.len()).collect().item()

all_probabilities = []
offset = 0

with tqdm(total=total_rows, unit="rows") as pbar:
    while True:
        batch = (
            presel_detvar_df
            .select(training_vars)
            .slice(offset, batch_size)
            .collect()
        )

        if batch.height == 0:
            break

        x = batch.to_numpy().astype(np.float64)
        x[np.isinf(x)] = np.nan

        probs = model.predict_proba(x)
        all_probabilities.append(probs)

        offset += batch.height
        pbar.update(batch.height)


# In[29]:


all_probabilities = np.vstack(all_probabilities)

num_probabilities = all_probabilities.shape[1]
for i in tqdm(range(num_probabilities)):
    presel_detvar_df = presel_detvar_df.with_columns(pl.DataFrame({
        f'prob_{train_category_labels[i]}': all_probabilities[:, i]
    }))


# In[30]:


presel_detvar_df.select(["filetype", "vartype", "run", "subrun", "event", 
                         "wc_kine_reco_Enu", "prob_1gNp"]).collect()



# In[31]:


presel_detvar_df.collect()["vartype"].value_counts()


# In[32]:


for prob in prob_categories:
    presel_detvar_df = presel_detvar_df.with_columns(pl.col(prob).fill_null(-1))

presel_detvar_df = presel_detvar_df.with_columns(
    pl.concat_list(prob_categories).list.arg_max().alias("reco_category_argmax_index")
)

expr = pl.when(reco_category_queries[0]).then(0)
for i in range(1, len(reco_categories)):
    expr = expr.when(reco_category_queries[i]).then(i)
expr = expr.otherwise(None)

presel_detvar_df = presel_detvar_df.with_columns(
    expr.cast(pl.Int32).alias("reco_category")
)

detvar_minimal_df = presel_detvar_df.select([
    "filetype", "vartype", "run", "subrun", "event", "reco_category",
    "wc_kine_reco_Enu", "wc_net_weight"
] + prob_categories)

detvar_minimal_df.select(["filetype", "vartype", "run", "subrun", "event",
                          "reco_category", "wc_kine_reco_Enu", "wc_net_weight"]).collect()



# In[33]:


vartypes = detvar_minimal_df.select("vartype").unique().collect()["vartype"].to_list()

if save_detvar_now:
    for vartype in vartypes:
        df_to_save = detvar_minimal_df.filter(pl.col("vartype") == vartype).collect()
        output_path = f"{intermediate_files_location}/minimal_detvar_{vartype}_df.root"

        columns = {}
        for col in df_to_save.columns:
            s = df_to_save[col]
            if isinstance(s.dtype, pl.List):
                columns[col] = ("list", s.fill_null([]).to_list())
            elif s.dtype in (pl.String, pl.Utf8):
                columns[col] = ("str", s.fill_null("").to_list())
            elif s.dtype == pl.Boolean:
                columns[col] = ("bool", s.fill_null(False).to_numpy())
            else:
                arr = s.to_numpy()
                if arr.dtype.kind in ("i", "u"):
                    columns[col] = ("int", arr.astype(np.int32))
                else:
                    columns[col] = ("float", arr.astype(np.float64))

        f = ROOT.TFile.Open(output_path, "RECREATE")
        tree = ROOT.TTree("tree", "tree")

        buffers = {}
        for col, (kind, _) in columns.items():
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

        n = df_to_save.height
        for i in tqdm(range(n), desc=f"writing {vartype}"):
            for col, (kind, data) in columns.items():
                buf = buffers[col]
                v = data[i]
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

        tree.Write()
        f.Close()
        print(f"wrote {n} events to {output_path}")


# In[ ]:


import uproot
import glob

detvar_root_paths = sorted(glob.glob(f"{intermediate_files_location}/minimal_detvar_*_df.root"))

detvar_arrs = {}
for path in detvar_root_paths:
    vartype = os.path.basename(path).removeprefix("minimal_detvar_").removesuffix("_df.root")
    with uproot.open(path) as f:
        t = f["tree"]
        detvar_arrs[vartype] = t.arrays(
            ["reco_category", "wc_kine_reco_Enu", "wc_net_weight"],
            library="np",
        )

# put CV first, the rest in alphabetical order
loaded_vartypes = sorted(detvar_arrs.keys(), key=lambda v: (v != "CV", v))

cmap = plt.get_cmap("tab10")
vartype_colors = {v: ("k" if v == "CV" else cmap(i % 10)) for i, v in enumerate(loaded_vartypes)}

fig, axs = plt.subplots(7, 5, figsize=(20, 20))
axs = axs.flatten()

bins = np.linspace(0, 2000, 21)
bin_centers = (bins[:-1] + bins[1:]) / 2

for i in tqdm(range(len(reco_categories))):

    signal_category_latex = reco_category_labels_latex[i]
    axnum = i

    for vartype in loaded_vartypes:
        a = detvar_arrs[vartype]
        sel = a["reco_category"] == i
        counts = np.histogram(
            a["wc_kine_reco_Enu"][sel],
            weights=a["wc_net_weight"][sel],
            bins=bins,
        )[0]
        lw = 2.0 if vartype == "CV" else 1.0
        axs[axnum].hist(
            bin_centers, weights=counts, bins=bins,
            histtype="step", color=vartype_colors[vartype], lw=lw, label=vartype,
        )

    if axnum == 19:
        axs[axnum].legend(ncol=4, loc="upper right", bbox_to_anchor=(-1, -0.5))
    if axnum in [15, 16, 17, 18, 19]:
        axs[axnum].set_xlabel(r"WC Reconstructed $E_\nu$ (MeV)")
    if axnum % 5 == 0:
        axs[axnum].set_ylabel("Counts (weighted)")
    axs[axnum].set_title(f"{signal_category_latex} Selection (detvar)")
    axs[axnum].set_xlim(0, 2000)

for axnum in range(len(axs)):
    if axnum > 19:
        axs[axnum].remove()

fig.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.15)
plt.show()


# In[ ]:


# reload with the RSE columns included so we can match between CV and each variation
detvar_full_arrs = {}
for path in detvar_root_paths:
    vartype = os.path.basename(path).removeprefix("minimal_detvar_").removesuffix("_df.root")
    with uproot.open(path) as f:
        t = f["tree"]
        detvar_full_arrs[vartype] = t.arrays(
            ["filetype", "run", "subrun", "event",
             "reco_category", "wc_kine_reco_Enu", "wc_net_weight"],
            library="np",
        )

def _to_polars(a):
    return pl.DataFrame({
        "filetype": a["filetype"],
        "run": a["run"],
        "subrun": a["subrun"],
        "event": a["event"],
        "reco_category": a["reco_category"],
        "wc_kine_reco_Enu": a["wc_kine_reco_Enu"],
        "wc_net_weight": a["wc_net_weight"],
    })

cv_pl = _to_polars(detvar_full_arrs["CV"])
variation_pls = {v: _to_polars(detvar_full_arrs[v]) for v in loaded_vartypes if v != "CV"}

fig, axs = plt.subplots(7, 5, figsize=(20, 20))
axs = axs.flatten()

bins = np.linspace(0, 2000, 21)
bin_centers = (bins[:-1] + bins[1:]) / 2

for i in tqdm(range(len(reco_categories))):

    signal_category_latex = reco_category_labels_latex[i]
    axnum = i

    cv_cat = cv_pl.filter(pl.col("reco_category") == i)
    cv_counts = np.histogram(
        cv_cat["wc_kine_reco_Enu"].to_numpy(),
        weights=cv_cat["wc_net_weight"].to_numpy(),
        bins=bins,
    )[0]
    total_cv_weight = float(cv_cat["wc_net_weight"].sum())

    axs[axnum].hist(bin_centers, weights=cv_counts, bins=bins,
                    histtype="step", color="k", lw=2, label="CV")

    if total_cv_weight == 0:
        axs[axnum].set_title(f"{signal_category_latex} (RSE-matched)")
        axs[axnum].set_xlim(0, 2000)
        continue

    for vartype, var_pl in variation_pls.items():
        var_cat = var_pl.filter(pl.col("reco_category") == i)

        rse_cols = ["filetype", "run", "subrun", "event"]
        matching_cv = cv_cat.join(var_cat.select(rse_cols), on=rse_cols, how="inner")
        matching_cv_weight = float(matching_cv["wc_net_weight"].sum())
        if matching_cv_weight == 0:
            continue
        match_weight = total_cv_weight / matching_cv_weight
        matching_var = var_cat.join(matching_cv.select(rse_cols), on=rse_cols, how="inner")

        matching_cv_counts = np.histogram(
            matching_cv["wc_kine_reco_Enu"].to_numpy(),
            weights=matching_cv["wc_net_weight"].to_numpy() * match_weight,
            bins=bins,
        )[0]
        matching_var_counts = np.histogram(
            matching_var["wc_kine_reco_Enu"].to_numpy(),
            weights=matching_var["wc_net_weight"].to_numpy() * match_weight,
            bins=bins,
        )[0]

        # same convention as make_det_variation_histogram in src/plot_helpers.py:
        # render variation as cv_counts * (matching_cv / matching_var)
        ratio = matching_cv_counts / matching_var_counts
        ratio = np.nan_to_num(ratio, nan=0, posinf=0, neginf=0)
        scaled_var = ratio * cv_counts

        axs[axnum].hist(bin_centers, weights=scaled_var, bins=bins,
                        histtype="step", color=vartype_colors[vartype], lw=1, label=vartype)

    if axnum == 19:
        axs[axnum].legend(ncol=4, loc="upper right", bbox_to_anchor=(-1, -0.5))
    if axnum in [15, 16, 17, 18, 19]:
        axs[axnum].set_xlabel(r"WC Reconstructed $E_\nu$ (MeV)")
    if axnum % 5 == 0:
        axs[axnum].set_ylabel("Counts (weighted)")
    axs[axnum].set_title(f"{signal_category_latex} (RSE-matched)")
    axs[axnum].set_xlim(0, 2000)

for axnum in range(len(axs)):
    if axnum > 19:
        axs[axnum].remove()

fig.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.15)
plt.show()


