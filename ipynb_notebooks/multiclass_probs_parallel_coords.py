# Parallel-coordinates explorer for the 20D NGEM multi-class BDT probability space.
# Adapted from fashion-mnist-parallel-coords.py.

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys

    import marimo as mo
    import numpy as np
    import polars as pl
    from wigglystuff import ParallelCoordinates

    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

    from src.file_locations import intermediate_files_location
    from src.signal_categories import train_category_labels

    prob_cols = ["prob_" + cat for cat in train_category_labels]
    return (
        ParallelCoordinates,
        intermediate_files_location,
        mo,
        np,
        pl,
        prob_cols,
        train_category_labels,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NGEM Multi-Class Probabilities — Parallel Coordinates

    Loads the neutrino test events with their 20-class BDT probability scores,
    samples a small subset, and visualizes the 20D probability space as an
    interactive parallel coordinates plot. Use the brushes on each axis to
    filter events; the run/subrun/event identifiers of brushed events are shown
    below.
    """)
    return


@app.cell
def _(intermediate_files_location, pl, prob_cols):
    training = "all_vars"

    keep_cols = [
        "filetype", "run", "subrun", "event",
        "used_for_training", "used_for_testing",
        "wc_kine_reco_Enu", "wc_net_weight",
        "del1g_simple_signal_category",
        *prob_cols,
    ]

    all_lf = pl.scan_parquet(f"{intermediate_files_location}/all_df.parquet")
    pred_lf = pl.scan_parquet(f"../training_outputs/{training}/predictions.parquet")

    merged_lf = all_lf.join(
        pred_lf, on=["filetype", "run", "subrun", "event"], how="left"
    ).filter(
        (pl.col("filetype") != "data")
        & (pl.col("filetype") != "isotropic_one_gamma_overlay")
    )

    counts = merged_lf.select(
        pl.col("used_for_training").sum().alias("n_train"),
        pl.col("used_for_testing").sum().alias("n_test"),
    ).collect()
    frac_test = counts["n_test"][0] / (counts["n_train"][0] + counts["n_test"][0])

    test_df = (
        merged_lf
        .filter(pl.col("used_for_testing") == True)
        .filter(pl.col("wc_kine_reco_Enu") > 0)
        .with_columns((pl.col("wc_net_weight") / frac_test).alias("wc_net_weight"))
        .select(keep_cols)
        .collect()
    )
    return (test_df,)


@app.cell
def _(n_samples_slider, np, pl, prob_cols, test_df, train_category_labels):
    rng = np.random.default_rng(42)
    n = min(n_samples_slider.value, test_df.height)
    sample_idx = rng.choice(test_df.height, size=n, replace=False)

    sampled_df = test_df[sample_idx.tolist()].with_columns(
        [pl.col(p).fill_null(-1.0) for p in prob_cols]
    )

    cat_ints = sampled_df["del1g_simple_signal_category"].to_list()
    true_cat_strs = [
        train_category_labels[i] if 0 <= i < len(train_category_labels) else "other"
        for i in cat_ints
    ]

    df = sampled_df.select(prob_cols).with_columns(
        pl.Series("true_category", true_cat_strs)
    )

    # HiPlot picks log scale when an axis's values are all positive and span
    # many orders of magnitude. The 0-row forces linear; the 1-row pins each
    # axis to span [0, 1] regardless of the sampled data's actual max.
    ghost = pl.DataFrame(
        {p: [0.0, 1.0] for p in prob_cols} | {"true_category": ["_ghost", "_ghost"]},
        schema={**{p: df.schema[p] for p in prob_cols}, "true_category": pl.Utf8},
    )
    df = pl.concat([df, ghost], how="vertical")
    return df, sampled_df


@app.cell(hide_code=True)
def _(mo):
    n_samples_slider = mo.ui.slider(
        start=50, stop=2000, step=50, value=1000, label="Number of samples"
    )
    n_samples_slider
    return (n_samples_slider,)


@app.cell(hide_code=True)
def _(ParallelCoordinates, df, mo):
    widget = mo.ui.anywidget(
        ParallelCoordinates(
            df,
            height=500,
            color_by="true_category",
            color_map={"_ghost": "rgba(0,0,0,0)"},
        )
    )
    widget
    return (widget,)


@app.cell(hide_code=True)
def _(mo, sampled_df, widget):
    real_idx = [i for i in widget.widget.filtered_indices if i < sampled_df.height]
    if len(real_idx) == 0:
        out = mo.md("_Brush axes above to select events._")
    else:
        selected = sampled_df[real_idx].select(
            ["filetype", "run", "subrun", "event", "del1g_simple_signal_category"]
        )
        out = mo.vstack(
            [
                mo.md(f"**{len(real_idx)} selected events** (run / subrun / event):"),
                mo.ui.table(selected.to_pandas(), selection=None),
            ]
        )
    out
    return


if __name__ == "__main__":
    app.run()
