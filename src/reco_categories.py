"""The nominal multi-class BDT selection: assigns each scored event exactly one reco category.

Each reco category is selected by a cut on its own BDT probability (or, where the threshold is
None, by being the argmax category). Events passing several cuts go to the category with the
lowest priority value, so the categories are mutually exclusive. Events with a BDT prediction that pass
none of the cuts (their argmax category has a cut they fail) are "unselected", so every scored event gets
exactly one category.

Used by the notebooks (as src.reco_categories) and by the scripts in src/, so it imports nothing
from the project: the functions take the BDT's category list (train_category_labels), whose order
is the prob_<category> column order for the argmax.
"""

import polars as pl

# reco category: (threshold on prob_<category>, or None for argmax; priority, lower claims overlapping events first)
RECO_CATEGORY_THRESHOLDS = {
    "1gNp":             (0.3, 1),
    "1g0p":             (0.9, 2),
    "1gNp1mu":          (0.5, 3),
    "1g0p1mu":          (0.2, 4),
    "1g_outFV":         (0.5, 5),
    "NC1pi0_Np":        (None, 6),
    "NC1pi0_0p":        (None, 7),
    "numuCC1pi0_Np":    (0.1, 9),
    "numuCC1pi0_0p":    (0.15, 8),   # 0p takes priority over Np
    "1pi0_outFV":       (0.1, 10),
    "nueCC_Np":         (0.05, 12),
    "nueCC_0p":         (0.05, 11),  # 0p takes priority over Np
    "numuCC_Np":        (0.5, 13),
    "numuCC_0p":        (0.5, 14),
    "other_outFV_dirt": (None, 15),
    "multi_pi0":        (0.02, 16),
    "eta_other":        (0.01, 17),
    "pi0_dalitz_decay": (0.1, 5.5),   # high priority for dalitz, rare topology
    "NC_no_gamma":      (None, 19),
    "ext":              (None, 20),
}


def reco_category_expr(categories, thresholds=RECO_CATEGORY_THRESHOLDS):
    """Polars expression for each event's reco category name: one of `categories`, "unselected" if it
    passes none of the cuts, or null if it has no BDT prediction (null probabilities).
    `categories` is the BDT's category list (train_category_labels). `thresholds` (same format as
    RECO_CATEGORY_THRESHOLDS) allows studying modified cuts."""
    if set(categories) != set(thresholds):
        raise ValueError(f"categories don't match the thresholds: {sorted(set(categories) ^ set(thresholds))}")
    argmax = pl.concat_list([f"prob_{cat}" for cat in categories]).list.arg_max()
    # the first matching cut, in priority order, wins
    expr = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.String))
    for cat in sorted(thresholds, key=lambda c: thresholds[c][1]):
        threshold = thresholds[cat][0]
        cut = (argmax == categories.index(cat)) if threshold is None else (pl.col(f"prob_{cat}") > threshold)
        expr = expr.when(cut).then(pl.lit(cat))
    return expr.when(argmax.is_not_null()).then(pl.lit("unselected")).otherwise(pl.lit(None, dtype=pl.String))


def get_reco_category_queries(categories, thresholds=RECO_CATEGORY_THRESHOLDS):
    """Mutually exclusive Polars selection expressions, one per reco category, in `categories`
    order (see reco_category_expr; the "unselected" events are in none of them)."""
    category = reco_category_expr(categories, thresholds)
    return [category == cat for cat in categories]
