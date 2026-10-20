"""Shared BDT setup for train.py, hyperparameter_tuning.py, and BDT inference.

Keeps the training variable lists, signal categories, train/validation/test split,
event selection, labels/weights, and feature preprocessing in one place, so the tuning
and the final training see exactly the same events, and inference sees exactly the
same feature preprocessing as training.
"""

import numpy as np
import polars as pl

from signal_categories import topological_category_labels, train_category_labels
from df_helpers import BDT_DATA_ROLE_FILETYPES, BDT_WHOLE_SAMPLE_FILETYPES
from ntuple_variables.variables import (
    wc_training_vars, combined_training_vars, lantern_training_vars,
    glee_training_vars, pandora_training_vars, pandora_scalar_training_vars,
    combined_postprocessing_training_vars,
)

# These events are never used for BDT training (training, validation, or the training-script
# test evaluation) in any signal_categories mode, and all their split flags are False: the data
# roles, and the prediction samples kept whole downstream (df_helpers.select_bdt_test_events),
# the fullosc sample and the dedicated numuCC rad. corr. simulation
NON_TRAINING_FILETYPES = BDT_DATA_ROLE_FILETYPES + BDT_WHOLE_SAMPLE_FILETYPES

# train_test_score split:
#   [0, 0.4)   training
#   [0.4, 0.5) validation (early stopping and hyperparameter tuning)
#   [0.5, 1)   testing (only used for evaluation and the final analysis)
TRAIN_SCORE_MAX = 0.4
VALIDATION_SCORE_MAX = 0.5

# How the event weights enter the XGBoost training:
#   "normalized": physics weights (wc_net_weight_open_data), scaled so the mean training weight
#                 is one. The scale doesn't change the unregularized optimum, but it puts
#                 min_child_weight and reg_lambda (sums of hessian*weight) in units of ~events.
#   "unit":       every event gets weight one, so the BDT learns the unweighted sample
#                 composition (class priors differ from the physics priors).
WEIGHT_MODES = ["normalized", "unit"]

# Features outside +-this (e.g. the -3.4e38 float sentinels in many Pandora variables) or
# non-finite are treated as missing
FEATURE_MISSING_THRESHOLD = 1e10


def get_training_vars(name):
    if name == "combined":
        training_vars = combined_training_vars
    elif name == "wc":
        training_vars = wc_training_vars
    elif name == "lantern":
        training_vars = lantern_training_vars
    elif name == "lantern_first_half":
        training_vars = lantern_training_vars[:len(lantern_training_vars) // 2]
    elif name == "glee":
        training_vars = glee_training_vars
    elif name == "pandora":
        training_vars = pandora_training_vars
    elif name == "pandora_scalars":
        training_vars = pandora_scalar_training_vars
    elif name == "only_wc_lantern_combined":
        training_vars = wc_training_vars + lantern_training_vars + combined_postprocessing_training_vars
    else:
        raise ValueError(f"Invalid training_vars: {name}")

    duplicates = sorted({var for var in training_vars if training_vars.count(var) > 1})
    if duplicates:
        raise ValueError(f"Duplicates in training_vars: {duplicates}")

    return list(training_vars)


def get_signal_categories(name):
    """Returns (signal_category_labels, signal_category_var)."""
    if name == "topological":
        return topological_category_labels, "topological_signal_category"
    elif name == "del1g_simple":
        return train_category_labels, "del1g_simple_signal_category"
    elif name == "nue_only":
        # Binary classification: 0 -> not_nue, 1 -> nue
        return ["not_nue", "nue"], "del1g_simple_signal_category"
    elif name == "nc_coh_1g_vs_bkg":
        return ["not_NC_coherent_1g_reweighted", "NC_coherent_1g_reweighted"], "filetype"
    else:
        raise ValueError(f"Invalid signal_categories: {name}")


def add_split_columns(df):
    """Adds used_for_training / used_for_validation / used_for_testing.

    For NON_TRAINING_FILETYPES, `~is_in(...)` is False, so all three flags are False
    regardless of train_test_score (Kleene: False & anything == False).
    """
    trainable = ~pl.col("filetype").is_in(NON_TRAINING_FILETYPES)
    score = pl.col("train_test_score")
    return df.with_columns([
        (trainable & (score < TRAIN_SCORE_MAX)).alias("used_for_training"),
        (trainable & (score >= TRAIN_SCORE_MAX) & (score < VALIDATION_SCORE_MAX)).alias("used_for_validation"),
        (trainable & (score >= VALIDATION_SCORE_MAX)).alias("used_for_testing"),
    ])


def select_training_events(df, signal_categories):
    """Events used for training/validation/testing (before the split)."""
    presel_df = df.filter(~pl.col("filetype").is_in(NON_TRAINING_FILETYPES))
    presel_df = presel_df.filter(pl.col("wc_kine_reco_Enu") > 0) # should already be applied in presel_df_train_vars.parquet

    if signal_categories == "nc_coh_1g_vs_bkg":
        # don't use Iso1g or Del1g events for training/testing
        presel_df = presel_df.filter(~pl.col("filetype").is_in(
            ["isotropic_one_gamma_overlay", "delete_one_gamma_overlay"]))

        # only use sampled NC Coherent 1g events, in order to get a coherent 1g shape without event weights
        presel_df = presel_df.filter(
            ((pl.col("filetype") == "NC_coherent_1g_reweighted") & (pl.col("coherent_1g_keep") == True))
            | (pl.col("filetype") != "NC_coherent_1g_reweighted")
        )
    else:
        # don't use NC Coherent 1g events for training/testing, since these are already included in the Del1g and Iso1g training
        presel_df = presel_df.filter(pl.col("filetype") != "NC_coherent_1g_reweighted")

    present = set(presel_df.select(pl.col("filetype").unique()).to_series().to_list())
    leaked = present & set(NON_TRAINING_FILETYPES)
    if leaked:
        raise ValueError(f"Filetypes that must never be used for BDT training were selected: {sorted(leaked)}")

    return presel_df


def get_labels(df, signal_categories):
    if signal_categories == "nc_coh_1g_vs_bkg":
        y = df.select((pl.col("filetype") == "NC_coherent_1g_reweighted") & pl.col("wc_truth_inFV"))
    else:
        _, signal_category_var = get_signal_categories(signal_categories)
        y = df.select(signal_category_var)
    return y.to_numpy().flatten().astype(int)


def get_physics_weights(df, signal_categories):
    """Physics (POT-normalized) event weights, unnormalized. The NC coherent 1g training
    uses sampled (unweighted) events, so every event has weight one there."""
    if signal_categories == "nc_coh_1g_vs_bkg":
        return np.ones(df.height)
    return df.select("wc_net_weight_open_data").to_numpy().flatten().astype(np.float64)


def get_training_weights(physics_weights, weight_mode, normalization):
    """Weights given to XGBoost. `normalization` is the mean physics weight of the
    training split, applied to every split so they share one scale."""
    if weight_mode == "normalized":
        return physics_weights / normalization
    elif weight_mode == "unit":
        return np.ones_like(physics_weights)
    else:
        raise ValueError(f"Invalid weight_mode: {weight_mode}, expected one of {WEIGHT_MODES}")


def class_prior_correction(y_train, training_weights, physics_weights, num_categories):
    """Per-class factors pi_physics / pi_training, from the training split.

    Multiplying BDT probabilities by these (and renormalizing) moves the class priors of
    a BDT trained with `training_weights` to the physics priors. All ones for the
    "normalized" weight mode.
    """
    train_frac = np.bincount(y_train, weights=training_weights, minlength=num_categories)
    phys_frac = np.bincount(y_train, weights=physics_weights, minlength=num_categories)
    train_frac = train_frac / train_frac.sum()
    phys_frac = phys_frac / phys_frac.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(train_frac > 0, phys_frac / train_frac, 0.0)


def apply_prior_correction(probs, correction):
    """probs: (n, num_categories) array of class probabilities."""
    corrected = probs * correction[np.newaxis, :]
    return corrected / corrected.sum(axis=1, keepdims=True)


def weighted_logloss(probs, y, weights, eps=1e-16):
    """Weighted mean log loss, matching XGBoost's logloss/mlogloss (including its 1e-16
    probability clip, which matters: some events get true-class probabilities ~1e-40).
    probs: (n, num_categories) array of class probabilities."""
    p_true = np.clip(probs[np.arange(len(y)), y], eps, 1.0)
    return float(-np.sum(weights * np.log(p_true)) / np.sum(weights))


class PhysicsWeightedLoglossMetric:
    """XGBoost custom eval metric: log loss with physics weights, whatever weights the eval
    sets were given for training, so the curves are comparable between weight modes.

    Each eval set is recognized by its number of events (they must all differ). Returns the
    log loss of the raw probabilities to XGBoost, and also records, per eval set and boosting
    iteration, the log loss after moving the class priors to the physics priors with
    `prior_correction` (identical for weight_mode "normalized"):
        self.history[set_name] = {"raw": [...], "prior_corrected": [...]}
    """

    def __init__(self, eval_sets, num_categories, prior_correction):
        """eval_sets: list of (set_name, y, physics_weights), in eval_set order."""
        self.__name__ = "physics_logloss" # the metric name in evals_result()
        self.num_categories = num_categories
        self.prior_correction = prior_correction
        self.sets_by_size = {}
        for set_name, y, w in eval_sets:
            if len(y) in self.sets_by_size:
                raise ValueError(f"Eval sets '{set_name}' and '{self.sets_by_size[len(y)][0]}' have the same size")
            self.sets_by_size[len(y)] = (set_name, np.asarray(y, dtype=int), np.asarray(w, dtype=np.float64))
        self.history = {set_name: {"raw": [], "prior_corrected": []} for set_name, _, _ in eval_sets}

    def __call__(self, y_true, y_score, sample_weight=None):
        set_name, y, w = self.sets_by_size[len(y_true)]
        probs = np.asarray(y_score, dtype=np.float64).reshape(len(y_true), -1)
        if probs.shape[1] == 1: # binary: probability of class 1
            probs = np.column_stack([1 - probs[:, 0], probs[:, 0]])
        raw = weighted_logloss(probs, y, w)
        self.history[set_name]["raw"].append(raw)
        self.history[set_name]["prior_corrected"].append(
            weighted_logloss(apply_prior_correction(probs, self.prior_correction), y, w))
        return raw


def bdt_feature_matrix(df, training_vars, chunk_size=200_000):
    """Feature matrix for BDT training or inference, as float32.

    Values that are non-finite or outside +-FEATURE_MISSING_THRESHOLD become NaN (missing).
    Filled in row chunks, so peak memory stays near the size of the float32 output.
    XGBoost casts inputs to float32 internally, so float32 here gives identical results.
    """
    x = np.empty((df.height, len(training_vars)), dtype=np.float32)
    for start in range(0, df.height, chunk_size):
        chunk = df.slice(start, chunk_size).select(training_vars).to_numpy().astype(np.float64)
        chunk[~(np.abs(chunk) <= FEATURE_MISSING_THRESHOLD)] = np.nan # also catches NaN and +-inf
        x[start:start + len(chunk)] = chunk
    return x
