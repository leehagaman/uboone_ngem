"""MINERvA axial-form-factor z-expansion weights.

This module converts MINERvA z-expansion axial form factors into per-event
weights by interpolating the existing GENIE MaCCQE spline weights at an
effective axial mass for each event's true Q^2.
"""

import numpy as np

# Branch names written to the spline-weights dataframe / output ROOT tree.
ZEXP_MINERVA_FA_BRANCH = "weight_minerva_FA"
ZEXP_PCA_BRANCHES = tuple(f"weight_spline_FAzexpPCA{i}" for i in range(1, 5))

# Physics constants and spline grids.
PION_MASS_GEV = 0.139570
T_CUT_GEV2 = 9.0 * PION_MASS_GEV * PION_MASS_GEV
AXIAL_FORM_FACTOR_Q2_ZERO = -1.2723

MA_CCQE_GRID_GEV = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], dtype=float)
ZEXP_SIGMA_VALUES = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)

# MINERvA z-expansion result from Nature 614, 48-53 (2023), supplementary table 4.
MINERVA_T0_GEV2 = -0.75
MINERVA_A_VALUES = np.array([-0.50, 1.50, -1.2, -0.1, 0.2, 0.46, -0.40, 0.15, -0.044], dtype=float)
MINERVA_A_ERRORS = np.array([0.31, 0.7, 1.9, 3.5], dtype=float)
MINERVA_A_CORRELATION = np.array(
    [
        [  1.0, 0.012, -0.93,  0.52],
        [0.012,   1.0, -0.32, -0.78],
        [-0.93, -0.32,   1.0, -0.27],
        [ 0.52, -0.78, -0.27,   1.0],
    ],
    dtype=float,
)
MINERVA_A_COVARIANCE = MINERVA_A_CORRELATION * np.outer(MINERVA_A_ERRORS, MINERVA_A_ERRORS)

def axial_form_factor_zexp(q2_gev2, a_values, t0_gev2=MINERVA_T0_GEV2):
    """Evaluate the z-expansion axial form factor F_A(Q^2)."""
    q2 = np.asarray(q2_gev2, dtype=float)
    a = np.asarray(a_values, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (
            np.sqrt(T_CUT_GEV2 + q2) - np.sqrt(T_CUT_GEV2 - t0_gev2)
        ) / (
            np.sqrt(T_CUT_GEV2 + q2) + np.sqrt(T_CUT_GEV2 - t0_gev2)
        )

    result = np.zeros_like(q2, dtype=float)
    for power, coeff in enumerate(a):
        result += coeff * np.power(z, power)
    return result

def effective_axial_mass_gev(q2_gev2, axial_form_factor):
    """Invert the dipole F_A form to an event-by-event effective M_A."""
    q2 = np.asarray(q2_gev2, dtype=float)
    f_a = np.asarray(axial_form_factor, dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = AXIAL_FORM_FACTOR_Q2_ZERO / f_a
        denom = np.sqrt(ratio) - 1.0
        return np.where(
            (ratio > 0.0) & (denom > 1e-10),
            np.sqrt(q2 / denom),
            np.nan,
        )

def interpolate_ma_spline_weights(ma_eff_gev, ma_weights):
    """Interpolate each event's MaCCQE spline weights at its effective M_A.

    Invalid, infinite, or negative interpolated weights are reset to 1.0.
    """
    ma_eff = np.asarray(ma_eff_gev, dtype=float)
    weights = np.asarray(ma_weights, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != len(MA_CCQE_GRID_GEV):
        raise ValueError(
            "ma_weights must have shape (n_events, 7), matching MaCCQE_UBGenie"
        )

    ma_safe = np.where(np.isnan(ma_eff), MA_CCQE_GRID_GEV[0], ma_eff)
    idx_lo = np.clip(
        np.searchsorted(MA_CCQE_GRID_GEV, ma_safe, side="right") - 1,
        0,
        len(MA_CCQE_GRID_GEV) - 2,
    )
    idx_hi = idx_lo + 1
    t = (ma_safe - MA_CCQE_GRID_GEV[idx_lo]) / (MA_CCQE_GRID_GEV[idx_hi] - MA_CCQE_GRID_GEV[idx_lo])

    rows = np.arange(len(ma_eff))
    result = weights[rows, idx_lo] * (1.0 - t) + weights[rows, idx_hi] * t
    bad = np.isnan(ma_eff) | np.isnan(result) | np.isinf(result) | (result < 0.0)
    result[bad] = 1.0
    return result

def complete_minerva_a_values(partial_a1_to_a4, t0_gev2=MINERVA_T0_GEV2):
    """Solve for a0 and a5-a8 from a1-a4 using F_A(0) and four sum rules."""
    a1_to_a4 = np.asarray(partial_a1_to_a4, dtype=float)
    if a1_to_a4.shape != (4,):
        raise ValueError("partial_a1_to_a4 must contain exactly four values")

    z0 = (
        np.sqrt(T_CUT_GEV2) - np.sqrt(T_CUT_GEV2 - t0_gev2)
    ) / (
        np.sqrt(T_CUT_GEV2) + np.sqrt(T_CUT_GEV2 - t0_gev2)
    )

    known_indices = np.arange(1, 5, dtype=float)
    unknown_indices = np.array([0, 5, 6, 7, 8], dtype=float)

    # Impose constrains from Eq. (13) in Nature 614, 48-53 (2023)
    matrix = np.vstack(
        [
            np.power(z0, unknown_indices),
            np.ones(5, dtype=float),
            unknown_indices,
            unknown_indices * (unknown_indices - 1.0),
            unknown_indices * (unknown_indices - 1.0) * (unknown_indices - 2.0),
        ]
    )
    rhs = np.array(
        [
            AXIAL_FORM_FACTOR_Q2_ZERO - np.sum(a1_to_a4 * np.power(z0, known_indices)),
            -np.sum(a1_to_a4),
            -np.sum(a1_to_a4 * known_indices),
            -np.sum(a1_to_a4 * known_indices * (known_indices - 1.0)),
            -np.sum(a1_to_a4 * known_indices * (known_indices - 1.0) * (known_indices - 2.0)),
        ],
        dtype=float,
    )

    a0_a5_to_a8 = np.linalg.solve(matrix, rhs)
    return np.array(
        [a0_a5_to_a8[0], *a1_to_a4, *a0_a5_to_a8[1:]],
        dtype=float,
    )

def _rows_from_vector_branch(values):
    """Return a Python row list for uproot vector data in several common forms."""
    if hasattr(values, "to_list"):
        return values.to_list()
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)

def _clean_ma_spline_weights(ma_spline_weights):
    expected_width = len(MA_CCQE_GRID_GEV)

    try:
        weights = np.asarray(ma_spline_weights, dtype=float)
    except (TypeError, ValueError):
        weights = None

    if weights is None or weights.ndim != 2 or weights.shape[1] != expected_width:
        rows = _rows_from_vector_branch(ma_spline_weights)
        weights = np.ones((len(rows), expected_width), dtype=float)
        for i, row in enumerate(rows):
            try:
                row_weights = np.asarray(row, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            if row_weights.size == expected_width:
                weights[i] = row_weights
    else:
        weights = weights.copy()

    weights[~np.isfinite(weights)] = 1.0
    return weights

def _weights_for_a_values(true_q2_gev2, ma_spline_weights, a_values):
    f_a = axial_form_factor_zexp(true_q2_gev2, a_values, MINERVA_T0_GEV2)
    ma_eff = effective_axial_mass_gev(true_q2_gev2, f_a)
    return interpolate_ma_spline_weights(ma_eff, ma_spline_weights)

def compute_minerva_zexp_weights(true_q2_gev2, ma_spline_weights):
    """Compute MINERvA z-expansion CV and PCA spline weights.

    Returns a dictionary with:
      * ``weight_minerva_FA``: shape (n_events,)
      * ``weight_spline_FAzexpPCA1`` ... ``PCA4``: each shape (n_events, 7)
    """
    q2 = np.asarray(true_q2_gev2, dtype=float)
    weights = _clean_ma_spline_weights(ma_spline_weights)
    if len(q2) != weights.shape[0]:
        raise ValueError("true_q2_gev2 and ma_spline_weights must have same length")

    cv_weights = _weights_for_a_values(q2, weights, MINERVA_A_VALUES)

    eigenvalues, eigenvectors = np.linalg.eigh(MINERVA_A_COVARIANCE)
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    result = {ZEXP_MINERVA_FA_BRANCH: cv_weights.astype(np.float32)}
    partial_cv = MINERVA_A_VALUES[1:5]

    for pca_i, branch in enumerate(ZEXP_PCA_BRANCHES):
        shift = np.sqrt(max(eigenvalues[pca_i], 0.0)) * eigenvectors[:, pca_i]
        sigma_columns = []
        for sigma in ZEXP_SIGMA_VALUES:
            if sigma == 0:
                sigma_columns.append(cv_weights)
                continue
            shifted_partial = partial_cv + sigma * shift
            full_a_values = complete_minerva_a_values(shifted_partial, MINERVA_T0_GEV2)
            sigma_columns.append(_weights_for_a_values(q2, weights, full_a_values))
        result[branch] = np.column_stack(sigma_columns).astype(np.float32)

    return result
