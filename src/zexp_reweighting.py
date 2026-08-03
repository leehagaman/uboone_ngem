"""Axial-form-factor z-expansion weights."""

from dataclasses import dataclass
import numpy as np

# Constants
MA_CCQE_GRID_GEV  = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], dtype=float)
ZEXP_SIGMA_VALUES = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)

# Branch names written to the spline-weights dataframe / output ROOT tree.
ZEXP_MINERVA_FA_BRANCH = "weight_minerva_FA"
ZEXP_PCA_BRANCHES = tuple(f"weight_spline_FAzexpPCA{i}" for i in range(1, 5))

ZEXP_CUSTOM_FA_BRANCH = "weight_custom_FA"
ZEXP_CUSTOM_A_BRANCHES = tuple(f"weight_spline_FAzexpA{i}" for i in range(1, 5))

ZEXP_MINERVA_K6_FA_BRANCH = "weight_minerva_hydrogen_k6_FA"
ZEXP_MINERVA_K6_BRANCHES = tuple(
    f"weight_spline_FAzexpMinervaK6PCA{i}" for i in range(1, 3)
)

ZEXP_MINERVA_K6_DIAGONAL_FA_BRANCH = (
    "weight_minerva_hydrogen_k6_diagonal_FA"
)
ZEXP_MINERVA_K6_DIAGONAL_BRANCHES = tuple(
    f"weight_spline_FAzexpMinervaK6A{i}" for i in range(1, 3)
)

ZEXP_MINERVA_K7_FA_BRANCH = "weight_minerva_hydrogen_k7_FA"
ZEXP_MINERVA_K7_BRANCHES = tuple(
    f"weight_spline_FAzexpMinervaK7PCA{i}" for i in range(1, 4)
)

ZEXP_LQCD_K6_FA_BRANCH = "weight_lqcd_k6_FA"
ZEXP_LQCD_K6_BRANCHES = tuple(
    f"weight_spline_FAzexpLQCDK6PCA{i}" for i in range(1, 3)
)

ZEXP_MINERVA_LQCD_K6_FA_BRANCH = "weight_minerva_lqcd_k6_FA"
ZEXP_MINERVA_LQCD_K6_BRANCHES = tuple(
    f"weight_spline_FAzexpMinervaLQCDK6PCA{i}" for i in range(1, 3)
)

# Used in Nature 614, 48-53 (2023)
PION_MASS_GEV = 0.139570
T_CUT_GEV2 = 9.0 * PION_MASS_GEV * PION_MASS_GEV
AXIAL_FORM_FACTOR_Q2_ZERO = -1.2723
MINERVA_T0_GEV2 = -0.75

# Used in arXiv:2512.14097 (2025)
ZEXP_T0_GEV2 = -0.50
ZEXP_T_CUT_GEV2 = 9.0 * 0.134**2
ZEXP_FA_Q2_ZERO = -1.2754

@dataclass(frozen=True)
class ZExpPrior:
    """Inputs needed to construct one complete z-expansion weight set."""

    name: str
    free_a_values: np.ndarray
    covariance: np.ndarray
    full_a_values: np.ndarray
    cv_branch: str
    variation_branches: tuple
    t0_gev2: float = ZEXP_T0_GEV2
    t_cut_gev2: float = ZEXP_T_CUT_GEV2
    fa_q2_zero: float = ZEXP_FA_Q2_ZERO
    use_pca: bool = True

    def __post_init__(self):
        free = np.array(self.free_a_values, dtype=float, copy=True)
        covariance = np.array(self.covariance, dtype=float, copy=True)
        full = np.array(self.full_a_values, dtype=float, copy=True)
        branches = tuple(self.variation_branches)
        n_free = len(full) - 5

        if n_free < 1 or free.shape != (n_free,):
            raise ValueError("full and free z-expansion coefficients are inconsistent")
        if covariance.shape != (n_free, n_free):
            raise ValueError("covariance dimension must match the free coefficients")
        if len(branches) != n_free:
            raise ValueError("one variation branch is required per free coefficient")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("covariance must be finite")
        if not np.allclose(covariance, covariance.T):
            raise ValueError("covariance must be symmetric")
        if not np.allclose(full[1:n_free + 1], free):
            raise ValueError("free coefficients must match the full central value")

        free.setflags(write=False)
        covariance.setflags(write=False)
        full.setflags(write=False)
        object.__setattr__(self, "free_a_values", free)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "full_a_values", full)
        object.__setattr__(self, "variation_branches", branches)

    @property
    def kmax(self):
        return len(self.full_a_values) - 1


def _negative_fa_coefficients(values):
    """Convert the paper's positive-gA convention to GENIE's negative F_A."""
    return -np.asarray(values, dtype=float)


MINERVA_LEGACY_PRIOR = ZExpPrior(
    name="MINERvA hydrogen 2023 kmax=8",
    free_a_values=np.array(
        [1.497803985, -1.210593528, -0.128820081, 0.176435518],
        dtype=float,
    ),
    covariance=np.array(
        [
            [ 0.096100,  0.002604, -0.547770,  0.564200],
            [ 0.002604,  0.490000, -0.425600, -1.911000],
            [-0.547770, -0.425600,  3.610000, -1.795500],
            [ 0.564200, -1.911000, -1.795500, 12.250000],
        ],
        dtype=float,
    ),
    full_a_values=np.array(
        [
            -0.501578990, 1.497803985, -1.210593528,
            -0.128820081, 0.176435518, 0.459613236,
            -0.400021063, 0.151125941, -0.043965018,
        ],
        dtype=float,
    ),
    cv_branch=ZEXP_MINERVA_FA_BRANCH,
    variation_branches=ZEXP_PCA_BRANCHES,
    t0_gev2=MINERVA_T0_GEV2,
    t_cut_gev2=T_CUT_GEV2,
    fa_q2_zero=AXIAL_FORM_FACTOR_Q2_ZERO,
)

CUSTOM_LEGACY_PRIOR = ZExpPrior(
    name="MINERvA hydrogen 2023 kmax=8 diagonal",
    free_a_values=MINERVA_LEGACY_PRIOR.free_a_values,
    covariance=np.diag(np.diag(MINERVA_LEGACY_PRIOR.covariance)),
    full_a_values=MINERVA_LEGACY_PRIOR.full_a_values,
    cv_branch=ZEXP_CUSTOM_FA_BRANCH,
    variation_branches=ZEXP_CUSTOM_A_BRANCHES,
    t0_gev2=MINERVA_LEGACY_PRIOR.t0_gev2,
    t_cut_gev2=MINERVA_LEGACY_PRIOR.t_cut_gev2,
    fa_q2_zero=MINERVA_LEGACY_PRIOR.fa_q2_zero,
    use_pca=False,
)

# A.S. Meyer et al., arXiv:2512.14097, Eqs. (31)--(41) and (56)--(58).
MINERVA_K6_PRIOR = ZExpPrior(
    name="MINERvA hydrogen 2025 kmax=6",
    free_a_values=_negative_fa_coefficients([-1.64778080, 0.94181417]),
    covariance=np.array(
        [[0.05554150, -0.03262482], [-0.03262482, 0.09151761]], dtype=float
    ),
    full_a_values=_negative_fa_coefficients(
        [
            0.61490770, -1.64778080, 0.94181417, 0.41239729,
            0.36611559, -1.18722194, 0.49976799,
        ]
    ),
    cv_branch=ZEXP_MINERVA_K6_FA_BRANCH,
    variation_branches=ZEXP_MINERVA_K6_BRANCHES,
)

MINERVA_K7_PRIOR = ZExpPrior(
    name="MINERvA hydrogen 2025 kmax=7",
    free_a_values=_negative_fa_coefficients(
        [-1.69373431, 0.80639393, 0.87442257]
    ),
    covariance=np.array(
        [
            [0.02771279, -0.00093839, -0.19136214],
            [-0.00093839, 0.11854816, -0.12082981],
            [-0.19136214, -0.12082981, 1.45823056],
        ],
        dtype=float,
    ),
    full_a_values=_negative_fa_coefficients(
        [
            0.62174048, -1.69373431, 0.80639393, 0.87442257,
            0.55213983, -2.61742963, 1.85900234, -0.40253521,
        ]
    ),
    cv_branch=ZEXP_MINERVA_K7_FA_BRANCH,
    variation_branches=ZEXP_MINERVA_K7_BRANCHES,
)

LQCD_K6_PRIOR = ZExpPrior(
    name="LQCD 2025 kmax=6",
    free_a_values=_negative_fa_coefficients([-1.72089706, 0.30982708]),
    covariance=np.array(
        [[0.00265598, -0.00562374], [-0.00562374, 0.01596000]], dtype=float
    ),
    full_a_values=_negative_fa_coefficients(
        [
            0.71742019, -1.72089706, 0.30982708, 1.62125837,
            -0.27506993, -1.25297945, 0.60044079,
        ]
    ),
    cv_branch=ZEXP_LQCD_K6_FA_BRANCH,
    variation_branches=ZEXP_LQCD_K6_BRANCHES,
)

MINERVA_LQCD_K6_PRIOR = ZExpPrior(
    name="MINERvA+LQCD 2025 kmax=6",
    free_a_values=_negative_fa_coefficients([-1.74307738, 0.37944565]),
    covariance=np.array(
        [[0.00241156, -0.00495246], [-0.00495246, 0.01406075]], dtype=float
    ),
    full_a_values=_negative_fa_coefficients(
        [
            0.71070233, -1.74307738, 0.37944565, 1.69894456,
            -0.60326876, -0.95690585, 0.51415945,
        ]
    ),
    cv_branch=ZEXP_MINERVA_LQCD_K6_FA_BRANCH,
    variation_branches=ZEXP_MINERVA_LQCD_K6_BRANCHES,
)

MINERVA_K6_DIAGONAL_PRIOR = ZExpPrior(
    name="MINERvA hydrogen 2025 kmax=6 diagonal",
    free_a_values=MINERVA_K6_PRIOR.free_a_values.copy(),
    covariance=np.diag(np.diag(MINERVA_K6_PRIOR.covariance)),
    full_a_values=MINERVA_K6_PRIOR.full_a_values.copy(),
    cv_branch=ZEXP_MINERVA_K6_DIAGONAL_FA_BRANCH,
    variation_branches=ZEXP_MINERVA_K6_DIAGONAL_BRANCHES,
    use_pca=False,
)

ZEXP_PRIORS = (
    MINERVA_LEGACY_PRIOR, # MINERvA Nature result with covariance, in PCA
    CUSTOM_LEGACY_PRIOR, # Use MINERvA Nature result with diagonal covariance, in a_k
    MINERVA_K6_PRIOR, # MINERvA k=6 result with covariance, in PCA
    MINERVA_K7_PRIOR, # MINERvA k=7 result with covariance, in PCA
    LQCD_K6_PRIOR, # LQCD k=6 result with covariance, in PCA
    MINERVA_LQCD_K6_PRIOR, # MINERvA + LQCD k=6 with covariance, in PCA
    MINERVA_K6_DIAGONAL_PRIOR, # MINERvA k=6 result with diagonal covariance, in a_k
)

# Get relevant branches from all priors
ZEXP_CV_BRANCHES = tuple(prior.cv_branch for prior in ZEXP_PRIORS)
ZEXP_VARIATION_BRANCHES = (
    *(
        branch
        for prior in ZEXP_PRIORS
        for branch in prior.variation_branches
    ),
)
ZEXP_ALL_BRANCHES = (*ZEXP_CV_BRANCHES, *ZEXP_VARIATION_BRANCHES)

def axial_form_factor_zexp(
    q2_gev2,
    a_values,
    t0_gev2=MINERVA_T0_GEV2,
    t_cut_gev2=T_CUT_GEV2,
):
    """Evaluate the z-expansion axial form factor F_A(Q^2)."""
    q2 = np.asarray(q2_gev2, dtype=float)
    a = np.asarray(a_values, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (
            np.sqrt(t_cut_gev2 + q2) - np.sqrt(t_cut_gev2 - t0_gev2)
        ) / (
            np.sqrt(t_cut_gev2 + q2) + np.sqrt(t_cut_gev2 - t0_gev2)
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

def interpolate_fa_spline_weights(q2_gev2, target_fa, ma_weights):
    """Interpolate or linearly extrapolate GENIE weights directly in F_A.

    The seven MaCCQE weights are treated as samples at the seven dipole
    F_A(Q^2, M_A) values.  This remains defined when the target z-expansion
    F_A cannot be represented by a real effective dipole mass.
    """
    q2 = np.asarray(q2_gev2, dtype=float)
    target = np.asarray(target_fa, dtype=float)
    weights = np.asarray(ma_weights, dtype=float)
    if q2.shape != target.shape or weights.shape != (len(q2), len(MA_CCQE_GRID_GEV)):
        raise ValueError("q2, target_fa, and ma_weights have incompatible shapes")

    fa_grid = AXIAL_FORM_FACTOR_Q2_ZERO / (
        1.0 + q2[:, None] / MA_CCQE_GRID_GEV[None, :] ** 2
    ) ** 2
    # F_A becomes more negative as M_A increases, so reverse into ascending
    # numerical order for row-wise interval lookup.
    x = fa_grid[:, ::-1]
    y = weights[:, ::-1]
    insertion = np.sum(x <= target[:, None], axis=1)
    idx_lo = np.clip(insertion - 1, 0, x.shape[1] - 2)
    idx_hi = idx_lo + 1
    rows = np.arange(len(q2))
    denominator = x[rows, idx_hi] - x[rows, idx_lo]

    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = (target - x[rows, idx_lo]) / denominator
        result = y[rows, idx_lo] + fraction * (
            y[rows, idx_hi] - y[rows, idx_lo]
        )

    # At Q^2=0 all dipole F_A grid points coincide and M_A is unidentifiable.
    # Retain the central M_A=1.1 weight rather than manufacture a large slope.
    degenerate = ~np.isfinite(denominator) | (np.abs(denominator) < 1e-14)
    result[degenerate] = weights[degenerate, 3]
    bad = ~np.isfinite(target) | ~np.isfinite(result) | (result < 0.0)
    result[bad] = 1.0
    return result

def quadratic_fa_spline_weights(
    q2_gev2,
    target_fa,
    ma_weights,
):
    """Evaluate each event's MaCCQE spline as a quadratic in F_A."""
    model = _prepare_quadratic_fa_splines(q2_gev2, ma_weights)
    return _evaluate_quadratic_fa_splines(target_fa, model)


def _prepare_quadratic_fa_splines(q2_gev2, ma_weights):
    q2 = np.asarray(q2_gev2, dtype=float)
    weights = np.asarray(ma_weights, dtype=float)
    expected_shape = (len(q2), len(MA_CCQE_GRID_GEV))
    if q2.ndim != 1 or weights.shape != expected_shape:
        raise ValueError("q2 and ma_weights have incompatible shapes")

    fa_grid = AXIAL_FORM_FACTOR_Q2_ZERO / (
        1.0 + q2[:, None] / MA_CCQE_GRID_GEV[None, :] ** 2
    ) ** 2
    center = fa_grid[:, 3]
    scale = np.ptp(fa_grid, axis=1)
    degenerate = ~np.isfinite(scale) | (np.abs(scale) < 1e-14)
    safe_scale = np.where(degenerate, 1.0, scale)
    x = (fa_grid - center[:, None]) / safe_scale[:, None]
    design = np.stack((np.ones_like(x), x, x**2), axis=2)
    gram = np.einsum("nki,nkj->nij", design, design)
    rhs = np.einsum("nki,nk->ni", design, weights)

    coefficients = np.zeros((len(q2), 3), dtype=float)
    valid = ~degenerate
    if np.any(valid):
        coefficients[valid] = np.linalg.solve(
            gram[valid], rhs[valid, :, None]
        )[:, :, 0]
    return center, safe_scale, coefficients, degenerate, weights[:, 3]

def _evaluate_quadratic_fa_splines(target_fa, model):
    target = np.asarray(target_fa, dtype=float)
    center, scale, coefficients, degenerate, central_weights = model
    if target.shape != center.shape:
        raise ValueError("target_fa and prepared spline model have incompatible shapes")

    target_x = (target - center) / scale
    result = (
        coefficients[:, 0]
        + coefficients[:, 1] * target_x
        + coefficients[:, 2] * target_x**2
    )
    result[degenerate] = central_weights[degenerate]
    bad = ~np.isfinite(target) | ~np.isfinite(result) | (result < 0.0)
    result[bad] = 1.0
    return result

def complete_zexp_a_values(
    free_a_values,
    kmax,
    t0_gev2,
    *,
    t_cut_gev2=T_CUT_GEV2,
    fa_q2_zero=AXIAL_FORM_FACTOR_Q2_ZERO,
):
    """Complete a1...a[kmax-4] using F_A(0) and four sum rules."""
    free = np.asarray(free_a_values, dtype=float)
    n_free = kmax - 4
    if kmax < 5 or free.shape != (n_free,):
        raise ValueError(f"kmax={kmax} requires exactly {n_free} free coefficients")

    z0 = (
        np.sqrt(t_cut_gev2) - np.sqrt(t_cut_gev2 - t0_gev2)
    ) / (
        np.sqrt(t_cut_gev2) + np.sqrt(t_cut_gev2 - t0_gev2)
    )
    known_indices = np.arange(1, n_free + 1, dtype=float)
    unknown_indices = np.array(
        [0, *range(n_free + 1, kmax + 1)], dtype=float
    )
    def falling(k, order):
        if order == 0:
            return np.ones_like(k)
        return np.prod([k - offset for offset in range(order)], axis=0)

    matrix = np.vstack(
        [
            np.power(z0, unknown_indices),
            *(falling(unknown_indices, order) for order in range(4)),
        ]
    )
    rhs = np.array(
        [
            fa_q2_zero - np.sum(free * np.power(z0, known_indices)),
            *(
                -np.sum(free * falling(known_indices, order))
                for order in range(4)
            ),
        ],
        dtype=float,
    )
    solved = np.linalg.solve(matrix, rhs)
    full = np.empty(kmax + 1, dtype=float)
    full[known_indices.astype(int)] = free
    full[unknown_indices.astype(int)] = solved
    return full

def complete_minerva_a_values(partial_a1_to_a4, t0_gev2=MINERVA_T0_GEV2):
    """Solve for a0 and a5-a8 from a1-a4 using F_A(0) and four sum rules."""
    return complete_zexp_a_values(
        partial_a1_to_a4,
        8,
        t0_gev2,
        t_cut_gev2=T_CUT_GEV2,
        fa_q2_zero=AXIAL_FORM_FACTOR_Q2_ZERO,
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

def _weights_for_a_values(
    true_q2_gev2,
    ma_spline_weights,
    a_values,
    t0_gev2=MINERVA_T0_GEV2,
    t_cut_gev2=T_CUT_GEV2,
    quadratic_model=None,
):
    f_a = axial_form_factor_zexp(
        true_q2_gev2, a_values, t0_gev2, t_cut_gev2
    )
    if quadratic_model is None:
        return quadratic_fa_spline_weights(
            true_q2_gev2, f_a, ma_spline_weights
        )
    return _evaluate_quadratic_fa_splines(f_a, quadratic_model)


def compute_zexp_weight_set(
    true_q2_gev2,
    ma_spline_weights,
    partial_a_values,
    covariance,
    cv_branch,
    variation_branches,
    *,
    use_pca,
    t0_gev2=MINERVA_T0_GEV2,
    t_cut_gev2=T_CUT_GEV2,
    fa_q2_zero=AXIAL_FORM_FACTOR_Q2_ZERO,
    kmax=None,
    full_cv_a_values=None,
    _quadratic_model=None,
):
    """Compute one central z-expansion weight and its sigma variations.

    When ``use_pca`` is true, the variations are the eigenvectors of the
    supplied covariance.  Otherwise, variation i shifts only a_i using the
    square root of covariance[i, i]; off-diagonal elements are ignored.
    """
    q2 = np.asarray(true_q2_gev2, dtype=float)
    if _quadratic_model is None:
        weights = _clean_ma_spline_weights(ma_spline_weights)
        quadratic_model = _prepare_quadratic_fa_splines(q2, weights)
    else:
        weights = np.asarray(ma_spline_weights, dtype=float)
        quadratic_model = _quadratic_model
    if len(q2) != weights.shape[0]:
        raise ValueError("true_q2_gev2 and ma_spline_weights must have same length")

    partial_cv = np.asarray(partial_a_values, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    branches = tuple(variation_branches)
    n_free = len(partial_cv)
    if cov.shape != (n_free, n_free):
        raise ValueError("free coefficients and covariance have incompatible shapes")
    if len(branches) != n_free:
        raise ValueError("one variation branch is required per free coefficient")
    if np.any(~np.isfinite(cov)) or np.any(np.diag(cov) < 0.0):
        raise ValueError("covariance must be finite with a non-negative diagonal")

    if kmax is None:
        kmax = n_free + 4
    if full_cv_a_values is None:
        full_cv = complete_zexp_a_values(
            partial_cv,
            kmax,
            t0_gev2,
            t_cut_gev2=t_cut_gev2,
            fa_q2_zero=fa_q2_zero,
        )
    else:
        full_cv = np.asarray(full_cv_a_values, dtype=float)
        if full_cv.shape != (kmax + 1,):
            raise ValueError(
                f"full_cv_a_values must contain exactly {kmax + 1} values"
            )
    cv_weights = _weights_for_a_values(
        q2, weights, full_cv, t0_gev2, t_cut_gev2, quadratic_model
    )
    result = {cv_branch: cv_weights.astype(np.float32)}

    if use_pca:
        eigenvalues, eigenvectors = np.linalg.eigh((cov + cov.T) / 2.0)
        sort_idx = np.argsort(eigenvalues)[::-1]
        shifts = (
            eigenvectors[:, sort_idx]
            * np.sqrt(np.clip(eigenvalues[sort_idx], 0.0, None))
        )
    else:
        shifts = np.diag(np.sqrt(np.diag(cov)))

    for variation_i, branch in enumerate(branches):
        shift = shifts[:, variation_i]
        sigma_columns = []
        for sigma in ZEXP_SIGMA_VALUES:
            if sigma == 0:
                sigma_columns.append(cv_weights)
                continue
            shifted_partial = partial_cv + sigma * shift
            full_a_values = complete_zexp_a_values(
                shifted_partial,
                kmax,
                t0_gev2,
                t_cut_gev2=t_cut_gev2,
                fa_q2_zero=fa_q2_zero,
            )
            sigma_columns.append(
                _weights_for_a_values(
                    q2,
                    weights,
                    full_a_values,
                    t0_gev2,
                    t_cut_gev2,
                    quadratic_model,
                )
            )
        result[branch] = np.column_stack(sigma_columns).astype(np.float32)

    return result

def compute_zexp_prior_weights(
    true_q2_gev2,
    ma_spline_weights,
    prior,
    *,
    _quadratic_model=None,
):
    """Compute the CV and variations described by one ZExpPrior."""
    return compute_zexp_weight_set(
        true_q2_gev2,
        ma_spline_weights,
        prior.free_a_values,
        prior.covariance,
        prior.cv_branch,
        prior.variation_branches,
        use_pca=prior.use_pca,
        t0_gev2=prior.t0_gev2,
        t_cut_gev2=prior.t_cut_gev2,
        fa_q2_zero=prior.fa_q2_zero,
        kmax=prior.kmax,
        full_cv_a_values=prior.full_a_values,
        _quadratic_model=_quadratic_model,
    )


def compute_zexp_weights(true_q2_gev2, ma_spline_weights):
    """Compute every configured z-expansion prior through the same path."""
    q2 = np.asarray(true_q2_gev2, dtype=float)
    weights = _clean_ma_spline_weights(ma_spline_weights)
    if len(q2) != weights.shape[0]:
        raise ValueError("true_q2_gev2 and ma_spline_weights must have same length")
    quadratic_model = _prepare_quadratic_fa_splines(q2, weights)

    result = {}
    for prior in ZEXP_PRIORS:
        result.update(
            compute_zexp_prior_weights(
                q2,
                weights,
                prior,
                _quadratic_model=quadratic_model,
            )
        )
    return result
