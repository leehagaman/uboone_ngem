"""
Physics functions for QED radiative corrections to neutrino scattering.

Reference: Tomalak et al., Phys. Rev. D 106, 093006 (2022)
https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.093006
"""

import numpy as np
from scipy.special import spence
from scipy.integrate import quad

# ── Physical constants [MeV] ───────────────────────────────────────────────────
alpha   = 1.0 / 137.036
m_e     = 0.511         # electron mass
m_mu    = 105.658       # muon mass
Delta_E = 20.0          # soft-photon cutoff ΔE


def Li2(z):
    """Dilogarithm Li₂(z) = spence(1−z)."""
    return spence(1.0 - z)


def j_collinear(x, eta):
    """
    Eq. (35): differential collinear photon distribution j(μ/m_ℓ, x, η).

    E_tree is E_mu + E_gamma (total leptonic jet energy)
    Δθ is the angle between the lepton direction and the photon direction

    x   = E_ℓ / E_tree  (lepton energy fraction; E_γ = (1−x) E_tree)
    eta = Δθ [rad] · E_tree / m_ℓ  (dimensionless cone parameter)
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        xe2 = x**2 * eta**2
        result = (alpha / np.pi) * (
              0.5 * (1.0 + x**2) / (1.0 - x) * np.log(1.0 + xe2)
            - x   / (1.0 - x)   * xe2 / (1.0 + xe2)
        )
    return np.where((x > 0) & (x < 1), result, 0.0)


def rad_frac_integrated_energy(Delta_theta_deg, E_tree, m_ell, dE=Delta_E):
    """
    Eq. (36): dσ^γ/dσ_LO as a function of cone half-angle.

    Integrates j(μ/m_ℓ, x, η) over x from 0 to 1 − ΔE/E_tree.

    Parameters
    ----------
    Delta_theta_deg : array_like  Cone half-angle Δθ [degrees]
    E_tree      : float       Tree-level jet energy [MeV]
    m_ell       : float       Charged-lepton mass [MeV]
    dE          : float       Soft-photon cutoff ΔE [MeV]
    """
    theta  = np.deg2rad(np.asarray(Delta_theta_deg, dtype=float))
    eta    = theta * E_tree / m_ell
    result = np.zeros_like(eta)
    mask   = eta > 0.0
    e      = eta[mask]
    log1e2 = np.log(1.0 + e**2)
    atane  = np.arctan(e)
    result[mask] = (alpha / np.pi) * (
          0.5  * Li2(-e**2)
        + 0.25 * log1e2**2
        + 0.25 * (2.0 / (1.0 + e**2) - 1.0 / e**2 - 3.0) * log1e2
        + 9.0  / 4.0
        + (log1e2 - e**2 / (1.0 + e**2)) * np.log(E_tree / dE)
        - atane**2
        - (1.0 + 1.0 / (1.0 + e**2)) * atane / e
    )
    return result


def rad_frac_x_eta(x, eta):
    """
    ∂j/∂η from differentiating Eq. (35).

    This represents collinear_gamma_xs / nominal_xs given x and eta.
    x and eta can be calculated given E_lep, E_gamma, and Δθ.

    x   = E_ℓ / E_tree
    eta = Δθ [rad] · E_tree / m_ℓ
    """
    xe2 = x**2 * eta**2
    with np.errstate(invalid='ignore', divide='ignore'):
        result = (alpha / np.pi) * (x**2 * eta) / ((1.0 - x) * (1.0 + xe2)) * (
            (1.0 + x**2) - 2.0 * x / (1.0 + xe2)
        )

    mask = (x > 0) & (x < 1) & (eta > 0)

    if np.ndim(result) == 0:
        return float(result) if mask else 0.0

    return np.where(mask, result, 0.0)
    

def rad_frac_kinematic(E_gamma, Delta_theta_deg, E_tree, m_ell=m_mu):

    """
    d²(dσ^γ/dσ_LO) / (dE_γ [MeV]  d(Δθ°)) — continuous 2D density.

    All energies in MeV. Jacobian: (π/180) / m_ℓ × ∂j/∂η
    """

    x   = 1.0 - E_gamma / E_tree
    eta = np.deg2rad(Delta_theta_deg) * E_tree / m_ell
    return (np.pi / 180.0) / m_ell * rad_frac_x_eta(x, eta)



def bin_fraction_integrated(E_lo, E_hi, theta_lo_deg, theta_hi_deg, E_tree):
    """
    Exact fraction of σ_LO in a (E_γ, Δθ) bin via 1D quadrature.

    The 2D integral over the bin reduces to a 1D integral by noting that
    integrating ∂j/∂η over η collapses to a boundary difference:

        fraction = ∫_{x_lo}^{x_hi} [j(x, η_hi) − j(x, η_lo)] dx

    where x = 1 − E_γ/E_tree  and  η = Δθ[rad]·E_tree/m_μ. All energies in MeV.
    The Jacobians from the (E_γ[MeV], Δθ°) → (x, η) change of variables
    cancel exactly, leaving a unit-weight 1D integral over x.
    """
    x_lo   = 1.0 - E_hi  / E_tree
    x_hi   = 1.0 - E_lo  / E_tree
    eta_lo = np.deg2rad(theta_lo_deg) * E_tree / m_mu
    eta_hi = np.deg2rad(theta_hi_deg) * E_tree / m_mu

    def integrand(x):
        return float(j_collinear(x, eta_hi) - j_collinear(x, eta_lo))

    result, _ = quad(integrand, x_lo, x_hi, limit=100, epsabs=1e-14, epsrel=1e-10)
    return max(0.0, result)
