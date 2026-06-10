"""Pure-Python reproduction of GENIE's hA2025 pion-FSI reweight (`hA2025reweight`).

Background.  Fan computes a final-state-interaction reweight that
swaps GENIE's INTRANUKE *hA2018* pion fate model for the new *hA2025* model, by
building GENIE master and running ``scripts/rwgtNtupleFSI.C`` on a Fermilab gpvm
(see ``BuildEventGenerators/``).  That macro clones the PeLEE
``nuselection/NeutrinoSelectionFilter`` tree and adds one float branch,
``hA2025reweight``, per event. This re-implements that workflow in python.

What the GENIE macro does, per event (rwgtNtupleFSI.C):
  1. target nucleus = mc_generator particle [1]; must be an ion with A>1, Z>1.
  2. find the *pre-FSI* nuclear remnant: the mc_generator particle whose status
     code is kIStFinalStateNuclearRemnant (15) points (via its `mother` field,
     used as an array index) at the remnant ion; its mass number `remnA` sets
     the target-A argument of the fate-fraction lookups.
  3. event_weight = product over every pion that was rescattered by INTRANUKE
     (rescatter fate code > 1) of  FracADep_2025(fate, KE, remnA)
                                  / FracADep_2018(fate, KE, remnA),
     where the ratio is taken as 1.0 whenever the 2018 fraction is 0.

FracADep returns the renormalised cross-section *fraction* for a given fate
(charge-exchange / inelastic / absorption / pi-production) of a pion with the
given kinetic energy KE (MeV, clamped to [1, 999]) on a nucleus of mass number
A (clamped to <=208).  hA2018 stores the fractions directly; hA2025 stores the
component cross sections (as log values) and forms the fractions on the fly.
Both interpolate over scattered (A, KE) points with ROOT's
``TGraph2D::Interpolate`` (Delaunay-triangulation linear interpolation), which we
reproduce with ``scipy.interpolate.LinearNDInterpolator`` on min-max-normalised
(A, KE) coordinates (the same normalisation ROOT::Math::Delaunay2D applies).

The fate codes match GENIE's INukeHadroFates2018/2025 enums (kIHAFtElas removed
in both, so the numbering is shared): CEx=2, Inelas=3, Abs=4, PiProd=7; the data
files store rescatter codes in this same convention (verified against the
reference file).  Pi+, pi-, pi0 share the same pion tables in GENIE.

See ``ipynb_notebooks``-style usage / validation in
``src`` test code and ``compute_pion_fsi_weights`` below.
"""

import os
import numpy as np
import triangle as _triangle
from matplotlib.tri import Triangulation as _Triangulation
from matplotlib.tri import LinearTriInterpolator as _LinearTriInterp
from scipy.interpolate import LinearNDInterpolator as _LinearND
from scipy.interpolate import CloughTocher2DInterpolator as _CloughTocher

# ── GENIE INTRANUKE data directory (shipped with the coworker's build) ───────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENIE_INTRANUKE_DIR = os.path.join(
    _REPO_ROOT, "BuildEventGenerators", "Generator",
    "data", "evgen", "intranuke", "tot_xsec",
)

# ── Fate codes (INukeHadroFates2018.h / 2025.h, shared numbering) ────────────
FATE_CEX = 2
FATE_INELAS = 3
FATE_ABS = 4
FATE_PIPRO = 7
_PION_FATES = (FATE_CEX, FATE_INELAS, FATE_ABS, FATE_PIPRO)

# ── KE clamps used inside FracADep (MeV) ─────────────────────────────────────
_KE_MIN = 1.0
_KE_MAX = 999.0
_A_MAX = 208

# GENIE works in GeV; FracADep wants KE in MeV (ke = KE / units::MeV).
_MEV_PER_GEV = 1000.0

# GHepStatus: pre-FSI nuclear remnant
_K_REMNANT_STATUS = 15

_PION_PDGS = (211, -211, 111)

# Nuclei lists baked into INukeHadroData2018/2025 LoadCrossSections (the file
# loops, in order, over pip<N>_<fate>.txt for these N).
_NUCLEI_2025 = {
    "cex":    [3, 27, 12, 56, 93, 209, 7],
    "inelas": [3, 27, 12, 56, 93, 209, 7],
    "abs":    [3, 27, 12, 56, 93, 209, 7],
    "pipro":  [27, 12, 56, 93, 209, 7],
    "tot":    [12, 27, 3, 56, 93, 209, 7],
}
_NUCLEI_2018 = {
    "cex":    [1, 2, 3, 4, 7, 9, 12, 16, 27, 48, 56, 58, 63, 93, 120, 165, 181, 209],
    "abs":    [1, 2, 3, 4, 7, 9, 12, 16, 27, 48, 56, 58, 63, 93, 120, 165, 181, 209],
    "inelas": [1, 2, 3, 4, 7, 9, 12, 16, 27, 40, 48, 56, 58, 63, 93, 120, 165, 181, 208, 209],
    "pipro":  [1, 2, 3, 4, 7, 9, 12, 16, 48, 56, 58, 63, 93, 120, 165, 181, 209],
}


# ── ROOT TGraph(filename) text reader ────────────────────────────────────────
def _read_tgraph_txt(path):
    """Read a 2-column text file the way ROOT's ``TGraph(filename, "%lg %lg")``
    does: parse the first two whitespace tokens of each line as doubles, and
    silently skip any line that does not start with two parseable numbers
    (e.g. the ``## ...`` comment headers in the hA2018 fraction files)."""
    xs, ys = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


class _Interp2D:
    """Scattered 2D interpolator over (A, KE) -> z used for every FracADep table.

    Every engine first normalises each axis to [-0.5, 0.5] with
    ``xN = (x - (xmax+xmin)/2) / (xmax-xmin)`` (min/max over *all* points,
    phantom origins included) -- exactly as ROOT::Math::Delaunay2D does -- so the
    very different A and KE scales get a 1:1 aspect ratio, and so a custom engine
    is compared on the same footing as the exact one.  Outside the convex hull
    the result is 0.

    ``engine`` selects the interpolation method:
      * ``"root628"`` (default): the bit-exact GENIE reproduction.  Delaunay
        triangulation with Shewchuk's *Triangle* library (flags "zQN") -- the
        SAME C code ROOT 6.28's non-CGAL build uses (math/mathcore/src/
        Delaunay2D.cxx), via the ``triangle`` package -- then linear barycentric
        interpolation, plus ROOT's grid-artefact guard (if the value is exactly
        0, retry at ``xN + 1e-4``).  Nevis's own ROOT 6.38 does NOT reproduce
        this (built with CGAL -> natural-neighbour interp), so we port 6.28.
      * ``"qhull"``: scipy ``LinearNDInterpolator`` (Qhull Delaunay) -- same
        piecewise-linear idea, different triangulation/tie-break; useful to see
        how sensitive the weights are to the triangulation choice.
      * ``"cubic"``: scipy ``CloughTocher2DInterpolator`` -- a smooth C1 cubic, a
        "what if we interpolated physically smoothly" alternative.
      * a *callable* ``factory(xn, yn, z)`` returning ``eval(xn_q, yn_q) -> z``:
        your own interpolation method, evaluated in the normalised space (the
        normalisation/dedup/phantom handling above is done for you).
    """

    def __init__(self, x, y, z, engine="root628"):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        z = np.asarray(z, float)
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        self._offx = -(xmax + xmin) / 2.0
        self._sx = 1.0 / (xmax - xmin)
        self._offy = -(ymax + ymin) / 2.0
        self._sy = 1.0 / (ymax - ymin)
        xn = (x + self._offx) * self._sx
        yn = (y + self._offy) * self._sy

        # Collapse exactly-coincident vertices to their first occurrence (the
        # phantom origins, and any repeated data point): Triangle and matplotlib's
        # point-finder dislike duplicates, and a Delaunay triangulation is defined
        # by the point *set* anyway, so this matches ROOT, whose Triangle build
        # likewise collapses coincident input.
        _, first = np.unique(np.column_stack([xn, yn]), axis=0, return_index=True)
        keep = np.sort(first)
        xn, yn, z = xn[keep], yn[keep], z[keep]

        self._engine = engine
        self._is_root628 = (engine == "root628")
        if engine == "root628":
            d = _triangle.triangulate({"vertices": np.column_stack([xn, yn])}, "zQN")
            tri = _Triangulation(xn, yn, triangles=d["triangles"])
            mpl = _LinearTriInterp(tri, z)
            self._eval = lambda xx, yy: np.ma.filled(mpl(xx, yy), 0.0).astype(float)
        elif engine == "qhull":
            nd = _LinearND(np.column_stack([xn, yn]), z, fill_value=0.0)
            self._eval = lambda xx, yy: nd(np.column_stack([xx, yy]))
        elif engine == "cubic":
            ct = _CloughTocher(np.column_stack([xn, yn]), z, fill_value=0.0)
            self._eval = lambda xx, yy: ct(np.column_stack([xx, yy]))
        elif callable(engine):
            self._eval = engine(xn, yn, z)
        else:
            raise ValueError("unknown interpolation engine: %r" % (engine,))

    def __call__(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        xx = (x + self._offx) * self._sx
        yy = (y + self._offy) * self._sy
        zz = np.asarray(self._eval(xx, yy), dtype=float)
        if self._is_root628:
            # ROOT: "Wrong zeros may appear when points sit on a regular grid"
            zero = zz == 0.0
            if zero.any():
                zz = zz.copy()
                zz[zero] = np.asarray(self._eval(xx[zero] + 0.0001, yy[zero]), float)
        return zz


# hA2025 LoadCrossSections preallocates each TGraph2D to this many points, but
# only fills the xsec>0 ones; the unfilled slots stay at (A=0, KE=0, log=0) and
# act as real "phantom" vertices in ROOT's Delaunay triangulation, extending the
# hull below the KE=25 threshold so that low-KE pions interpolate to a nonzero
# value instead of 0.  Reproducing them is required for an exact weight match.
# (Duplicate origin points collapse to a single Delaunay vertex, so one suffices;
# its presence also pulls the min-max normalisation origin to (0,0), matching
# ROOT::Math::Delaunay2D.)
_PREALLOC_2025 = {"cex": 294, "inelas": 294, "abs": 294, "pipro": 252}


def _load_2025_table(fate, engine="root628", include_phantom=True):
    """hA2025: TPipA_<fate>, points (A, KE, log(xsec)) keeping only xsec>0,
    matching the ``if (y>0) SetPoint(A, KE, log(y))`` loop, plus the phantom
    (0,0,0) vertices left by the preallocation (see _PREALLOC_2025).

    ``include_phantom=False`` drops those phantom origins -- physically the more
    defensible choice (no fictitious A=0 nucleus), but then sub-threshold pions
    fall outside the hull and get fraction 0 / no reweight."""
    xs_all, ys_all, zs_all = [], [], []
    nreal = 0
    for n in _NUCLEI_2025[fate]:
        path = os.path.join(GENIE_INTRANUKE_DIR, "2025", "pipA_%s" % fate,
                            "pip%d_%s.txt" % (n, fate))
        ke, y = _read_tgraph_txt(path)
        m = y > 0
        nreal += int(m.sum())
        xs_all.append(np.full(m.sum(), float(n)))
        ys_all.append(ke[m])
        zs_all.append(np.log(y[m]))
    if include_phantom and _PREALLOC_2025[fate] > nreal:
        # one phantom vertex at the origin (duplicates collapse to one)
        xs_all.append(np.array([0.0]))
        ys_all.append(np.array([0.0]))
        zs_all.append(np.array([0.0]))
    return _Interp2D(np.concatenate(xs_all), np.concatenate(ys_all),
                     np.concatenate(zs_all), engine=engine)


def _load_2018_table(fate, engine="root628"):
    """hA2018: TfracPipA_<fate>, points (A, KE, fraction) with no positivity
    filter (the loader adds every point).  The 2018 tables have no phantoms."""
    xs_all, ys_all, zs_all = [], [], []
    for n in _NUCLEI_2018[fate]:
        path = os.path.join(GENIE_INTRANUKE_DIR, "pipA_%s_frac" % fate,
                            "pip%d_%s_frac.txt" % (n, fate))
        ke, frac = _read_tgraph_txt(path)
        xs_all.append(np.full(len(ke), float(n)))
        ys_all.append(ke)
        zs_all.append(frac)
    return _Interp2D(np.concatenate(xs_all), np.concatenate(ys_all),
                     np.concatenate(zs_all), engine=engine)


_FATE_TO_KEY = {FATE_CEX: "cex", FATE_INELAS: "inelas",
                FATE_ABS: "abs", FATE_PIPRO: "pipro"}

# Cache of (engine, include_phantom) -> (T2025, T2018) table dicts, so repeated
# calls (and the default bit-exact path) don't rebuild interpolators.
_TABLE_CACHE = {}


def make_tables(engine="root628", include_phantom=True):
    """Return ``(T2025, T2018)`` dicts of the four fate interpolators each, built
    with the given interpolation ``engine`` (see ``_Interp2D``) and phantom
    choice.  Cached.  ``engine`` may be a string or your own
    ``factory(xn, yn, z) -> eval`` callable (cached by id)."""
    key = (engine if isinstance(engine, str) else id(engine), include_phantom)
    if key not in _TABLE_CACHE:
        t2025 = {f: _load_2025_table(f, engine=engine, include_phantom=include_phantom)
                 for f in ("cex", "inelas", "abs", "pipro")}
        t2018 = {f: _load_2018_table(f, engine=engine)
                 for f in ("cex", "inelas", "abs", "pipro")}
        _TABLE_CACHE[key] = (t2025, t2018)
    return _TABLE_CACHE[key]


# Default bit-exact tables, built once on import.
_T2025, _T2018 = make_tables("root628", include_phantom=True)


def _frac_adep_2018(fate, ke, A, tables=None):
    """Vectorised INukeHadroData2018::FracADep for pions.  ke (MeV), A arrays
    already clamped by the caller.  ``tables`` defaults to the bit-exact set."""
    t = _T2018 if tables is None else tables
    fc = t["cex"](A, ke)
    fi = t["inelas"](A, ke)
    fa = t["abs"](A, ke)
    fp = t["pipro"](A, ke)
    total = fc + fi + fa + fp
    num = np.select([fate == FATE_CEX, fate == FATE_INELAS,
                     fate == FATE_ABS, fate == FATE_PIPRO],
                    [fc, fi, fa, fp], default=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total != 0.0, num / total, 0.0)


def _frac_adep_2025(fate, ke, A, tables=None):
    """Vectorised INukeHadroData2025::FracADep for pions.  The tables store
    log(xsec); the total reaction xsec cancels in the renormalised fraction."""
    t = _T2025 if tables is None else tables
    cex = np.exp(t["cex"](A, ke))
    inel = np.exp(t["inelas"](A, ke))
    ab = np.exp(t["abs"](A, ke))
    pp = np.exp(t["pipro"](A, ke))
    total = cex + inel + ab + pp
    num = np.select([fate == FATE_CEX, fate == FATE_INELAS,
                     fate == FATE_ABS, fate == FATE_PIPRO],
                    [cex, inel, ab, pp], default=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total != 0.0, num / total, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# hA2025c -- hA2025 with pion-charge considerations (v0, charge-exchange only)
# ─────────────────────────────────────────────────────────────────────────────
# GENIE's hA model uses the pi+ -on-A cross sections for pi+, pi-, AND pi0 alike.
# That is charge-blind, which mainly matters for charge exchange (CEx), whose rate
# scales with the number of *target nucleons of the right type*:
#     pi+ + n -> pi0 + p   (off a NEUTRON)   -> rate ~ N
#     pi- + p -> pi0 + n   (off a PROTON)    -> rate ~ Z
#     pi0 + p -> pi+ + n   and   pi0 + n -> pi- + p   (off EITHER) -> rate ~ Z+N = A
# By detailed balance / charge symmetry the elementary single-CEx cross sections
# are ~equal, so relative to the pi+ baseline the CEx fate fraction picks up a
# per-pion factor:
#     pi+ :  N / Nref      (only the nucleus being more/less neutron-rich than the
#                           reference nucleus the pi+ table represents at this A)
#     pi- :  Z / Nref
#     pi0 :  A / Nref      (the two-channel "doubling": ~2 for isoscalar, ~A/N)
# where Nref(A) is the neutron number the pi+ table represents at mass number A
# (interpolated from the actual N of the hA2025 CEx data nuclei).  v0 applies this
# to CEx only (the dominant, cleanest term); abs/inelas/pipro keep factor 1.0.
# This is a leading-order nucleon-counting model: it omits the renormalisation of
# the other fate fractions, the Delta Clebsch-Gordan / KE structure, and the
# absorption isospin weights -- refinements for later versions.
#
# Neutrons of the 2025 CEx data nuclei (He3, Li7, C12, Al27, Fe56, Nb93, Bi209):
_NREF_A = np.array([3, 7, 12, 27, 56, 93, 209], dtype=float)
_NREF_N = _NREF_A - np.array([2, 3, 6, 13, 26, 41, 83], dtype=float)  # N = A - Z


def _n_ref(A):
    """Reference neutron number the pi+ hA table represents at mass number A
    (linear interpolation of the CEx data nuclei's actual N vs A)."""
    return np.interp(A, _NREF_A, _NREF_N)


def _charge_correction(fate, pion_pdg, Z, N):
    """hA2025c per-pion factor C(fate, charge, Z, N): 1.0 except for charge
    exchange, where the rate scales with the available target nucleons relative
    to the pi+ baseline N_ref(A).  Vectorised.

        pion   CEx channel(s)                       available    factor
        pi+    pi+ n -> pi0 p   (off a neutron)      N            N / Nref
        pi-    pi- p -> pi0 n   (off a proton)       Z            Z / Nref
        pi0    pi0 p -> pi+ n  &  pi0 n -> pi- p     Z + N = A    A / Nref
    """
    nref = _n_ref(Z + N)
    c_piplus = N / nref           # pi+ : charge-exchanges off neutrons
    c_piminus = Z / nref          # pi- : charge-exchanges off protons
    c_pizero = (Z + N) / nref     # pi0 : off either nucleon (two channels)
    c_cex = np.select(
        [pion_pdg == 211, pion_pdg == -211, pion_pdg == 111],
        [c_piplus, c_piminus, c_pizero],
        default=1.0,               # non-pion -> no charge correction
    )
    return np.where(fate == FATE_CEX, c_cex, 1.0)


# ── ion PDG decoding (PDG ion code convention 10LZZZAAAI) ────────────────────
def _is_ion(pdg):
    return (pdg > 1000000000) & (pdg < 1999999999)


def _ion_A(pdg):
    return (pdg // 10) - 1000 * (pdg // 10000)


def _ion_Z(pdg):
    return (pdg // 10000) - 1000 * (pdg // 10000000)


def compute_pion_fsi_weights(
    file_path,
    tree_path="nuselection/NeutrinoSelectionFilter",
    entry_start=0,
    entry_stop=None,
    return_queries=False,
    engine="root628",
    include_phantom=True,
):
    """Read a ROOT file and return ``(hA2025, additional_hA2025c)`` per-event
    weight arrays (see :func:`compute_pion_fsi_weights_from_arrays`).

    With the defaults (``engine="root628"``, ``include_phantom=True``) the
    ``hA2025`` array is the bit-exact GENIE reproduction.  Pass a different
    ``engine`` (``"qhull"``, ``"cubic"``, or your own ``factory(xn, yn, z) ->
    eval`` callable) and/or ``include_phantom=False`` to test alternative
    interpolation choices -- only the fate-fraction interpolation changes.

    Reads with uproot; no ROOT/GENIE/cvmfs dependency.  With
    ``return_queries=True`` a third element, the per-reweighted-pion dict, is
    appended (the raw ingredients, handy for debugging / validation plots).
    """
    import uproot

    branches = [
        "mc_generator_pdg", "mc_generator_mother", "mc_generator_rescatter",
        "mc_generator_statuscode", "mc_generator_E",
        "mc_generator_px", "mc_generator_py", "mc_generator_pz",
    ]
    with uproot.open(file_path) as f:
        arr = f[tree_path].arrays(branches, entry_start=entry_start,
                                  entry_stop=entry_stop, library="np")
    return compute_pion_fsi_weights_from_arrays(
        arr["mc_generator_pdg"], arr["mc_generator_mother"],
        arr["mc_generator_rescatter"], arr["mc_generator_statuscode"],
        arr["mc_generator_E"], arr["mc_generator_px"],
        arr["mc_generator_py"], arr["mc_generator_pz"],
        engine=engine, include_phantom=include_phantom,
        return_queries=return_queries)


def compute_pion_fsi_weights_from_arrays(pdg, mother, rescatter, status, E, px, py, pz,
                                engine="root628", include_phantom=True,
                                return_queries=False):
    """Per-event pion-FSI weights from already-loaded ``mc_generator_*`` arrays
    (each an object array of variable-length per-event vectors).

    Same logic as :func:`compute_pion_fsi_weights`, but takes the branches
    directly so a caller that has already opened the file (e.g. the create_df
    pipeline reading the nuselection tree) does not re-read it.

    Returns ``(hA2025, additional_hA2025c)``, two arrays aligned to the input
    event order:
      * ``hA2025`` -- the bit-exact GENIE hA2018->hA2025 reweight, and
      * ``additional_hA2025c`` -- the extra per-event factor (product of the
        per-pion :func:`_charge_correction`) that turns hA2025 into the
        charge-aware hA2025c: ``weight_hA2025c = hA2025 * additional_hA2025c``.
        1.0 for events with no charge-exchange pion / an isoscalar remnant.
    With ``return_queries=True`` a third element, the per-reweighted-pion dict,
    is appended.
    """
    t2025, t2018 = make_tables(engine, include_phantom)

    n_events = len(pdg)

    # First pass: per-event validity + remnant A, and collect the flat list of
    # reweightable pion queries (event index, fate, KE_MeV, remnA).
    remnA = np.zeros(n_events, dtype=float)
    cal_weight = np.zeros(n_events, dtype=bool)

    # per reweightable-pion queries: event index, fate, KE (MeV), remnant A,
    # remnant Z, and the pion's own pdg (charge) -- the last two only used by
    # the hA2025c charge correction.
    q_evt, q_fate, q_ke, q_A, q_Z, q_pdg = [], [], [], [], [], []

    for i in range(n_events):
        p = pdg[i]
        if len(p) < 2:
            continue
        target_pdg = p[1]
        if not (1000000000 < target_pdg < 1999999999):
            continue
        A = _ion_A(target_pdg)
        Z = _ion_Z(target_pdg)
        if A <= 1 or Z <= 1:
            continue

        # pre-FSI remnant: last particle with status==15 -> its mother is the
        # array index of the remnant ion.
        st = status[i]
        remnant_trackid = -999
        rem_idx = np.where(st == _K_REMNANT_STATUS)[0]
        if rem_idx.size:
            remnant_trackid = int(mother[i][rem_idx[-1]])
        if remnant_trackid == -999:
            continue
        if not (0 <= remnant_trackid < len(p)):
            continue
        rem_pdg = p[remnant_trackid]
        if not (1000000000 < rem_pdg < 1999999999):
            continue
        rA = _ion_A(rem_pdg)
        rZ = _ion_Z(rem_pdg)
        remnA[i] = rA
        cal_weight[i] = True

        # collect reweightable pions: |pdg| in {211,111} and rescatter>1.
        pp = p
        rr = rescatter[i]
        is_pi = (pp == 211) | (pp == -211) | (pp == 111)
        sel = is_pi & (rr > 1)
        idx = np.where(sel)[0]
        if idx.size == 0:
            continue
        Ei = E[i][idx].astype(float)
        pxi = px[i][idx].astype(float)
        pyi = py[i][idx].astype(float)
        pzi = pz[i][idx].astype(float)
        m2 = Ei * Ei - (pxi * pxi + pyi * pyi + pzi * pzi)
        mass = np.sqrt(np.clip(m2, 0.0, None))
        ke_mev = (Ei - mass) * _MEV_PER_GEV
        for j, ridx in enumerate(idx):
            q_evt.append(i)
            q_fate.append(int(rr[ridx]))
            q_ke.append(ke_mev[j])
            q_A.append(rA)
            q_Z.append(rZ)
            q_pdg.append(int(pp[ridx]))

    weights = np.ones(n_events, dtype=float)       # hA2025
    additional = np.ones(n_events, dtype=float)    # additional hA2025c factor
    if not q_evt:
        if return_queries:
            empty = np.array([], dtype=float)
            return weights, additional, {k: empty for k in
                             ("evt", "fate", "ke", "A", "Z", "pdg",
                              "f18", "f25", "ratio", "charge_corr")}
        return weights, additional

    q_evt = np.asarray(q_evt)
    q_fate = np.asarray(q_fate)
    q_ke = np.clip(np.asarray(q_ke, dtype=float), _KE_MIN, _KE_MAX)
    q_A = np.asarray(q_A, dtype=float)        # raw remnant A (for nucleon counting)
    q_Z = np.asarray(q_Z, dtype=float)
    q_pdg = np.asarray(q_pdg)
    q_A_lookup = np.minimum(q_A, _A_MAX)      # clamped A for the FracADep tables

    # Only fates handled by FracADep (2,3,4,7) contribute; others -> ratio 1.
    handled = np.isin(q_fate, _PION_FATES)

    f18_all = np.full(len(q_evt), np.nan)
    f25_all = np.full(len(q_evt), np.nan)
    ratio = np.ones(len(q_evt), dtype=float)
    if handled.any():
        f18 = _frac_adep_2018(q_fate[handled], q_ke[handled], q_A_lookup[handled], t2018)
        f25 = _frac_adep_2025(q_fate[handled], q_ke[handled], q_A_lookup[handled], t2025)
        f18_all[handled] = f18
        f25_all[handled] = f25
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio[handled] = np.where(f18 != 0.0, f25 / f18, 1.0)

    # per-pion charge correction (hA2025c); 1.0 except for charge-exchange pions.
    charge_corr = _charge_correction(q_fate, q_pdg, q_Z, q_A - q_Z)

    # hA2025 weight  = product over the event's pions of the hA2018->hA2025 ratio
    # additional fac = product over the event's pions of the charge correction
    np.multiply.at(weights, q_evt, ratio)
    np.multiply.at(additional, q_evt, charge_corr)

    if return_queries:
        queries = {"evt": q_evt, "fate": q_fate, "ke": q_ke, "A": q_A, "Z": q_Z,
                   "pdg": q_pdg, "f18": f18_all, "f25": f25_all, "ratio": ratio,
                   "charge_corr": charge_corr}
        return weights, additional, queries
    return weights, additional


if __name__ == "__main__":
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else (
        "/nevis/riverside/data/leehagaman/ngem/other_files/"
        "checkout_MCC9.10_Run4b_v10_04_07_20_BNB_nu_overlay_retuple_retuple_hist_FSIrwgt.root"
    )
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    w, add = compute_pion_fsi_weights(fp, entry_stop=n)
    print("computed %d weights: hA2025 mean %.5f, additional hA2025c mean %.5f"
          % (len(w), w.mean(), add.mean()))
