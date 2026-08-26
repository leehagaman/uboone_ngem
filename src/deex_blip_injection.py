"""Inject simulated nuclear de-excitation photon blips into MC events.

The MicroBooNE production GENIE tune did not simulate nuclear de-excitation
photons. This module adds them as an alternative-universe prediction: for each
MC event it samples the de-excitation photon content predicted by the
GENIE+INCL+++MARLEY model (see deexcitation_photon_study_marley.py) and
injects the corresponding *reconstructed* blips, taken from the dedicated
low-energy isotropic single-photon overlay production (see
lowE_iso1g_blip_response.py). blip_postprocessing then recomputes every blip
variable on the augmented blip lists, producing "<variable>_deexadded" twins
of every nominal variable. The systematic bracket is the comparison of the
nominal (no de-excitation) and _deexadded (MARLEY de-excitation) variables.

How one event is processed:
1. Match the event's (CC/NC, QE/MEC/RES/DIS, N_p^35, N_n^35) to the MARLEY
   event library and take the library event nearest in true energy transfer
   q0 (randomly among the MATCH_KNEAREST nearest, so donor usage stays
   approximately uniform). The matched donor supplies a list of true
   de-excitation photon energies (possibly empty). Nucleon counts use the
   same KE > 35 MeV final-state-primaries definition on both sides, because
   raw counts are not comparable across generators (INCL+MARLEY evaporates
   many soft nucleons that the production hA FSI model does not).
2. For each photon with E <= MAX_PHOTON_E_MEV, sample a donor event from the
   low-E isotropic 1-gamma library in the same photon-energy bin. The donor's
   truth-matched reco blips are translated rigidly so the donor's photon
   origin lands on this event's true neutrino vertex (the photon direction is
   the donor's random isotropic direction); translated blips that fall
   outside the TPC active volume are dropped.
3. The donor blips (with their reco energy, dx, dw, nplanes, touchtrk,
   dead-wire flags, and track-proximity values) are appended to copies of the
   event's blip lists.

Which events are injected: filetypes with GENIE truth (not data / ext /
NuWro fake data / single-photon overlays), with scattering type QE, RES, DIS,
or MEC. Coherent and other rare channels get no photons by construction
(coherent scattering leaves the nucleus in its ground state; the MARLEY
sample contains no such events). Neutrino flavor is ignored in the matching
(the numu-only MARLEY library is used for all flavors; de-excitation physics
does not depend on the lepton flavor).

Documented approximations (all second-order vs the no-deex/MARLEY bracket):
- Rigid translation ignores space-charge differences between the donor and
  target locations (~cm effects at tens-of-cm scales).
- Per-plane readout context cannot be translated: nplanes, nwires and
  dead-wire flags reflect the donor location's channel conditions. Donor
  vertices are uniform in the TPC, so detector-average attrition is modeled,
  but position-correlated dead regions at the target are not. Drift-dependent
  (x) reconstruction differences are likewise frozen at the donor value.
- proxtrkdist / touchtrk measure isolation from the donor event's cosmics,
  a statistically equivalent but not event-correct environment.
- Detector-boundary effects are split between the two sides: on the donor
  side, even with the DONOR_MIN_DWALL_CM cut some donor photons exit the
  detector before depositing, so the donor blip sets carry a residual
  donor-side truncation (a slight underestimate of the injected blip
  content); on the target side this is handled explicitly by dropping
  translated blips that would land outside the TPC active volume
  (is_blip_in_tpc at injection time), so photons that would escape from the
  target vertex correctly lose their far blips.
- MARLEY photons above MAX_PHOTON_E_MEV (0.06% of photons) are skipped:
  the donor sample only covers 0-15 MeV.
- Possible merging of an injected blip with a nearby real deposit is not
  modeled.

Donor libraries are built once and cached as parquet in
intermediate_files_location (delete the caches to rebuild after changing the
source files). Per-event bookkeeping columns (deex_marley_donor_index,
deex_iso1g_donor_indices, deex_n_photons*, deex_marley_match_level) are kept
in the output dataframe so donor-usage uniformity can be checked from all_df.

Sampling is deterministic per (run, subrun, event), so reprocessing
reproduces the same injection.

The de-excitation model is pluggable: a model is just a library table of
(is_cc, scattype, np35, nn35, q0_mev, photon_energies) rows. To add another
model, add a builder here and construct DeexInjector(model="<name>").
"""

import os

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from file_locations import intermediate_files_location, other_files_location

MARLEY_GST_FILE = os.path.join(
    other_files_location, "generator_files", "deex",
    "GENIE_INCL_MARLEY_numu_flux_500k.gst.root")
LOWE_ISO1G_NTUPLE = os.path.join(
    other_files_location, "lowE_1g_files", "lowE_iso1g_1000job_ntuple.root")

DEFAULT_MODEL = "marley"

MAX_PHOTON_E_MEV = 15.0    # iso1g donor coverage; rarer MARLEY photons are skipped
DEEX_LIBRARY_E_MAX_MEV = 30.0  # de-excitation photons in the MARLEY gst end here
NUCLEON_COUNT_CAP = 4      # np35/nn35 matched as 0,1,2,3,>=4
MATCH_KNEAREST = 16        # sample uniformly among this many nearest-q0 donors
ISO1G_EBIN_MEV = 0.25      # photon-energy bin width for iso1g donor lookup
ISO1G_MIN_DONORS = 5       # widen the energy window until at least this many
DONOR_MIN_DWALL_CM = 50.0  # iso1g donors must start this far from every TPC wall

# GENIE scattering-type codes (wc_truth_nuScatType) that get de-excitation
SCATTYPE_QE, SCATTYPE_DIS, SCATTYPE_RES, SCATTYPE_MEC = 1, 3, 4, 10
INJECTED_SCATTYPES = (SCATTYPE_QE, SCATTYPE_DIS, SCATTYPE_RES, SCATTYPE_MEC)

# filetypes with no (usable) GENIE truth: never injected, _deexadded == nominal
NO_INJECTION_FILETYPES = (
    "data", "ext", "nuwro_fake_data",
    "isotropic_one_gamma_overlay", "delete_one_gamma_overlay",
)

PROTON_MASS_GEV = 0.938272088
NEUTRON_MASS_GEV = 0.939565422

# blip list columns that do_blip_postprocessing reads; injected blips must
# provide every one of these
BLIP_AUG_COLUMNS = [
    "blip_x", "blip_y", "blip_z", "blip_energy", "blip_dx", "blip_dw",
    "blip_nplanes", "blip_touchtrk", "blip_pl2_bydeadwire", "blip_proxtrkdist",
]
# offsets are stored relative to the donor photon origin; positions are
# rebuilt at injection time
_ISO1G_BLIP_FIELDS = ["off_x", "off_y", "off_z", "energy", "dx", "dw",
                      "nplanes", "touchtrk", "pl2_bydeadwire", "proxtrkdist"]

MARLEY_LIBRARY_PARQUET = os.path.join(
    intermediate_files_location, "deex_marley_library.parquet")
ISO1G_LIBRARY_PARQUET = os.path.join(
    intermediate_files_location, "deex_iso1g_library.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# library builders (run once, cached as parquet)
# ─────────────────────────────────────────────────────────────────────────────

def build_marley_library(cache_path=MARLEY_LIBRARY_PARQUET):
    """Event library from the GENIE+INCL+MARLEY gst file: matching keys plus
    the list of de-excitation photon energies per event."""
    import polars as pl
    print(f"building MARLEY de-excitation library from {MARLEY_GST_FILE}")
    tree = uproot.open(MARLEY_GST_FILE)["gst"]
    a = tree.arrays(["pdgf", "Ef", "qel", "mec", "res", "dis", "cc",
                     "Ev", "El"])

    is_p = a["pdgf"] == 2212
    is_n = a["pdgf"] == 2112
    np35 = np.asarray(ak.sum(is_p & (a["Ef"] - PROTON_MASS_GEV > 0.035), axis=1))
    nn35 = np.asarray(ak.sum(is_n & (a["Ef"] - NEUTRON_MASS_GEV > 0.035), axis=1))

    is_deex = (a["pdgf"] == 22) & (a["Ef"] < DEEX_LIBRARY_E_MAX_MEV / 1000.0)
    photons = [
        [1000.0 * e for e in evt] for evt in ak.to_list(a["Ef"][is_deex])
    ]

    scattype = (SCATTYPE_QE * np.asarray(a["qel"])
                + SCATTYPE_DIS * np.asarray(a["dis"])
                + SCATTYPE_RES * np.asarray(a["res"])
                + SCATTYPE_MEC * np.asarray(a["mec"])).astype(np.int32)

    df = pl.DataFrame({
        "is_cc": np.asarray(a["cc"]).astype(bool),
        "scattype": scattype,
        "np35": np35.astype(np.int32),
        "nn35": nn35.astype(np.int32),
        "q0_mev": np.asarray((a["Ev"] - a["El"]) * 1000.0, dtype=np.float32),
        "photon_energies_mev": photons,
    })
    df.write_parquet(cache_path)
    print(f"  {df.height:,} events saved to {cache_path}")
    return df


def build_iso1g_library(cache_path=ISO1G_LIBRARY_PARQUET):
    """Donor library from the low-E isotropic 1-gamma production: per event,
    the true photon energy and the truth-matched reco blips as offsets from
    the photon origin, with every field do_blip_postprocessing needs.

    Donors must start >= DONOR_MIN_DWALL_CM from every TPC wall so their blip
    sets carry minimal donor-side boundary truncation; target-side boundary
    effects are restored by the TPC clipping at injection time. No photon-ray
    requirement is applied (that would bias the donor directions inward)."""
    import polars as pl
    from check_ntuple_alignment import check_ntuple_alignment
    from blip_postprocessing import (TPC_X_MIN, TPC_X_MAX, TPC_Y_MIN,
                                     TPC_Y_MAX, TPC_Z_MIN, TPC_Z_MAX)

    print(f"building iso1g de-excitation donor library from {LOWE_ISO1G_NTUPLE}")
    f = uproot.open(LOWE_ISO1G_NTUPLE)
    bad = [r["tree"] for r in check_ntuple_alignment(f)
           if not r["aligned"] and "eventweight" not in r["tree"]]
    if bad:
        raise RuntimeError(f"misaligned trees in {LOWE_ISO1G_NTUPLE}: {bad}")

    pf = f["wcpselection/T_PFeval"].arrays(
        ["truth_pdg", "truth_mother", "truth_startMomentum", "truth_startXYZT"])
    prim = (pf["truth_pdg"] == 22) & (pf["truth_mother"] == 0)
    mom = np.asarray(ak.flatten(pf["truth_startMomentum"][prim]))
    pos = np.asarray(ak.flatten(pf["truth_startXYZT"][prim]))
    E_mev = mom[:, 3] * 1000.0
    vtx = pos[:, :3]

    d_wall = np.minimum.reduce([
        vtx[:, 0] - TPC_X_MIN, TPC_X_MAX - vtx[:, 0],
        vtx[:, 1] - TPC_Y_MIN, TPC_Y_MAX - vtx[:, 1],
        vtx[:, 2] - TPC_Z_MIN, TPC_Z_MAX - vtx[:, 2],
    ])

    ns = f["nuselection/NeutrinoSelectionFilter"].arrays(
        ["blip_x", "blip_y", "blip_z", "blip_energy", "blip_dx", "blip_dw",
         "blip_nplanes", "blip_touchtrk", "blip_pl2_bydeadwire",
         "blip_proxtrkdist", "blip_true_g4id"])
    matched = ns["blip_true_g4id"] >= 0

    keep = (d_wall >= DONOR_MIN_DWALL_CM) & (E_mev <= MAX_PHOTON_E_MEV + 0.5)
    keep_idx = np.where(keep)[0]

    def _lists(field, values):
        return [list(values[i]) for i in keep_idx]

    off_x = ns["blip_x"][matched] - vtx[:, 0]
    off_y = ns["blip_y"][matched] - vtx[:, 1]
    off_z = ns["blip_z"][matched] - vtx[:, 2]
    field_values = {
        "off_x": ak.to_list(off_x), "off_y": ak.to_list(off_y),
        "off_z": ak.to_list(off_z),
        "energy": ak.to_list(ns["blip_energy"][matched]),
        "dx": ak.to_list(ns["blip_dx"][matched]),
        "dw": ak.to_list(ns["blip_dw"][matched]),
        "nplanes": ak.to_list(ns["blip_nplanes"][matched]),
        "touchtrk": ak.to_list(ns["blip_touchtrk"][matched]),
        "pl2_bydeadwire": ak.to_list(ns["blip_pl2_bydeadwire"][matched]),
        "proxtrkdist": ak.to_list(ns["blip_proxtrkdist"][matched]),
    }
    df = pl.DataFrame(
        {"true_E_mev": E_mev[keep_idx].astype(np.float32)}
        | {f"blip_{name}": _lists(name, vals)
           for name, vals in field_values.items()}
    )
    df.write_parquet(cache_path)
    print(f"  {df.height:,} donor events saved to {cache_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# injector
# ─────────────────────────────────────────────────────────────────────────────

class DeexInjector:
    """Samples de-excitation photons for an event and provides the donor
    blips to inject. Build once and reuse (module-level get_injector())."""

    def __init__(self, model=DEFAULT_MODEL):
        import polars as pl
        if model != "marley":
            raise ValueError(f"unknown de-excitation model '{model}'; "
                             "add a library builder for it in deex_blip_injection.py")
        self.model = model

        if not os.path.exists(MARLEY_LIBRARY_PARQUET):
            build_marley_library()
        if not os.path.exists(ISO1G_LIBRARY_PARQUET):
            build_iso1g_library()

        lib = pl.read_parquet(MARLEY_LIBRARY_PARQUET)
        self._photons = lib["photon_energies_mev"].to_list()
        is_cc = lib["is_cc"].to_numpy()
        scattype = lib["scattype"].to_numpy()
        np35 = np.minimum(lib["np35"].to_numpy(), NUCLEON_COUNT_CAP)
        nn35 = np.minimum(lib["nn35"].to_numpy(), NUCLEON_COUNT_CAP)
        q0 = lib["q0_mev"].to_numpy()

        # match index: key tuple -> (sorted q0 array, event indices sorted by q0)
        # four fallback levels, coarser keys dropping nucleon counts then current
        self._index = [{} for _ in range(4)]
        key_funcs = [
            lambda i: (bool(is_cc[i]), int(scattype[i]), int(np35[i]), int(nn35[i])),
            lambda i: (bool(is_cc[i]), int(scattype[i]), int(np35[i])),
            lambda i: (bool(is_cc[i]), int(scattype[i])),
            lambda i: (int(scattype[i]),),
        ]
        order = np.argsort(q0, kind="stable")
        for i in order:
            for level, kf in enumerate(key_funcs):
                self._index[level].setdefault(kf(i), []).append(i)
        for level in range(4):
            for k, idxs in self._index[level].items():
                idxs = np.asarray(idxs)
                self._index[level][k] = (q0[idxs], idxs)

        iso = pl.read_parquet(ISO1G_LIBRARY_PARQUET)
        self._iso_E = iso["true_E_mev"].to_numpy()
        self._iso_blips = {
            name: iso[f"blip_{name}"].to_list() for name in _ISO1G_BLIP_FIELDS
        }
        self._iso_bins = {}
        for i, E in enumerate(self._iso_E):
            self._iso_bins.setdefault(int(E / ISO1G_EBIN_MEV), []).append(i)
        self._iso_bins = {k: np.asarray(v) for k, v in self._iso_bins.items()}

    def sample_marley_event(self, is_cc, scattype, np35, nn35, q0_mev, rng):
        """-> (photon energies [MeV], donor index, match level 0-3)."""
        keys = [
            (bool(is_cc), int(scattype), min(int(np35), NUCLEON_COUNT_CAP),
             min(int(nn35), NUCLEON_COUNT_CAP)),
            (bool(is_cc), int(scattype), min(int(np35), NUCLEON_COUNT_CAP)),
            (bool(is_cc), int(scattype)),
            (int(scattype),),
        ]
        entry = None
        for level, key in enumerate(keys):
            candidate = self._index[level].get(key)
            if candidate is not None and (len(candidate[1]) >= MATCH_KNEAREST
                                          or level == len(keys) - 1):
                entry = candidate
                break
        if entry is None:
            return [], -1, -1
        q0_sorted, idx_sorted = entry
        pos = np.searchsorted(q0_sorted, q0_mev)
        lo = max(0, pos - MATCH_KNEAREST // 2)
        hi = min(len(idx_sorted), lo + MATCH_KNEAREST)
        lo = max(0, hi - MATCH_KNEAREST)
        donor = int(idx_sorted[int(rng.integers(lo, hi))])
        return self._photons[donor], donor, level

    def sample_iso1g_donor(self, E_mev, rng):
        """-> donor index in the iso1g library for this photon energy."""
        b = int(E_mev / ISO1G_EBIN_MEV)
        width = 0
        while True:
            cands = np.concatenate([
                self._iso_bins.get(bb, np.empty(0, dtype=int))
                for bb in range(b - width, b + width + 1)
            ])
            if len(cands) >= ISO1G_MIN_DONORS or width > 8:
                break
            width += 1
        if len(cands) == 0:
            return -1
        return int(cands[int(rng.integers(len(cands)))])

    def donor_blips(self, iso_idx):
        """-> dict of per-blip field arrays (offsets relative to photon origin)."""
        return {name: self._iso_blips[name][iso_idx]
                for name in _ISO1G_BLIP_FIELDS}


def rebuild_libraries():
    """Delete the cached donor libraries and rebuild them from the raw source
    ROOT files. Called at the start of a create_df --create_file_dfs run so
    every production starts from scratch, fully logged and reproducible."""
    global _INJECTOR
    _INJECTOR = None
    for path in (MARLEY_LIBRARY_PARQUET, ISO1G_LIBRARY_PARQUET):
        if os.path.exists(path):
            os.remove(path)
            print(f"deleted cached de-excitation library {path}")
    build_marley_library()
    build_iso1g_library()


_INJECTOR = None


def get_injector():
    global _INJECTOR
    if _INJECTOR is None:
        _INJECTOR = DeexInjector()
    return _INJECTOR


def _event_seed(run, subrun, event):
    """Deterministic per-event seed, stable across processing runs."""
    x = (np.uint64(run) * np.uint64(1_000_003)
         + np.uint64(subrun)) * np.uint64(1_000_003) + np.uint64(event)
    return int(x % np.uint64(2**63 - 1))


REQUIRED_EVENT_COLUMNS = [
    "wc_truth_isCC", "wc_truth_nuScatType", "wc_truth_nuEnergy",
    "wc_true_outgoing_lepton_energy",
    "wc_true_num_prim_protons_35", "wc_true_num_prim_neutrons_35",
    "wc_truth_vtxX", "wc_truth_vtxY", "wc_truth_vtxZ",
]


def _empty_bookkeeping(df):
    n = len(df)
    return pd.DataFrame({
        "deex_marley_donor_index": np.full(n, -1, dtype=np.int64),
        "deex_marley_match_level": np.full(n, -1, dtype=np.int64),
        "deex_n_photons": np.zeros(n, dtype=np.int64),
        "deex_n_photons_injected": np.zeros(n, dtype=np.int64),
        "deex_n_blips_injected": np.zeros(n, dtype=np.int64),
        "deex_iso1g_donor_indices": [""] * n,
    }, index=df.index)


def build_deexadded_blip_lists(df):
    """For every event build the augmented (original + injected) blip lists.

    Returns (aug_lists, bookkeeping_df):
    - aug_lists: {column name in BLIP_AUG_COLUMNS: list of per-event lists},
      or None when nothing in the chunk is injectable (data/ext files, or
      files without the required truth columns)
    - bookkeeping_df: per-event injection bookkeeping (donor indices for
      uniformity checks, photon/blip counts, match level)
    """
    from blip_postprocessing import is_blip_in_tpc

    # chunk-level gate: data/ext (and other no-truth) files don't carry the
    # truth columns at all, so bail out before touching them (and before
    # loading the donor libraries)
    filetypes_arr = df["filetype"].to_numpy()
    injectable_ft = ~np.isin(filetypes_arr, NO_INJECTION_FILETYPES)
    if not injectable_ft.any():
        return None, _empty_bookkeeping(df)
    missing = [c for c in REQUIRED_EVENT_COLUMNS if c not in df.columns]
    if missing:
        print(f"    WARNING: deex injection skipped for this chunk "
              f"(filetypes {sorted(set(filetypes_arr[injectable_ft]))}): "
              f"missing truth columns {missing}")
        return None, _empty_bookkeeping(df)

    inj = get_injector()

    runs = df["run"].to_numpy()
    subruns = df["subrun"].to_numpy()
    events = df["event"].to_numpy()
    filetypes = df["filetype"].to_numpy()
    is_cc = df["wc_truth_isCC"].to_numpy()
    scattype = df["wc_truth_nuScatType"].to_numpy()
    nu_E = df["wc_truth_nuEnergy"].to_numpy()          # MeV
    lep_E = df["wc_true_outgoing_lepton_energy"].to_numpy()  # MeV
    np35 = df["wc_true_num_prim_protons_35"].to_numpy()
    nn35 = df["wc_true_num_prim_neutrons_35"].to_numpy()
    vtx_x = df["wc_truth_vtxX"].to_numpy()
    vtx_y = df["wc_truth_vtxY"].to_numpy()
    vtx_z = df["wc_truth_vtxZ"].to_numpy()

    orig = {c: df[c].to_numpy() for c in BLIP_AUG_COLUMNS}

    aug_lists = {c: [] for c in BLIP_AUG_COLUMNS}
    bk_marley_idx, bk_match_level = [], []
    bk_n_photons, bk_n_injected, bk_n_blips = [], [], []
    bk_iso_idxs = []

    from tqdm import tqdm
    for i in tqdm(range(len(df)), desc="Injecting MARLEY de-excitation blips",
                  mininterval=10):
        base = {}
        for c in BLIP_AUG_COLUMNS:
            v = orig[c][i]
            base[c] = list(v) if (v is not None and hasattr(v, "__len__")) else []

        eligible = (
            filetypes[i] not in NO_INJECTION_FILETYPES
            and not np.isnan(float(scattype[i]) if scattype[i] is not None else np.nan)
            and int(scattype[i]) in INJECTED_SCATTYPES
            and np.isfinite(nu_E[i]) and np.isfinite(lep_E[i])
            and np.isfinite(np35[i]) and np.isfinite(nn35[i])
            and np.isfinite(vtx_x[i]) and np.isfinite(vtx_y[i])
            and np.isfinite(vtx_z[i])
        )

        # match level: -1 = ineligible by design (COH, nu-e, missing truth),
        # -2 = eligible but no library match found (should never happen)
        marley_idx, match_level = -1, -1
        n_photons = n_injected = n_blips = 0
        iso_idxs = []

        if eligible:
            rng = np.random.default_rng(_event_seed(runs[i], subruns[i], events[i]))
            q0 = nu_E[i] - lep_E[i]
            photon_Es, marley_idx, match_level = inj.sample_marley_event(
                is_cc[i], scattype[i], np35[i], nn35[i], q0, rng)
            if marley_idx < 0:
                match_level = -2
            n_photons = len(photon_Es)
            for E_gamma in photon_Es:
                if E_gamma > MAX_PHOTON_E_MEV:
                    continue
                iso_idx = inj.sample_iso1g_donor(E_gamma, rng)
                if iso_idx < 0:
                    continue
                n_injected += 1
                iso_idxs.append(iso_idx)
                blips = inj.donor_blips(iso_idx)
                for j in range(len(blips["off_x"])):
                    x = vtx_x[i] + blips["off_x"][j]
                    y = vtx_y[i] + blips["off_y"][j]
                    z = vtx_z[i] + blips["off_z"][j]
                    if not is_blip_in_tpc(x, y, z):
                        continue
                    n_blips += 1
                    base["blip_x"].append(x)
                    base["blip_y"].append(y)
                    base["blip_z"].append(z)
                    for name in ("energy", "dx", "dw", "nplanes", "touchtrk",
                                 "pl2_bydeadwire", "proxtrkdist"):
                        base[f"blip_{name}"].append(blips[name][j])

        for c in BLIP_AUG_COLUMNS:
            aug_lists[c].append(np.asarray(base[c]))
        bk_marley_idx.append(marley_idx)
        bk_match_level.append(match_level)
        bk_n_photons.append(n_photons)
        bk_n_injected.append(n_injected)
        bk_n_blips.append(n_blips)
        bk_iso_idxs.append(",".join(str(k) for k in iso_idxs))

    n_matched = sum(1 for m in bk_marley_idx if m >= 0)
    n_inelig = sum(1 for lv in bk_match_level if lv == -1)
    n_failed = sum(1 for lv in bk_match_level if lv == -2)
    n_with_gamma = sum(1 for n in bk_n_photons if n > 0)
    print(f"    de-excitation injection ({DEFAULT_MODEL}): "
          f"{n_matched}/{len(df)} events matched "
          f"({n_inelig} skipped by design: COH / nu-e / no GENIE truth), "
          f"{n_with_gamma} with >= 1 photon "
          f"({100.0 * n_with_gamma / max(n_matched, 1):.1f}% of matched), "
          f"{sum(bk_n_injected)} photons and {sum(bk_n_blips)} blips injected")
    if n_failed:
        print(f"    WARNING: {n_failed} eligible events found no MARLEY library "
              f"match (match_level == -2) — this should not happen")

    bookkeeping = pd.DataFrame({
        "deex_marley_donor_index": np.asarray(bk_marley_idx, dtype=np.int64),
        "deex_marley_match_level": np.asarray(bk_match_level, dtype=np.int64),
        "deex_n_photons": np.asarray(bk_n_photons, dtype=np.int64),
        "deex_n_photons_injected": np.asarray(bk_n_injected, dtype=np.int64),
        "deex_n_blips_injected": np.asarray(bk_n_blips, dtype=np.int64),
        "deex_iso1g_donor_indices": bk_iso_idxs,
    }, index=df.index)
    return aug_lists, bookkeeping
