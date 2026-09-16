import numpy as np

# Reference implementation from Afroditi Papadopoulou:
#     preselection: https://github.com/afropapp13/myPreSelection/blob/MicroBooNE_Atmospherics/neutrino_selection.cxx
#     reco selection: https://github.com/afropapp13/myEvents/blob/MicroBooNE_Atmospherics/reco_selection.cxx
#     truth signal: https://github.com/afropapp13/myEvents/blob/MicroBooNE_Atmospherics/true_selection.cxx

MUON_MOMENTUM_THRESHOLD = 0.1
MUON_MOMENTUM_MAX = 1.2
PROTON_MOMENTUM_THRESHOLD = 0.3
PROTON_MOMENTUM_MAX = 1.0
CHARGED_PION_MOMENTUM_THRESHOLD = 0.07

BINDING_ENERGY_GEV = 0.04
PROTON_REMOVAL_ENERGY_GEV = 0.0309

MUON_MASS = 0.106        # GeV/c^2, as in STV_Tools.cxx
PROTON_MASS = 0.938272
NEUTRON_MASS = 0.939565

TRACK_SCORE_CUT = 0.5
PROTON_LLR_PID_SCORE = 0.05

# Fiducial box of Tools::inFV (myClasses/Constants.h: FVx, FVy, FVz and 10 cm borders)
FVX = 256.0
FVY = 230.0
FVZ = 1036.0
BORDERX = 10.0
BORDERY = 10.0
BORDERZ = 10.0

REQUIRED_RECO_SELECTION_COLUMNS = [
    "wc_kine_reco_Enu",
    "wc_match_isFC",
    "wc_numu_score",
    "pandora_reco_nu_vtx_sce_x",
    "pandora_reco_nu_vtx_sce_y",
    "pandora_reco_nu_vtx_sce_z",
    "pandora_n_pfps",
    "pandora_pfp_generation_v",
    "pandora_trk_score_v",
    "pandora_nslice",
    "pandora_trk_llr_pid_score_v",
    "pandora_pfpdg",
    "pandora_trk_sce_start_x_v",
    "pandora_trk_sce_start_y_v",
    "pandora_trk_sce_start_z_v",
    "pandora_trk_sce_end_x_v",
    "pandora_trk_sce_end_y_v",
    "pandora_trk_sce_end_z_v",
    "pandora_trk_range_muon_mom_v",
    "pandora_trk_energy_proton_v",
    "pandora_trk_mcs_muon_mom_v",
    "pandora_trk_theta_v",
    "pandora_trk_phi_v",
]

TRUE_VERTEX_COLUMNS = ["wc_truth_vtxX", "wc_truth_vtxY", "wc_truth_vtxZ"]

REQUIRED_TRUE_1MU1P_COLUMNS = [
    "wc_truth_nuPdg",
    "wc_truth_isCC",
    *TRUE_VERTEX_COLUMNS,
    "wc_truth_pdg",
    "wc_truth_mother",
    "wc_truth_startMomentum",
]

# the STV_Tools.cxx Return<X>() quantities, in its order
STV_BRANCH_NAMES = [
    "kMiss", "EMiss", "PMissMinus", "PMiss",
    "Pt", "Ptx", "Pty",
    "PnPerp", "PnPerpx", "PnPerpy", "PnPar", "PL", "Pn",
    "DeltaAlphaT", "DeltaAlpha3Dq", "DeltaAlpha3DMu", "DeltaPhiT", "DeltaPhi3D",
    "ECal", "ECalMB", "EQE", "Q2", "A",
]
STV_COLUMN_NAMES = {branch_name: f"afro_1mu1p_{branch_name}" for branch_name in STV_BRANCH_NAMES}
TRUE_STV_COLUMN_NAMES = {branch_name: f"afro_1mu1p_true_{branch_name}" for branch_name in STV_BRANCH_NAMES}
INVALID_STV_VALUE = -9999.0

def in_fv(x, y, z):
    return (
        x < (FVX - BORDERX)
        and x > BORDERX
        and y < (FVY / 2.0 - BORDERY)
        and y > (-FVY / 2.0 + BORDERY)
        and z < (FVZ - BORDERZ)
        and z > BORDERZ
    )

def is_missing(x):
    return x is None or (isinstance(x, (float, np.floating)) and np.isnan(x))

def is_meson_or_antimeson(pdg):
    pdg = abs(int(pdg))
    return (
        (pdg < 9900000)
        and ((pdg // 1000) % 10 == 0)
        and ((pdg // 100) % 10 != 0)
        and not (901 <= pdg <= 930)
        and (pdg != 110)
        and (pdg != 990)
        and (pdg != 998)
        and (pdg != 999)
        and (pdg != 100)
    )

def get_vector_value(x, index):
    if is_missing(x):
        raise IndexError
    return x[index]

def momentum_mag(momentum):
    return np.sqrt(momentum[0]**2 + momentum[1]**2 + momentum[2]**2)

def vector3(data, prefix, index):
    return np.array(
        [
            get_vector_value(data[f"{prefix}_x_v"], index),
            get_vector_value(data[f"{prefix}_y_v"], index),
            get_vector_value(data[f"{prefix}_z_v"], index),
        ]
    )

def dist(a, b):
    return np.linalg.norm(a - b)

def spherical_to_cartesian(mag, theta, phi):
    return np.array(
        [
            mag * np.sin(theta) * np.cos(phi),
            mag * np.sin(theta) * np.sin(phi),
            mag * np.cos(theta),
        ]
    )

FAILED_RECO_EVENT = (False, None, None)

def evaluate_true_1mu1p_kinematics(data):
    """
    Returns (is_signal, muon_vector, proton_vector) with the 3-momenta in GeV/c, or
    (0, None, None) when the event is not signal.
    """
    try:
        if not (data["wc_truth_nuPdg"] == 14 and data["wc_truth_isCC"] == 1):
            return 0, None, None
        vertex = [data[col] for col in TRUE_VERTEX_COLUMNS]
        if any(is_missing(v) for v in vertex) or not in_fv(*vertex):
            return 0, None, None

        truth_pdgs = data["wc_truth_pdg"]
        truth_mothers = data["wc_truth_mother"]
        truth_start_momenta = data["wc_truth_startMomentum"]
        if is_missing(truth_pdgs) or is_missing(truth_mothers) or is_missing(truth_start_momenta):
            return 0, None, None

        muon_indices = []
        proton_indices = []
        n_charged_pions = 0
        n_neutral_pions = 0
        n_heavier_mesons = 0

        for particle_i in range(len(truth_pdgs)):
            pdg = truth_pdgs[particle_i]
            if is_missing(pdg) or is_missing(truth_mothers[particle_i]):
                continue
            if int(truth_mothers[particle_i]) != 0:
                continue  # Geant4 secondary, not a GENIE final-state primary

            p_mag = momentum_mag(truth_start_momenta[particle_i])
            abs_pdg = abs(int(pdg))

            if int(pdg) == 13 and p_mag >= MUON_MOMENTUM_THRESHOLD:
                muon_indices.append(particle_i)
            elif int(pdg) == 2212 and p_mag >= PROTON_MOMENTUM_THRESHOLD:
                proton_indices.append(particle_i)
            elif abs_pdg == 211 and p_mag >= CHARGED_PION_MOMENTUM_THRESHOLD:
                n_charged_pions += 1
            elif int(pdg) == 111:
                n_neutral_pions += 1
            elif abs_pdg != 211 and is_meson_or_antimeson(pdg):
                n_heavier_mesons += 1

        is_signal = (
            len(muon_indices) == 1
            and len(proton_indices) == 1
            and n_charged_pions == 0
            and n_neutral_pions == 0
            and n_heavier_mesons == 0
        )
        if not is_signal:
            return 0, None, None

        muon_vector = np.asarray(truth_start_momenta[muon_indices[0]][:3], dtype=float)
        proton_vector = np.asarray(truth_start_momenta[proton_indices[0]][:3], dtype=float)
        if np.linalg.norm(muon_vector) > MUON_MOMENTUM_MAX or np.linalg.norm(proton_vector) > PROTON_MOMENTUM_MAX:
            return 0, None, None
        return 1, muon_vector, proton_vector
    except (IndexError, TypeError, ValueError):
        return 0, None, None

def evaluate_true_1mu1p_event(data):
    return evaluate_true_1mu1p_kinematics(data)[0]

def _unit_vectors(v):
    n = np.linalg.norm(v, axis=1)
    return np.where(n[:, None] > 0, v / np.where(n > 0, n, 1.0)[:, None], v)

def _acos_deg(x):
    return np.degrees(np.arccos(np.clip(x, -1.0, 1.0)))

def compute_stv_arrays(muon_vector, proton_vector, muon_energy, proton_energy):
    """Vectorised port of STV_Tools.cxx (myClasses, MicroBooNE_Atmospherics).
    muon_vector / proton_vector are (N, 3) in GeV/c, the energies (N,) in GeV.
    Returns {branch_name: (N,) float64} keyed by STV_BRANCH_NAMES; rows with
    non-finite or degenerate inputs come out as NaN.  Used for both the reco
    (Pandora candidates) and the true (GENIE muon + proton) STVs."""
    mu = np.asarray(muon_vector, dtype=float).reshape(-1, 3)
    pr = np.asarray(proton_vector, dtype=float).reshape(-1, 3)
    Emu = np.asarray(muon_energy, dtype=float).reshape(-1)
    Ep = np.asarray(proton_energy, dtype=float).reshape(-1)

    Mm, Mp, Mn = MUON_MASS, PROTON_MASS, NEUTRON_MASS
    DeltaM2 = Mn**2 - Mp**2
    BE = BINDING_ENERGY_GEV

    with np.errstate(divide="ignore", invalid="ignore"):
        mu_x, mu_y, mu_z = mu[:, 0], mu[:, 1], mu[:, 2]
        pr_x, pr_y, pr_z = pr[:, 0], pr[:, 1], pr[:, 2]
        mu_t_mag = np.hypot(mu_x, mu_y)
        pr_t_mag = np.hypot(pr_x, pr_y)
        mu_mag = np.linalg.norm(mu, axis=1)
        pr_mag = np.linalg.norm(pr, axis=1)

        pt_x = mu_x + pr_x
        pt_y = mu_y + pr_y
        Pt = np.hypot(pt_x, pt_y)

        DeltaAlphaT = _acos_deg((-mu_x * pt_x - mu_y * pt_y) / (mu_t_mag * Pt))
        DeltaPhiT = _acos_deg((-mu_x * pr_x - mu_y * pr_y) / (mu_t_mag * pr_t_mag))

        proton_KE = Ep - Mp
        ECal = Emu + proton_KE + BE

        cos_theta_mu = mu_z / mu_mag
        EQE_num = 2 * (Mn - BE) * Emu - (BE**2 - 2 * Mn * BE + Mm**2 + DeltaM2)
        EQE_den = 2 * (Mn - BE - Emu + mu_mag * cos_theta_mu)
        EQE = EQE_num / EQE_den

        # q = nu(0, 0, ECal, ECal) - muon
        Q2 = -((ECal - Emu)**2 - (mu_x**2 + mu_y**2 + (ECal - mu_z)**2))

        # unit_z x mu_t = (-mu_y, mu_x, 0)
        Ptx = (-mu_y * pt_x + mu_x * pt_y) / mu_t_mag
        Pty = -(mu_x * pt_x + mu_y * pt_y) / mu_t_mag

        miss_E = Emu + Ep - ECal
        miss_z = mu_z + pr_z - ECal
        EMiss = np.abs(miss_E)
        PMiss = np.sqrt(pt_x**2 + pt_y**2 + miss_z**2)
        PMissMinus = (Emu - mu_z) + (Ep - pr_z)

        kMiss2 = Mp**2 * (Pt**2 + Mp**2) / (PMissMinus * (2 * Mp - PMissMinus)) - Mp**2
        kMiss = np.sqrt(kMiss2)
        A = PMissMinus / Mp

        ECalMB = Emu + proton_KE + PROTON_REMOVAL_ENERGY_GEV
        q_x, q_y, q_z = -mu_x, -mu_y, ECalMB - mu_z
        q_mag = np.sqrt(q_x**2 + q_y**2 + q_z**2)

        PL = mu_z + pr_z - ECalMB
        Pn = np.sqrt(Pt**2 + PL**2)

        DeltaAlpha3Dq = _acos_deg((q_x * pt_x + q_y * pt_y + q_z * PL) / (q_mag * Pn))
        DeltaAlpha3DMu = _acos_deg(-(mu_x * pt_x + mu_y * pt_y + mu_z * PL) / (mu_mag * Pn))
        DeltaPhi3D = _acos_deg((q_x * pr_x + q_y * pr_y + q_z * pr_z) / (q_mag * pr_mag))

        PnPerp = Pn * np.sin(np.radians(DeltaAlpha3Dq))
        PnPar = Pn * np.cos(np.radians(DeltaAlpha3Dq))

        q_unit = _unit_vectors(np.stack([q_x, q_y, q_z], axis=1))
        qT_unit = _unit_vectors(np.stack([q_x, q_y, np.zeros_like(q_x)], axis=1))
        # w = qT_unit x unit_z = (qT_y, -qT_x, 0)
        w_x, w_y = qT_unit[:, 1], -qT_unit[:, 0]
        PnPerpx = w_x * pt_x + w_y * pt_y
        # q_unit x w = (q_z w_y... ) written out: (-q_uz*w_y, q_uz*w_x, q_ux*w_y - q_uy*w_x)
        c_x = -q_unit[:, 2] * w_y
        c_y = q_unit[:, 2] * w_x
        c_z = q_unit[:, 0] * w_y - q_unit[:, 1] * w_x
        PnPerpy = c_x * pt_x + c_y * pt_y + c_z * PL

    return {
        "kMiss": kMiss, "EMiss": EMiss, "PMissMinus": PMissMinus, "PMiss": PMiss,
        "Pt": Pt, "Ptx": Ptx, "Pty": Pty,
        "PnPerp": PnPerp, "PnPerpx": PnPerpx, "PnPerpy": PnPerpy, "PnPar": PnPar,
        "PL": PL, "Pn": Pn,
        "DeltaAlphaT": DeltaAlphaT, "DeltaAlpha3Dq": DeltaAlpha3Dq, "DeltaAlpha3DMu": DeltaAlpha3DMu,
        "DeltaPhiT": DeltaPhiT, "DeltaPhi3D": DeltaPhi3D,
        "ECal": ECal, "ECalMB": ECalMB, "EQE": EQE, "Q2": Q2, "A": A,
    }

def stv_columns_from_vectors(muon_vector, proton_vector, is_valid, column_names):
    """STV columns for N events from muon / proton 3-vectors (GeV/c; rows where is_valid
    is False may hold anything).  Energies are rebuilt from the momenta with the
    STV_Tools masses, as the reference does for both reco and truth.  column_names maps
    branch name -> output column (STV_COLUMN_NAMES or TRUE_STV_COLUMN_NAMES).  Returns
    {column: (N,) float64}, INVALID_STV_VALUE where is_valid is False or the result is
    not finite."""
    is_valid = np.asarray(is_valid, dtype=bool)
    out = {col: np.full(len(is_valid), INVALID_STV_VALUE) for col in column_names.values()}
    if not is_valid.any():
        return out
    mu = np.asarray(muon_vector, dtype=float).reshape(-1, 3)[is_valid]
    pr = np.asarray(proton_vector, dtype=float).reshape(-1, 3)[is_valid]
    muon_energy = np.sqrt(np.sum(mu**2, axis=1) + MUON_MASS**2)
    proton_energy = np.sqrt(np.sum(pr**2, axis=1) + PROTON_MASS**2)
    values = compute_stv_arrays(mu, pr, muon_energy, proton_energy)
    for branch_name, col in column_names.items():
        out[col][is_valid] = np.nan_to_num(values[branch_name], nan=INVALID_STV_VALUE,
                                           posinf=INVALID_STV_VALUE, neginf=INVALID_STV_VALUE)
    return out

def evaluate_reco_event(data):
    """Reference reco selection (neutrino_selection.cxx + reco_selection.cxx) plus the
    Wire-Cell cuts.  Returns (passes, muon_vector, proton_vector) with the Pandora
    candidate 3-momenta in GeV/c, or FAILED_RECO_EVENT; the STVs are computed for all
    events at once afterwards with stv_columns_from_vectors."""
    try:
        # Wire-Cell requirements on top of the Pandora-based reference selection: generic neutrino
        # selection, full containment and the numu BDT score, mainly for cosmic rejection.  
        if not (
            (data["wc_kine_reco_Enu"] > 0)
            and (data["wc_match_isFC"] == 1)
            and (data["wc_numu_score"] > 0.9)
        ):
            return FAILED_RECO_EVENT

        candidate_index = []
        reco_shower_count = 0
        reco_track_count = 0

        for pfp_idx in range(int(data["pandora_n_pfps"])):
            if get_vector_value(data["pandora_pfp_generation_v"], pfp_idx) != 2:
                continue

            if get_vector_value(data["pandora_trk_score_v"], pfp_idx) <= TRACK_SCORE_CUT:
                reco_shower_count += 1
            else:
                reco_track_count += 1
                candidate_index.append(pfp_idx)

        if reco_shower_count != 0 or reco_track_count != 2 or data["pandora_nslice"] != 1:
            return FAILED_RECO_EVENT

        first_pid_score = get_vector_value(data["pandora_trk_llr_pid_score_v"], candidate_index[0])
        second_pid_score = get_vector_value(data["pandora_trk_llr_pid_score_v"], candidate_index[1])

        if first_pid_score > second_pid_score:
            candidate_muon_index = candidate_index[0]
            candidate_proton_index = candidate_index[1]
        else:
            candidate_muon_index = candidate_index[1]
            candidate_proton_index = candidate_index[0]

        candidate_proton_pid_score = get_vector_value(
            data["pandora_trk_llr_pid_score_v"], candidate_proton_index
        )

        if (
            get_vector_value(data["pandora_pfpdg"], candidate_muon_index) != 13
            or get_vector_value(data["pandora_pfpdg"], candidate_proton_index) != 13
        ):
            return FAILED_RECO_EVENT

        vertex_vector = np.array(
            [
                data["pandora_reco_nu_vtx_sce_x"],
                data["pandora_reco_nu_vtx_sce_y"],
                data["pandora_reco_nu_vtx_sce_z"],
            ]
        )
        if not in_fv(*vertex_vector):
            return FAILED_RECO_EVENT

        muon_start_vector = vector3(data, "pandora_trk_sce_start", candidate_muon_index)
        muon_end_vector = vector3(data, "pandora_trk_sce_end", candidate_muon_index)
        if not in_fv(*muon_start_vector) or not in_fv(*muon_end_vector):
            return FAILED_RECO_EVENT

        proton_start_vector = vector3(data, "pandora_trk_sce_start", candidate_proton_index)
        proton_end_vector = vector3(data, "pandora_trk_sce_end", candidate_proton_index)
        if not in_fv(*proton_start_vector) or not in_fv(*proton_end_vector):
            return FAILED_RECO_EVENT

        muon_momentum = get_vector_value(
            data["pandora_trk_range_muon_mom_v"], candidate_muon_index
        )
        proton_ke_gev = get_vector_value(
            data["pandora_trk_energy_proton_v"], candidate_proton_index
        )
        proton_e_gev = proton_ke_gev + PROTON_MASS
        proton_momentum = np.sqrt(proton_e_gev**2 - PROTON_MASS**2)

        if muon_momentum < MUON_MOMENTUM_THRESHOLD or proton_momentum < PROTON_MOMENTUM_THRESHOLD:
            return FAILED_RECO_EVENT
        if muon_momentum > MUON_MOMENTUM_MAX or proton_momentum > PROTON_MOMENTUM_MAX:
            return FAILED_RECO_EVENT

        mcs_muon_momentum = get_vector_value(
            data["pandora_trk_mcs_muon_mom_v"], candidate_muon_index
        )
        reso = np.abs(muon_momentum - mcs_muon_momentum) / muon_momentum
        if reso > 0.25:
            return FAILED_RECO_EVENT

        if (
            dist(vertex_vector, muon_start_vector) > dist(vertex_vector, muon_end_vector)
            or dist(vertex_vector, proton_start_vector) > dist(vertex_vector, proton_end_vector)
        ):
            return FAILED_RECO_EVENT

        if dist(muon_start_vector, proton_start_vector) > dist(
            muon_end_vector, proton_end_vector
        ):
            return FAILED_RECO_EVENT

        if candidate_proton_pid_score >= PROTON_LLR_PID_SCORE:
            return FAILED_RECO_EVENT

        muon_theta = get_vector_value(data["pandora_trk_theta_v"], candidate_muon_index)
        muon_phi = get_vector_value(data["pandora_trk_phi_v"], candidate_muon_index)
        proton_theta = get_vector_value(data["pandora_trk_theta_v"], candidate_proton_index)
        proton_phi = get_vector_value(data["pandora_trk_phi_v"], candidate_proton_index)

        candidate_muon_vector = spherical_to_cartesian(muon_momentum, muon_theta, muon_phi)
        candidate_proton_vector = spherical_to_cartesian(proton_momentum, proton_theta, proton_phi)
        return True, candidate_muon_vector, candidate_proton_vector
    except (IndexError, TypeError, ValueError):
        return FAILED_RECO_EVENT
