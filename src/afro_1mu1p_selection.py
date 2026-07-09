import numpy as np

class STVTools:
    MUON_MASS = 0.106
    PROTON_MASS = 0.938272
    NEUTRON_MASS = 0.939565

    def __init__(self, muon_vector, proton_vector, muon_energy, proton_energy):
        muon_vector = np.asarray(muon_vector, dtype=float)
        proton_vector = np.asarray(proton_vector, dtype=float)

        Mm = self.MUON_MASS
        Mp = self.PROTON_MASS
        Mn = self.NEUTRON_MASS
        DeltaM2 = Mn**2 - Mp**2
        BE = 0.04

        muon_trans = np.array([muon_vector[0], muon_vector[1], 0.])
        muon_trans_mag = np.linalg.norm(muon_trans)
        muon_long = np.array([0., 0., muon_vector[2]])

        proton_trans = np.array([proton_vector[0], proton_vector[1], 0.])
        proton_trans_mag = np.linalg.norm(proton_trans)
        proton_long = np.array([0., 0., proton_vector[2]])

        proton_KE = proton_energy - Mp
        pt_vector = muon_trans + proton_trans
        self.fPt = np.linalg.norm(pt_vector)

        cos_dat = np.dot(-muon_trans, pt_vector) / (muon_trans_mag * self.fPt)
        self.fDeltaAlphaT = np.degrees(np.arccos(np.clip(cos_dat, -1., 1.)))
        if self.fDeltaAlphaT > 180.: self.fDeltaAlphaT -= 180.
        if self.fDeltaAlphaT < 0.: self.fDeltaAlphaT += 180.

        cos_dpt = np.dot(-muon_trans, proton_trans) / (muon_trans_mag * proton_trans_mag)
        self.fDeltaPhiT = np.degrees(np.arccos(np.clip(cos_dpt, -1., 1.)))
        if self.fDeltaPhiT > 180.: self.fDeltaPhiT -= 180.
        if self.fDeltaPhiT < 0.: self.fDeltaPhiT += 180.

        self.fECal = muon_energy + proton_KE + BE

        muon_mag = np.linalg.norm(muon_vector)
        cos_theta_mu = muon_vector[2] / muon_mag
        EQE_num = 2 * (Mn - BE) * muon_energy - (BE**2 - 2 * Mn * BE + Mm**2 + DeltaM2)
        EQE_den = 2 * (Mn - BE - muon_energy + muon_mag * cos_theta_mu)
        self.fEQE = EQE_num / EQE_den

        muon_4v = np.array([muon_vector[0], muon_vector[1], muon_vector[2], muon_energy])
        nu_4v = np.array([0., 0., self.fECal, self.fECal])
        q_4v = nu_4v - muon_4v
        self.fQ2 = -self._mag2(q_4v)

        unit_z = np.array([0., 0., 1.])
        self.fPtx = np.dot(np.cross(unit_z, muon_trans), pt_vector) / muon_trans_mag
        self.fPty = -np.dot(muon_trans, pt_vector) / muon_trans_mag

        proton_4v = np.array([proton_vector[0], proton_vector[1], proton_vector[2], proton_energy])
        miss_4v = muon_4v + proton_4v - nu_4v
        self.fEMiss = abs(miss_4v[3])
        self.fPMiss = np.linalg.norm(miss_4v[:3])
        self.fPMissMinus = (muon_energy - muon_vector[2]) + (proton_energy - proton_vector[2])

        kMiss_num = self.fPt**2 + Mp**2
        kMiss_den = self.fPMissMinus * (2 * Mp - self.fPMissMinus)
        kMiss2 = Mp**2 * kMiss_num / kMiss_den - Mp**2
        self.fkMiss = np.sqrt(kMiss2)
        self.fA = self.fPMissMinus / Mp

        MA = 22 * Mn + 18 * Mp - 0.34381
        self.fECalMB = muon_energy + proton_KE + 0.0309
        nu_4v_MB = np.array([0., 0., self.fECalMB, self.fECalMB])
        q_4v_MB = nu_4v_MB - muon_4v

        self.fPL = muon_vector[2] + proton_vector[2] - self.fECalMB
        pn_vector = np.array([pt_vector[0], pt_vector[1], self.fPL])

        q_vector = q_4v_MB[:3]
        qT_vector = np.array([q_vector[0], q_vector[0], 0.])
        q_vector_unit = self._safe_unit(q_vector)
        qT_vector_unit = self._safe_unit(qT_vector)

        self.fPn = np.sqrt(self.fPt**2 + self.fPL**2)

        q_mag = np.linalg.norm(q_vector)
        cos_a3dq = np.dot(q_vector, pn_vector) / (q_mag * self.fPn)
        self.fDeltaAlpha3Dq = np.degrees(np.arccos(np.clip(cos_a3dq, -1., 1.)))
        if self.fDeltaAlpha3Dq > 180.: self.fDeltaAlpha3Dq -= 180.
        if self.fDeltaAlpha3Dq < 0.: self.fDeltaAlpha3Dq += 180.

        cos_a3dmu = np.dot(-muon_vector, pn_vector) / (muon_mag * self.fPn)
        self.fDeltaAlpha3DMu = np.degrees(np.arccos(np.clip(cos_a3dmu, -1., 1.)))
        if self.fDeltaAlpha3DMu > 180.: self.fDeltaAlpha3DMu -= 180.
        if self.fDeltaAlpha3DMu < 0.: self.fDeltaAlpha3DMu += 180.

        proton_mag = np.linalg.norm(proton_vector)
        cos_p3d = np.dot(q_vector, proton_vector) / (q_mag * proton_mag)
        self.fDeltaPhi3D = np.degrees(np.arccos(np.clip(cos_p3d, -1., 1.)))
        if self.fDeltaPhi3D > 180.: self.fDeltaPhi3D -= 180.
        if self.fDeltaPhi3D < 0.: self.fDeltaPhi3D += 180.

        self.fPnPerp = self.fPn * np.sin(np.radians(self.fDeltaAlpha3Dq))
        self.fPnPar = self.fPn * np.cos(np.radians(self.fDeltaAlpha3Dq))
        self.fPnPerpx = np.dot(np.cross(qT_vector_unit, unit_z), pn_vector)
        self.fPnPerpy = np.dot(np.cross(q_vector_unit, np.cross(qT_vector_unit, unit_z)), pn_vector)

    @staticmethod
    def _mag2(four_vector):
        p = four_vector[:3]
        E = four_vector[3]
        return E**2 - np.dot(p, p)

    @staticmethod
    def _safe_unit(v):
        n = np.linalg.norm(v)
        return v / n if n != 0 else v

    def ReturnkMiss(self): return self.fkMiss
    def ReturnEMiss(self): return self.fEMiss
    def ReturnPMissMinus(self): return self.fPMissMinus
    def ReturnPMiss(self): return self.fPMiss
    def ReturnPt(self): return self.fPt
    def ReturnPtx(self): return self.fPtx
    def ReturnPty(self): return self.fPty
    def ReturnPnPerp(self): return self.fPnPerp
    def ReturnPnPerpx(self): return self.fPnPerpx
    def ReturnPnPerpy(self): return self.fPnPerpy
    def ReturnPnPar(self): return self.fPnPar
    def ReturnPL(self): return self.fPL
    def ReturnPn(self): return self.fPn
    def ReturnDeltaAlphaT(self): return self.fDeltaAlphaT
    def ReturnDeltaAlpha3Dq(self): return self.fDeltaAlpha3Dq
    def ReturnDeltaAlpha3DMu(self): return self.fDeltaAlpha3DMu
    def ReturnDeltaPhiT(self): return self.fDeltaPhiT
    def ReturnDeltaPhi3D(self): return self.fDeltaPhi3D
    def ReturnECal(self): return self.fECal
    def ReturnECalMB(self): return self.fECalMB
    def ReturnEQE(self): return self.fEQE
    def ReturnQ2(self): return self.fQ2
    def ReturnA(self): return self.fA

PROTON_MASS = 0.938272  # GeV/c^2
TRACK_SCORE_CUT = 0.5
PROTON_LLR_PID_SCORE = 0.05

FVX = 256.0
FVY = 232.0
FVZ = 1037.0
BORDERX = 10.0
BORDERY = 10.0
BORDERZ = 10.0

REQUIRED_RECO_SELECTION_COLUMNS = [
    "wc_kine_reco_Enu",
    "wc_match_isFC",
    "wc_numu_score",
    "wc_reco_muonMomentum",
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

REQUIRED_TRUE_1MU1P_COLUMNS = [
    "wc_truth_nuPdg",
    "wc_truth_isCC",
    "wc_truth_pdg",
    "wc_truth_startMomentum",
]

STV_RETURN_METHODS = [
    "ReturnkMiss",
    "ReturnEMiss",
    "ReturnPMissMinus",
    "ReturnPMiss",
    "ReturnPt",
    "ReturnPtx",
    "ReturnPty",
    "ReturnPnPerp",
    "ReturnPnPerpx",
    "ReturnPnPerpy",
    "ReturnPnPar",
    "ReturnPL",
    "ReturnPn",
    "ReturnDeltaAlphaT",
    "ReturnDeltaAlpha3Dq",
    "ReturnDeltaAlpha3DMu",
    "ReturnDeltaPhiT",
    "ReturnDeltaPhi3D",
    "ReturnECal",
    "ReturnECalMB",
    "ReturnEQE",
    "ReturnQ2",
    "ReturnA",
]

STV_BRANCH_NAMES = {
    method_name: method_name.replace("Return", "", 1)
    for method_name in STV_RETURN_METHODS
}
STV_COLUMN_NAMES = {
    branch_name: f"afro_1mu1p_{branch_name}"
    for branch_name in STV_BRANCH_NAMES.values()
}
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

def stv_values(stv_tool):
    return {
        STV_COLUMN_NAMES[branch_name]: float(getattr(stv_tool, method_name)())
        for method_name, branch_name in STV_BRANCH_NAMES.items()
    }

def default_stv_values():
    return {
        column_name: INVALID_STV_VALUE
        for column_name in STV_COLUMN_NAMES.values()
    }

def failed_reco_event():
    return False, default_stv_values()

def evaluate_true_1mu1p_event(data):
    try:
        if not (abs(data["wc_truth_nuPdg"]) == 14 and data["wc_truth_isCC"] == 1):
            return 0

        truth_pdgs = data["wc_truth_pdg"]
        truth_start_momenta = data["wc_truth_startMomentum"]
        if is_missing(truth_pdgs) or is_missing(truth_start_momenta):
            return 0

        n_muons = 0
        n_protons = 0
        n_charged_pions = 0
        n_neutral_pions = 0
        n_heavier_mesons = 0

        for particle_i in range(len(truth_pdgs)):
            pdg = truth_pdgs[particle_i]
            if is_missing(pdg):
                continue

            p_mag = momentum_mag(truth_start_momenta[particle_i])
            abs_pdg = abs(int(pdg))

            if abs_pdg == 13 and p_mag > 0.1:
                n_muons += 1
            elif abs_pdg == 2212 and p_mag > 0.25:
                n_protons += 1
            elif abs_pdg == 211 and p_mag > 0.07:
                n_charged_pions += 1
            elif int(pdg) == 111:
                n_neutral_pions += 1
            elif int(pdg) != 111 and abs_pdg != 211 and is_meson_or_antimeson(pdg):
                n_heavier_mesons += 1

        return int(
            n_muons == 1
            and n_protons == 1
            and n_charged_pions == 0
            and n_neutral_pions == 0
            and n_heavier_mesons == 0
        )
    except (IndexError, TypeError, ValueError):
        return 0

def evaluate_reco_event(data):
    try:
        if not (
            (data["wc_kine_reco_Enu"] > 0)
            and (data["wc_match_isFC"] == 1)
            and (data["wc_numu_score"] > 0.9)
            and (get_vector_value(data["wc_reco_muonMomentum"], 3) > 0)
        ):
            return failed_reco_event()

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
            return failed_reco_event()

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
            return failed_reco_event()

        vertex_vector = np.array(
            [
                data["pandora_reco_nu_vtx_sce_x"],
                data["pandora_reco_nu_vtx_sce_y"],
                data["pandora_reco_nu_vtx_sce_z"],
            ]
        )
        if not in_fv(*vertex_vector):
            return failed_reco_event()

        muon_start_vector = vector3(data, "pandora_trk_sce_start", candidate_muon_index)
        muon_end_vector = vector3(data, "pandora_trk_sce_end", candidate_muon_index)
        if not in_fv(*muon_start_vector) or not in_fv(*muon_end_vector):
            return failed_reco_event()

        proton_start_vector = vector3(data, "pandora_trk_sce_start", candidate_proton_index)
        proton_end_vector = vector3(data, "pandora_trk_sce_end", candidate_proton_index)
        if not in_fv(*proton_start_vector) or not in_fv(*proton_end_vector):
            return failed_reco_event()

        muon_momentum = get_vector_value(
            data["pandora_trk_range_muon_mom_v"], candidate_muon_index
        )
        proton_ke_gev = get_vector_value(
            data["pandora_trk_energy_proton_v"], candidate_proton_index
        )
        proton_e_gev = proton_ke_gev + PROTON_MASS
        proton_momentum = np.sqrt(proton_e_gev**2 - PROTON_MASS**2)

        if muon_momentum < 0.1 or proton_momentum < 0.25:
            return failed_reco_event()

        mcs_muon_momentum = get_vector_value(
            data["pandora_trk_mcs_muon_mom_v"], candidate_muon_index
        )
        reso = np.abs(muon_momentum - mcs_muon_momentum) / muon_momentum
        if reso > 0.25:
            return failed_reco_event()

        if (
            dist(vertex_vector, muon_start_vector) > dist(vertex_vector, muon_end_vector)
            or dist(vertex_vector, proton_start_vector) > dist(vertex_vector, proton_end_vector)
        ):
            return failed_reco_event()

        if dist(muon_start_vector, proton_start_vector) > dist(
            muon_end_vector, proton_end_vector
        ):
            return failed_reco_event()

        if candidate_proton_pid_score >= PROTON_LLR_PID_SCORE:
            return failed_reco_event()

        muon_theta = get_vector_value(data["pandora_trk_theta_v"], candidate_muon_index)
        muon_phi = get_vector_value(data["pandora_trk_phi_v"], candidate_muon_index)
        proton_theta = get_vector_value(data["pandora_trk_theta_v"], candidate_proton_index)
        proton_phi = get_vector_value(data["pandora_trk_phi_v"], candidate_proton_index)

        candidate_muon_vector = spherical_to_cartesian(muon_momentum, muon_theta, muon_phi)
        candidate_proton_vector = spherical_to_cartesian(proton_momentum, proton_theta, proton_phi)
        muon_e_gev = np.sqrt(muon_momentum**2 + STVTools.MUON_MASS**2)

        stv_tool = STVTools(
            candidate_muon_vector,
            candidate_proton_vector,
            muon_e_gev,
            proton_e_gev,
        )
        return True, stv_values(stv_tool)
    except (IndexError, TypeError, ValueError):
        return failed_reco_event()
