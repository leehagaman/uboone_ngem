
from .pandora_variables import *
from .glee_variables import *
from .lantern_variables import *
from .wc_variables import *

combined_postprocessing_training_vars = [
    "wc_pandora_dist",
    "wc_lantern_dist",
    "lantern_pandora_dist",
]

# variables to throw away after postprocessing in order to make reasonable pandas df sizes
vector_columns = [
    "wc_kine_energy_particle",
    "wc_kine_particle_type",
    "wc_truth_id",
    "wc_truth_pdg",
    "wc_truth_mother",
    "wc_truth_startMomentum",
    "wc_truth_startXYZT",
    "wc_truth_endXYZT",
    "wc_reco_id",
    "wc_reco_pdg",
    "wc_reco_mother",
    "wc_reco_startMomentum",
    "wc_reco_startXYZT",
    "wc_reco_endXYZT",

    "lantern_showerPhScore",

    "blip_x",
    "blip_y",
    "blip_z",
    "blip_dx",
    "blip_dw",
    "blip_energy",
    "blip_true_g4id",
    "blip_true_pdg",
    "blip_true_energy",

    "glee_sss_candidate_veto_score",
    "glee_sss3d_shower_score",

    "lantern_showerIsSecondary",
    "lantern_showerPID",
    "lantern_showerPhScore",
    "lantern_showerElScore",
    "lantern_showerMuScore",
    "lantern_showerPiScore",
    "lantern_showerPrScore",
    "lantern_showerCharge",
    "lantern_showerPurity",
    "lantern_showerComp",
    "lantern_showerPrimaryScore",
    "lantern_showerFromNeutralScore",
    "lantern_showerFromChargedScore",
    "lantern_showerCosTheta",
    "lantern_showerCosThetaY",
    "lantern_showerDistToVtx",
    "lantern_showerStartDirX",
    "lantern_showerStartDirY",
    "lantern_showerStartDirZ",

    "lantern_trackIsSecondary",
    "lantern_trackClassified",
    "lantern_trackCharge",
    "lantern_trackComp",
    "lantern_trackPurity",
    "lantern_trackPrimaryScore",
    "lantern_trackFromNeutralScore",
    "lantern_trackFromChargedScore",
    "lantern_trackCosTheta",
    "lantern_trackCosThetaY",
    "lantern_trackDistToVtx",
    "lantern_trackStartDirX",
    "lantern_trackStartDirY",
    "lantern_trackStartDirZ",
    "lantern_trackElScore",
    "lantern_trackPhScore",
    "lantern_trackMuScore",
    "lantern_trackPiScore",
    "lantern_trackPrScore",
    "lantern_trackPID",

    "lantern_showerRecoE",
    "lantern_trackRecoE",

    "wc_Trecchargeblob_spacepoints_x",
    "wc_Trecchargeblob_spacepoints_y",
    "wc_Trecchargeblob_spacepoints_z",

    "wc_WCPMTInfoPePred",
    "wc_WCPMTInfoPeMeas",
    "wc_WCPMTInfoPeMeasErr",

    "wc_reco_muonMomentum",
    "wc_reco_showerMomentum",
    "wc_truth_muonMomentum",

    "true_gamma_energies",
    "true_gamma_pairconversion_xs",
    "true_gamma_pairconversion_ys",
    "true_gamma_pairconversion_zs",
    "wc_true_gamma_pairconversion_spacepoint_min_distances",

]
vector_columns += [f"pandora_{var}" for var in pandora_vector_vars]
vector_columns += [f"glee_{var}" for var in glee_vector_vars]

combined_training_vars = wc_training_vars + glee_training_vars + pandora_training_vars + lantern_training_vars + combined_postprocessing_training_vars
# leave blip, nanosecond timing, CRT, spacepoint SSV, and PMT vars for analysis of the selection after the combined BDT for more interpretability
