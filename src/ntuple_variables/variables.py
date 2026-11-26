
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

    "blip_x",
    "blip_y",
    "blip_z",
    "blip_dx",
    "blip_dw",
    "blip_energy",
    "blip_true_g4id",
    "blip_true_pdg",
    "blip_true_energy",
    "blip_charge",
    "blip_nplanes",
    "blip_proxtrkdist",
    "blip_proxtrkid",
    "blip_touchtrk",
    "blip_touchtrkid",
    "blip_pl0_nwires",
    "blip_pl1_nwires",
    "blip_pl2_nwires",
    "blip_pl0_bydeadwire",
    "blip_pl1_bydeadwire",
    "blip_pl2_bydeadwire",

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
    "wc_Trecchargeblob_spacepoints_real_cluster_id",

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
vector_columns += [f"pandora_{var}" for var in pandora_pelee_vector_vars if var not in pandora_vector_vars]
vector_columns += [f"glee_{var}" for var in glee_vector_vars]

combined_training_vars_with_duplicates = wc_training_vars + glee_training_vars + pandora_training_vars + lantern_training_vars + combined_postprocessing_training_vars
# leave blip, nanosecond timing, CRT, spacepoint SSV, and PMT vars for analysis of the selection after the combined BDT for more interpretability

# output from checking_bdt_variables.ipynb (see plots/bdt_var_redundancy_report.csv)
redundant_bdt_vars = [
    'wc_cosmict_2_filled',  # identical of wc_cosmict_4_filled
    'wc_cosmict_2_flag_inside',  # conditional copy of wc_cosmict_4_flag_inside given wc_cosmict_4_filled
    'wc_cosmict_2_angle_beam',  # identical of wc_cosmict_2_theta
    'wc_cosmict_3_filled',  # identical of wc_cosmict_5_filled
    'wc_cosmict_3_flag_inside',  # identical of wc_cosmict_5_flag_inside
    'wc_cosmict_3_angle_beam',  # identical of wc_cosmict_5_angle_beam
    'wc_cosmict_3_theta',  # identical of wc_cosmict_5_angle_beam
    'wc_cosmict_7_angle_beam',  # identical of wc_cosmict_7_theta
    'wc_cosmic_flag',  # monotonic transform of wc_cosmict_flag_9
    'wc_cme_length',  # identical of wc_spt_shower_total_length
    'wc_cme_angle_beam',  # identical of wc_spt_angle_beam
    'wc_anc_shower_main_length',  # identical of wc_spt_shower_main_length
    'wc_anc_shower_total_length',  # identical of wc_spt_shower_total_length
    'wc_gap_energy',  # identical of wc_cme_energy
    'wc_gap_flag_single_shower',  # identical of wc_stw_1_flag_single_shower
    'wc_shw_sp_hol_1_energy',  # identical of wc_shw_sp_hol_2_energy
    'wc_mgo_energy',  # identical of wc_cme_energy
    'wc_mgt_flag_single_shower',  # identical of wc_stw_1_flag_single_shower
    'wc_mip_quality_energy',  # identical of wc_cme_energy
    'wc_shw_sp_br_filled',  # identical of wc_shw_sp_filled
    'wc_shw_sp_br1_flag',  # constant
    'wc_shw_sp_br1_1_flag',  # constant
    'wc_shw_sp_br1_1_energy',  # identical of wc_shw_sp_lol_3_energy
    'wc_shw_sp_br1_2_flag',  # constant
    'wc_shw_sp_br1_2_energy',  # identical of wc_shw_sp_lol_3_energy
    'wc_shw_sp_br1_3_flag',  # constant
    'wc_shw_sp_br1_3_energy',  # identical of wc_shw_sp_lol_3_energy
    'wc_shw_sp_br1_3_flag_sg_topology',  # identical of wc_shw_sp_br1_1_flag_sg_topology
    'wc_shw_sp_br1_3_flag_sg_trajectory',  # identical of wc_shw_sp_br1_1_flag_sg_trajectory
    'wc_shw_sp_br1_1_vtx_n_segs',  # identical of wc_shw_sp_lol_3_vtx_n_segs
    'wc_shw_sp_br1_1_n_segs',  # identical of wc_shw_sp_br1_3_n_shower_segs
    'wc_shw_sp_br1_1_sg_length',  # identical of wc_shw_sp_br1_3_sg_length
    'wc_shw_sp_br1_2_n_shower_segs',  # identical of wc_shw_sp_br1_3_n_shower_segs
    'wc_shw_sp_br3_1_energy',  # identical of wc_shw_sp_lol_3_energy
    'wc_shw_sp_br3_1_sg_flag_trajectory',  # identical of wc_shw_sp_br1_1_flag_sg_trajectory
    'wc_shw_sp_br3_1_sg_flag_topology',  # identical of wc_shw_sp_br1_1_flag_sg_topology
    'wc_shw_sp_br3_2_energy',  # identical of wc_shw_sp_lol_3_energy
    'wc_shw_sp_br3_2_total_main_length',  # identical of wc_shw_sp_lol_3_shower_main_length
    'wc_shw_sp_br3_2_total_length',  # identical of wc_shw_sp_br1_2_shower_length
    'wc_shw_sp_br3_4_energy',  # identical of wc_shw_sp_lol_3_energy
    'wc_shw_sp_br3_7_energy',  # identical of wc_shw_sp_lol_3_energy
    'wc_shw_sp_br3_7_sg_length',  # identical of wc_shw_sp_br1_3_sg_length
    'wc_shw_sp_br3_7_main_length',  # identical of wc_shw_sp_lol_3_shower_main_length
    'wc_shw_sp_br3_8_energy',  # identical of wc_shw_sp_lol_3_energy
    'wc_shw_sp_br3_8_shower_main_length',  # identical of wc_shw_sp_lol_3_shower_main_length
    'wc_shw_sp_br3_8_shower_length',  # identical of wc_shw_sp_br1_2_shower_length
    'wc_shw_sp_br3_1_n_shower_segments',  # identical of wc_shw_sp_br1_3_n_shower_segs
    'wc_shw_sp_br3_1_sg_length',  # identical of wc_shw_sp_br1_3_sg_length
    'wc_shw_sp_br3_1_total_main_length',  # identical of wc_shw_sp_lol_3_shower_main_length
    'wc_shw_sp_br3_1_total_length',  # identical of wc_shw_sp_br1_2_shower_length
    'wc_shw_sp_br3_4_total_length',  # identical of wc_shw_sp_lol_3_shower_main_length
    'wc_shw_sp_br3_8_n_main_segs',  # identical of wc_shw_sp_br1_3_n_shower_main_segs
    'wc_shw_sp_br4_2_energy',  # identical of wc_shw_sp_lol_3_energy
    'wc_shw_sp_br4_1_shower_main_length',  # identical of wc_shw_sp_lol_3_shower_main_length
    'wc_shw_sp_br4_1_shower_total_length',  # identical of wc_shw_sp_br1_2_shower_length
    'wc_shw_sp_br4_1_energy',  # identical of wc_shw_sp_lol_3_energy
    'wc_shw_sp_br4_1_n_vtx_segs',  # identical of wc_shw_sp_lol_3_vtx_n_segs
    'wc_shw_sp_br4_1_n_main_segs',  # identical of wc_shw_sp_br1_3_n_shower_main_segs
    'wc_stem_dir_flag_single_shower',  # identical of wc_stw_1_flag_single_shower
    'wc_stem_dir_energy',  # identical of wc_cme_energy
    'wc_stem_len_energy',  # identical of wc_cme_energy
    'wc_brm_shower_total_length',  # identical of wc_spt_shower_total_length
    'wc_shw_sp_lem_shower_total_length',  # identical of wc_shw_sp_br1_2_shower_length
    'wc_shw_sp_lem_shower_main_length',  # identical of wc_shw_sp_lol_3_shower_main_length
    'wc_shw_sp_lem_shower_num_segs',  # identical of wc_shw_sp_br1_3_n_shower_segs
    'wc_shw_sp_lem_shower_num_main_segs',  # identical of wc_shw_sp_br1_3_n_shower_main_segs
    'wc_stw_1_energy',  # identical of wc_cme_energy
    'wc_stw_1_dQ_dx',  # identical of wc_spt_max_dQ_dx
    'wc_spt_angle_beam_1',  # identical of wc_anc_angle
    'wc_spt_n_vtx_segs',  # conditional copy of wc_vis_2_n_vtx_segs given wc_stw_1_flag_single_shower
    'wc_shw_sp_energy',  # identical of wc_shw_sp_lol_3_energy
    'wc_shw_sp_stem_length',  # conditional copy of wc_shw_sp_br1_3_sg_length given wc_shw_sp_filled
    'wc_shw_sp_length_main',  # conditional copy of wc_shw_sp_lol_3_shower_main_length given wc_shw_sp_filled
    'wc_shw_sp_length_total',  # conditional copy of wc_shw_sp_br1_2_shower_length given wc_shw_sp_filled
    'wc_shw_sp_flag_stem_trajectory',  # identical of wc_shw_sp_br1_1_flag_sg_trajectory
    'wc_vis_1_tmp_length1',  # identical of wc_vis_1_tmp_length2
    'wc_vis_2_angle_beam',  # conditional copy of wc_spt_angle_beam given wc_stw_1_flag_single_shower
    'wc_vis_2_iso_angle1',  # conditional copy of wc_spt_angle_drift given wc_stw_1_flag_single_shower
    'wc_shw_sp_pio_filled',  # identical of wc_shw_sp_filled
    'glee_reco_asso_showers',  # identical of pandora_n_showers
    'glee_reco_slice_num',  # monotonic transform of pandora_nslice
    'glee_reco_vertex_size',  # identical of pandora_nslice
    'pandora_shr_tkfit_dedx_U',  # identical of pandora_shr_tkfit_dedx_U_alt
    'pandora_shr_tkfit_dedx_V',  # identical of pandora_shr_tkfit_dedx_V_alt
    'pandora_shr_tkfit_dedx_Y',  # identical of pandora_shr_tkfit_dedx_Y_alt
    'pandora_shr_tkfit_nhits_U',  # identical of pandora_shr_tkfit_nhits_U_alt
    'pandora_shr_tkfit_nhits_V',  # identical of pandora_shr_tkfit_nhits_V_alt
    'pandora_shr_tkfit_nhits_Y',  # identical of pandora_shr_tkfit_nhits_Y_alt
    'lantern_prim_track_proton_num',  # identical of lantern_prim_track_proton_num_10MeV
    'lantern_prim_track_proton_num_5MeV',  # identical of lantern_prim_track_proton_num_10MeV
]

combined_training_vars = [var for var in combined_training_vars_with_duplicates if var not in redundant_bdt_vars]

# pandora MCS has some known issues
not_using_for_training_vars = ["pandora_trk_energy_muon_mcs", 
                               "pandora_max_len_trk_trk_mcs_muon_mom_v",
                               "pandora_max2_len_trk_trk_mcs_muon_mom_v",
                               "pandora_max3_len_trk_trk_mcs_muon_mom_v",
]

# nugraph hit fractions see some data/pred disagreement, better to use pfng2semlabel for these
not_using_for_training_vars += ["pandora_max_len_trk_pfng2shrfrac", "pandora_max_len_trk_pfng2mipfrac", "pandora_max_len_trk_pfng2hipfrac", "pandora_max_len_trk_pfng2mclfrac", "pandora_max_len_trk_pfng2dfsfrac"]
not_using_for_training_vars += ["pandora_max2_len_trk_pfng2shrfrac", "pandora_max2_len_trk_pfng2mipfrac", "pandora_max2_len_trk_pfng2hipfrac", "pandora_max2_len_trk_pfng2mclfrac", "pandora_max2_len_trk_pfng2dfsfrac"]
not_using_for_training_vars += ["pandora_max3_len_trk_pfng2shrfrac", "pandora_max3_len_trk_pfng2mipfrac", "pandora_max3_len_trk_pfng2hipfrac", "pandora_max3_len_trk_pfng2mclfrac", "pandora_max3_len_trk_pfng2dfsfrac"]

# nugraph diffuse has some known data/prediction disagreement
not_using_for_training_vars += [
    "pandora_slcng2dfs",
    "pandora_clung2dfs",

    "pandora_max_len_trk_pfng2semlabel",
    "pandora_max2_len_trk_pfng2semlabel",
    "pandora_max3_len_trk_pfng2semlabel",

    "pandora_max_len_trk_pfng2dfsavrg",
    "pandora_max2_len_trk_pfng2dfsavrg",
    "pandora_max3_len_trk_pfng2dfsavrg",
]


combined_training_vars = [var for var in combined_training_vars if var not in not_using_for_training_vars]


# check that no list variable defined or imported in this module contains duplicate entries
from collections import Counter

_vars_with_duplicates = {}
for _name, _value in list(globals().items()):
    if _name.startswith("_"):
        continue
    if not isinstance(_value, list):
        continue
    _counts = Counter(_value)
    _dups = [_item for _item, _count in _counts.items() if _count > 1]
    if _dups:
        _vars_with_duplicates[_name] = _dups

if _vars_with_duplicates:
    for _name, _dups in _vars_with_duplicates.items():
        print(f"Duplicate entries in '{_name}': {_dups}")
    raise ValueError(f"Found duplicate entries in variable lists: {sorted(_vars_with_duplicates)}")
