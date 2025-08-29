
wc_T_bdt_vars = [              # variables involved with BDT training (if you want to train your own BDT, lots of these variables will be useful)
                               # many of these variables describe specific features of the WC spacepoints using this code: https://github.com/BNLIF/wire-cell-pid/blob/master/src/NeutrinoID_nue_tagger.h
                               # here, we just include higher level outputs:
    "nue_score",                    # BDT score for nue selection, used for the WC inclusive nueCC analysis
    "numu_score",                   # BDT score for numu selection, used for the WC inclusive numuCC selections
    "nc_delta_score",               # BDT score for NC Delta selection
    "nc_pio_score",                 # BDT score for NC pi0 selection
    "numu_cc_flag",                 # flag, -1 means not generic selected, 0 means generic selected, 1 means cut-based numuCC selected. We often use "numu_cc_flag >= 0" to apply generic neutrino selection.
    "shw_sp_n_20mev_showers",       # number of reco 20 MeV showers
]

wc_T_eval_vars = [             # variables involved with low level reconstruction and truth information
    "run",                          # run number
    "subrun",                       # subrun number
    "event",                        # event number
    "match_isFC",                   # reconstructed cluster is fully contained (FC), boolean
    "truth_nuEnergy",               # true neutrino energy (MeV)
    "truth_nuPdg",                  # true neutrino pdg code
    "truth_isCC",                   # true interaction type is charged current, boolean
    "match_completeness_energy",    # the true energy deposited in the clusters that are 3D-matched with the reconstructed neutrino clusters (MeV)
    "truth_energyInside",           # the true energy deposited in the TPC Fiducial Volume (MeV)
    "truth_vtxInside",              # boolean, true neutrino vertex is inside the TPC Fiducial Volume
    "weight_cv",                    # GENIE MicroBooNE tune event weight (which should be corrected by also using weight_spline)
    "weight_spline",                # additional weight to correct the GENIE tune for certain events
    "flash_measPe",                 # measured flash PE
    "flash_predPe",                 # predicted flash PE
]
wc_T_eval_data_vars = [        # same as above, but for data files we do not attempt to load any truth information
    "run",
    "subrun",
    "event",
    "match_isFC",
    "flash_measPe",
    "flash_predPe",
]

wc_T_kine_vars = [             # variables involved with kinematic reconstruction
    "kine_reco_Enu",                # reconstructed neutrino energy (MeV). "kine_reco_Enu > 0" is another way to apply generic neutrino selection.
    "kine_energy_particle",         # energy of each reco particle
    "kine_particle_type",           # pdg code of each reco particle
]

wc_T_pf_vars = [               # variables involved with individual particles
    "truth_NprimPio",
    "truth_NCDelta",
    "reco_nuvtxX",
    "reco_nuvtxY",
    "reco_nuvtxZ",
    "reco_muonMomentum",            # reconstructed muon momentum 4-vector (p_x, p_y, p_z, p_t), in (GeV/c, GeV/c, GeV/c, GeV)
    "reco_showerMomentum",          # reconstructed primary shower momentum 4-vector (p_x, p_y, p_z, p_t), in (GeV/c, GeV/c, GeV/c, GeV)
    "reco_showervtxX",
    "reco_showervtxY",
    "reco_showervtxZ",
    "truth_vtxX",                   # true neutrino vertex x (cm)
    "truth_vtxY",                   # true neutrino vertex y (cm)
    "truth_vtxZ",                   # true neutrino vertex z (cm)
    "truth_corr_nuvtxX",            # true neutrino vertex x (cm), corrected for SCE
    "truth_corr_nuvtxY",            # true neutrino vertex y (cm), corrected for SCE
    "truth_corr_nuvtxZ",            # true neutrino vertex z (cm), corrected for SCE

    # These variables are related to individual true particles
    "truth_Ntrack",
    "truth_id",
    "truth_pdg",
    "truth_mother",
    "truth_startMomentum",
    "truth_startXYZT",
    "truth_endXYZT",

    # These variables are related to individual reco particles
    "reco_Ntrack",
    "reco_id",
    "reco_pdg",
    "reco_mother",
    "reco_startMomentum",
    "reco_startXYZT",
    "reco_endXYZT",

    # Nanosecond timing
    #"evtTimeNS",      # for data
    "evtTimeNS_cor",  # for MC
]

wc_T_pf_data_vars = [          # same as above, but for data files we do not attempt to load any truth information
    "reco_nuvtxX",
    "reco_nuvtxY",
    "reco_nuvtxZ",
    "reco_muonMomentum",
    "reco_showerMomentum",
    "reco_showervtxX",
    "reco_showervtxY",
    "reco_showervtxZ",
    "reco_Ntrack",
    "reco_id",
    "reco_pdg",
    "reco_mother",
    "reco_startMomentum",
    "reco_startXYZT",
    "reco_endXYZT",
    "evtTimeNS",      # for data
    #"evtTimeNS_cor",  # for MC
]

# some of these variables only exist in
wc_T_spacepoints_vars = [
    "Trec_spacepoints_x",
    "Trec_spacepoints_y",
    "Trec_spacepoints_z",
    "Trec_spacepoints_q",
    "Treccharge_spacepoints_x",
    "Treccharge_spacepoints_y",
    "Treccharge_spacepoints_z",
    "Treccharge_spacepoints_q",
    "Trecchargeblob_spacepoints_x",
    "Trecchargeblob_spacepoints_y",
    "Trecchargeblob_spacepoints_z",
    "Trecchargeblob_spacepoints_q",
]

# These are only available if the run_wcanatree.fcl ntuple making was ran with these options:
#    SaveTclusterSpacePoints: true
#    SaveTrueEDepSpacePoints: true
wc_extra_T_spacepoints_vars = [
    "Tcluster_spacepoints_x",
    "Tcluster_spacepoints_y",
    "Tcluster_spacepoints_z",
    "Tcluster_spacepoints_q",
    "TrueEDep_spacepoints_startx",
    "TrueEDep_spacepoints_starty",
    "TrueEDep_spacepoints_startz",
    "TrueEDep_spacepoints_endx",
    "TrueEDep_spacepoints_endy",
    "TrueEDep_spacepoints_endz",
    "TrueEDep_spacepoints_edep",
]


# These are the scalar variables saved for the numu tagger
numu3_var = [
    "numu_cc_flag_3",
    "numu_cc_3_particle_type",
    "numu_cc_3_max_length",
    "numu_cc_3_track_length",#numu_cc_3_acc_track_length'
    "numu_cc_3_max_length_all",
    "numu_cc_3_max_muon_length",
    "numu_cc_3_n_daughter_tracks",
    "numu_cc_3_n_daughter_all"
]
cosmict24_var = [
    "cosmict_flag_2",
    "cosmict_2_filled",
    "cosmict_2_particle_type",
    "cosmict_2_n_muon_tracks",
    "cosmict_2_total_shower_length",
    "cosmict_2_flag_inside",
    "cosmict_2_angle_beam",
    "cosmict_2_flag_dir_weak",
    "cosmict_2_dQ_dx_end",
    "cosmict_2_dQ_dx_front",
    "cosmict_2_theta",
    "cosmict_2_phi",
    "cosmict_2_valid_tracks",
    "cosmict_flag_4", 
    "cosmict_4_filled",
    "cosmict_4_flag_inside",
    "cosmict_4_angle_beam",
    "cosmict_4_connected_showers"
]
cosmict35_var = [
    "cosmict_flag_3",
    "cosmict_3_filled",
    "cosmict_3_flag_inside",
    "cosmict_3_angle_beam",
    "cosmict_3_flag_dir_weak",
    "cosmict_3_dQ_dx_end",
    "cosmict_3_dQ_dx_front",
    "cosmict_3_theta",
    "cosmict_3_phi",
    "cosmict_3_valid_tracks",
    "cosmict_flag_5",  
    "cosmict_5_filled",
    "cosmict_5_flag_inside",
    "cosmict_5_angle_beam",
    "cosmict_5_connected_showers"
]
cosmict6_var = ["cosmict_flag_6", 
    "cosmict_6_filled",
    "cosmict_6_flag_dir_weak",
    "cosmict_6_flag_inside",
    "cosmict_6_angle"
]
cosmict7_var = [
    "cosmict_flag_7",
    "cosmict_7_filled",
    "cosmict_7_flag_sec",
    "cosmict_7_n_muon_tracks",
    "cosmict_7_total_shower_length",
    "cosmict_7_flag_inside",
    "cosmict_7_angle_beam",
    "cosmict_7_flag_dir_weak",
    "cosmict_7_dQ_dx_end",
    "cosmict_7_dQ_dx_front",
    "cosmict_7_theta",
    "cosmict_7_phi"
]
cosmict8_var = [
    "cosmict_flag_8", 
    "cosmict_8_filled",
    "cosmict_8_flag_out",
    "cosmict_8_muon_length",
    "cosmict_8_acc_length"
]
cosmict9_var = [
    "cosmict_flag_9",
    "cosmic_flag",
    "cosmic_filled"
]
overall_var = [
    "cosmict_flag",
    "numu_cc_flag"
]
all_numu_scalars = []
all_numu_scalars += numu3_var
all_numu_scalars += cosmict24_var
all_numu_scalars += cosmict35_var
all_numu_scalars += cosmict6_var
all_numu_scalars += cosmict7_var
all_numu_scalars += cosmict8_var
all_numu_scalars += cosmict9_var
all_numu_scalars += overall_var
all_numu_scalars += ["cosmict_flag_1", "kine_reco_Enu", "match_isFC"]


# These are the vector variables saved for the numu tagger
var_numu1 = [#'weight',
             #'numu_cc_flag',
             #'cosmict_flag_1',
             #'numu_cc_flag_1',
             'numu_cc_1_particle_type',
             'numu_cc_1_length',
             'numu_cc_1_medium_dQ_dx',
             'numu_cc_1_dQ_dx_cut',
             'numu_cc_1_direct_length',
             'numu_cc_1_n_daughter_tracks',
             'numu_cc_1_n_daughter_all']
var_numu2 = [#'weight',
             #'numu_cc_flag',
             #'cosmict_flag_1',
             #'numu_cc_flag_2',
             'numu_cc_2_length',
             'numu_cc_2_total_length',
             'numu_cc_2_n_daughter_tracks',
             'numu_cc_2_n_daughter_all']
var_cos10 = [#'weight',
             #'numu_cc_flag',
             #'cosmict_flag_1',
             #'cosmict_flag_10',
             #'cosmict_10_flag_inside',
             'cosmict_10_vtx_z',
             'cosmict_10_flag_shower',
             'cosmict_10_flag_dir_weak',
             'cosmict_10_angle_beam',
             'cosmict_10_length']
all_numu_vectors = []
all_numu_vectors += var_numu1
all_numu_vectors += var_numu2
all_numu_vectors += var_cos10

# These are the scalar variables saved for the nue tagger
taggerCMEAMC_var = ["cme_mu_energy","cme_energy","cme_mu_length","cme_length",
                "cme_angle_beam","anc_angle","anc_max_angle","anc_max_length",
                "anc_acc_forward_length","anc_acc_backward_length","anc_acc_forward_length1",
                "anc_shower_main_length","anc_shower_total_length","anc_flag_main_outside"]
taggerGAP_var = ["gap_flag_prolong_u","gap_flag_prolong_v","gap_flag_prolong_w","gap_flag_parallel",
                 "gap_n_points","gap_n_bad","gap_energy","gap_num_valid_tracks","gap_flag_single_shower"]
taggerHOL_var = ["hol_1_n_valid_tracks","hol_1_min_angle","hol_1_energy","hol_1_flag_all_shower","hol_1_min_length",
               "hol_2_min_angle","hol_2_medium_dQ_dx","hol_2_ncount","lol_3_angle_beam","lol_3_n_valid_tracks",
               "lol_3_min_angle","lol_3_vtx_n_segs","lol_3_shower_main_length","lol_3_n_out","lol_3_n_sum"]
taggerMGOMGT_var = ["mgo_energy","mgo_max_energy","mgo_total_energy","mgo_n_showers","mgo_max_energy_1",
                    "mgo_max_energy_2","mgo_total_other_energy","mgo_n_total_showers","mgo_total_other_energy_1",
                   "mgt_flag_single_shower","mgt_max_energy","mgt_total_other_energy","mgt_max_energy_1",
                   "mgt_e_indirect_max_energy","mgt_e_direct_max_energy","mgt_n_direct_showers",
                    "mgt_e_direct_total_energy","mgt_flag_indirect_max_pio","mgt_e_indirect_total_energy"]
taggerMIPQUALITY_var = ["mip_quality_energy","mip_quality_overlap","mip_quality_n_showers","mip_quality_n_tracks",
                        "mip_quality_flag_inside_pi0","mip_quality_n_pi0_showers","mip_quality_shortest_length",
                        "mip_quality_acc_length","mip_quality_shortest_angle","mip_quality_flag_proton"]
taggerBR1_var = ["br1_1_shower_type","br1_1_vtx_n_segs","br1_1_energy","br1_1_n_segs","br1_1_flag_sg_topology",
                 "br1_1_flag_sg_trajectory","br1_1_sg_length","br1_2_n_connected","br1_2_max_length",
                "br1_2_n_connected_1","br1_2_n_shower_segs","br1_2_max_length_ratio","br1_2_shower_length",
                 "br1_3_n_connected_p","br1_3_max_length_p","br1_3_n_shower_main_segs"]
taggerBR3_var = ["br3_1_energy","br3_1_n_shower_segments","br3_1_sg_flag_trajectory","br3_1_sg_direct_length",
                "br3_1_sg_length","br3_1_total_main_length","br3_1_total_length","br3_1_iso_angle",
                 "br3_1_sg_flag_topology","br3_2_n_ele","br3_2_n_other","br3_2_other_fid","br3_4_acc_length",
                 "br3_4_total_length","br3_7_min_angle","br3_8_max_dQ_dx","br3_8_n_main_segs"]
taggerBR4TRE_var = ["br4_1_shower_main_length","br4_1_shower_total_length","br4_1_min_dis","br4_1_energy",
                    "br4_1_flag_avoid_muon_check","br4_1_n_vtx_segs","br4_1_n_main_segs","br4_2_ratio_45",
                   "br4_2_ratio_35","br4_2_ratio_25","br4_2_ratio_15","br4_2_ratio1_45","br4_2_ratio1_35",
                   "br4_2_ratio1_25","br4_2_ratio1_15","br4_2_iso_angle","br4_2_iso_angle1","br4_2_angle",
                   "tro_3_stem_length","tro_3_n_muon_segs"]
taggerVIS1_var = ["vis_1_n_vtx_segs","vis_1_energy","vis_1_num_good_tracks","vis_1_max_angle",
                  "vis_1_max_shower_angle","vis_1_tmp_length1","vis_1_tmp_length2"]
taggerVIS2_var = ["vis_2_n_vtx_segs","vis_2_min_angle","vis_2_min_weak_track","vis_2_angle_beam","vis_2_min_angle1",
                 "vis_2_iso_angle1","vis_2_min_medium_dQ_dx","vis_2_min_length","vis_2_sg_length","vis_2_max_angle",
                 "vis_2_max_weak_track"]
taggerPI01_var = ["pio_1_mass","pio_1_pio_type","pio_1_energy_1","pio_1_energy_2","pio_1_dis_1","pio_1_dis_2","pio_mip_id"]
taggerSTEMDIRBR2_var = ["stem_dir_flag_single_shower","stem_dir_angle","stem_dir_energy","stem_dir_angle1",
                        "stem_dir_angle2","stem_dir_angle3","stem_dir_ratio","br2_num_valid_tracks",
                        "br2_n_shower_main_segs","br2_max_angle","br2_sg_length","br2_flag_sg_trajectory"]
taggerSTLLEMBRM_var = ["stem_len_energy","stem_len_length","stem_len_flag_avoid_muon_check","stem_len_num_daughters",
                      "stem_len_daughter_length","brm_n_mu_segs","brm_Ep","brm_acc_length","brm_shower_total_length",
                      "brm_connected_length","brm_n_size","brm_acc_direct_length","brm_n_shower_main_segs",
                       "brm_n_mu_main","lem_shower_main_length","lem_n_3seg","lem_e_charge","lem_e_dQdx",
                       "lem_shower_num_main_segs"]
taggerSTWSPT_var = ["stw_1_energy","stw_1_dis","stw_1_dQ_dx","stw_1_flag_single_shower","stw_1_n_pi0",
                    "stw_1_num_valid_tracks","spt_shower_main_length","spt_shower_total_length","spt_angle_beam",
                    "spt_angle_vertical","spt_max_dQ_dx","spt_angle_beam_1","spt_angle_drift","spt_angle_drift_1",
                    "spt_num_valid_tracks","spt_n_vtx_segs","spt_max_length"]
taggerMIP_var = ["mip_energy","mip_n_end_reduction","mip_n_first_mip","mip_n_first_non_mip","mip_n_first_non_mip_1",
                "mip_n_first_non_mip_2","mip_vec_dQ_dx_0","mip_vec_dQ_dx_1","mip_max_dQ_dx_sample",
                "mip_n_below_threshold","mip_n_below_zero","mip_n_lowest","mip_n_highest","mip_lowest_dQ_dx",
                 "mip_highest_dQ_dx","mip_medium_dQ_dx","mip_stem_length","mip_length_main","mip_length_total",
                 "mip_angle_beam","mip_iso_angle","mip_n_vertex","mip_n_good_tracks","mip_E_indirect_max_energy",
                 "mip_flag_all_above","mip_min_dQ_dx_5","mip_n_other_vertex","mip_n_stem_size",
                 "mip_flag_stem_trajectory","mip_min_dis"]
taggerAdditional_Var = ["mip_vec_dQ_dx_2","mip_vec_dQ_dx_3","mip_vec_dQ_dx_4","mip_vec_dQ_dx_5","mip_vec_dQ_dx_6",
                        "mip_vec_dQ_dx_7","mip_vec_dQ_dx_8","mip_vec_dQ_dx_9","mip_vec_dQ_dx_10","mip_vec_dQ_dx_11",
                        "mip_vec_dQ_dx_12","mip_vec_dQ_dx_13","mip_vec_dQ_dx_14","mip_vec_dQ_dx_15",
                        "mip_vec_dQ_dx_16","mip_vec_dQ_dx_17","mip_vec_dQ_dx_18","mip_vec_dQ_dx_19"]
all_nue_scalars = []
all_nue_scalars += taggerCMEAMC_var
all_nue_scalars += taggerGAP_var
all_nue_scalars += taggerHOL_var
all_nue_scalars += taggerMGOMGT_var
all_nue_scalars += taggerMIPQUALITY_var
all_nue_scalars += taggerBR1_var
all_nue_scalars += taggerBR3_var
all_nue_scalars += taggerBR4TRE_var
all_nue_scalars += taggerSTEMDIRBR2_var
all_nue_scalars += taggerSTLLEMBRM_var
all_nue_scalars += taggerSTWSPT_var
all_nue_scalars += taggerMIP_var
all_nue_scalars += taggerVIS1_var
all_nue_scalars += taggerVIS2_var
all_nue_scalars += taggerPI01_var
all_nue_scalars += taggerAdditional_Var
all_nue_scalars += ["kine_reco_Enu", "match_isFC"]

# These are the vector variables saved for the nue tagger
taggerTRO5_var = ["tro_5_v_max_angle","tro_5_v_min_angle","tro_5_v_max_length","tro_5_v_iso_angle","tro_5_v_n_vtx_segs",
                "tro_5_v_min_count","tro_5_v_max_count","tro_5_v_energy"]
taggerTRO4_var = ["tro_4_v_dir2_mag","tro_4_v_angle","tro_4_v_angle1","tro_4_v_angle2","tro_4_v_length","tro_4_v_length1",
                "tro_4_v_medium_dQ_dx","tro_4_v_end_dQ_dx","tro_4_v_energy","tro_4_v_shower_main_length","tro_4_v_flag_shower_trajectory"]
taggerTRO2_var = ["tro_2_v_energy","tro_2_v_stem_length","tro_2_v_iso_angle","tro_2_v_max_length","tro_2_v_angle"]
taggerTRO1_var = ["tro_1_v_particle_type","tro_1_v_flag_dir_weak","tro_1_v_min_dis","tro_1_v_sg1_length","tro_1_v_shower_main_length",
                "tro_1_v_max_n_vtx_segs","tro_1_v_tmp_length","tro_1_v_medium_dQ_dx","tro_1_v_dQ_dx_cut","tro_1_v_flag_shower_topology"]
taggerSTW4_var = ["stw_4_v_angle","stw_4_v_dis","stw_4_v_energy"]
taggerSTW3_var = ["stw_3_v_angle","stw_3_v_dir_length","stw_3_v_energy","stw_3_v_medium_dQ_dx"]
taggerSTW2_var = ["stw_2_v_medium_dQ_dx","stw_2_v_energy","stw_2_v_angle","stw_2_v_dir_length","stw_2_v_max_dQ_dx"]
taggerSIG2_var = ["sig_2_v_energy","sig_2_v_shower_angle","sig_2_v_flag_single_shower","sig_2_v_medium_dQ_dx",
                "sig_2_v_start_dQ_dx"]
taggerSIG1_var = ["sig_1_v_angle","sig_1_v_flag_single_shower","sig_1_v_energy","sig_1_v_energy_1"]
taggerPI02_var = ["pio_2_v_dis2","pio_2_v_angle2","pio_2_v_acc_length"]#"pio_mip_id"
taggerLOL2_var = ["lol_2_v_length","lol_2_v_angle","lol_2_v_type","lol_2_v_vtx_n_segs","lol_2_v_energy",
                 "lol_2_v_shower_main_length","lol_2_v_flag_dir_weak"]
taggerLOL1_var = ["lol_1_v_energy","lol_1_v_vtx_n_segs","lol_1_v_nseg","lol_1_v_angle"]
taggerBR3TAGGER6_var = ["br3_6_v_angle","br3_6_v_angle1","br3_6_v_flag_shower_trajectory","br3_6_v_direct_length",
               "br3_6_v_length","br3_6_v_n_other_vtx_segs","br3_6_v_energy"]

taggerBR3TAGGER5_var = ["br3_5_v_dir_length","br3_5_v_total_length","br3_5_v_flag_avoid_muon_check","br3_5_v_n_seg",
               "br3_5_v_angle","br3_5_v_sg_length","br3_5_v_energy","br3_5_v_n_segs","br3_5_v_shower_main_length",
                "br3_5_v_shower_total_length"]
taggerBR3TAGGER3_var = ["br3_3_v_energy","br3_3_v_angle","br3_3_v_dir_length","br3_3_v_length"]
all_nue_vectors = []
all_nue_vectors += taggerTRO5_var
all_nue_vectors += taggerTRO4_var
all_nue_vectors += taggerTRO2_var
all_nue_vectors += taggerTRO1_var
all_nue_vectors += taggerSTW4_var
all_nue_vectors += taggerSTW3_var
all_nue_vectors += taggerSTW2_var
all_nue_vectors += taggerSIG2_var
all_nue_vectors += taggerSIG1_var
all_nue_vectors += taggerPI02_var
all_nue_vectors += taggerLOL2_var
all_nue_vectors += taggerLOL1_var
all_nue_vectors += taggerBR3TAGGER6_var
all_nue_vectors += taggerBR3TAGGER5_var
all_nue_vectors += taggerBR3TAGGER3_var

numu_bdt_score_variables = [
    "cosmict_10_score",
    "numu_1_score",
    "numu_2_score"
]

nue_bdt_score_variables = [
    "tro_5_score",
    "tro_4_score",
    "tro_2_score",
    "tro_1_score",
    "stw_4_score",
    "stw_3_score",
    "stw_2_score",
    "sig_2_score",
    "sig_1_score",
    "pio_2_score",
    "lol_2_score",
    "lol_1_score",
    "br3_6_score",
    "br3_5_score",
    "br3_3_score"
]

kine_scalar_vars = [
    "kine_reco_add_energy",
    "kine_pio_mass",
    "kine_pio_flag",
    "kine_pio_vtx_dis",
    "kine_pio_energy_1",
    "kine_pio_theta_1",
    "kine_pio_phi_1",
    "kine_pio_dis_1",
    "kine_pio_energy_2",
    "kine_pio_theta_2",
    "kine_pio_phi_2",
    "kine_pio_dis_2",
    "kine_pio_angle"
]

# The WC nueCC BDT was trained with all_nue_scalars + nue_bdt_score_variables
# The WC numuCC BDT was trained with all_numu_scalars + numu_bdt_score_variables
# The WC NC Delta BDT was trained with: all_numu_scalars[:-2] + all_nue_scalars[:-2] + numu_bdt_score_variables + nue_bdt_score_variables + kine_scalar_vars
#     (did not include match_isFC or kine_reco_Enu)

# All variables used for all WC BDTs, including match_isFC and kine_reco_Enu, doesn't include nue_score, numu_score, nc_delta_score, or nc_pio_score
wc_T_BDT_training_vars = all_numu_scalars + all_nue_scalars[:-2] + numu_bdt_score_variables + nue_bdt_score_variables
wc_T_KINEvars_training_vars = kine_scalar_vars
wc_training_vars = wc_T_BDT_training_vars + wc_T_KINEvars_training_vars

# adding in particle multiplicities for this multi-class BDT, rather than cutting on them later
# add wc prefix to wc_training_vars
wc_training_vars = [f"wc_{var}" for var in wc_training_vars]
wc_training_vars += ["wc_reco_num_protons", "wc_reco_num_other_tracks"]

wc_training_only_vars = [var for var in wc_training_vars if var not in ["wc_kine_reco_Enu", "wc_match_isFC", "wc_reco_num_protons", "wc_reco_num_other_tracks"]]

wc_T_BDT_including_training_vars = list(set(wc_T_bdt_vars + wc_T_BDT_training_vars))
wc_T_KINEvars_including_training_vars = list(set(wc_T_kine_vars + wc_T_KINEvars_training_vars))


blip_vars = [
    "nblips_saved",
    "blip_x",
    "blip_y",
    "blip_z",
    "blip_dx",
    "blip_dw",
    "blip_energy",
    "blip_true_g4id",
    "blip_true_pdg",
    "blip_true_energy",
]

# more technical blip variables
extra_blip_vars = [
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
]

# TODO: add more relevant PeLEE variables here
pelee_vars = [
    "shr_energy",
    "reco_nu_vtx_x",
    "reco_nu_vtx_y",
    "reco_nu_vtx_z",
    "reco_nu_vtx_sce_x",
    "reco_nu_vtx_sce_y",
    "reco_nu_vtx_sce_z",
]

# TODO: add more relevant gLEE variables here
glee_vars = [
    "sss_candidate_veto_score",
    "sss3d_shower_score",
]

# TODO: add more relevant LANTERN variables here
lantern_vars = [
    "nShowers",
    "showerPhScore",
    "vtxX",
    "vtxY",
    "vtxZ",
]

wc_postprocessing_training_vars = [
    "wc_reco_shower_theta",
    "wc_reco_shower_phi",
    "wc_reco_distance_to_boundary",
    "wc_reco_backwards_projected_dist",
]

blip_postprocessing_vars = [
    "blip_closest_upstream_distance",
    "blip_closest_upstream_angle",
    "blip_closest_upstream_impact_parameter",
    "blip_closest_upstream_energy",
    "blip_closest_upstream_dx",
    "blip_closest_upstream_dw",
]

glee_postprocessing_training_vars = [
    "glee_max_ssv_score",
    "glee_max_3d_shower_score",
]

lantern_postprocessing_training_vars = [
    "lantern_max_showerPhScore",
    "lantern_second_max_showerPhScore",
]

combined_postprocessing_training_vars = [
    "wc_pandora_dist",
    "wc_lantern_dist",
    "lantern_pandora_dist",
]

combined_training_vars = wc_training_vars + wc_postprocessing_training_vars + glee_postprocessing_training_vars + lantern_postprocessing_training_vars
# leave blip, nanosecond timing, spacepoint SSV, and PMT vars for analysis of the selection after the combined BDT for more interpretability
