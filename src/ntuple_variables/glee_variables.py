glee_scalar_non_training_vars = [
    "reco_vertex_x",
    "reco_vertex_y",
    "reco_vertex_z",
]

glee_CRT_variables = [
    "CRT_dt",
    "CRT_hits_PE", # vector
    "CRT_hits_time", # vector
    "CRT_hits_x", # vector
    "CRT_hits_y", # vector
    "CRT_hits_z", # vector
    "CRT_min_hit_PE",
    "CRT_min_hit_time",
    "CRT_min_hit_x",
    "CRT_min_hit_y",
    "CRT_min_hit_z",
    "CRT_veto_hit_PE", # vector
    "CRT_veto_nhits",
]

glee_scalar_vars = [
    "reco_asso_showers",
    "reco_asso_tracks",
    "reco_slice_num",
    "reco_slice_objects", # number of pfparticles in the slice (showers plus tracks plus secondaries)
    #"reco_slice_shower_num_matched_signal", # bad variable, filled for true NC Delta, weird values from -1e9 to 1e9 otherwise
    #"reco_slice_track_num_matched_signal", # bad variable, filled for true NC Delta, weird values from -1e9 to 1e9 otherwise
    "reco_vertex_dist_to_SCB",
    "reco_vertex_dist_to_active_TPC",
    "reco_vertex_in_SCB",
    "reco_vertex_size",

    # SSV was trained with coherent gamma signal and NC Pi0 background
    # PSV was trained on NC Delta Rad, but nu_overlay might be better?

    # unassociated hits on each plane grouped by DBscan
    # also there is more candadate-by-candidate stuff in vector variables
    # (don't use RMS stuff for that, extra model dependent)
    # matched variable tells us in truth whether this is from a pi0 shower
    # remerge could be used to add extra energy to the main shower
    "sss_num_associated_hits",
    "sss_num_candidates",
    "sss_num_unassociated_hits",
    "sss_num_unassociated_hits_below_threshold",
    # conv_ranked: closest candidate
    "sss2d_conv_ranked_angle_to_shower",
    "sss2d_conv_ranked_conv",
    "sss2d_conv_ranked_en",
    "sss2d_conv_ranked_invar",
    "sss2d_conv_ranked_ioc",
    "sss2d_conv_ranked_num_planes",
    "sss2d_conv_ranked_pca",
    # invar_ranked: best pi0 invariant mass
    "sss2d_invar_ranked_angle_to_shower",
    "sss2d_invar_ranked_conv",
    "sss2d_invar_ranked_en",
    "sss2d_invar_ranked_invar",
    "sss2d_invar_ranked_ioc",
    "sss2d_invar_ranked_num_planes",
    "sss2d_invar_ranked_pca",
    # ioc_ranked: best-pointing candidate
    "sss2d_ioc_ranked_angle_to_shower",
    "sss2d_ioc_ranked_conv",
    "sss2d_ioc_ranked_en",
    "sss2d_ioc_ranked_invar",
    "sss2d_ioc_ranked_ioc",
    "sss2d_ioc_ranked_num_planes",
    "sss2d_ioc_ranked_pca",


    # 3D, introduced later, looped over cosmic slices and looked for showers
    "sss3d_num_showers",
    # invar_ranked: best pi0 invariant mass
    "sss3d_invar_ranked_conv",
    "sss3d_invar_ranked_en",
    "sss3d_invar_ranked_id",
    "sss3d_invar_ranked_implied_invar",
    "sss3d_invar_ranked_implied_opang",
    "sss3d_invar_ranked_invar",
    "sss3d_invar_ranked_ioc",
    "sss3d_invar_ranked_opang",
    # ioc_ranked: best-pointing candidate
    "sss3d_ioc_ranked_conv",
    "sss3d_ioc_ranked_en",
    "sss3d_ioc_ranked_id",
    "sss3d_ioc_ranked_implied_invar",
    "sss3d_ioc_ranked_implied_opang",
    "sss3d_ioc_ranked_invar",
    "sss3d_ioc_ranked_ioc",
    "sss3d_ioc_ranked_opang",

    # DBscan run again with different requirements to look for proton stubs rather than showers
    # "trackstub_associated_hits", # always zero!
    "trackstub_num_candidate_groups",
    "trackstub_num_candidates",
    "trackstub_num_unassociated_hits",
    "trackstub_unassociated_hits_below_threshold",
]

# isolation variables, looks for unassociated hits around the proton
glee_vector_vars = [
    "isolation_min_dist_trk_shr",
    "isolation_min_dist_trk_unassoc",
    "isolation_nearest_shr_hit_to_trk_time",
    "isolation_nearest_shr_hit_to_trk_wire",
    "isolation_nearest_unassoc_hit_to_trk_time",
    "isolation_nearest_unassoc_hit_to_trk_wire",
    "isolation_num_shr_hits_win_10cm_trk",
    "isolation_num_shr_hits_win_1cm_trk",
    "isolation_num_shr_hits_win_2cm_trk",
    "isolation_num_shr_hits_win_5cm_trk",
    "isolation_num_unassoc_hits_win_10cm_trk",
    "isolation_num_unassoc_hits_win_1cm_trk",
    "isolation_num_unassoc_hits_win_2cm_trk",
    "isolation_num_unassoc_hits_win_5cm_trk",

    # "trackstub_candidate_veto_score", # always empty!
    "trackstub_candidate_in_nu_slice",
    "trackstub_candidate_num_hits",
    "trackstub_candidate_num_wires",
    "trackstub_candidate_num_ticks",
    "trackstub_candidate_plane",
    "trackstub_candidate_PCA",
    "trackstub_candidate_mean_ADC",
    "trackstub_candidate_ADC_RMS",
    "trackstub_candidate_min_dist",
    "trackstub_candidate_min_impact_parameter_to_shower",
    "trackstub_candidate_min_conversion_dist_to_shower_start",
    "trackstub_candidate_min_ioc_to_shower_start",
    "trackstub_candidate_ioc_based_length",
    "trackstub_candidate_wire_tick_based_length",
    "trackstub_candidate_mean_ADC_first_half",
    "trackstub_candidate_mean_ADC_second_half",
    "trackstub_candidate_mean_ADC_first_to_second_ratio",
    "trackstub_candidate_track_angle_wrt_shower_direction",
    "trackstub_candidate_linear_fit_chi2",
    "trackstub_candidate_energy",
]

glee_postprocessing_vars = [
    "glee_dist_ranked_isolation_min_dist_trk_shr",
    "glee_dist_ranked_isolation_min_dist_trk_unassoc",
    "glee_dist_ranked_isolation_nearest_shr_hit_to_trk_time",
    "glee_dist_ranked_isolation_nearest_shr_hit_to_trk_wire",
    "glee_dist_ranked_isolation_nearest_unassoc_hit_to_trk_time",
    "glee_dist_ranked_isolation_nearest_unassoc_hit_to_trk_wire",
    "glee_dist_ranked_isolation_num_shr_hits_win_10cm_trk",
    "glee_dist_ranked_isolation_num_shr_hits_win_1cm_trk",
    "glee_dist_ranked_isolation_num_shr_hits_win_2cm_trk",
    "glee_dist_ranked_isolation_num_shr_hits_win_5cm_trk",
    "glee_dist_ranked_isolation_num_unassoc_hits_win_10cm_trk",
    "glee_dist_ranked_isolation_num_unassoc_hits_win_1cm_trk",
    "glee_dist_ranked_isolation_num_unassoc_hits_win_2cm_trk",
    "glee_dist_ranked_isolation_num_unassoc_hits_win_5cm_trk",

    "glee_dist_ranked_trackstub_candidate_in_nu_slice",
    "glee_dist_ranked_trackstub_candidate_num_hits",
    "glee_dist_ranked_trackstub_candidate_num_wires",
    "glee_dist_ranked_trackstub_candidate_num_ticks",
    "glee_dist_ranked_trackstub_candidate_plane",
    "glee_dist_ranked_trackstub_candidate_PCA",
    "glee_dist_ranked_trackstub_candidate_mean_ADC",
    "glee_dist_ranked_trackstub_candidate_ADC_RMS",
    "glee_dist_ranked_trackstub_candidate_min_dist",
    "glee_dist_ranked_trackstub_candidate_min_impact_parameter_to_shower",
    "glee_dist_ranked_trackstub_candidate_min_conversion_dist_to_shower_start",
    "glee_dist_ranked_trackstub_candidate_min_ioc_to_shower_start",
    "glee_dist_ranked_trackstub_candidate_ioc_based_length",
    "glee_dist_ranked_trackstub_candidate_wire_tick_based_length",
    "glee_dist_ranked_trackstub_candidate_mean_ADC_first_half",
    "glee_dist_ranked_trackstub_candidate_mean_ADC_second_half",
    "glee_dist_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio",
    "glee_dist_ranked_trackstub_candidate_track_angle_wrt_shower_direction",
    "glee_dist_ranked_trackstub_candidate_linear_fit_chi2",
    "glee_dist_ranked_trackstub_candidate_energy",
    "glee_energy_ranked_trackstub_candidate_in_nu_slice",
    "glee_energy_ranked_trackstub_candidate_num_hits",
    "glee_energy_ranked_trackstub_candidate_num_wires",
    "glee_energy_ranked_trackstub_candidate_num_ticks",
    "glee_energy_ranked_trackstub_candidate_plane",
    "glee_energy_ranked_trackstub_candidate_PCA",
    "glee_energy_ranked_trackstub_candidate_mean_ADC",
    "glee_energy_ranked_trackstub_candidate_ADC_RMS",
    "glee_energy_ranked_trackstub_candidate_min_dist",
    "glee_energy_ranked_trackstub_candidate_min_impact_parameter_to_shower",
    "glee_energy_ranked_trackstub_candidate_min_conversion_dist_to_shower_start",
    "glee_energy_ranked_trackstub_candidate_min_ioc_to_shower_start",
    "glee_energy_ranked_trackstub_candidate_ioc_based_length",
    "glee_energy_ranked_trackstub_candidate_wire_tick_based_length",
    "glee_energy_ranked_trackstub_candidate_mean_ADC_first_half",
    "glee_energy_ranked_trackstub_candidate_mean_ADC_second_half",
    "glee_energy_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio",
    "glee_energy_ranked_trackstub_candidate_track_angle_wrt_shower_direction",
    "glee_energy_ranked_trackstub_candidate_linear_fit_chi2",
    "glee_energy_ranked_trackstub_candidate_energy",
    "glee_conv_ranked_trackstub_candidate_in_nu_slice",
    "glee_conv_ranked_trackstub_candidate_num_hits",
    "glee_conv_ranked_trackstub_candidate_num_wires",
    "glee_conv_ranked_trackstub_candidate_num_ticks",
    "glee_conv_ranked_trackstub_candidate_plane",
    "glee_conv_ranked_trackstub_candidate_PCA",
    "glee_conv_ranked_trackstub_candidate_mean_ADC",
    "glee_conv_ranked_trackstub_candidate_ADC_RMS",
    "glee_conv_ranked_trackstub_candidate_min_dist",
    "glee_conv_ranked_trackstub_candidate_min_impact_parameter_to_shower",
    "glee_conv_ranked_trackstub_candidate_min_conversion_dist_to_shower_start",
    "glee_conv_ranked_trackstub_candidate_min_ioc_to_shower_start",
    "glee_conv_ranked_trackstub_candidate_ioc_based_length",
    "glee_conv_ranked_trackstub_candidate_wire_tick_based_length",
    "glee_conv_ranked_trackstub_candidate_mean_ADC_first_half",
    "glee_conv_ranked_trackstub_candidate_mean_ADC_second_half",
    "glee_conv_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio",
    "glee_conv_ranked_trackstub_candidate_track_angle_wrt_shower_direction",
    "glee_conv_ranked_trackstub_candidate_linear_fit_chi2",
    "glee_conv_ranked_trackstub_candidate_energy",
    "glee_impact_param_ranked_trackstub_candidate_in_nu_slice",
    "glee_impact_param_ranked_trackstub_candidate_num_hits",
    "glee_impact_param_ranked_trackstub_candidate_num_wires",
    "glee_impact_param_ranked_trackstub_candidate_num_ticks",
    "glee_impact_param_ranked_trackstub_candidate_plane",
    "glee_impact_param_ranked_trackstub_candidate_PCA",
    "glee_impact_param_ranked_trackstub_candidate_mean_ADC",
    "glee_impact_param_ranked_trackstub_candidate_ADC_RMS",
    "glee_impact_param_ranked_trackstub_candidate_min_dist",
    "glee_impact_param_ranked_trackstub_candidate_min_impact_parameter_to_shower",
    "glee_impact_param_ranked_trackstub_candidate_min_conversion_dist_to_shower_start",
    "glee_impact_param_ranked_trackstub_candidate_min_ioc_to_shower_start",
    "glee_impact_param_ranked_trackstub_candidate_ioc_based_length",
    "glee_impact_param_ranked_trackstub_candidate_wire_tick_based_length",
    "glee_impact_param_ranked_trackstub_candidate_mean_ADC_first_half",
    "glee_impact_param_ranked_trackstub_candidate_mean_ADC_second_half",
    "glee_impact_param_ranked_trackstub_candidate_mean_ADC_first_to_second_ratio",
    "glee_impact_param_ranked_trackstub_candidate_track_angle_wrt_shower_direction",
    "glee_impact_param_ranked_trackstub_candidate_linear_fit_chi2",
    "glee_impact_param_ranked_trackstub_candidate_energy",
]

glee_training_vars = ["glee_" + var for var in glee_scalar_vars] + glee_postprocessing_vars

glee_vars = glee_scalar_vars + glee_vector_vars + glee_scalar_non_training_vars

# Variables from singlephotonana/eventweight_tree (GENIE truth quantities)
glee_eventweight_vars = [
    "GTruth_ResNum",         # resonance number (GENIE resonance code), -1 for non-resonance
    "GTruth_Gint",           # GENIE interaction type code
    "GTruth_Gscatter",       # GENIE scattering type code
    "GTruth_gW",             # hadronic invariant mass W (GeV)
    "GTruth_gQ2",            # momentum transfer Q^2 (GeV^2)
    "GTruth_gq2",            # momentum transfer q^2 = -Q^2 (GeV^2)
    "GTruth_gX",             # Bjorken x
    "GTruth_gY",             # inelasticity y
    "GTruth_gT",             # momentum transfer t (GeV^2), used for coherent scattering
    "GTruth_probability",    # interaction probability
    "GTruth_Xsec",           # cross section (1e-38 cm^2)
    "GTruth_DiffXsec",       # differential cross section
    "GTruth_GPhaseSpace",    # phase space flag
    "GTruth_tgtPDG",         # target PDG code
    "GTruth_tgtA",           # target atomic mass number
    "GTruth_tgtZ",           # target atomic number
    "GTruth_ProbePDG",       # probe PDG code
    "GTruth_IsCharm",        # is charm production
    "GTruth_IsStrange",      # is strange production
    "GTruth_IsSeaQuark",     # struck nucleon is sea quark
    "GTruth_NumPiPlus",      # number of pi+ in hadronic system
    "GTruth_NumPi0",         # number of pi0 in hadronic system
    "GTruth_NumPiMinus",     # number of pi- in hadronic system
    "GTruth_NumProton",      # number of protons in hadronic system
    "GTruth_NumNeutron",     # number of neutrons in hadronic system
    "GTruth_DecayMode",      # resonance decay mode
    "GTruth_weight",         # GENIE event weight
    "GTruth_vertexX",        # interaction vertex x (cm)
    "GTruth_vertexY",        # interaction vertex y (cm)
    "GTruth_vertexZ",        # interaction vertex z (cm)
    "GTruth_vertexT",        # interaction vertex t (ns)
    "GTruth_ProbeP4x",       # probe 4-momentum px (GeV/c)
    "GTruth_ProbeP4y",       # probe 4-momentum py (GeV/c)
    "GTruth_ProbeP4z",       # probe 4-momentum pz (GeV/c)
    "GTruth_ProbeP4E",       # probe 4-momentum E (GeV)
    "GTruth_HitNucP4x",      # struck nucleon 4-momentum px (GeV/c)
    "GTruth_HitNucP4y",      # struck nucleon 4-momentum py (GeV/c)
    "GTruth_HitNucP4z",      # struck nucleon 4-momentum pz (GeV/c)
    "GTruth_HitNucP4E",      # struck nucleon 4-momentum E (GeV)
    "GTruth_HitNucPos",      # struck nucleon position in nucleus
    "GTruth_TgtP4x",         # target 4-momentum px (GeV/c)
    "GTruth_TgtP4y",         # target 4-momentum py (GeV/c)
    "GTruth_TgtP4z",         # target 4-momentum pz (GeV/c)
    "GTruth_TgtP4E",         # target 4-momentum E (GeV)
    "GTruth_FShadSystP4x",   # final state hadronic system 4-momentum px (GeV/c)
    "GTruth_FShadSystP4y",   # final state hadronic system 4-momentum py (GeV/c)
    "GTruth_FShadSystP4z",   # final state hadronic system 4-momentum pz (GeV/c)
    "GTruth_FShadSystP4E",   # final state hadronic system 4-momentum E (GeV)
    "GTruth_StrangeHadronPDG",  # strange hadron PDG code
    "GTruth_CharmHadronPDG",    # charm hadron PDG code
]
