
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
    # TODO: add more variables for trackstub_candidate stuff?
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
]

# saved the min value for each of these, probabably 
# TODO: should use elements with min glee_min_isolation_min_dist_trk_unassoc
glee_postprocessing_vars = [
    "glee_min_isolation_min_dist_trk_shr",
    "glee_min_isolation_min_dist_trk_unassoc",
    "glee_min_isolation_nearest_shr_hit_to_trk_time",
    "glee_min_isolation_nearest_shr_hit_to_trk_wire",
    "glee_min_isolation_nearest_unassoc_hit_to_trk_time",
    "glee_min_isolation_nearest_unassoc_hit_to_trk_wire",
    "glee_sum_isolation_num_shr_hits_win_10cm_trk",
    "glee_sum_isolation_num_shr_hits_win_1cm_trk",
    "glee_sum_isolation_num_shr_hits_win_2cm_trk",
    "glee_sum_isolation_num_shr_hits_win_5cm_trk",
    "glee_sum_isolation_num_unassoc_hits_win_10cm_trk",
    "glee_sum_isolation_num_unassoc_hits_win_1cm_trk",
    "glee_sum_isolation_num_unassoc_hits_win_2cm_trk",
    "glee_sum_isolation_num_unassoc_hits_win_5cm_trk",
]

glee_training_vars = ["glee_" + var for var in glee_scalar_vars] + glee_postprocessing_vars

glee_vars = glee_scalar_vars + glee_vector_vars
