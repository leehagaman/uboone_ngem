
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
    "reco_slice_objects",
    "reco_slice_shower_num_matched_signal",
    "reco_slice_track_num_matched_signal",
    "reco_vertex_dist_to_SCB",
    "reco_vertex_dist_to_active_TPC",
    "reco_vertex_in_SCB",
    "reco_vertex_size",

    "sss2d_conv_ranked_angle_to_shower",
    "sss2d_conv_ranked_conv",
    "sss2d_conv_ranked_en",
    "sss2d_conv_ranked_invar",
    "sss2d_conv_ranked_ioc",
    "sss2d_conv_ranked_num_planes",
    "sss2d_conv_ranked_pca",
    "sss2d_invar_ranked_angle_to_shower",
    "sss2d_invar_ranked_conv",
    "sss2d_invar_ranked_en",
    "sss2d_invar_ranked_invar",
    "sss2d_invar_ranked_ioc",
    "sss2d_invar_ranked_num_planes",
    "sss2d_invar_ranked_pca",
    "sss2d_ioc_ranked_angle_to_shower",
    "sss2d_ioc_ranked_conv",
    "sss2d_ioc_ranked_en",
    "sss2d_ioc_ranked_invar",
    "sss2d_ioc_ranked_ioc",
    "sss2d_ioc_ranked_num_planes",
    "sss2d_ioc_ranked_pca",
    "sss3d_invar_ranked_conv",
    "sss3d_invar_ranked_en",
    "sss3d_invar_ranked_id",
    "sss3d_invar_ranked_implied_invar",
    "sss3d_invar_ranked_implied_opang",
    "sss3d_invar_ranked_invar",
    "sss3d_invar_ranked_ioc",
    "sss3d_invar_ranked_opang",
    "sss3d_ioc_ranked_conv",
    "sss3d_ioc_ranked_en",
    "sss3d_ioc_ranked_id",
    "sss3d_ioc_ranked_implied_invar",
    "sss3d_ioc_ranked_implied_opang",
    "sss3d_ioc_ranked_invar",
    "sss3d_ioc_ranked_ioc",
    "sss3d_ioc_ranked_opang",
    "sss3d_num_showers",
    "sss_num_associated_hits",
    "sss_num_candidates",
    "sss_num_unassociated_hits",
    "sss_num_unassociated_hits_below_threshold",

    # "trackstub_associated_hits", # always zero!
    "trackstub_num_candidate_groups",
    "trackstub_num_candidates",
    "trackstub_num_unassociated_hits",
    "trackstub_unassociated_hits_below_threshold",
]

# TODO: determine if we should also have trackstub and second shower veto variables here with some type of postprocessing
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
