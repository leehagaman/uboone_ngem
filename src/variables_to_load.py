
wc_T_bdt_vars = [              # variables involved with BDT training (if you want to train your own BDT, lots of these variables will be useful)
                            # many of these variables describe specific features of the WC spacepoints using this code: https://github.com/BNLIF/wire-cell-pid/blob/master/src/NeutrinoID_nue_tagger.h
                            # here, we just include higher level outputs:
    "nue_score",                    # BDT score for nue selection, used for the WC inclusive nueCC analysis
    "numu_score",                   # BDT score for numu selection, used for the WC inclusive numuCC selections
    "nc_delta_score",               # BDT score for NC Delta selection
    "nc_pio_score",                 # BDT score for NC pi0 selection
    "numu_cc_flag",                 # flag, -1 means not generic selected, 0 means generic selected, 1 means cut-based numuCC selected. We often use "numu_cc_flag >= 0" to apply generic neutrino selection.
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
]
wc_T_eval_data_vars = [        # same as above, but for data files we do not attempt to load any truth information
    "run",
    "subrun",
    "event",
    "match_isFC",
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
]

