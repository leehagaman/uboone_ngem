NC1pi0_query = "normal_overlay and wc_truth_inFV and wc_truth_isNC and not wc_truth_NCDeltaRad and wc_truth_1pi0"
NC1pi0Np_query = f"{NC1pi0_query} and wc_truth_Np"
del1g_detailed_categories = [
    ("NCDeltaRad_1gNp",                 "normal_overlay and wc_truth_inFV and wc_truth_NCDeltaRad and wc_truth_0pi0 and wc_truth_Np",                                                           "xkcd:light yellow", None),
    ("NCDeltaRad_1g0p",                 "normal_overlay and wc_truth_inFV and wc_truth_NCDeltaRad and wc_truth_0pi0 and wc_truth_0p",                                                           "xkcd:pumpkin", None),
    ("numuCCDeltaRad_1gNp",             "normal_overlay and wc_truth_inFV and wc_truth_numuCCDeltaRad and wc_truth_0pi0 and wc_truth_Np",                                                       "xkcd:royal blue", None),
    ("numuCCDeltaRad_1g0p",             "normal_overlay and wc_truth_inFV and wc_truth_numuCCDeltaRad and wc_truth_0pi0 and wc_truth_0p",                                                       "xkcd:baby blue", None),

    ("NC1pi0_Np_outFV_gamma",   """normal_overlay and wc_truth_inFV and wc_truth_isNC and not wc_truth_NCDeltaRad and wc_truth_1pi0 and wc_truth_Np
                                    and true_num_gamma_pairconvert_in_FV < 2""".strip().replace("\n", ""),                                                          "#FFE6E6", None),
    ("NC1pi0_Np_lowE_gamma",    """normal_overlay and wc_truth_inFV and wc_truth_isNC and not wc_truth_NCDeltaRad and wc_truth_1pi0 and wc_truth_Np
                                    and not (true_num_gamma_pairconvert_in_FV < 2) and true_num_gamma_pairconvert_in_FV_20_MeV < 2""".strip().replace("\n", ""),    "#FFB8B8", None),
    ("NC1pi0_Np_misclustered_gamma",    """normal_overlay and wc_truth_inFV and wc_truth_isNC and not wc_truth_NCDeltaRad and wc_truth_1pi0 and wc_truth_Np
                                    and not (true_num_gamma_pairconvert_in_FV < 2) and not (true_num_gamma_pairconvert_in_FV_20_MeV < 2)
                                    and wc_true_gamma_pairconversion_spacepoint_max_min_distance > 5""".strip().replace("\n", ""),                                  "#FF8A8A", None),
    ("NC1pi0_Np_other",         """normal_overlay and wc_truth_inFV and wc_truth_isNC and not wc_truth_NCDeltaRad and wc_truth_1pi0 and wc_truth_Np
                                    and not (true_num_gamma_pairconvert_in_FV < 2) and not (true_num_gamma_pairconvert_in_FV_20_MeV < 2)
                                    and not (wc_true_gamma_pairconversion_spacepoint_max_min_distance > 5)""".strip().replace("\n", ""),                            "#FF5C5C", None),
    
    ("NC1pi0_0p_outFV_gamma",   """normal_overlay and wc_truth_inFV and wc_truth_isNC and not wc_truth_NCDeltaRad and wc_truth_1pi0 and wc_truth_0p
                                    and true_num_gamma_pairconvert_in_FV < 2""".strip().replace("\n", ""),                                                          "#FFE6E6", None),
    ("NC1pi0_0p_lowE_gamma",    """normal_overlay and wc_truth_inFV and wc_truth_isNC and not wc_truth_NCDeltaRad and wc_truth_1pi0 and wc_truth_0p
                                    and not (true_num_gamma_pairconvert_in_FV < 2) and true_num_gamma_pairconvert_in_FV_20_MeV < 2""".strip().replace("\n", ""),    "#FFB8B8", None),
    ("NC1pi0_0p_misclustered_gamma",    """normal_overlay and wc_truth_inFV and wc_truth_isNC and not wc_truth_NCDeltaRad and wc_truth_1pi0 and wc_truth_0p
                                    and not (true_num_gamma_pairconvert_in_FV < 2) and not (true_num_gamma_pairconvert_in_FV_20_MeV < 2)
                                    and wc_true_gamma_pairconversion_spacepoint_max_min_distance > 5""".strip().replace("\n", ""),                                  "#FF8A8A", None),
    ("NC1pi0_0p_other",         """normal_overlay and wc_truth_inFV and wc_truth_isNC and not wc_truth_NCDeltaRad and wc_truth_1pi0 and wc_truth_0p
                                    and not (true_num_gamma_pairconvert_in_FV < 2) and not (true_num_gamma_pairconvert_in_FV_20_MeV < 2)
                                    and not (wc_true_gamma_pairconversion_spacepoint_max_min_distance > 5)""".strip().replace("\n", ""),                            "#FF5C5C", None),

    ("numuCC1pi0_Np",   "normal_overlay and wc_truth_inFV and wc_truth_numuCC and not wc_truth_numuCCDeltaRad and wc_truth_1pi0 and wc_truth_Np",                                   "xkcd:blue", None),
    ("numuCC1pi0_0p",   "normal_overlay and wc_truth_inFV and wc_truth_numuCC and not wc_truth_numuCCDeltaRad and wc_truth_1pi0 and wc_truth_0p",                                   "xkcd:lightblue", None),
    ("1pi0_outFV",      "normal_overlay and not (wc_truth_inFV) and wc_truth_1pi0",                                                                                                 "xkcd:light pink", None),
    ("nueCC_Np",        "normal_overlay and wc_truth_inFV and wc_truth_nueCC and wc_truth_Np",                                                                                      "xkcd:seafoam", None),
    ("nueCC_0p",        "normal_overlay and wc_truth_inFV and wc_truth_nueCC and wc_truth_0p",                                                                                      "xkcd:electric green", None),
    ("multi_pi0",       "normal_overlay and wc_truth_inFV and wc_truth_notnueCC and (wc_truth_multi_pi0 or (wc_truth_1pi0 and (wc_truth_NCDeltaRad or wc_truth_numuCCDeltaRad)))",  "xkcd:ice blue", None), # also includes pi0 + Delta radiative
    ("0pi0",            "normal_overlay and wc_truth_inFV and wc_truth_notnueCC and wc_truth_0pi0 and not (wc_truth_NCDeltaRad or wc_truth_numuCCDeltaRad)",                        "xkcd:azure", None),
    ("other_outFV",     "normal_overlay and not (wc_truth_inFV) and not (wc_truth_1pi0)",                                                                                           "xkcd:bright purple", None),
    ("dirt",        "filetype == 'dirt_overlay'",                                                                                                                                   "xkcd:brown", None),
    ("ext",         "filetype == 'ext'",                                                                                                                                            "xkcd:green", None),
    ("del1g_Np",        "del1g_overlay and wc_truth_inFV and wc_truth_Np and wc_truth_0mu",                                                                                         "xkcd:yellow", "++++"),
    ("del1g_0p",        "del1g_overlay and wc_truth_inFV and wc_truth_0p and wc_truth_0mu",                                                                                         "xkcd:orange", "++++"),
    ("del1g_Np1mu",     "del1g_overlay and wc_truth_inFV and wc_truth_Np and wc_truth_1mu",                                                                                         "xkcd:cyan", "++++"),
    ("del1g_0p1mu",     "del1g_overlay and wc_truth_inFV and wc_truth_0p and wc_truth_1mu",                                                                                         "xkcd:aqua", "++++"),
    ("del1g_outFV",     "del1g_overlay and not wc_truth_inFV",                                                                                                                      "xkcd:pink", "++++"),
    ("iso1g",           "iso1g_overlay and wc_truth_inFV",                                                                                                                          "xkcd:turquoise", "++++"),
    ("iso1g_outFV",     "iso1g_overlay and not wc_truth_inFV",                                                                                                                      "xkcd:gray", "++++"),
]
del1g_detailed_category_queries = [cat[1] for cat in del1g_detailed_categories]
del1g_detailed_category_labels = [cat[0] for cat in del1g_detailed_categories]
del1g_detailed_category_colors = [cat[2] for cat in del1g_detailed_categories]
del1g_detailed_category_hatches = [cat[3] for cat in del1g_detailed_categories]
del1g_detailed_category_labels_latex = [
    r"NC $\Delta\rightarrow N \gamma$ $Np$",
    r"NC $\Delta\rightarrow N \gamma$ $0p$",
    r"$\nu_\mu$ CC $\Delta\rightarrow N \gamma$ $Np$",
    r"$\nu_\mu$ CC $\Delta\rightarrow N \gamma$ $0p$",
    r"NC $1\pi^0$ $Np$ out-FV $\gamma$",
    r"NC $1\pi^0$ $Np$ low-E $\gamma$",
    r"NC $1\pi^0$ $Np$ misclustered $\gamma$",
    r"NC $1\pi^0$ $Np$ other",
    r"NC $1\pi^0$ $0p$ out-FV $\gamma$",
    r"NC $1\pi^0$ $0p$ low-E $\gamma$",
    r"NC $1\pi^0$ $0p$ misclustered $\gamma$",
    r"NC $1\pi^0$ $0p$ other",
    r"NC $1\pi^0$ $0p$",
    r"$\nu_\mu$ CC $1\pi^0$ $Np$",
    r"$\nu_\mu$ CC $1\pi^0$ $0p$",
    r"$1\pi^0$ out FV",
    r"$\nu_e$ CC $Np$",
    r"$\nu_e$ CC $0p$",
    r"multi $\pi^0$",
    r"$0\pi^0$",
    r"other out FV",
    r"dirt",
    r"ext",
    r"del1g $1\gamma Np$",
    r"del1g $1\gamma 0p$",
    r"del1g $1\gamma Np 1\mu$",
    r"del1g $1\gamma 0p 1\mu$",
    r"del1g $1\gamma$ out FV",
    r"iso1g $1\gamma 0p$",
    r"iso1g $1\gamma$ out FV",
]

def get_cut_from_del1g(name):
    for line in del1g_detailed_categories:
        if line[0] == name:
            return line[1]
    raise ValueError(f"Category not found in del1g_detailed_categories! {name}")

del1g_simple_categories = [
    ("1gNp",                f"({get_cut_from_del1g('del1g_Np')}) or ({get_cut_from_del1g('NCDeltaRad_1gNp')})",                                         "xkcd:yellow", None),
    ("1g0p",                f"({get_cut_from_del1g('del1g_0p')}) or ({get_cut_from_del1g('NCDeltaRad_1g0p')}) or ({get_cut_from_del1g('iso1g')})",      "xkcd:orange", None),
    ("1gNp1mu",             f"({get_cut_from_del1g('del1g_Np1mu')}) or ({get_cut_from_del1g('numuCCDeltaRad_1gNp')})",                                  "xkcd:cyan", None),
    ("1g0p1mu",             f"({get_cut_from_del1g('del1g_0p1mu')}) or ({get_cut_from_del1g('numuCCDeltaRad_1g0p')})",                                  "xkcd:aqua", None),
    ("1g_outFV",            f"({get_cut_from_del1g('del1g_outFV')}) or ({get_cut_from_del1g('iso1g_outFV')})",                                          "xkcd:pink", None),
    ("NC1pi0_Np",           f"({get_cut_from_del1g('NC1pi0_Np_outFV_gamma')}) or ({get_cut_from_del1g('NC1pi0_Np_lowE_gamma')}) or ({get_cut_from_del1g('NC1pi0_Np_misclustered_gamma')}) or ({get_cut_from_del1g('NC1pi0_Np_other')})", "xkcd:red", None),
    ("NC1pi0_0p",           f"({get_cut_from_del1g('NC1pi0_0p_outFV_gamma')}) or ({get_cut_from_del1g('NC1pi0_0p_lowE_gamma')}) or ({get_cut_from_del1g('NC1pi0_0p_misclustered_gamma')}) or ({get_cut_from_del1g('NC1pi0_0p_other')})", "xkcd:red", None),
    ("numuCC1pi0_Np",           get_cut_from_del1g('numuCC1pi0_Np'),                                                                                    "xkcd:blue", None),
    ("numuCC1pi0_0p",           get_cut_from_del1g('numuCC1pi0_0p'),                                                                                    "xkcd:lightblue", None),
    ("1pi0_outFV",              get_cut_from_del1g('1pi0_outFV'),                                                                                       "xkcd:light pink", None),
    ("nueCC_Np",                get_cut_from_del1g('nueCC_Np'),                                                                                         "xkcd:seafoam", None),
    ("nueCC_0p",                get_cut_from_del1g('nueCC_0p'),                                                                                         "xkcd:electric green", None),
    ("multi_pi0",               get_cut_from_del1g('multi_pi0'),                                                                                        "xkcd:ice blue", None), # also includes pi0 + Delta radiative
    ("0pi0",                    get_cut_from_del1g('0pi0'),                                                                                             "xkcd:azure", None),
    ("other_outFV_dirt",    f"({get_cut_from_del1g('other_outFV')}) or ({get_cut_from_del1g('dirt')})",                                                 "xkcd:bright purple", None),
    ("ext",                     get_cut_from_del1g('ext'),                                                                                              "xkcd:green", None),
]
del1g_simple_category_queries = [cat[1] for cat in del1g_simple_categories]
del1g_simple_category_labels = [cat[0] for cat in del1g_simple_categories]
del1g_simple_category_colors = [cat[2] for cat in del1g_simple_categories]
del1g_simple_category_hatches = [cat[3] for cat in del1g_simple_categories]
del1g_simple_category_labels_latex = [
    r"$1\gamma Np$",
    r"$1\gamma 0p$",
    r"$1\gamma Np 1\mu$",
    r"$1\gamma 0p 1\mu$",
    r"$1\gamma$ out FV",
    r"$NC 1\pi^0$ $Np$",
    r"$NC 1\pi^0$ $0p$",
    r"$\nu_\mu$ CC $1\pi^0$ $Np$",
    r"$\nu_\mu$ CC $1\pi^0$ $0p$",
    r"$1\pi^0$ out FV",
    r"$\nu_e$ CC $Np$",
    r"$\nu_e$ CC $0p$",
    r"multi-$\pi^0$",
    r"$0\pi^0$",
    r"other out-FV/dirt",
    r"ext",
]


topological_categories = [
    ("1gNp",        "(normal_overlay or del1g_overlay or iso1g_overlay) and wc_truth_inFV and wc_truth_0e and wc_truth_1g and wc_truth_Np and wc_truth_0mu",    "xkcd:yellow", None),
    ("1g0p",        "(normal_overlay or del1g_overlay or iso1g_overlay) and wc_truth_inFV and wc_truth_0e and wc_truth_1g and wc_truth_0p and wc_truth_0mu",    "xkcd:orange", None),
    ("1gNp1mu",     "(normal_overlay or del1g_overlay or iso1g_overlay) and wc_truth_inFV and wc_truth_0e and wc_truth_1g and wc_truth_Np and wc_truth_1mu",    "xkcd:cyan", None),
    ("1g0p1mu",     "(normal_overlay or del1g_overlay or iso1g_overlay) and wc_truth_inFV and wc_truth_0e and wc_truth_1g and wc_truth_0p and wc_truth_1mu",    "xkcd:aqua", None),
    ("1g_outFV",    "(normal_overlay or del1g_overlay or iso1g_overlay) and not (wc_truth_inFV) and wc_truth_0e and wc_truth_1g",                               "xkcd:pink", None),
    ("2gNp",        "(normal_overlay or del1g_overlay or iso1g_overlay) and wc_truth_inFV and wc_truth_0e and wc_truth_2g and wc_truth_Np and wc_truth_0mu",    "xkcd:red", None),
    ("2g0p",        "(normal_overlay or del1g_overlay or iso1g_overlay) and wc_truth_inFV and wc_truth_0e and wc_truth_2g and wc_truth_0p and wc_truth_0mu",    "xkcd:salmon", None),
    ("2gNp1mu",     "(normal_overlay or del1g_overlay or iso1g_overlay) and wc_truth_inFV and wc_truth_0e and wc_truth_2g and wc_truth_Np and wc_truth_1mu",    "xkcd:blue", None),
    ("2g0p1mu",     "(normal_overlay or del1g_overlay or iso1g_overlay) and wc_truth_inFV and wc_truth_0e and wc_truth_2g and wc_truth_0p and wc_truth_1mu",    "xkcd:light blue", None),
    ("2g_outFV",    "(normal_overlay or del1g_overlay or iso1g_overlay) and not (wc_truth_inFV) and wc_truth_0e and wc_truth_2g",                               "xkcd:bright purple", None),
    ("1eNp",        "(normal_overlay or del1g_overlay or iso1g_overlay) and wc_truth_1e and wc_truth_Np",                                                       "xkcd:seafoam", None),
    ("1e0p",        "(normal_overlay or del1g_overlay or iso1g_overlay) and wc_truth_1e and wc_truth_0p",                                                       "xkcd:electric green", None),
    ("0g",          "(normal_overlay or del1g_overlay or iso1g_overlay) and wc_truth_0e and wc_truth_0g",                                                       "xkcd:gray", None),
    ("3plusg",      "(normal_overlay or del1g_overlay or iso1g_overlay) and wc_truth_0e and wc_truth_3plusg",                                                   "xkcd:beige", None),
    ("dirt",        "filetype == 'dirt_overlay'",                                                                           "xkcd:brown", None),
    ("ext",         "filetype == 'ext'",                                                                                    "xkcd:green", None),
]
topological_category_queries = [cat[1] for cat in topological_categories]
topological_category_labels = [cat[0] for cat in topological_categories]
topological_category_colors = [cat[2] for cat in topological_categories]
topological_category_hatches = [cat[3] for cat in topological_categories]
topological_category_labels_latex = [
    r"$1\gamma Np$",
    r"$1\gamma 0p$",
    r"$1\gamma Np 1\mu$",
    r"$1\gamma 0p 1\mu$",
    r"$1\gamma$ out FV",
    r"$2\gamma Np$",
    r"$2\gamma 0p$",
    r"$2\gamma Np 1\mu$",
    r"$2\gamma 0p 1\mu$",
    r"$2\gamma$ out FV",
    r"$1e Np$",
    r"$1e 0p$",
    r"$0\gamma$",
    r"$3+\gamma$",
    r"dirt",
    r"ext",
]

filetype_categories = [
    ("nc_pi0_overlay", "filetype == 'nc_pi0_overlay'", "xkcd:red", None),
    ("nu_overlay",     "filetype == 'nu_overlay'",     "xkcd:blue", None),
    ("dirt_overlay",   "filetype == 'dirt_overlay'",   "xkcd:brown", None),
    ("ext",            "filetype == 'ext'",            "xkcd:green", None),
    ("del1g_overlay",   "filetype == 'delete_one_gamma_overlay'",   "xkcd:yellow", "++++"),
    ("iso1g_overlay",   "filetype == 'isotropic_one_gamma_overlay'",   "xkcd:turquoise", "++++"),
]
filetype_category_queries = [cat[1] for cat in filetype_categories]
filetype_category_labels = [cat[0] for cat in filetype_categories]
filetype_category_colors = [cat[2] for cat in filetype_categories]
filetype_category_hatches = [cat[3] for cat in filetype_categories]
filetype_category_labels_latex = [cat[0] for cat in filetype_categories]

# TODO: add categories that separate out events with vertex blips and neutron blips
# Maybe use some variable for "true effective vertex blip energy", as a sum of all charged particle energies produced at the vertex after accounting for quenching?
# Add category for photonuclear absorption
