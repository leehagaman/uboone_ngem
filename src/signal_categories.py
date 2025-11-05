import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import polars as pl

red_hex_colors = [mpl.colors.rgb2hex(color) for color in plt.cm.Reds(np.linspace(0, 1, 10))]
blue_hex_colors = [mpl.colors.rgb2hex(color) for color in plt.cm.Blues(np.linspace(0, 1, 10))]

del1g_detailed_categories = [
    ("NCDeltaRad_1gNp",                 "pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_NCDeltaRad') & pl.col('wc_truth_0pi0') & pl.col('wc_truth_Np') & ~pl.col('wc_true_has_pi0_dalitz_decay')",                                                           "xkcd:light yellow", None),
    ("NCDeltaRad_1g0p",                 "pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_NCDeltaRad') & pl.col('wc_truth_0pi0') & pl.col('wc_truth_0p') & ~pl.col('wc_true_has_pi0_dalitz_decay')",                                                           "xkcd:pumpkin", None),
    ("numuCCDeltaRad_1gNp",             "pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_numuCCDeltaRad') & pl.col('wc_truth_0pi0') & pl.col('wc_truth_Np') & ~pl.col('wc_true_has_pi0_dalitz_decay')",                                                       "xkcd:royal blue", None),
    ("numuCCDeltaRad_1g0p",             "pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_numuCCDeltaRad') & pl.col('wc_truth_0pi0') & pl.col('wc_truth_0p') & ~pl.col('wc_true_has_pi0_dalitz_decay')",                                                       "xkcd:baby blue", None),

    ("NC1pi0_Np_photonuc",   """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_isNC') & ~pl.col('wc_truth_NCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_Np')
                                    & pl.col('wc_true_has_photonuclear_absorption') & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                                                           red_hex_colors[0], None),
    ("NC1pi0_Np_outFV_gamma",   """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_isNC') & ~pl.col('wc_truth_NCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_Np')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & (pl.col('true_num_gamma_pairconvert_in_FV') < 2) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),              red_hex_colors[1], None),
    ("NC1pi0_Np_lowE_gamma",    """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_isNC') & ~pl.col('wc_truth_NCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_Np')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & ~(pl.col('true_num_gamma_pairconvert_in_FV') < 2)
                                    & (pl.col('true_num_gamma_pairconvert_in_FV_20_MeV') < 2) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                                                   red_hex_colors[2], None),
    ("NC1pi0_Np_misclustered_gamma",    """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_isNC') & ~pl.col('wc_truth_NCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_Np')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & ~(pl.col('true_num_gamma_pairconvert_in_FV') < 2)
                                    & ~(pl.col('true_num_gamma_pairconvert_in_FV_20_MeV') < 2)
                                    & (pl.col('wc_true_gamma_pairconversion_spacepoint_max_min_distance') > 5) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                                  red_hex_colors[3], None),
    ("NC1pi0_Np_other",         """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_isNC') & ~pl.col('wc_truth_NCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_Np')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & ~(pl.col('true_num_gamma_pairconvert_in_FV') < 2)
                                    & ~(pl.col('true_num_gamma_pairconvert_in_FV_20_MeV') < 2)
                                    & (pl.col('wc_true_gamma_pairconversion_spacepoint_max_min_distance').is_null() | (pl.col('wc_true_gamma_pairconversion_spacepoint_max_min_distance') <= 5)) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                            red_hex_colors[4], None),
    
    ("NC1pi0_0p_photonuc",   """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_isNC') & ~pl.col('wc_truth_NCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_0p')
                                    & pl.col('wc_true_has_photonuclear_absorption') & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                                                           red_hex_colors[5], None),
    ("NC1pi0_0p_outFV_gamma",   """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_isNC') & ~pl.col('wc_truth_NCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_0p')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & (pl.col('true_num_gamma_pairconvert_in_FV') < 2) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),              red_hex_colors[6], None),
    ("NC1pi0_0p_lowE_gamma",    """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_isNC') & ~pl.col('wc_truth_NCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_0p')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & ~(pl.col('true_num_gamma_pairconvert_in_FV') < 2)
                                    & (pl.col('true_num_gamma_pairconvert_in_FV_20_MeV') < 2) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                                                   red_hex_colors[7], None),
    ("NC1pi0_0p_misclustered_gamma",    """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_isNC') & ~pl.col('wc_truth_NCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_0p')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & ~(pl.col('true_num_gamma_pairconvert_in_FV') < 2)
                                    & ~(pl.col('true_num_gamma_pairconvert_in_FV_20_MeV') < 2)
                                    & (pl.col('wc_true_gamma_pairconversion_spacepoint_max_min_distance') > 5) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                                  red_hex_colors[8], None),
    ("NC1pi0_0p_other",         """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_isNC') & ~pl.col('wc_truth_NCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_0p')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & ~(pl.col('true_num_gamma_pairconvert_in_FV') < 2)
                                    & ~(pl.col('true_num_gamma_pairconvert_in_FV_20_MeV') < 2)
                                    & (pl.col('wc_true_gamma_pairconversion_spacepoint_max_min_distance').is_null() | (pl.col('wc_true_gamma_pairconversion_spacepoint_max_min_distance') <= 5)) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                            red_hex_colors[9], None),


    ("numuCC1pi0_Np_photonuc",   """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_numuCC') & ~pl.col('wc_truth_numuCCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_Np')
                                    & pl.col('wc_true_has_photonuclear_absorption') & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                                                           blue_hex_colors[0], None),
    ("numuCC1pi0_Np_outFV_gamma",   """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_numuCC') & ~pl.col('wc_truth_numuCCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_Np')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & (pl.col('true_num_gamma_pairconvert_in_FV') < 2) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),              blue_hex_colors[1], None),
    ("numuCC1pi0_Np_lowE_gamma",    """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_numuCC') & ~pl.col('wc_truth_numuCCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_Np')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & ~(pl.col('true_num_gamma_pairconvert_in_FV') < 2)
                                    & (pl.col('true_num_gamma_pairconvert_in_FV_20_MeV') < 2) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                                                   blue_hex_colors[2], None),
    ("numuCC1pi0_Np_misclustered_gamma",    """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_numuCC') & ~pl.col('wc_truth_numuCCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_Np')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & ~(pl.col('true_num_gamma_pairconvert_in_FV') < 2)
                                    & ~(pl.col('true_num_gamma_pairconvert_in_FV_20_MeV') < 2)
                                    & (pl.col('wc_true_gamma_pairconversion_spacepoint_max_min_distance') > 5) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                                  blue_hex_colors[3], None),
    ("numuCC1pi0_Np_other",         """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_numuCC') & ~pl.col('wc_truth_numuCCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_Np')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & ~(pl.col('true_num_gamma_pairconvert_in_FV') < 2)
                                    & ~(pl.col('true_num_gamma_pairconvert_in_FV_20_MeV') < 2)
                                    & (pl.col('wc_true_gamma_pairconversion_spacepoint_max_min_distance').is_null() | (pl.col('wc_true_gamma_pairconversion_spacepoint_max_min_distance') <= 5)) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                            blue_hex_colors[4], None),

    ("numuCC1pi0_0p_photonuc",   """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_numuCC') & ~pl.col('wc_truth_numuCCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_0p')
                                    & pl.col('wc_true_has_photonuclear_absorption') & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                                                           blue_hex_colors[5], None),
    ("numuCC1pi0_0p_outFV_gamma",   """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_numuCC') & ~pl.col('wc_truth_numuCCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_0p')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & (pl.col('true_num_gamma_pairconvert_in_FV') < 2) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),              blue_hex_colors[6], None),
    ("numuCC1pi0_0p_lowE_gamma",    """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_numuCC') & ~pl.col('wc_truth_numuCCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_0p')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & ~(pl.col('true_num_gamma_pairconvert_in_FV') < 2)
                                    & (pl.col('true_num_gamma_pairconvert_in_FV_20_MeV') < 2) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                                                   blue_hex_colors[7], None),
    ("numuCC1pi0_0p_misclustered_gamma",    """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_numuCC') & ~pl.col('wc_truth_numuCCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_0p')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & ~(pl.col('true_num_gamma_pairconvert_in_FV') < 2)
                                    & ~(pl.col('true_num_gamma_pairconvert_in_FV_20_MeV') < 2)
                                    & (pl.col('wc_true_gamma_pairconversion_spacepoint_max_min_distance') > 5) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                                  blue_hex_colors[8], None),
    ("numuCC1pi0_0p_other",         """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_numuCC') & ~pl.col('wc_truth_numuCCDeltaRad') & pl.col('wc_truth_1pi0') & pl.col('wc_truth_0p')
                                    & ~pl.col('wc_true_has_photonuclear_absorption') & ~(pl.col('true_num_gamma_pairconvert_in_FV') < 2)
                                    & ~(pl.col('true_num_gamma_pairconvert_in_FV_20_MeV') < 2)
                                    & (pl.col('wc_true_gamma_pairconversion_spacepoint_max_min_distance').is_null() | (pl.col('wc_true_gamma_pairconversion_spacepoint_max_min_distance') <= 5)) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                            blue_hex_colors[9], None),

    ("1pi0_outFV",      "pl.col('normal_overlay') & ~pl.col('wc_truth_inFV') & pl.col('wc_truth_1pi0') & ~pl.col('wc_true_has_pi0_dalitz_decay')",                                                                                                     "xkcd:lavender", None),
    ("nueCC_Np",        "pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_nueCC') & pl.col('wc_truth_Np') & ~pl.col('wc_true_has_pi0_dalitz_decay')",                                                                                          "xkcd:seafoam", None),
    ("nueCC_0p",        "pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_nueCC') & pl.col('wc_truth_0p') & ~pl.col('wc_true_has_pi0_dalitz_decay')",                                                                                          "xkcd:electric green", None),
    ("numuCC_Np",       """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_numuCC') & pl.col('wc_truth_Np')
                                    & pl.col('wc_truth_0pi0') & ~(pl.col('wc_truth_numuCCDeltaRad') | (pl.col('true_num_prim_gamma') >= 2)) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                                       "xkcd:azure", None),
    ("numuCC_0p",       """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_numuCC') & pl.col('wc_truth_0p')
                                    & pl.col('wc_truth_0pi0') & ~(pl.col('wc_truth_numuCCDeltaRad') | (pl.col('true_num_prim_gamma') >= 2)) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),                                       "xkcd:electric blue", None),
    ("pi0_dalitz_decay", "pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_true_has_pi0_dalitz_decay')",                                                                                          "xkcd:dark mint", None),
    
    ("multi_pi0",       """pl.col('normal_overlay') & pl.col('wc_truth_inFV')
                    & (pl.col('wc_truth_multi_pi0') | (pl.col('wc_truth_1pi0') & (pl.col('wc_truth_NCDeltaRad') | pl.col('wc_truth_numuCCDeltaRad'))))
                    & ~(pl.col('wc_true_has_pi0_dalitz_decay') | pl.col('wc_truth_nueCC'))""".strip().replace("\n", ""),      "xkcd:ice blue", None), # also includes pi0 + Delta radiative

    ("eta_other",       """pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_notnueCC') & (pl.col('true_num_prim_gamma') >= 2)
                                    & ~pl.col('wc_truth_multi_pi0') & ~pl.col('wc_truth_1pi0') & ~(pl.col('wc_truth_NCDeltaRad') | pl.col('wc_truth_numuCCDeltaRad')) & ~pl.col('wc_true_has_pi0_dalitz_decay')""".strip().replace("\n", ""),             "xkcd:light aqua", None),
    ("NC_no_gamma",     "pl.col('normal_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_notnueCC') & pl.col('wc_truth_notnumuCC') & pl.col('wc_truth_0pi0') & ~pl.col('wc_truth_NCDeltaRad') & ~pl.col('wc_true_has_pi0_dalitz_decay')",                                  "xkcd:burnt sienna", None),
    ("other_outFV",     "pl.col('normal_overlay') & ~pl.col('wc_truth_inFV') & ~(pl.col('wc_truth_1pi0') & ~pl.col('wc_true_has_pi0_dalitz_decay'))",                                                                                               "xkcd:bright purple", None),
    ("dirt",        "pl.col('filetype') == 'dirt_overlay'",                                                                                                                                       "xkcd:brown", None),
    ("ext",         "pl.col('filetype') == 'ext'",                                                                                                                                                "xkcd:green", None),
    ("del1g_Np",        "pl.col('del1g_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_Np') & pl.col('wc_truth_0mu')",                                                                                             "xkcd:yellow", "++++"),
    ("del1g_0p",        "pl.col('del1g_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_0p') & pl.col('wc_truth_0mu')",                                                                                             "xkcd:orange", "++++"),
    ("del1g_Np1mu",     "pl.col('del1g_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_Np') & pl.col('wc_truth_1mu')",                                                                                             "xkcd:cyan", "++++"),
    ("del1g_0p1mu",     "pl.col('del1g_overlay') & pl.col('wc_truth_inFV') & pl.col('wc_truth_0p') & pl.col('wc_truth_1mu')",                                                                                             "xkcd:aqua", "++++"),
    ("del1g_outFV",     "pl.col('del1g_overlay') & ~pl.col('wc_truth_inFV')",                                                                                                                          "xkcd:pink", "++++"),
    ("iso1g",           "pl.col('iso1g_overlay') & pl.col('wc_truth_inFV')",                                                                                                                              "xkcd:turquoise", "++++"),
    ("iso1g_outFV",     "pl.col('iso1g_overlay') & ~pl.col('wc_truth_inFV')",                                                                                                                          "xkcd:gray", "++++"),
    ("data",         "pl.col('filetype') == 'data'",                                                                                                                                              "xkcd:black", None),
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

    r"NC $1\pi^0$ $Np$ photonuc",
    r"NC $1\pi^0$ $Np$ out-FV $\gamma$",
    r"NC $1\pi^0$ $Np$ low-E $\gamma$",
    r"NC $1\pi^0$ $Np$ misclustered $\gamma$",
    r"NC $1\pi^0$ $Np$ other",

    r"NC $1\pi^0$ $0p$ photonuc",
    r"NC $1\pi^0$ $0p$ out-FV $\gamma$",
    r"NC $1\pi^0$ $0p$ low-E $\gamma$",
    r"NC $1\pi^0$ $0p$ misclustered $\gamma$",
    r"NC $1\pi^0$ $0p$ other",

    r"$\nu_\mu$ CC $1\pi^0$ $Np$ photonuc",
    r"$\nu_\mu$ CC $1\pi^0$ $Np$ out-FV $\gamma$",
    r"$\nu_\mu$ CC $1\pi^0$ $Np$ low-E $\gamma$",
    r"$\nu_\mu$ CC $1\pi^0$ $Np$ misclustered $\gamma$",
    r"$\nu_\mu$ CC $1\pi^0$ $Np$ other",

    r"$\nu_\mu$ CC $1\pi^0$ $0p$ photonuc",
    r"$\nu_\mu$ CC $1\pi^0$ $0p$ out-FV $\gamma$",
    r"$\nu_\mu$ CC $1\pi^0$ $0p$ low-E $\gamma$",
    r"$\nu_\mu$ CC $1\pi^0$ $0p$ misclustered $\gamma$",
    r"$\nu_\mu$ CC $1\pi^0$ $0p$ other",
    
    r"$1\pi^0$ out FV",

    r"$\nu_e$ CC $Np$",
    r"$\nu_e$ CC $0p$",

    r"$\nu_\mu$ CC $Np$",
    r"$\nu_\mu$ CC $0p$",

    r"$\pi^0$ Dalitz decay",
    r"multi $\pi^0$",
    r"Other $2\gamma$ ($\eta$)",
    r"NC no $\gamma$",
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
    r"data",
]
del1g_detailed_categories_dic = {del1g_detailed_category_labels[i]: i for i in range(len(del1g_detailed_category_labels))}

def get_cut_from_del1g(name):
    for line in del1g_detailed_categories:
        if line[0] == name:
            return line[1]
    raise ValueError(f"Category not found in del1g_detailed_categories! {name}")

del1g_simple_categories = [
    ("1gNp",                f"({get_cut_from_del1g('del1g_Np')}) | ({get_cut_from_del1g('NCDeltaRad_1gNp')})",                                         "xkcd:yellow", None),
    ("1g0p",                f"({get_cut_from_del1g('del1g_0p')}) | ({get_cut_from_del1g('NCDeltaRad_1g0p')}) | ({get_cut_from_del1g('iso1g')})",      "xkcd:orange", None),
    ("1gNp1mu",             f"({get_cut_from_del1g('del1g_Np1mu')}) | ({get_cut_from_del1g('numuCCDeltaRad_1gNp')})",                                  "xkcd:cyan", None),
    ("1g0p1mu",             f"({get_cut_from_del1g('del1g_0p1mu')}) | ({get_cut_from_del1g('numuCCDeltaRad_1g0p')})",                                  "xkcd:aqua", None),
    ("1g_outFV",            f"({get_cut_from_del1g('del1g_outFV')}) | ({get_cut_from_del1g('iso1g_outFV')})",                                          "xkcd:pink", None),
    ("NC1pi0_Np",           f"({get_cut_from_del1g('NC1pi0_Np_photonuc')}) | ({get_cut_from_del1g('NC1pi0_Np_outFV_gamma')}) | ({get_cut_from_del1g('NC1pi0_Np_lowE_gamma')}) | ({get_cut_from_del1g('NC1pi0_Np_misclustered_gamma')}) | ({get_cut_from_del1g('NC1pi0_Np_other')})", "xkcd:red", None),
    ("NC1pi0_0p",           f"({get_cut_from_del1g('NC1pi0_0p_photonuc')}) | ({get_cut_from_del1g('NC1pi0_0p_outFV_gamma')}) | ({get_cut_from_del1g('NC1pi0_0p_lowE_gamma')}) | ({get_cut_from_del1g('NC1pi0_0p_misclustered_gamma')}) | ({get_cut_from_del1g('NC1pi0_0p_other')})", "xkcd:light red", None),
    ("numuCC1pi0_Np",       f"({get_cut_from_del1g('numuCC1pi0_Np_photonuc')}) | ({get_cut_from_del1g('numuCC1pi0_Np_outFV_gamma')}) | ({get_cut_from_del1g('numuCC1pi0_Np_lowE_gamma')}) | ({get_cut_from_del1g('numuCC1pi0_Np_misclustered_gamma')}) | ({get_cut_from_del1g('numuCC1pi0_Np_other')})", "xkcd:blue", None),
    ("numuCC1pi0_0p",       f"({get_cut_from_del1g('numuCC1pi0_0p_photonuc')}) | ({get_cut_from_del1g('numuCC1pi0_0p_outFV_gamma')}) | ({get_cut_from_del1g('numuCC1pi0_0p_lowE_gamma')}) | ({get_cut_from_del1g('numuCC1pi0_0p_misclustered_gamma')}) | ({get_cut_from_del1g('numuCC1pi0_0p_other')})", "xkcd:lightblue", None),
    ("1pi0_outFV",              get_cut_from_del1g('1pi0_outFV'),                                                                                       "xkcd:light pink", None),
    ("nueCC_Np",                get_cut_from_del1g('nueCC_Np'),                                                                                         "xkcd:seafoam", None),
    ("nueCC_0p",                get_cut_from_del1g('nueCC_0p'),                                                                                         "xkcd:electric green", None),
    ("numuCC_Np",               get_cut_from_del1g('numuCC_Np'),                                                                                        "xkcd:azure", None),
    ("numuCC_0p",               get_cut_from_del1g('numuCC_0p'),                                                                                        "xkcd:electric blue", None),
    ("pi0_dalitz_decay",        get_cut_from_del1g('pi0_dalitz_decay'),                                                                                "xkcd:dark mint", None),
    ("multi_pi0",               get_cut_from_del1g('multi_pi0'),                                                                                        "xkcd:ice blue", None), # also includes pi0 + Delta radiative
    ("eta_other",               get_cut_from_del1g('eta_other'),                                                                                        "xkcd:light aqua", None),
    ("NC_no_gamma",             get_cut_from_del1g('NC_no_gamma'),                                                                                      "xkcd:burnt sienna", None),
    ("other_outFV_dirt",    f"({get_cut_from_del1g('other_outFV')}) | ({get_cut_from_del1g('dirt')})",                                                 "xkcd:bright purple", None),
    ("ext",                     get_cut_from_del1g('ext'),                                                                                              "xkcd:green", None),
    ("data",                    get_cut_from_del1g('data'),                                                                                             "xkcd:black", None),
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
    r"NC $1\pi^0$ $Np$",
    r"NC $1\pi^0$ $0p$",
    r"$\nu_\mu$ CC $1\pi^0$ $Np$",
    r"$\nu_\mu$ CC $1\pi^0$ $0p$",
    r"$1\pi^0$ out FV",
    r"$\nu_e$ CC $Np$",
    r"$\nu_e$ CC $0p$",
    r"$\nu_\mu$ CC $Np$",
    r"$\nu_\mu$ CC $0p$",
    r"$\pi^0$ Dalitz decay",
    r"Multi-$\pi^0$",
    r"Other $2\gamma$ ($\eta$)",
    r"NC no $\gamma$",
    r"Other out-FV/dirt",
    r"EXT",
    r"data",
]
del1g_simple_categories_dic = {del1g_simple_category_labels[i]: i for i in range(len(del1g_simple_category_labels))}

train_category_queries = []
train_category_labels = []
train_category_colors = []
train_category_hatches = []
train_category_labels_latex = []
for i in range(len(del1g_simple_category_labels)):
    if 'data' not in del1g_simple_category_labels[i]:
        train_category_queries.append(del1g_simple_category_queries[i])
        train_category_labels.append(del1g_simple_category_labels[i])
        train_category_colors.append(del1g_simple_category_colors[i])
        train_category_hatches.append(del1g_simple_category_hatches[i])
        train_category_labels_latex.append(del1g_simple_category_labels_latex[i])
train_category_dic = {train_category_labels[i]: i for i in range(len(train_category_labels))}

topological_categories = [
    ("1gNp",        "(pl.col('normal_overlay') | pl.col('del1g_overlay') | pl.col('iso1g_overlay')) & pl.col('wc_truth_inFV') & pl.col('wc_truth_0e') & pl.col('wc_truth_1g') & pl.col('wc_truth_Np') & pl.col('wc_truth_0mu')",    "xkcd:yellow", None),
    ("1g0p",        "(pl.col('normal_overlay') | pl.col('del1g_overlay') | pl.col('iso1g_overlay')) & pl.col('wc_truth_inFV') & pl.col('wc_truth_0e') & pl.col('wc_truth_1g') & pl.col('wc_truth_0p') & pl.col('wc_truth_0mu')",    "xkcd:orange", None),
    ("1gNp1mu",     "(pl.col('normal_overlay') | pl.col('del1g_overlay') | pl.col('iso1g_overlay')) & pl.col('wc_truth_inFV') & pl.col('wc_truth_0e') & pl.col('wc_truth_1g') & pl.col('wc_truth_Np') & pl.col('wc_truth_1mu')",    "xkcd:cyan", None),
    ("1g0p1mu",     "(pl.col('normal_overlay') | pl.col('del1g_overlay') | pl.col('iso1g_overlay')) & pl.col('wc_truth_inFV') & pl.col('wc_truth_0e') & pl.col('wc_truth_1g') & pl.col('wc_truth_0p') & pl.col('wc_truth_1mu')",    "xkcd:aqua", None),
    ("1g_outFV",    "(pl.col('normal_overlay') | pl.col('del1g_overlay') | pl.col('iso1g_overlay')) & ~pl.col('wc_truth_inFV') & pl.col('wc_truth_0e') & pl.col('wc_truth_1g')",                               "xkcd:pink", None),
    ("2gNp",        "(pl.col('normal_overlay') | pl.col('del1g_overlay') | pl.col('iso1g_overlay')) & pl.col('wc_truth_inFV') & pl.col('wc_truth_0e') & pl.col('wc_truth_2g') & pl.col('wc_truth_Np') & pl.col('wc_truth_0mu')",    "xkcd:red", None),
    ("2g0p",        "(pl.col('normal_overlay') | pl.col('del1g_overlay') | pl.col('iso1g_overlay')) & pl.col('wc_truth_inFV') & pl.col('wc_truth_0e') & pl.col('wc_truth_2g') & pl.col('wc_truth_0p') & pl.col('wc_truth_0mu')",    "xkcd:salmon", None),
    ("2gNp1mu",     "(pl.col('normal_overlay') | pl.col('del1g_overlay') | pl.col('iso1g_overlay')) & pl.col('wc_truth_inFV') & pl.col('wc_truth_0e') & pl.col('wc_truth_2g') & pl.col('wc_truth_Np') & pl.col('wc_truth_1mu')",    "xkcd:blue", None),
    ("2g0p1mu",     "(pl.col('normal_overlay') | pl.col('del1g_overlay') | pl.col('iso1g_overlay')) & pl.col('wc_truth_inFV') & pl.col('wc_truth_0e') & pl.col('wc_truth_2g') & pl.col('wc_truth_0p') & pl.col('wc_truth_1mu')",    "xkcd:light blue", None),
    ("2g_outFV",    "(pl.col('normal_overlay') | pl.col('del1g_overlay') | pl.col('iso1g_overlay')) & ~pl.col('wc_truth_inFV') & pl.col('wc_truth_0e') & pl.col('wc_truth_2g')",                               "xkcd:bright purple", None),
    ("1eNp",        "(pl.col('normal_overlay') | pl.col('del1g_overlay') | pl.col('iso1g_overlay')) & pl.col('wc_truth_1e') & pl.col('wc_truth_Np')",                                                       "xkcd:seafoam", None),
    ("1e0p",        "(pl.col('normal_overlay') | pl.col('del1g_overlay') | pl.col('iso1g_overlay')) & pl.col('wc_truth_1e') & pl.col('wc_truth_0p')",                                                       "xkcd:electric green", None),
    ("0g",          "(pl.col('normal_overlay') | pl.col('del1g_overlay') | pl.col('iso1g_overlay')) & pl.col('wc_truth_0e') & pl.col('wc_truth_0g')",                                                       "xkcd:gray", None),
    ("3plusg",      "(pl.col('normal_overlay') | pl.col('del1g_overlay') | pl.col('iso1g_overlay')) & pl.col('wc_truth_0e') & pl.col('wc_truth_3plusg')",                                                   "xkcd:beige", None),
    ("dirt",        "pl.col('filetype') == 'dirt_overlay'",                                                                           "xkcd:brown", None),
    ("ext",         "pl.col('filetype') == 'ext'",                                                                                    "xkcd:green", None),
    ("data",        "pl.col('filetype') == 'data'",                                                                                   "xkcd:black", None),
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
    r"data",
]
topological_categories_dic = {topological_category_labels[i]: i for i in range(len(topological_category_labels))}

filetype_categories = [
    ("nc_pi0_overlay", "pl.col('filetype') == 'nc_pi0_overlay'",                 "xkcd:red", None),
    ("nu_overlay",     "pl.col('filetype') == 'nu_overlay'",                     "xkcd:blue", None),
    ("nue_overlay",    "pl.col('filetype') == 'nue_overlay'",                    "xkcd:green", None),
    ("dirt_overlay",   "pl.col('filetype') == 'dirt_overlay'",                   "xkcd:brown", None),
    ("ext",            "pl.col('filetype') == 'ext'",                            "xkcd:bright green", None),
    ("del1g_overlay",  "pl.col('filetype') == 'delete_one_gamma_overlay'",       "xkcd:yellow", "++++"),
    ("iso1g_overlay",  "pl.col('filetype') == 'isotropic_one_gamma_overlay'",    "xkcd:turquoise", "++++"),
    ("data",           "pl.col('filetype') == 'data'",                           "xkcd:black", None),
]
filetype_category_queries = [cat[1] for cat in filetype_categories]
filetype_category_labels = [cat[0] for cat in filetype_categories]
filetype_category_colors = [cat[2] for cat in filetype_categories]
filetype_category_hatches = [cat[3] for cat in filetype_categories]
filetype_category_labels_latex = [cat[0] for cat in filetype_categories]

# TODO: add categories that separate out events with vertex blips and neutron blips
# Maybe use some variable for "true effective vertex blip energy", as a sum of all charged particle energies produced at the vertex after accounting for quenching?
# Add category for photonuclear absorption
