
topological_categories = [
    ("1gNp",        "normal_overlay and wc_truth_inFV and wc_truth_0e and wc_truth_1g and wc_truth_Np and wc_truth_0mu",    "xkcd:yellow"),
    ("1g0p",        "normal_overlay and wc_truth_inFV and wc_truth_0e and wc_truth_1g and wc_truth_0p and wc_truth_0mu",    "xkcd:orange"),
    ("1gNp1mu",     "normal_overlay and wc_truth_inFV and wc_truth_0e and wc_truth_1g and wc_truth_Np and wc_truth_1mu",    "xkcd:cyan"),
    ("1g0p1mu",     "normal_overlay and wc_truth_inFV and wc_truth_0e and wc_truth_1g and wc_truth_0p and wc_truth_1mu",    "xkcd:aqua"),
    ("1g_outFV",    "normal_overlay and not (wc_truth_inFV) and wc_truth_0e and wc_truth_1g",                               "xkcd:pink"),
    ("2gNp",        "normal_overlay and wc_truth_inFV and wc_truth_0e and wc_truth_2g and wc_truth_Np and wc_truth_0mu",    "xkcd:red"),
    ("2g0p",        "normal_overlay and wc_truth_inFV and wc_truth_0e and wc_truth_2g and wc_truth_0p and wc_truth_0mu",    "xkcd:salmon"),
    ("2gNp1mu",     "normal_overlay and wc_truth_inFV and wc_truth_0e and wc_truth_2g and wc_truth_Np and wc_truth_1mu",    "xkcd:blue"),
    ("2g0p1mu",     "normal_overlay and wc_truth_inFV and wc_truth_0e and wc_truth_2g and wc_truth_0p and wc_truth_1mu",    "xkcd:lightblue"),
    ("2g_outFV",    "normal_overlay and not (wc_truth_inFV) and wc_truth_0e and wc_truth_2g",                               "xkcd:bright purple"),
    ("1eNp",        "normal_overlay and wc_truth_1e and wc_truth_Np",                                                       "xkcd:seafoam"),
    ("1e0p",        "normal_overlay and wc_truth_1e and wc_truth_0p",                                                       "xkcd:electric green"),
    ("0g",          "normal_overlay and wc_truth_0e and wc_truth_0g",                                                       "xkcd:gray"),
    ("3plusg",      "normal_overlay and wc_truth_0e and wc_truth_3plusg",                                                   "xkcd:beige"),
    ("dirt",        "filetype == 'dirt_overlay'",                                                                           "xkcd:brown"),
    ("ext",         "filetype == 'ext'",                                                                                    "xkcd:green"),
]
topological_category_queries = [cat[1] for cat in topological_categories]
topological_category_labels = [cat[0] for cat in topological_categories]
topological_category_colors = [cat[2] for cat in topological_categories]

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

physics_categories = [
    ("NCDeltaRad_1gNp", "normal_overlay and wc_truth_inFV and wc_truth_NCDeltaRad and wc_truth_0pi0 and wc_truth_Np",                                           "xkcd:yellow"),
    ("NCDeltaRad_1g0p", "normal_overlay and wc_truth_inFV and wc_truth_NCDeltaRad and wc_truth_0pi0 and wc_truth_0p",                                           "xkcd:orange"),
    ("NC1pi0_Np",       "normal_overlay and wc_truth_inFV and wc_truth_isNC and not wc_truth_NCDeltaRad and wc_truth_1pi0 and wc_truth_Np and wc_truth_0mu",    "xkcd:red"),
    ("NC1pi0_0p",       "normal_overlay and wc_truth_inFV and wc_truth_isNC and not wc_truth_NCDeltaRad and wc_truth_1pi0 and wc_truth_0p and wc_truth_0mu",    "xkcd:salmon"),
    ("numuCC1pi0_Np",   "normal_overlay and wc_truth_inFV and wc_truth_numuCC and wc_truth_1pi0 and wc_truth_Np",                                               "xkcd:cyan"),
    ("numuCC1pi0_0p",   "normal_overlay and wc_truth_inFV and wc_truth_numuCC and wc_truth_1pi0 and wc_truth_0p",                                               "xkcd:aqua"),
    ("nueCC_Np",        "normal_overlay and wc_truth_inFV and wc_truth_nueCC and wc_truth_Np",                                                                  "xkcd:seafoam"),
    ("nueCC_0p",        "normal_overlay and wc_truth_inFV and wc_truth_nueCC and wc_truth_0p",                                                                  "xkcd:electric green"),
    ("multi_pi0",       "normal_overlay and wc_truth_inFV and wc_truth_notnueCC and (wc_truth_multi_pi0 or (wc_truth_1pi0 and wc_truth_NCDeltaRad))",           "xkcd:blue"), # also includes pi0 + NC Delta radiative
    ("0pi0",            "normal_overlay and wc_truth_inFV and wc_truth_notnueCC and wc_truth_0pi0 and not (wc_truth_inFV and wc_truth_NCDeltaRad)",             "xkcd:lightblue"),
    ("1pi0_outFV",      "normal_overlay and not (wc_truth_inFV) and wc_truth_1pi0",                                                                             "xkcd:pink"),
    ("other_outFV",     "normal_overlay and not (wc_truth_inFV) and not (wc_truth_1pi0)",                                                                       "xkcd:bright purple"),
    ("dirt",            "filetype == 'dirt_overlay'",                                                                                                           "xkcd:brown"),
    ("ext",             "filetype == 'ext'",                                                                                                                    "xkcd:green"),
]
physics_category_queries = [cat[1] for cat in physics_categories]
physics_category_labels = [cat[0] for cat in physics_categories]
physics_category_colors = [cat[2] for cat in physics_categories]


filetype_categories = [
    ("nc_pi0_overlay", "filetype == 'nc_pi0_overlay'", "xkcd:red"),
    ("nu_overlay",     "filetype == 'nu_overlay'",     "xkcd:blue"),
    ("dirt_overlay",   "filetype == 'dirt_overlay'",   "xkcd:green"),
    ("ext",            "filetype == 'ext'",            "xkcd:yellow"),
]
filetype_category_queries = [cat[1] for cat in filetype_categories]
filetype_category_labels = [cat[0] for cat in filetype_categories]
filetype_category_colors = [cat[2] for cat in filetype_categories]


# TODO: add categories that separate out events with vertex blips and neutron blips
# Maybe use some variable for "true effective vertex blip energy", as a sum of all charged particle energies produced at the vertex after accounting for quenching?
# Add category for photonuclear absorption
