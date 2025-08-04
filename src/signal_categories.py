
topological_categories = [
    ("1gNp",        "normal_overlay and wc_truth_inFV and wc_truth_1g and wc_truth_Np and wc_truth_0mu",    "xkcd:yellow"),
    ("1g0p",        "normal_overlay and wc_truth_inFV and wc_truth_1g and wc_truth_0p and wc_truth_0mu",    "xkcd:orange"),
    ("1gNp1mu",     "normal_overlay and wc_truth_inFV and wc_truth_1g and wc_truth_Np and wc_truth_1mu",    "xkcd:cyan"),
    ("1g0p1mu",     "normal_overlay and wc_truth_inFV and wc_truth_1g and wc_truth_0p and wc_truth_1mu",    "xkcd:aqua"),
    ("2gNp",        "normal_overlay and wc_truth_inFV and wc_truth_2g and wc_truth_Np and wc_truth_0mu",    "xkcd:red"),
    ("2g0p",        "normal_overlay and wc_truth_inFV and wc_truth_2g and wc_truth_0p and wc_truth_0mu",    "xkcd:salmon"),
    ("2gNp1mu",     "normal_overlay and wc_truth_inFV and wc_truth_2g and wc_truth_Np and wc_truth_1mu",    "xkcd:blue"),
    ("2g0p1mu",     "normal_overlay and wc_truth_inFV and wc_truth_2g and wc_truth_0p and wc_truth_1mu",    "xkcd:lightblue"),
    ("1g_outFV",    "normal_overlay and not (wc_truth_inFV) and wc_truth_1g",                               "xkcd:pink"),
    ("2g_outFV",    "normal_overlay and not (wc_truth_inFV) and wc_truth_2g",                               "xkcd:bright purple"),
    ("0g",          "normal_overlay and wc_truth_0g",                                                       "xkcd:gray"),
    ("3plusg",      "normal_overlay and wc_truth_3plusg",                                                   "xkcd:beige"),
    ("dirt",        "filetype == 'dirt_overlay'",                                                           "xkcd:brown"),
    ("ext",         "filetype == 'ext'",                                                                    "xkcd:green"),
]
topological_category_queries = [cat[1] for cat in topological_categories]
topological_category_labels = [cat[0] for cat in topological_categories]
topological_category_colors = [cat[2] for cat in topological_categories]

physics_categories = [
    ("NCDeltaRad_1gNp", "normal_overlay and wc_truth_inFV and wc_truth_NCDelta == 1 and wc_truth_0pi0 and wc_truth_Np",                     "xkcd:yellow"),
    ("NCDeltaRad_1g0p", "normal_overlay and wc_truth_inFV and wc_truth_NCDelta == 1 and wc_truth_0pi0 and wc_truth_0p",                     "xkcd:orange"),
    ("NC1pi0_Np",       "normal_overlay and wc_truth_inFV and wc_truth_NCDelta == 0 and wc_truth_1pi0 and wc_truth_Np and wc_truth_0mu",    "xkcd:cyan"),
    ("NC1pi0_0p",       "normal_overlay and wc_truth_inFV and wc_truth_NCDelta == 0 and wc_truth_1pi0 and wc_truth_0p and wc_truth_0mu",    "xkcd:aqua"),
    ("numuCC1pi0_Np",   "normal_overlay and wc_truth_inFV and wc_truth_NCDelta == 0 and wc_truth_1pi0 and wc_truth_Np and wc_truth_1mu",    "xkcd:red"),
    ("numuCC1pi0_0p",   "normal_overlay and wc_truth_inFV and wc_truth_NCDelta == 0 and wc_truth_1pi0 and wc_truth_0p and wc_truth_1mu",    "xkcd:salmon"),
    ("multi_pi0",       "normal_overlay and wc_truth_inFV and (wc_truth_multi_pi0 or (wc_truth_1pi0 and wc_truth_NCDelta == 1))",           "xkcd:blue"), # also includes pi0 + NC Delta radiative
    ("0pi0",            "normal_overlay and wc_truth_inFV and wc_truth_0pi0 and not (wc_truth_inFV and wc_truth_NCDelta == 1)",             "xkcd:lightblue"),
    ("1pi0_outFV",      "normal_overlay and not (wc_truth_inFV) and wc_truth_1pi0",                                                         "xkcd:pink"),
    ("other_outFV",     "normal_overlay and not (wc_truth_inFV) and not (wc_truth_1pi0)",                                                   "xkcd:bright purple"),
    ("dirt",            "filetype == 'dirt_overlay'",                                                                                       "xkcd:brown"),
    ("ext",             "filetype == 'ext'",                                                                                                "xkcd:green"),
]
physics_category_queries = [cat[1] for cat in physics_categories]
physics_category_labels = [cat[0] for cat in physics_categories]
physics_category_colors = [cat[2] for cat in physics_categories]
