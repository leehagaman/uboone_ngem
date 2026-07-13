"""Per-run-period POT and trigger reference numbers.

These come from Zarko's getDataInfo.py run over each file's run/subrun list;
see ipynb_notebooks/pot_and_trigger_processing.ipynb for the run/subrun list
generation, the exact getDataInfo.py commands, and the parsed text outputs.
    beam-on data POT          = tor875_wcut
    beam-on data num triggers = E1DCNT_wcut
    EXT num triggers          = EXT

create_df.py imports these dicts to assign each file's POT in _get_file_metadata
and to build the weighting configs in get_weight_configs.
"""

# TODO: Update all of this when we have full data files

# beam-on (open data) POT and trigger counts, by detailed_run_period
open_data_POT = {
    "1":  2.807e19,
    "3":  9.471e18,
    "4a": 2.082e19,
    "4b": 3.731e19,
}
open_data_num_triggers = {
    "1":  6247112,
    "3":  2301101,
    "4a": 4801880,
    "4b": 8523291,
}

# beam-off (EXT) trigger counts, by detailed_run_period
ext_num_triggers = {
    "1":  49551151,
    "2":  152116269,
    "3":  143426455,
    "4a": 27734396,
    "4b": 84966908,
    "4c": 53405830,
    "4d": 76274342,
    "5":  109124930,
}

# Which beam-on data period's POT/trigger ratio to use to convert each EXT
# period's trigger count into a POT-equivalent.  Run 2 has no open data (its EXT
# is paired with run 3 open data); runs 4c/4d/5 are paired with run 4b open data.
# will change when we have full data files
ext_pot_normalizing_period = {
    "1": "1", 
    "2": "3",
    "3": "3", 
    "4a": "4a",
    "4b": "4b", 
    "4c": "4b", 
    "4d": "4b", 
    "5": "4b",
}

# Expected full runs-1-5 dataset POT per run period, used for the "full
# prediction" weighting config (overlays normalized to the projected final data
# POT in each period rather than to the small open-data subset).  Keys are
# normalizing_run_periods of the full_pred config: "1", "2", "3", "4a", "4nota"
# (runs 4b/4c/4d/4bcd combined), "5".
# from DocDB 41725
# will change when we have full data files
expected_full_dataset_data_POT = {
    "1":  1.67e20+5.89e19+8.93e19, # run 1, run 1A open trigger, run 1B open trigger
    "2":  2.61e20,
    "3":  2.57e20,
    "4a": 4.50e19,
    "4nota": 1.36e20+8.95e19+4.93e19, # run 4b, run 4c, run 4d
    "5":  1.48e20,
}
