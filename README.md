# MicroBooNE Next-Gen Electromagnetic Shower Search (NGEM)

Some overview slides are available on the MicroBooNE DocDB [here](https://microboone-docdb.fnal.gov/cgi-bin/sso/ShowDocument?docid=46041).


## Downloading Input Files
```
# kinit -f {YOUR_USERNAME}@FNAL.GOV

source download_input_files.sh {YOUR_USERNAME} {LOCATION_WHERE_YOU_WANT_DATA_FILES}
nohup bash download_input_files.sh lhagaman /nevis/riverside/data/leehagaman/ngem/data_files > download_nohup.out &
```


## Python environment
```
uv pip install ipykernel numpy pandas matplotlib uproot umap-learn tqdm xgboost nbconvert polars root
```

To avoid making ipynb plots visible on github:

```
git config --global filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
```

To set up your input and output paths, edit this file to point towards your own directories. If you're using Nevis servers, you can leave the read-only data_files_location as it is.
```
src/file_locations.py
```


## Full Re-processing
To run every step below in one go (create_df, then the rw and detvar systematics in
parallel, then BDT training; stops at the first failure, each step logging to its usual
`*_nohup.out` file):

```
nohup bash run_full_reprocessing.sh all_vars_2026_10_23 > reprocess_nohup.out 2>&1 &
```

The argument is the training name (default `all_vars_<YYYY_MM_DD>`). Add `--no_train` to
skip the BDT training step:

```
nohup bash run_full_reprocessing.sh --no_train > reprocess_nohup.out 2>&1 &
```


## Creating Dataframes
You can add --frac_events (-f) 0.01 to load only 1% of the events from each file, making this faster (and less RAM consuming) for small tests. You can also add --just_one_file to only process one file for small tests.

You can also add -m to add memory logging printouts, to check if you're running out of memory.

This takes a bit of time, might want to run it in the background with nohup or tmux.

```
python src/create_df.py -f 0.01 --create_file_dfs --merge_file_dfs

nohup python -u src/create_df.py -m --create_file_dfs --merge_file_dfs > create_file_dfs_nohup.out 2>&1 &
nohup python -u src/create_df.py -m --merge_file_dfs > merge_file_dfs_nohup.out 2>&1 &

```


## Creating Systematic Dataframes
Similar as above, but now we only load systematic files after WC generic neutrino preselection.

```
python src/create_rw_syst_df.py -f 0.01 --just_one_file
nohup python -u src/create_rw_syst_df.py -m > weights_nohup.out 2>&1 &

python src/create_detvar_df.py -f 0.01
nohup python -u src/create_detvar_df.py -m > detvar_nohup.out 2>&1 &
```

`create_rw_syst_df.py` also writes `spline_weights_df.parquet` (the per-knob spline
weights), read in the same pass from each overlay file's `spline_weights` tree -- this
replaces the old standalone `create_splines_df.py`.


## Training Multi-Class BDT

```
nohup python -u src/train.py --name all_vars_2026_10_22 > train_nohup.out 2>&1 &

# hyperparameter tuning at a fixed learning rate (outputs in hyperparameter_tuning_outputs/; analyze a study with
# ipynb_notebooks/hyperparameter_tuning_analysis.ipynb). A new study first tries the XGBoost defaults, then the
# best hyperparameters of the studies given with --start_from.
nohup python -u src/hyperparameter_tuning.py --study_name hp_2026_10_26 --start_from hyperparameter_tuning_outputs/hp_2026_10_25_best_params.json --nthread 64 --learning_rate 0.3 --max_boost_rounds 1500 --n_trials 50 > hp_nohup.out 2>&1 &

# learning rate scan with the tuned hyperparameters (it prints the src/compare_bdt_trainings.py command to compare them)
nohup bash run_learning_rate_scan.sh hyperparameter_tuning_outputs/hp_2026_10_26_best_params.json > lr_scan_nohup.out 2>&1 &

# train with the tuned hyperparameters (unit event weights by default; add --weighted for the physics
# weights, with hyperparameters tuned with --weighted); --learning_rate overrides the tuned one
nohup python -u src/train.py --name tuned_2026_10_25 --params_json hyperparameter_tuning_outputs/hp_2026_10_25_best_params.json > train_nohup.out 2>&1 &
```

Training uses the `train_test_score` split [0, 0.4) train | [0.4, 0.5) validation (early stopping and
tuning) | [0.5, 1) test. Events never trained on (data, NuWro fake data, fullosc, the rad. corr. sim,
and true NC/numuCC Delta radiative events) are `not_trained`. This `bdt_role` is decided only in
`add_bdt_role_columns` (`src/bdt_training_setup.py`) and written to `predictions.parquet` together with
`bdt_prediction_weight_factor` (0 for train/validation, 2 for test, 1 for not_trained); predictions that
use BDT scores just apply it with `df_helpers.select_bdt_test_events`.
To compare weight modes, use the physics-weighted log loss: `val_physics_weighted_logloss_raw` / `val_physics_weighted_logloss`
(after moving the class priors to the physics priors) of each study's best trial, and for trainings
`physics_weighted_loss_curves.png/.csv` and `physics_weighted_logloss_at_best_iteration` in `training_config.json`.

## Creating PROfit root file
```
python src/save_PROfit_rootfiles.py            # runs 1-5 open data: minimal_withspline_df.root + minimal_detvar_<vartype>_df.root
python src/save_PROfit_rootfiles.py --nuwro    # NuWro fake-data study: minimal_withspline_nuwro_df.root + minimal_detvar_nuwro_<vartype>_df.root
```

## Creating Many Plots

```
python src/plot_many_variables.py --num_plots 3

nohup python -u src/plot_many_variables.py > many_plots_nohup.out 2>&1 &

nohup python -u src/plot_many_detvar_variables.py --clear-prev > many_detvar_plots_nohup.out 2>&1 &

# Sometimes polars causes segfaults? Can check using dmesg.
```

