# MicroBooNE Next-Gen Electromagnetic Shower Search (NGEM)


## Downloading Input Files
```
# kinit -f {YOUR_USERNAME}@FNAL.GOV

source download_input_files.sh {YOUR_USERNAME} {LOCATION_WHERE_YOU_WANT_DATA_FILES}
nohup bash download_input_files.sh lhagaman /nevis/riverside/data/leehagaman/ngem/data_files > download_nohup.out &
```


## Python environment
```
uv pip install numpy pandas matplotlib uproot umap-learn tqdm xgboost nbconvert polars root
```

To avoid making ipynb plots visible on github:

```
git config --global filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
```


## Creating Dataframes
You can add --frac_events (-f) 0.01 to load only 1% of the events from each file, making this faster (and less RAM consuming) for small tests. You can also add --just_one_file to only process one file for small tests.

You can also add -m to add memory logging printouts, to check if you're running out of memory.

This takes a bit of time, might want to run it in the background with nohup or tmux.

```
python src/create_df.py -f 0.01 --create_file_dfs
python src/create_df.py -f 0.01 --merge_file_dfs

nohup python -u src/create_df.py -m --create_file_dfs > create_file_dfs_nohup.out 2>&1 &
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


## Training Multi-Class BDT

```
nohup python -u src/train.py --name all_vars > train_nohup.out 2>&1 &

python src/train.py --name all_vars_small
```

## Creating Many Plots

```
python src/plot_many_variables.py --num_plots 3

nohup python -u src/plot_many_variables.py > many_plots_nohup.out 2>&1 &

nohup python -u src/plot_many_detvar_variables.py --clear-prev > many_detvar_plots_nohup.out 2>&1 &

# Sometimes polars causes segfaults? Can check using dmesg.
```

