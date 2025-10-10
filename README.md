# MicroBooNE Next-Gen Electromagnetic Shower Search (NGEM)


## Downloading Input Files
```
# kinit -f {YOUR_USERNAME}@FNAL.GOV

source download_input_files.sh {YOUR_USERNAME} {LOCATION_WHERE_YOU_WANT_DATA_FILES}
source download_input_files.sh lhagaman /nevis/riverside/data/leehagaman/ngem/data_files
```

## Python environment
```
uv pip install numpy pandas matplotlib uproot umap-learn tqdm xgboost
```

## Creating Dataframes
You can add --frac_events (-f) 0.05 to load only 5% of the events from each file, making this faster (and less RAM consuming) for small tests. You can also add -m to add memory logging printouts, to check if you're running out of memory.

This takes a bit of time, might want to run it in the background.

```
python src/create_df.py -f 0.01
nohup python -u src/create_df.py > nohup.out 2>&1 &
```

## Training Multi-Class BDT

```
python src/train.py --name first_combined_training
python src/train.py --name first_wc_training --training_vars wc
python src/train.py --name first_lantern_training --training_vars lantern

python src/train.py --name mixed_del1g_iso_training --signal_categories del1g_simple

```

