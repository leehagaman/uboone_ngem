# MicroBooNE Next-Gen Electromagnetic Shower Search (NGEM)


## Downloading Input Files
```
# kinit -f {YOUR_USERNAME}@FNAL.GOV

source download_input_files.sh {YOUR_USERNAME} {LOCATION_WHERE_YOU_WANT_DATA_FILES}
source download_input_files.sh lhagaman /nevis/riverside/data/lhagaman/ngem/data_files
```

## Python environment
```
uv pip install numpy pandas matplotlib uproot umap-learn tqdm xgboost
```

## Creating Dataframes
You can add --frac_events (-f) 0.05 to load only 5% of the events from each file, making this faster (and less RAM consuming) for small tests.

Note that with large root files, this could use a lot of RAM before extra variables are thrown away after pre-processing. On Nevis computing, you can run this step after ssh-ing into hopper.nevis.columbia.edu from houston.nevis.columbia.edu, which has 128 GB of RAM rather than 32 GB. This takes a bit of time, might want to run it in the background.

```
python src/create_df.py
nohup python -u src/create_df.py > nohup.out 2>&1 &
```

## Training Multi-Class BDT
```
python src/train.py --name first_combined_training
python src/train.py --name first_wc_training --training_vars wc
python src/train.py --name first_lantern_training --training_vars lantern

python src/train.py --name first_combined_physics_training --signal_categories physics
```

