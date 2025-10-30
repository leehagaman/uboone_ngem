# MicroBooNE Next-Gen Electromagnetic Shower Search (NGEM)


## Downloading Input Files
```
# kinit -f {YOUR_USERNAME}@FNAL.GOV

source download_input_files.sh {YOUR_USERNAME} {LOCATION_WHERE_YOU_WANT_DATA_FILES}
nohup bash download_input_files.sh lhagaman /nevis/riverside/data/leehagaman/ngem/data_files > download_nohup.out &
```

## Python environment
```
uv pip install numpy pandas matplotlib uproot umap-learn tqdm xgboost nbconvert
```

To avoid making ipynb plots visible on github:

```
git config --global filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
```

## Creating Dataframes
You can add --frac_events (-f) 0.05 to load only 5% of the events from each file, making this faster (and less RAM consuming) for small tests. You can also add -m to add memory logging printouts, to check if you're running out of memory.

This takes a bit of time, might want to run it in the background.

```
python src/create_df.py -f 0.01 -m
python src/create_df.py -f 0.01 -m --just_one_file
nohup python -u src/create_df.py -m > nohup.out 2>&1 &

python src/create_rw_syst_df.py -f 0.01
nohup python -u src/create_rw_syst_df.py -m > weights_nohup.out 2>&1 &


python src/create_rw_syst_df.py -f 0.01 --just_one_file

```

## Training Multi-Class BDT

```
python src/train.py --name first_combined_training
python src/train.py --name first_wc_training --training_vars wc
python src/train.py --name first_lantern_training --training_vars lantern

python src/train.py --name mixed_del1g_iso_training --signal_categories del1g_simple

python src/train.py --name mixed_wc_training --training_vars wc
python src/train.py --name mixed_lantern_training --training_vars lantern

python src/train.py --name nue_only_lantern_training --training_vars lantern --signal_categories nue_only # perfect performance, data/pred difference
python src/train.py --name nue_only_lantern_first_half_training --training_vars lantern_first_half --signal_categories nue_only # perfect performance, data/pred difference
python src/train.py --name nue_only_lantern_key_training --training_vars lantern_key_vars --signal_categories nue_only # perfect performance, data/pred difference
python src/train.py --name nue_only_lantern_key_2_training --training_vars lantern_key_2_vars --signal_categories nue_only # good performance, data/pred consistent
python src/train.py --name nue_only_lantern_key_other_2_training --training_vars lantern_key_other_2_vars --signal_categories nue_only

python src/train.py --name all_vars

python src/train.py --name with_numu_generic_pandora --training_vars pandora
python src/train.py --name with_numu_generic_glee --training_vars glee
python src/train.py --name only_pandora_scalars --training_vars pandora_scalars
python src/train.py --name only_wc_lantern_combined --training_vars only_wc_lantern_combined

python src/train.py --name only_pandora_scalars_first_half --training_vars pandora_scalars_first_half
python src/train.py --name only_pandora_scalars_second_half --training_vars pandora_scalars_second_half


nohup python -u src/train.py --name all_vars > train_nohup.out 2>&1 &

python src/train.py --name all_vars_small

```

