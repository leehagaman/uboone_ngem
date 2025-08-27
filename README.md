# MicroBooNE Next-Gen Electromagnetic Shower Search (NGEM)


## Downloading Input Files

```
# kinit -f {YOUR_USERNAME}@FNAL.GOV

bash download_input_files.sh {YOUR_USERNAME} 

```

## Creating Dataframes

You can add --frac_events (-f) 0.05 to load only 5% of the events and make this faster for small tests.

```
python src/create_df.py
```

## Training Multi-Class BDT

```
python src/train.py --name first_combined_training
python src/train.py --name first_wc_training --training_vars wc

```
