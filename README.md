# MicroBooNE Next-Gen Electromagnetic Shower Search (NGEM)


## Downloading Input Files

```
# kinit -f {YOUR_USERNAME}@FNAL.GOV

bash download_input_files.sh {YOUR_USERNAME} 

```

## Creating Dataframes

```
python src/create_df.py
```

## Training Multi-Class BDT

```
python src/train.py --name first_multiclass_training
```
