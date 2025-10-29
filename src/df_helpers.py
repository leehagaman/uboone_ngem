import polars as pl

def align_columns_for_concat(dfs):
    # Find all columns across all DataFrames
    all_cols = {col for df in dfs for col in df.columns}

    # Build a mapping of column -> dtype (prefer from first df that has it)
    dtype_map = {}
    for df in dfs:
        for col, dtype in zip(df.columns, df.dtypes):
            dtype_map.setdefault(col, dtype)

    aligned = []
    for df in dfs:
        missing = all_cols - set(df.columns)
        if missing:
            # Add missing columns as nulls, cast to the desired dtype
            defaults = [
                pl.lit(None).cast(dtype_map[c]).alias(c)
                for c in missing
            ]
            df = df.with_columns(defaults)
        # Ensure consistent column order
        df = df.select(sorted(all_cols))
        aligned.append(df)

    return aligned


# compress a pandas dataframe to reduce memory usage
def compress_df(df):

    print("compressing dataframe by changing column dtypes...")

    # ignoring overflow warnings, we expect that occassionally
    """with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        for c in df.select_dtypes("float64"):
            df[c] = df[c].astype("float32")

        for c in df.select_dtypes("int64"):
            df[c] = pd.to_numeric(df[c], downcast="integer")
    """

    for c in df.select_dtypes("object"):
        unique_ratio = df[c].nunique() / len(df)
        if unique_ratio < 0.5:
            df[c] = df[c].astype("category")

    # dropping entirely NA columns to avoid pandas FutureWarning during concat
    #df = df.loc[:, df.notna().any(axis=0)]

    return df
