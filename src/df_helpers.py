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
    for i, df in enumerate(dfs):
        # Store original filetype values before any operations
        original_filetype = None
        if "filetype" in df.columns:
            original_filetype = df["filetype"].clone()
            # Validate filetype before processing
            empty_or_null_count_before = df.filter(
                (pl.col("filetype") == '') | pl.col("filetype").is_null()
            ).height
            if empty_or_null_count_before > 0:
                raise ValueError(f"DataFrame {i}: filetype has {empty_or_null_count_before} empty/null values BEFORE alignment!")
        
        missing = all_cols - set(df.columns)
        if missing:
            # Add missing columns as nulls, cast to the desired dtype
            # For string columns, use empty string "" as a placeholder that we'll check for
            defaults = []
            for c in missing:
                defaults.append(pl.lit(None).cast(dtype_map[c]).alias(c))
            df = df.with_columns(defaults)
        
        # Ensure consistent column order
        df = df.select(sorted(all_cols))
        
        # Restore and validate filetype column after all transformations
        if original_filetype is not None:
            # Check if filetype was corrupted during select operation
            empty_or_null_count_after = df.filter(
                (pl.col("filetype") == '') | pl.col("filetype").is_null()
            ).height
            if empty_or_null_count_after > 0:
                # Debug: print the first problematic row
                problem_row = df.filter(
                    (pl.col("filetype") == '') | pl.col("filetype").is_null()
                ).head(1)
                print(f"ERROR in align_columns_for_concat DataFrame {i}: Found {empty_or_null_count_after} rows with empty/null filetype AFTER column operations")
                print(f"Example row columns: {problem_row.columns[:10]}")
                if "filename" in problem_row.columns:
                    print(f"Example filename: {problem_row['filename'][0]}")
                print(f"Original filetype unique values: {original_filetype.unique().to_list()}")
                print(f"After transform filetype unique values: {df['filetype'].unique().to_list()}")
                raise ValueError(f"align_columns_for_concat DataFrame {i}: filetype column was corrupted during column alignment!")
        
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
