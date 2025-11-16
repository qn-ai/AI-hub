import pandas as pd

# your dataframe: df

# 1. Find all columns starting with "y_"
y_cols = [col for col in df.columns if col.lower().startswith("y_")]

# 2. Select only y_ columns with NO missing values
y_cols_no_missing = [col for col in y_cols if df[col].notna().all()]

# 3. Build a new dataframe:
#    - keep all original columns
#    - but filter y_ columns to only those with no missing
other_cols = [col for col in df.columns if col not in y_cols]

new_df = df[other_cols + y_cols_no_missing]

# Optional: print them
print("Total y_ columns:", len(y_cols))
print("y_ columns with NO missing:", len(y_cols_no_missing))
print(y_cols_no_missing)
