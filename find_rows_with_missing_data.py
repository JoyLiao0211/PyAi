import pandas as pd
import numpy as np

for dataset in ["fashion","mnist"]:
    print("running",dataset,"test data ...")
    df=pd.read_csv(f"./{dataset}_test.csv")
    rows_missing = df[df.isnull().any(axis=1)]
    count_rows = len(df[df.isnull().any(axis=1)])
    print(count_rows,"rows with missing values:\n",list(rows_missing.index))

'''
result:

running fashion test data ...
10 rows with missing values:
[1877, 2516, 2707, 2910, 3760, 3918, 6027, 8347, 8582, 9368]
running mnist test data ...
10 rows with missing values:
[326, 1801, 2389, 3300, 3392, 4122, 5403, 7710, 7928, 9490]

'''
