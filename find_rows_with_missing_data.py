import pandas as pd
import numpy as np

dataset="fashion"
datatype="train"
for dataset,datatype in [(a,b) for a in ["fashion","mnist"] for b in ["train","test"]]:
    print(dataset,datatype)
    df=pd.read_csv(f"./{dataset}_{datatype}.csv")
    rows_with_missing_data = df[df.isnull().any(axis=1)]
    count_rows_with_missing_data = len(df[df.isnull().any(axis=1)])
    print(count_rows_with_missing_data,":",list(rows_with_missing_data.index))

'''
result:

fashion train
0 : []
fashion test
10 : [1877, 2516, 2707, 2910, 3760, 3918, 6027, 8347, 8582, 9368]
mnist train
0 : []
mnist test
10 : [326, 1801, 2389, 3300, 3392, 4122, 5403, 7710, 7928, 9490]

'''
