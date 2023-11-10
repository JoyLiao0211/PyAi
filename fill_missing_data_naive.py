import pandas as pd
import numpy as np
import cv2 as cv

dataset="fashion"
datatype="test"
ids=[1877, 2516, 2707, 2910, 3760, 3918, 6027, 8347, 8582, 9368]
print(dataset,datatype)
df=pd.read_csv(f"./{dataset}_{datatype}.csv")
df_drop=df.drop(ids)
df_mean=df
df_median=df
col_means={col:int(df[col].mean()) for col in df.columns}
col_medians={col:int(df[col].median()) for col in df.columns}

for i in ids:
    img_col_mean=np.zeros((28,28,3))
    img_col_median=np.zeros((28,28,3))
    print(i)
    for (x,y,col) in [(x,y,f"pixel{x*28+y+1}") for x in range(28) for y in range(28)]:
        if df[col][i] not in range(256):
            df_mean[col][i]=col_means[col]
            df_median[col][i]=col_medians[col]
        img_col_mean[x][y]=[df_mean[col][i]]*3
        img_col_median[x][y]=[df_median[col][i]]*3
    cv.imwrite(f"./image/column_mean/{dataset}_{datatype}_{i}.png",np.uint8(img_col_mean))
    cv.imwrite(f"./image/column_median/{dataset}_{datatype}_{i}.png",np.uint8(img_col_median))

df_drop.to_csv(f"{dataset}_{datatype}_drop.csv",index=False)
df_mean.to_csv(f"{dataset}_{datatype}_col_mean.csv",index=False)
df_median.to_csv(f"{dataset}_{datatype}_col_median.csv",index=False)

"-----------------------------------------------------------------"

dataset="mnist"
datatype="test"
ids=[326, 1801, 2389, 3300, 3392, 4122, 5403, 7710, 7928, 9490]
print(dataset,datatype)
df=pd.read_csv(f"./{dataset}_{datatype}.csv")
df_drop=df.drop(ids)
df_mean=df
df_median=df
col_means={col:int(df[col].mean()) for col in df.columns}
col_medians={col:int(df[col].median()) for col in df.columns}

for i in ids:
    img_col_mean=np.zeros((28,28,3))
    img_col_median=np.zeros((28,28,3))
    print(i)
    for (x,y,col) in [(x,y,f"{x+1}x{y+1}") for x in range(28) for y in range(28)]:
        if df[col][i] not in range(256):
            df_mean[col][i]=col_means[col]
            df_median[col][i]=col_medians[col]
        img_col_mean[x][y]=[df_mean[col][i]]*3
        img_col_median[x][y]=[df_median[col][i]]*3
    cv.imwrite(f"./image/column_mean/{dataset}_{datatype}_{i}.png",np.uint8(img_col_mean))
    cv.imwrite(f"./image/column_median/{dataset}_{datatype}_{i}.png",np.uint8(img_col_median))

df_drop.to_csv(f"{dataset}_{datatype}_drop.csv",index=False)
df_mean.to_csv(f"{dataset}_{datatype}_col_mean.csv",index=False)
df_median.to_csv(f"{dataset}_{datatype}_col_median.csv",index=False)
