import pandas as pd
import numpy as np
import cv2 as cv

# a function that returns column name for a pixel
def col_name(dataset:str,x:int,y:int)->str:
    if dataset=="fashion":
        return f"pixel{x*28+y+1}"
    else:
        return f"{x+1}x{y+1}"

# images with missing values
ids={
    "fashion":[1877, 2516, 2707, 2910, 3760, 3918, 6027, 8347, 8582, 9368],
    "mnist":[326, 1801, 2389, 3300, 3392, 4122, 5403, 7710, 7928, 9490]
}

for dataset in ["fashion","mnist"]:
    print("running",dataset,"test data ...")
    df=pd.read_csv(f"./{dataset}_test.csv")
    df_drop=df.drop(ids[dataset])
    df_mean=df
    df_median=df
    col_means={col:int(df[col].mean()) for col in df.columns}
    col_medians={col:int(df[col].median()) for col in df.columns}

    for i in ids[dataset]:
        img_col_mean=np.zeros((28,28,3))
        img_col_median=np.zeros((28,28,3))
        print("running image",i)
        for (x,y,col) in [(x,y,col_name(dataset,x,y)) for x in range(28) for y in range(28)]:
            if df[col][i] not in range(256):
                df_mean[col][i]=col_means[col]
                df_median[col][i]=col_medians[col]
            img_col_mean[x][y]=[df_mean[col][i]]*3
            img_col_median[x][y]=[df_median[col][i]]*3
        cv.imwrite(f"./image/column_mean/{dataset}_test_{i}.png",np.uint8(img_col_mean))
        cv.imwrite(f"./image/column_median/{dataset}_test_{i}.png",np.uint8(img_col_median))

    df_drop.to_csv(f"{dataset}_test_drop.csv",index=False)
    df_mean.to_csv(f"{dataset}_test_col_mean.csv",index=False)
    df_median.to_csv(f"{dataset}_test_col_median.csv",index=False)
