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
    print("running",dataset,"test data...")
    df=pd.read_csv(f"./{dataset}_test.csv")
    for i in ids[dataset]:
        # create an empty image with 3 channels: B,G,R
        img=np.zeros((28,28,3))
        print("running image",i)
        # x, y: x and y of the pixel
        # col: the column name
        for (x,y,col) in [(x,y,col_name(x,y)) for x in range(28) for y in range(28)]:
            if df[col][i] not in range(256):
                # mark missing pixel as red
                img[x][y]=[0,0,255]
            else:
                # leave it unchanged
                img[x][y][j]=[df[col][i]]*3
        cv.imwrite(f"./image/image_with_missing/{dataset}_test_{i}_preview.png",np.uint8(img))

