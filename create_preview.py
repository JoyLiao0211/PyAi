import pandas as pd
import numpy as np
import cv2 as cv

dataset="fashion"
datatype="test"
ids=[1877, 2516, 2707, 2910, 3760, 3918, 6027, 8347, 8582, 9368]
print(dataset,datatype)
df=pd.read_csv(f"./{dataset}_{datatype}.csv")

for i in ids:
    img=np.zeros((28,28,3))
    print(i)
    for (x,y,z) in [(x,y,x*28+y+1) for x in range(28) for y in range(28)]:
        if df[f"pixel{z}"][i] not in range(256):
            print(i,x,y,df[f"pixel{z}"][i])
            img[x][y][2]=255
        else:
            for j in range(3):
                img[x][y][j]=df[f"pixel{z}"][i]
    cv.imwrite(f"./image/image_with_missing/{dataset}_{datatype}_{i}_preview.png",np.uint8(img))

"-----------------------------------------------------------------"

dataset="mnist"
ids=[326, 1801, 2389, 3300, 3392, 4122, 5403, 7710, 7928, 9490]
print(dataset,datatype)
df=pd.read_csv(f"./{dataset}_{datatype}.csv")

for i in ids:
    print(i)
    img=np.zeros((28,28,3))
    for (x,y,z) in [(x,y,x*28+y+1) for x in range(28) for y in range(28)]:
        if df[f"{x+1}x{y+1}"][i] not in range(256):
            print(i,x,y,df[f"{x+1}x{y+1}"][i])
            img[x][y][2]=255
        else:
            for j in range(3):
                img[x][y][j]=df[f"{x+1}x{y+1}"][i]
    cv.imwrite(f"./image/{dataset}_{datatype}_{i}_preview.png",np.uint8(img))