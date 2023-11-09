import pandas as pd
import cv2 as cv
import numpy as np
import random

def column(x,y,dataset):
    if dataset=="fashion":
        return f"pixel{x*28+y+1}"
    else:
        return f"{x+1}x{y+1}"

for dataset,datatype in [(a,b) for a in ["fashion","mnist"] for b in ["train","test"]]:
    print(dataset,datatype)
    df=pd.read_csv(f"./{dataset}_{datatype}.csv")
    ids=[random.randint(0,len(df)) for i in range(10)]
    for i in ids:
        img=np.zeros((28,28))
        for x,y,col in [(x,y,column(x,y,dataset)) for x in range(28) for y in range(28)]:
            img[x][y]=(df[col][i])
        cv.imwrite(f"image/image_origin/{dataset}_{datatype}_{i}.png",img)
        for x,y,col in ([(x,y,column(x,y,dataset)) for x in range(28) for y in [13,14]]+[(x,y,column(x,y,dataset)) for x in [13,14] for y in range(28)]):
            img[x][y]=255
        cv.imwrite(f"image/image_cross/{dataset}_{datatype}_{i}.png",img)

