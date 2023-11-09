import pandas as pd
import numpy as np
import cv2 as cv

dataset="fashion"
datatype="test"
ids=[1877, 2516, 2707, 2910, 3760, 3918, 6027, 8347, 8582, 9368]
print(dataset,datatype)
df=pd.read_csv(f"./{dataset}_{datatype}.csv")
df_N8_mean=df
df_N8_median=df
df_N4_mean=df
df_N4_median=df

for i in ids:
    img=np.zeros((28,28,3))
    img_median_N8=np.zeros((28,28,3))
    img_mean_N8=np.zeros((28,28,3))
    img_median_N4=np.zeros((28,28,3))
    img_mean_N4=np.zeros((28,28,3))

    print(i)
    for (x,y,col) in [(x,y,f"pixel{x*28+y+1}") for x in range(28) for y in range(28)]:
        if df[col][i] not in range(256):
            N8=[]
            for u,v,col in [(x+u,y+v,f"pixel{(x+u)*28+(y+v)+1}") for u in range(-1,2) for v in range(-1,2) if (x+u) in range(28) and (y+v) in range(28)]:
                if df[col][i] in range(256):
                    N8.append(df[col][i])
            N4=[]
            for u,v,col in [(x+u,y+v,f"pixel{(x+u)*28+(y+v)+1}") for u,v in [(0,-1),(-1,0),(0,1),(1,0)] if (x+u) in range(28) and (y+v) in range(28)]:
                if df[col][i] in range(256):
                    N4.append(df[col][i])
            df_N8_mean[col][i]=int(np.mean(N8))
            df_N8_median[col][i]=int(np.median(N8))
            df_N4_mean[col][i]=int(np.mean(N4))
            df_N4_median[col][i]=int(np.median(N4))
            img[x][y]=[0,0,255]
            img_mean_N8[x][y]=[int(np.mean(N8))]*3
            img_median_N8[x][y]=[int(np.median(N8))]*3
            img_mean_N4[x][y]=[int(np.mean(N4))]*3
            img_median_N4[x][y]=[int(np.median(N4))]*3
        else:
            img[x][y]=[df[col][i]]*3
            img_mean_N4[x][y]=[df[col][i]]*3
            img_median_N4[x][y]=[df[col][i]]*3
            img_mean_N8[x][y]=[df[col][i]]*3
            img_median_N8[x][y]=[df[col][i]]*3
    cv.imwrite(f"./image/image_with_missing/{dataset}_{datatype}_{i}.png",np.uint8(img))
    cv.imwrite(f"./image/N4_mean/{dataset}_{datatype}_{i}.png",np.uint8(img_mean_N4))
    cv.imwrite(f"./image/N4_median/{dataset}_{datatype}_{i}.png",np.uint8(img_median_N4))
    cv.imwrite(f"./image/N8_mean/{dataset}_{datatype}_{i}.png",np.uint8(img_mean_N8))
    cv.imwrite(f"./image/N8_median/{dataset}_{datatype}_{i}.png",np.uint8(img_median_N8))

df_N8_mean.to_csv(f"{dataset}_{datatype}_N8_mean.csv",index=False)
df_N8_median.to_csv(f"{dataset}_{datatype}_N8_median.csv",index=False)
df_N4_mean.to_csv(f"{dataset}_{datatype}_N4_mean.csv",index=False)
df_N4_median.to_csv(f"{dataset}_{datatype}_N4_median.csv",index=False)

"-----------------------------------------------------------------"

dataset="mnist"
datatype="test"
ids=[326, 1801, 2389, 3300, 3392, 4122, 5403, 7710, 7928, 9490]
print(dataset,datatype)
df=pd.read_csv(f"./{dataset}_{datatype}.csv")
df_N8_mean=df
df_N8_median=df
df_N4_mean=df
df_N4_median=df

for i in ids:
    img=np.zeros((28,28,3))
    img_median_N8=np.zeros((28,28,3))
    img_mean_N8=np.zeros((28,28,3))
    img_median_N4=np.zeros((28,28,3))
    img_mean_N4=np.zeros((28,28,3))

    print(i)
    for (x,y,col) in [(x,y,f"{x+1}x{y+1}") for x in range(28) for y in range(28)]:
        if df[col][i] not in range(256):
            N8=[]
            for u,v,col in [(x+u,y+v,f"{x+u+1}x{y+v+1}") for u in range(-1,2) for v in range(-1,2) if (x+u) in range(28) and (y+v) in range(28)]:
                if df[col][i] in range(256):
                    N8.append(df[col][i])
            N4=[]
            for u,v,col in [(x+u,y+v,f"{x+u+1}x{y+v+1}") for u,v in [(0,-1),(-1,0),(0,1),(1,0)] if (x+u) in range(28) and (y+v) in range(28)]:
                if df[col][i] in range(256):
                    N4.append(df[col][i])
            df_N8_mean[col][i]=int(np.mean(N8))
            df_N8_median[col][i]=int(np.median(N8))
            df_N4_mean[col][i]=int(np.mean(N4))
            df_N4_median[col][i]=int(np.median(N4))
            img[x][y]=[0,0,255]
            img_mean_N8[x][y]=[int(np.mean(N8))]*3
            img_median_N8[x][y]=[int(np.median(N8))]*3
            img_mean_N4[x][y]=[int(np.mean(N4))]*3
            img_median_N4[x][y]=[int(np.median(N4))]*3
        else:
            img[x][y]=[df[col][i]]*3
            img_mean_N4[x][y]=[df[col][i]]*3
            img_median_N4[x][y]=[df[col][i]]*3
            img_mean_N8[x][y]=[df[col][i]]*3
            img_median_N8[x][y]=[df[col][i]]*3
    cv.imwrite(f"./image/image_with_missing/{dataset}_{datatype}_{i}.png",np.uint8(img))
    cv.imwrite(f"./image/N4_mean/{dataset}_{datatype}_{i}.png",np.uint8(img_mean_N4))
    cv.imwrite(f"./image/N4_median/{dataset}_{datatype}_{i}.png",np.uint8(img_median_N4))
    cv.imwrite(f"./image/N8_mean/{dataset}_{datatype}_{i}.png",np.uint8(img_mean_N8))
    cv.imwrite(f"./image/N8_median/{dataset}_{datatype}_{i}.png",np.uint8(img_median_N8))

df_N8_mean.to_csv(f"{dataset}_{datatype}_N8_mean.csv",index=False)
df_N8_median.to_csv(f"{dataset}_{datatype}_N8_median.csv",index=False)
df_N4_mean.to_csv(f"{dataset}_{datatype}_N4_mean.csv",index=False)
df_N4_median.to_csv(f"{dataset}_{datatype}_N4_median.csv",index=False)
