import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset="fashion"
datatype="train"
for dataset,datatype in [(a,b) for a in ["fashion","mnist"] for b in ["train","test"]]:
    print(dataset,datatype)
    df1=pd.read_csv(f"./{dataset}_{datatype}.csv")
    df_x=df1.drop(labels="label",axis=1)
    X=np.array(df_x)
    df_y=df1[["label"]]
    Y=np.array(df_y)
    X0=X.flatten()
    plt.hist(X0,bins=range(0,257,16),density=True)
    plt.xlabel("Gray levels")
    plt.ylabel("Occurrence rate")
    plt.title(f"Histogram of all {len(X)} {datatype} data of {dataset} dataset")
    plt.savefig(f"./image/gray_level_histogram/{dataset}_{datatype}_histogram.png",dpi=500)
    plt.clf()


