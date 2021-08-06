# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 01:52:15 2021

@author: Amartya
"""
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

air=pd.read_csv("F:\Softwares\Data Science Assignments\Python-Assignment\H-Clustering\\EastWestAirlines.csv")
air.columns
air.isnull().sum()


def norm(i):
    x=((i-i.min())/(i.max()-i.min()))
    return(x)

data=norm(air)

dendogram=sch.dendrogram(sch.linkage(data,method='complete'))
clust=AgglomerativeClustering(n_clusters=6,linkage='complete',affinity="Euclidean").fit(data)
b=clust.labels_

Cluster_Labels=pd.Series(b)

air['Clusters']=Cluster_Labels
air.Clusters.value_counts()


##KMeans Clustering


twss=[]
for i in range(1,21):
    kmeans=KMeans(n_clusters=i,random_state=0)
    kmeans.fit(data)
    twss.append(kmeans.inertia_)

plt.plot(range(1,21),twss)
plt.title("Elbow Curve or Scree Plot")
plt.xlabel("No. of Clusters")
plt.ylabel("Total Within Sum of Squares")

kmeans=KMeans(n_clusters=9,random_state=0)
pred_y=kmeans.fit_predict(data)

K_Cluster=pd.DataFrame(pred_y)
air['K_Cluster']=K_Cluster
air.K_Cluster.value_counts()
air.groupby('K_Cluster').mean()

air.to_csv("Airlines.csv",index=False)
