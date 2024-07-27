import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import os
import csv
import time

# Reading data from the CSV file
start = time.time()
script_dir = os.path.dirname(__file__)
fileA =os.path.join(script_dir,  'Data', 'Point.csv')
fileB =os.path.join(script_dir,  'Data', 'Point_kmeans.csv')
data = pd.read_csv(fileA)

# Selecting the columns relevant to the K-Means algorithm
X = data[['X', 'Y', 'Z']]
ids = data['ID']  # Extracting the IDs

# Setting the number of clusters, K
K = 500

# Running the K-Means algorithm
kmeans = KMeans(n_clusters=K)
kmeans.fit(X)

# Getting the labels and cluster centers
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Finding the IDs of the selected centroids
centroid_ids = []
for center in cluster_centers:
    distances = np.linalg.norm(X - center, axis=1)
    centroid_id = ids.iloc[np.argmin(distances)]
    centroid_ids.append(centroid_id)

# Printing the cluster assignments and their corresponding centroid IDs
print("Cluster Assignments:", labels)
print("Centroid IDs:", centroid_ids)

# Adding the 'cluster' column to the original dataframe
list_cluster_asso=[centroid_ids[i] for i in kmeans.labels_]
data['cluster'] = list_cluster_asso

# Writing the new dataframe to a CSV file
data.to_csv(fileB, index=False)
end = time.time()
print(f"Time taken: {(end - start) }seconds")