import pandas as pd
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering, 
                             SpectralClustering, MeanShift, AffinityPropagation,
                             MiniBatchKMeans, Birch, BisectingKMeans, OPTICS)
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def apply_clustering(dataset, cluster_model, description):

    cluster_model.fit(dataset.values)
    siluetas = silhouette_score(X=dataset.values, labels=cluster_model.labels_)
    calinski = calinski_harabasz_score(X=dataset.values, labels=cluster_model.labels_)
    davies = davies_bouldin_score(X=dataset.values, labels=cluster_model.labels_)

    row = [description, siluetas, calinski, davies]

    return cluster_model.labels_, row

df_data = pd.read_csv("estimated_properties.csv")

df_values = df_data.drop(columns=["sequence", "Activity"])

matrix_exploration = []

df_labels = pd.DataFrame()

df_labels["Activity"] = df_data["Activity"].values
df_labels["sequence"] = df_data["sequence"].values

# Kmeans
k=8
kmeans_instance = KMeans(n_clusters=8)
description = f"k_means_k_{k}"
labels, row = apply_clustering(df_values, kmeans_instance, description)
    
matrix_exploration.append(row)

df_labels[description] = labels

# Minibatch
mini_batch_kmeans = MiniBatchKMeans(n_clusters=k)
description = f"minibatch_k_means_k_{k}"
labels, row = apply_clustering(df_values, mini_batch_kmeans, description)

matrix_exploration.append(row)

df_labels[description] = labels

# Agglomerative
k=2
agglomerative_instance = AgglomerativeClustering(n_clusters=k)
description = f"Agglomerative_k_{k}"
labels, row = apply_clustering(df_values, agglomerative_instance, description)

matrix_exploration.append(row)

df_labels[description] = labels

# Birch
k=3
birch_instance = Birch(n_clusters=k)
description = f"Birch_k_{k}"
labels, row = apply_clustering(df_values, birch_instance, description)

matrix_exploration.append(row)

df_labels[description] = labels

# Spectral
k=8
spectral_instance = SpectralClustering(n_clusters=k)
description = f"Spectral_k_{k}"
labels, row = apply_clustering(df_values, spectral_instance, description)

matrix_exploration.append(row)

df_labels[description] = labels

# Bisecting
bisecting_kmeans = BisectingKMeans(n_clusters=k)
description = f"Bisecting_k_{k}"
labels, row = apply_clustering(df_values, bisecting_kmeans, description)

matrix_exploration.append(row)

df_labels[description] = labels

# DBSCAN
dbscan_instance = DBSCAN()
description = "DBSCAN"
labels, row = apply_clustering(df_values, dbscan_instance, description)
matrix_exploration.append(row)
df_labels[description] = labels

# optics
optics_instance = OPTICS()
description = "OPTICS"
labels, row = apply_clustering(df_values, optics_instance, description)
matrix_exploration.append(row)
df_labels[description] = labels

# MeanShift
meanshift_instance = MeanShift()
description = "MeanShift"
labels, row = apply_clustering(df_values, meanshift_instance, description)
matrix_exploration.append(row)
df_labels[description] = labels

# AffinityPropagation
AffinityPropagation_instance = AffinityPropagation()
description = "AffinityPropagation"
labels, row = apply_clustering(df_values, AffinityPropagation_instance, description)
matrix_exploration.append(row)
df_labels[description] = labels

df_performances = pd.DataFrame(data=matrix_exploration, columns=["description", "siluetas",
                                                                 "calinski", "davies"])

df_performances.to_csv("performances.csv", index=False)
df_labels.to_csv("labels.csv", index=False)

