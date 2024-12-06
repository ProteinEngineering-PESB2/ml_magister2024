import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift, AffinityPropagation
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.utils import shuffle

df_data = pd.read_csv("estimated_properties.csv")

df_pos = df_data[df_data["Activity"] == 1]
df_neg = df_data[df_data["Activity"] == 0]
df_neg = shuffle(df_neg, n_samples=len(df_pos), random_state=42)

df_data = pd.concat([df_pos, df_neg])

df_data = shuffle(df_data, n_samples=500, random_state=42)

df_values = df_data.drop(columns=["sequence","Activity"])
df_labels = df_data[["sequence","Activity"]]

pca_instance = PCA(
    n_components=2,
    random_state=42
)

pca_values = pca_instance.fit_transform(df_values.values)

tsne_instance = TSNE(
    n_components=2
)

tsne_values = tsne_instance.fit_transform(df_values.values)

df_pca = pd.DataFrame(data=pca_values, columns=["p0", "p1"])
df_pca["label"] = df_labels["Activity"].values

df_tsne = pd.DataFrame(data=tsne_values, columns=["p0", "p1"])
df_tsne["label"] = df_labels["Activity"].values

sns.scatterplot(data=df_pca, x="p0", y="p1", hue="label")
plt.show()

sns.scatterplot(data=df_tsne, x="p0", y="p1", hue="label")
plt.show()

kmeans_instance = KMeans()
kmeans_instance.fit(df_values.values)

df_cluster = pd.DataFrame(data=df_values.values, columns=df_values.columns)
df_cluster["kmeans_label"] = kmeans_instance.labels_

print(df_cluster["kmeans_label"].value_counts())

df_pca["label_c"] = kmeans_instance.labels_
df_tsne["label_c"] = kmeans_instance.labels_

sns.scatterplot(data=df_pca, x="p0", y="p1", hue="label_c", palette="Set1")
plt.show()

sns.scatterplot(data=df_tsne, x="p0", y="p1", hue="label_c", palette="Set1")
plt.show()

df_cluster["Activity"] = df_labels["Activity"].values

df_cluster.groupby(by=["Activity", "kmeans_label"]).count().to_csv("groupby.csv")

df_cluster.to_csv("clustering_kmeans.csv", index=False)
df_pca.to_csv("pca_df.csv", index=False)
df_tsne.to_csv("tsne_df.csv", index=False)