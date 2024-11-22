import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib
import sys
import sklearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings('ignore')

data = pd.read_csv('ciqual.csv', sep=',', quotechar='"')
# print(data)

values = data.iloc[:, 4:].values
print(values)
values_scaled = StandardScaler().fit_transform(values)


pca = PCA()


values_pca = pca(0.95).fit_transform(values_scaled)



print(pca.explained_variance_ratio_)
labels = data.iloc[:, :3].values
# print(labels)

# V, v = values.make_blobs(random_state=42)



# for k in range(2,12):
#     kmeans = KMeans(n_clusters=k, random_state=42).fit_predict(values)
#     # labels = kmeans.fit_predict(data)
#     # inertias.append(kmeans.inertia_)
#     print(k)
#     print(silhouette_score(values_pca, kmeans))

silhouette_scores = []

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(values_pca)
    score = silhouette_score(values_pca, cluster_labels)
    silhouette_scores.append(score)

# Tracer les scores
plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.title("Silhouette Scores pour différents nombres de clusters")
plt.xlabel("Nombre de clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# # Appliquer le clustering KMeans
# kmeans = KMeans(n_clusters=4, random_state=42)
# y_pred = kmeans.fit_predict(X)

# # Visualisation
# plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='X')
# plt.title("Clustering avec KMeans")
# plt.show()