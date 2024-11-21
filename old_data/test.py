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


np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings('ignore')

data = pd.read_csv('filtered_data.csv', sep=';')
# print(data)

X = data.iloc[:, 1:13].values
labels = data.iloc[:, 0].values

# print(X)
# print(labels)

Y, y = make_blobs(random_state=42)

value = [0,0]

for i in np.arange(2, 20):
    kmeans = KMeans(n_clusters=i).fit_predict(Y)
    if value[0] < silhouette_score(Y, kmeans, metric='euclidean'):
        value[0] = silhouette_score(Y, kmeans, metric='euclidean')
        value[1] = i
    print(silhouette_score(Y, kmeans, metric='euclidean'))
    print()
print(value)

kmeans = KMeans(n_clusters=value[1])
clustering = kmeans.fit_predict(Y)



print(kmeans)

PCA = PCA(n_components=2)

X_pca = PCA.fit_transform(Y)

# colors = ['red','yellow','blue','pink']

# K-means
plt.scatter(X_pca[:, 0], X_pca[:, 1], c= clustering)

for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
    plt.show()

# Agglomerative
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c= aglom, cmap=matplotlib.colors.ListedColormap(colors))

# for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
#     plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
#     plt.show()