import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib
import sys
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from matplotlib.colors import SymLogNorm
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler , MinMaxScaler, RobustScaler
from matplotlib.ticker import FuncFormatter


def compute_distances(values_pca, selected_index):
    target_point = values_pca[selected_index]
    distances = np.linalg.norm(values_pca - target_point, axis=1)
    return distances

def get_nearest_neighbors(distances, selected_index, n_neighbors=10):
    distances[selected_index] = np.inf  # Exclure l'aliment sélectionné
    nearest_indices = np.argsort(distances)[:n_neighbors]
    return nearest_indices

def display_neighbors(data, nearest_indices, distances):
    print("\nLes 10 aliments les plus proches sont:")
    for idx, neighbor_idx in enumerate(nearest_indices):
        aliment = data.iloc[neighbor_idx]['name']
        distance = distances[idx]
        print(f"{idx + 1}. {aliment} - Distance: {distance:.4f}")
        print("Valeurs nutritionnelles:")
        nutrition_values = data.iloc[neighbor_idx, 4:-1]  # Exclure la colonne 'cluster'
        print(nutrition_values.to_string())
        print("---------------------------")

def search_cluster(data, n_clusters, values_pca, selected_item):
     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
     kmeans.fit(values_pca)

     kmeans_labels = kmeans.labels_

     data['cluster'] = kmeans_labels

     item_cluster = data.loc[selected_item, 'cluster']

     # Filtrer les données pour ne garder que les aliments du même cluster
     same_cluster_indices = data[data['cluster'] == item_cluster].index

     # Extraire les valeurs PCA des aliments du même cluster
     values_pca_same_cluster = values_pca[same_cluster_indices]

     # Recalculer l'index de l'aliment sélectionné dans le sous-ensemble
     selected_index_in_cluster = np.where(same_cluster_indices == selected_item)[0][0]

     # Calcul des distances dans le même cluster
     distances = compute_distances(values_pca_same_cluster, selected_index_in_cluster)

     # Trouver les voisins les plus proches dans le même cluster
     nearest_indices_in_cluster = get_nearest_neighbors(distances, selected_index_in_cluster)

     # Obtenir les indices originaux des voisins
     nearest_indices = same_cluster_indices[nearest_indices_in_cluster]

     # Afficher les 10 voisins les plus proches
     display_neighbors(data, nearest_indices, distances[nearest_indices_in_cluster])
