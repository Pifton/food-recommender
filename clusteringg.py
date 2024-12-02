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
    # Calcule la distance euclidienne entre un point donné (selected_index)
    # et tous les autres points dans l'espace transformé (values_pca).
    target_point = values_pca[selected_index]
    # Tableau des distances pour chaque point par rapport au point de référence.
    distances = np.linalg.norm(values_pca - target_point, axis=1)
    return distances

def get_nearest_neighbors(distances, selected_index, n_neighbors=10):
    # Exclure l'aliment sélectionné en lui donnant une valeure infini
    distances[selected_index] = np.inf
    # Index des 10 voisins les plus proches
    neigbor_idx = np.argsort(distances)[:n_neighbors]
    return neigbor_idx

def display_neighbors(data, neigbor_idx, distances):
    print("\nLes 10 aliments les plus proches sont:")
    # Afficher les 10 aliments les plus proches
    # pos correspond a la position de l'aliment dans la liste => ordre plus proche au plus lointain
    # neighbor_idx correspond a l'indexe de l'aliment
    for pos, neighbor_idx in enumerate(neigbor_idx):
        aliment = data.iloc[neighbor_idx]['name']
        distance = distances[pos]
        print(f"{pos + 1}. {aliment} - Distance: {distance:.4f}")
        print("Valeurs nutritionnelles:")
        # Exclure la colonne 'cluster'
        nutrition_values = data.iloc[neighbor_idx, 4:-1]
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
     neigbor_idx_in_cluster = get_nearest_neighbors(distances, selected_index_in_cluster)

     # Obtenir les indices originaux des voisins
     neigbor_idx = same_cluster_indices[neigbor_idx_in_cluster]

     # Afficher les 10 voisins les plus proches
     display_neighbors(data, neigbor_idx, distances[neigbor_idx_in_cluster])
