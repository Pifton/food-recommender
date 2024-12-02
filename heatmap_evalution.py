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
from sklearn.preprocessing import StandardScaler , MinMaxScaler, RobustScaler
from matplotlib.ticker import FuncFormatter
from data_prep import prepare_data


def check_heatmap(data):
    method = [
        # Les 3 meilleurs methodes en fonction des score de silhouette et d'inertie
        ('RobustScaler', 'KernelPCA', {'kernel': 'poly', 'degree': 2}, 6),
        ('RobustScaler', 'PCA', {'n_components': 5}, 7),
        ('StandardScaler', 'KernelPCA', {'kernel': 'poly', 'degree': 3}, 4),
    ]

    # Autres methodes paraissant moins optimales mais interessantes
    """('RobustScaler', 'KernelPCA', {'kernel': 'poly', 'degree': 3}, 4),
    ('RobustScaler', 'KernelPCA', {'kernel': 'poly', 'degree': 4}, 4),
    ('RobustScaler', 'KernelPCA', {'kernel': 'poly', 'degree': 5}, 4),
    ('RobustScaler', 'PCA', {'n_components': 0.9}, 6),
    ('RobustScaler', 'PCA', {'n_components': 0.95}, 6),
    ('RobustScaler', 'PCA', {'n_components': 0.99}, 8),
    ('RobustScaler', 'PCA', {'n_components': 10}, 8),

    ('StandardScaler', 'KernelPCA', {'kernel': 'poly', 'degree': 2}, 4),
    ('StandardScaler', 'KernelPCA', {'kernel': 'poly', 'degree': 4}, 4),
    ('StandardScaler', 'KernelPCA', {'kernel': 'poly', 'degree': 5}, 4)"""
    
    for scaler, decomposer, params, n_clusters in method:
        print(f"\n##### {scaler} ##### {decomposer} #### {params} #### {n_clusters}")
        values, values_pca, labels = prepare_data(data, scaler, decomposer, params=params)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(values_pca)

        labels = kmeans.labels_

        file_name = f"{scaler}, {decomposer}, {params}, {n_clusters}"
        title = f"{scaler}, {decomposer}, {params}, N_clusters: {n_clusters}"

        heatmap_graph(values_pca, title, labels, n_clusters, file_name)


def heatmap_graph(data, title, labels, n_clusters, debug_file):
    # Calculer les moyennes par cluster
    clusters_means = []
    for cluster_id in range(n_clusters):
        cluster_data = data[labels == cluster_id]  # Filtrer les données par cluster
        cluster_mean = np.mean(cluster_data, axis=0)  # Moyenne par composante
        clusters_means.append(cluster_mean)

    # Convertir les résultats en DataFrame
    df = pd.DataFrame(
        clusters_means,
        index=[f"K {i+1}" for i in range(n_clusters)],
        columns=[f"{i+1}" for i in range(data.shape[1])]
    )

    # Sauvegarde des données pour débogage
    with open(f"./heatmaps/{debug_file}.txt", "w") as f:
        f.write("=== Debugging Heatmap Data ===\n")
        f.write("Data PCA/KernelPCA (first 10 rows):\n")
        np.savetxt(f, data[:10], fmt="%.4f", delimiter=", ")
        f.write("\nCluster Labels (first 10 labels):\n")
        f.write(", ".join(map(str, labels[:10])) + "\n")
        f.write("\nCluster Means:\n")
        f.write(df.to_string())
        f.write("\n")

    # vmin = np.percentile(df.values, 5)
    # vmax = np.percentile(df.values, 95)

    vmin = df.values.min()  
    vmax = df.values.max()

    print(f"vmin: {vmin}, vmax: {vmax}") 

    # Créer la heatmap
    plt.figure(figsize=(15, 6))
    # sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax), cbar_kws={'label': 'Valeur PCA','format': LogFormatter()})
    print(title)
    if title == "RobustScaler, PCA, {'n_components': 5}, N_clusters: 7":
        ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", norm=SymLogNorm(linthresh=1, vmin=vmin, vmax=vmax, base=10), cbar_kws={'label': 'Valeur PCA'})
    else :
        ax = sns.heatmap(df, cmap="viridis", norm=SymLogNorm(linthresh=1, vmin=vmin, vmax=vmax, base=10), cbar_kws={'label': 'Valeur PCA'})

    # Obtenir le colorbar
    cbar = ax.collections[0].colorbar

    # Définir le formateur personnalisé pour afficher des nombres entiers avec séparateurs d'espaces
    cbar.formatter = FuncFormatter(lambda x, pos: '{:,.0f}'.format(x).replace(',', ' '))
    cbar.update_ticks()
    plt.title(title)
    plt.xlabel("Components")
    plt.ylabel("Clusters")
    plt.tight_layout()
    plt.savefig(f"./heatmaps/{title.replace(' ', '_')}.png")
    plt.show()
    plt.close()