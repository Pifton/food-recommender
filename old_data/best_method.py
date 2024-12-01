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

def prepare_data(file, preprocesser, decomposer, params=None):
    np.set_printoptions(threshold=sys.maxsize)
    warnings.filterwarnings('ignore')
    data = pd.read_csv(file, sep=',', quotechar='"')
    values = data.iloc[:, 4:].values
    # Test pour differents scalers
    if preprocesser == "StandardScaler":
        values_scaled = StandardScaler().fit_transform(values)
    elif preprocesser == "MinMaxScaler":
        values_scaled = MinMaxScaler().fit_transform(values)
    elif preprocesser == "RobustScaler":
        values_scaled = RobustScaler().fit_transform(values)
    # Test pour differents decomposers
    # print(values_scaled)
    if decomposer == "PCA":
        pca = PCA(**params)
        values_pca = pca.fit_transform(values_scaled)
        # print(values_pca)
    elif decomposer == "KernelPCA":
        pca = KernelPCA(**params)
        values_pca = pca.fit_transform(values_scaled)
        # print(values_pca)

    labels = data.iloc[:, :3].values

    return values, values_pca, labels

def clustering_score(data, k_values):
    results = {"k": [], "Silhouette": [], "Inertia": []}
    silhouette_values_per_k = []
    cluster_labels_per_k = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)

        # Silhouette Score
        silhouette = silhouette_score(data, kmeans.labels_)
        results["Silhouette"].append(silhouette)

        silhouette_values_per_k.append(silhouette_samples(data, kmeans.labels_))
        cluster_labels_per_k.append(kmeans.labels_)

        # Inertia
        inertia = kmeans.inertia_
        results["Inertia"].append(inertia)

        # print("k: ", k, "silhouette: ", silhouette, "inertia: ", inertia)
    print("### Silhouette ###")
    print(results["Silhouette"])
    print("### Inertia ###")
    print(results["Inertia"])

    return results, silhouette_values_per_k, cluster_labels_per_k

def check_scores(data, k_values):
    scalers = ["StandardScaler", "MinMaxScaler", "RobustScaler"]
    # scalers = ["RobustScaler"]
    decomposers = [
        ("PCA", {"n_components": 0.9}),
        ("PCA", {"n_components": 0.95}),
        ("PCA", {"n_components": 0.99}),
        ("PCA", {"n_components": 5}),
        ("PCA", {"n_components": 10}),
        ("KernelPCA", {"kernel": "rbf", "gamma": 0.1}),
        ("KernelPCA", {"kernel": "rbf", "gamma": 0.5}),
        ("KernelPCA", {"kernel": "poly", "degree": 2}),
        ("KernelPCA", {"kernel": "poly", "degree": 3}),
        ("KernelPCA", {"kernel": "poly", "degree": 4}),
        ("KernelPCA", {"kernel": "poly", "degree": 5}),
        ("KernelPCA", {"kernel": "cosine"}),

        # ne fonctionne pas
        ("KernelPCA", {"kernel": "sigmoid", "gamma": 0.5}),
        
    ]
    saved_results = {}

    for scaler in scalers:
        print(f"\n##### {scaler} #####\n")
        for decomposer, params in decomposers:
            print(f"\n    ##### {decomposer} #####")
            print("Parameters: ", params)
            values, values_pca, labels = prepare_data(data, scaler, decomposer, params=params)

            results, silhouette_values_per_k, cluster_labels_per_k= clustering_score(values_pca, k_values)
            # print(silhouette_values_per_k)
            saved_results[f"{scaler}, {decomposer}"] = results

            title = f"{scaler}, {decomposer}, {params}"
            # draw_graph(k_values, results["Inertia"], results["Silhouette"], title)

def draw_graph(k_values, iniertia_score, silhouette_score, title):
    plt.figure(figsize=(15, 6))


    plt.subplot(1, 2, 1)
    plt.plot(k_values, iniertia_score, marker="o", label="Inertia")
    plt.title("Inertia Score (Elbow method)")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia Score")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(range(min(k_values), max(k_values) + 1, 1))

    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_score, marker="o", label="Silhouette")
    plt.title("Silhouette Score")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(range(min(k_values), max(k_values) + 1, 1))

    plt.suptitle(title, fontsize=16, fontweight="bold")

    plt.savefig(f"./optimal_clusters/{title.replace(' ', '_')}.png")
    plt.close()

        

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
    plt.show()
    plt.savefig(f"./heatmaps/{title.replace(' ', '_')}.png")
    plt.close()

def main():
    file = "../data/ciqual.csv"

    # k_values = range(2, 12)
    # check_scores(file, k_values)
    
    check_heatmap(file)
    

main()


    