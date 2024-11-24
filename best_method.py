import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib
import sys
import sklearn
from sklearn.cluster import KMeans
# from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler , MinMaxScaler, RobustScaler

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
    if decomposer == "PCA":
        pca = PCA(**params)
        values_pca = pca.fit_transform(values_scaled)
        # print(pca.explained_variance_ratio_)
    elif decomposer == "KernelPCA":
        pca = KernelPCA(**params)
        values_pca = pca.fit_transform(values_scaled)
        # print(pca.explained_variance_ratio_)
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
        # ("PCA", {"n_components": 0.9}),
        # ("PCA", {"n_components": 0.95}),
        # ("PCA", {"n_components": 0.99}),
        # ("PCA", {"n_components": 5}),
        # ("PCA", {"n_components": 10}),
        # ("KernelPCA", {"kernel": "rbf", "gamma": 0.1}),
        # ("KernelPCA", {"kernel": "rbf", "gamma": 0.5}),
        # ("KernelPCA", {"kernel": "poly", "degree": 3}),
        # # ("KernelPCA", {"kernel": "sigmoid", "gamma": 0.5}),
        # ("KernelPCA", {"kernel": "cosine"}),

        ("KernelPCA", {"kernel": "poly", "degree": 2}),
        ("KernelPCA", {"kernel": "poly", "degree": 4}),
        ("KernelPCA", {"kernel": "poly", "degree": 5}),
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
            draw_graph(k_values, results["Inertia"], results["Silhouette"], title)



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

    # plt.show(block=False)
    # close after 5 seconds
    # plt.pause(2)
    plt.savefig(f"./optimal_clusters/{title.replace(' ', '_')}.png")
    plt.close()

        

def main():
    k_values = range(2, 28)
    file = "ciqual.csv"
    check_scores(file, k_values)
    

main()



# def silhouette_plot(k_values,silhouette_values_per_k,cluster_labels_per_k, title):
#     plt.figure(figsize=(10, 7))  # Taille du graphique

#     silhouette_values = silhouette_values_per_k[-1]  # Dernier k (car on affiche tout à la fin)
#     cluster_labels = cluster_labels_per_k[-1]
#     num_clusters = len(np.unique(cluster_labels))  # Nombre de clusters pour ce k

#     y_ticks = []  # Liste pour positionner les étiquettes sur l'axe Y
#     y_lower = 0  # Départ de l'axe Y
#     height_per_cluster = 1 / num_clusters  # Hauteur égale pour chaque cluster

#     for i in range(num_clusters):  # Parcourt tous les clusters
#         # Filtre les valeurs de silhouette pour le cluster courant
#         cluster_silhouette_vals = silhouette_values[cluster_labels == i]
#         cluster_silhouette_vals.sort()  # Trie pour un affichage propre

#         # Normalisation de l'axe Y pour que tous les clusters aient la même hauteur
#         y_upper = y_lower + height_per_cluster
#         y_positions = np.linspace(y_lower, y_upper, len(cluster_silhouette_vals))

#         # Remplit l'espace pour le cluster courant
#         plt.fill_betweenx(
#             y_positions,
#             0,
#             cluster_silhouette_vals,
#             alpha=0.7,
#             label=f"Cluster {i}"
#         )

#         # Ajoute la position du texte pour l'étiquette du cluster
#         y_ticks.append((y_lower + y_upper) / 2)  # Position centrale pour le texte

#         y_lower = y_upper  # Passe au prochain segment

#     # Ajoute une ligne verticale pour la moyenne des scores de silhouette
#     avg_silhouette_score = np.mean(silhouette_values)
#     plt.axvline(x=avg_silhouette_score, color="red", linestyle="--", label="Moyenne")

#     # Configuration de l'axe Y
#     plt.yticks(y_ticks, [f"Cluster {i}" for i in range(num_clusters)])
#     plt.gca().invert_yaxis()  # Optionnel : pour inverser les clusters si besoin

#     # Configuration du graphique
#     plt.title(title)
#     plt.xlabel("Silhouette Coefficient Values")
#     plt.ylabel("Cluster Label")
#     plt.legend(loc="best")
#     plt.tight_layout()

#     # Affiche le graphique
#     plt.show()