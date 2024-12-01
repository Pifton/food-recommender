import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib
import sys
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler , MinMaxScaler, RobustScaler
from data_prep import prepare_data

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

     #    silhouette_values_per_k.append(silhouette_samples(data, kmeans.labels_))
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

     #    # ne fonctionne pas
     #    ("KernelPCA", {"kernel": "sigmoid", "gamma": 0.5}),
        
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
    plt.show()

#     plt.savefig(f"./optimal_clusters/{title.replace(' ', '_')}.png")
    plt.close()
