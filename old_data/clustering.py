# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Charger le dataset des valeurs nutritionnelles
# Remplacez 'nutrition_data.csv' par votre fichier ou source de données
data = pd.read_csv('filtered_data.csv')

# Prétraitement des données
# Supposons que les colonnes numériques sont les valeurs nutritionnelles que nous allons utiliser pour le clustering
nutrition_features = data.select_dtypes(include=[np.number])
scaler = StandardScaler()
nutrition_scaled = scaler.fit_transform(nutrition_features)

# Appliquer l'algorithme de K-means pour le clustering
kmeans = KMeans(n_clusters=8, random_state=0)  # n_clusters à ajuster selon vos besoins
data['cluster'] = kmeans.fit_predict(nutrition_scaled)

# Représentation graphique des clusters (en supposant 2D avec PCA si nécessaire)
plt.figure(figsize=(10, 7))
plt.scatter(nutrition_scaled[:, 0], nutrition_scaled[:, 1], c=data['cluster'], cmap='viridis', marker='o')
plt.xlabel('Nutrition Feature 1')
plt.ylabel('Nutrition Feature 2')
plt.title('Clustering des Aliments (K-Means)')
plt.colorbar(label='Cluster')
plt.show()

# Évaluer les métriques d’apprentissage
# Exemple de calculs que vous pouvez tester en commentaire pour ajuster les paramètres de clustering
# Silhouette score
silhouette_avg = silhouette_score(nutrition_scaled, data['cluster'])
print(f"Silhouette Score: {silhouette_avg}")

# Davies-Bouldin score
db_score = davies_bouldin_score(nutrition_scaled, data['cluster'])
print(f"Davies-Bouldin Score: {db_score}")

# Trouver les 10 voisins les plus proches d'un aliment donné
# Remplacez 'sample_index' par l'index de l'aliment pour lequel vous voulez des alternatives
sample_index = 15  # Exemple, aliment à la position 0 du dataset
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(nutrition_scaled)
distances, indices = nbrs.kneighbors([nutrition_scaled[sample_index]])

# Afficher les indices des voisins les plus proches et leurs distances
print("10 voisins les plus proches de l'aliment donné:")
for i, index in enumerate(indices[0]):
    print(f"Aliment {index} avec distance {distances[0][i]}")

# Notez: Vous pouvez affiner k-means en ajustant 'n_clusters' et comparer les scores de silhouette ou Davies-Bouldin.
