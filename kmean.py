import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
def load_data(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        print(f"Fichier {filename} introuvable.")
        exit()

# Préparation des données
def prepare_data(data):
    encoder = OneHotEncoder()
    encoded_cols = encoder.fit_transform(data[['alim_grp_nom_fr', 'alim_ssgrp_nom_fr', 'alim_ssssgrp_nom_fr']].fillna('Unknown')).toarray()
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out())
    
    # Colonnes nutritionnelles
    numeric_cols = ['calories', 'total_fat', 'saturated_fat', 'polyunsaturated_fatty_acids', 
                    'monounsaturated_fatty_acids', 'protein', 'carbohydrate', 'sugars', 'lactose', 
                    'fiber', 'calcium', 'vitamin_a', 'vitamin_b5', 'vitamin_b6', 'vitamin_b12', 
                    'vitamin_d', 'vitamin_c', 'vitamin_e', 'vitamin_k', 'unsaturated_fat']
    
    # Remplissage des valeurs manquantes
    numeric_data = data[numeric_cols].fillna(0)
    scaler = StandardScaler()
    scaled_numeric_data = scaler.fit_transform(numeric_data)
    scaled_df = pd.DataFrame(scaled_numeric_data, columns=numeric_cols)
    
    # Combiner les données finales
    final_df = pd.concat([data[['name']], scaled_df, encoded_df], axis=1)
    return final_df, scaler, numeric_cols

# Déterminer le nombre optimal de clusters
def find_optimal_k(data, max_k=10):
    inertias = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))
    
    # Visualisation
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(k_values, inertias, 'g-', label='Inertie')
    ax2.plot(k_values, silhouette_scores, 'b-', label='Indice de silhouette')
    ax1.set_xlabel('Nombre de clusters (k)')
    ax1.set_ylabel('Inertie', color='g')
    ax2.set_ylabel('Silhouette', color='b')
    plt.title('Méthode du coude et silhouette')
    fig.tight_layout()
    plt.show()
    
    while True:
        try:
            optimal_k = int(input("Entrez le nombre optimal de clusters selon les graphiques : "))
            if 2 <= optimal_k <= max_k:
                return optimal_k
            print(f"Veuillez entrer un nombre entre 2 et {max_k}.")
        except ValueError:
            print("Entrée non valide. Veuillez entrer un entier.")

# Appliquer le clustering
def apply_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data.drop(columns=['name']))
    return data, kmeans

# Rechercher un aliment spécifique
def search_food(data, query):
    results = data[data['name'].str.contains(query, case=False, na=False)]
    if results.empty:
        print("Aucun aliment trouvé.")
        return None
    print("Résultats trouvés :")
    print(results[['name']])
    
    try:
        choice = int(input("Entrez l'index de l'aliment : "))
        if choice in results.index:
            return data.loc[choice]
        print("Index non valide.")
        return None
    except ValueError:
        print("Entrée non valide.")
        return None

# Restaurer les valeurs originales des colonnes sélectionnées
def restore_original_values(data, scaler, columns):
    data_restored = data.copy()
    data_restored[columns] = scaler.inverse_transform(data[columns])
    return data_restored

# Trouver les 10 plus proches voisins
def find_neighbors(data, selected_row, scaler, columns, n_neighbors=10):
    neighbors_model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean')
    neighbors_model.fit(data[columns])  # Ajustement uniquement sur les colonnes nutritionnelles
    distances, indices = neighbors_model.kneighbors([selected_row[columns].values])
    neighbors = data.iloc[indices[0][1:]]  # Exclure l'aliment lui-même
    
    # Restaurer les données nutritionnelles originales
    restored_neighbors = restore_original_values(neighbors, scaler, columns)
    
    # Affichage des données
    nutrition_order = ['calories', 'protein', 'total_fat', 'saturated_fat', 
                       'polyunsaturated_fatty_acids', 'monounsaturated_fatty_acids']
    print("Les 10 aliments les plus proches :")
    print(restored_neighbors[['name', 'cluster'] + nutrition_order])
    return restored_neighbors

# Programme principal
def main():
    filename = "ciqual.csv"
    data = load_data(filename)
    
    print("Chargement des données terminé.")
    final_data, scaler, nutrition_columns = prepare_data(data)
    
    print("Données préparées.")
    optimal_k = find_optimal_k(final_data.drop(columns=['name']))
    
    clustered_data, kmeans = apply_clustering(final_data, optimal_k)
    
    while True:
        query = input("Recherchez un aliment (ou 'exit' pour quitter) : ")
        if query.lower() == 'exit':
            break
        selected = search_food(clustered_data, query)
        if selected is not None:
            find_neighbors(clustered_data, selected, scaler, nutrition_columns)

# Exécution
if __name__ == "__main__":
    main()
