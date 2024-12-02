from pre_evaluation import check_scores
from heatmap_evalution import check_heatmap
from pretraitement import pretraitement
from food_selection import select_food
from data_prep import prepare_data
from clusteringg import search_cluster

def main():
     # Fonction pour pretraiter notre fichier xls et ainsi supprimer des colonnes inutiles 
     # et le transformer en csv
     '''pretraitement()'''
     # chemin du fichier sur lequel nous réalisons le clustering
     file = "./data/ciqual.csv"
     # test avec différents nombre de clusters
     k_values = range(2, 22)

     # check_scores nous permet de comparer plusieurs methodes de clustering dans le but 
     # de faire un pretraitement des meilleurs methodes
     '''check_scores(file, k_values)'''

     # Check_heatmap nous ert pour evaluer la variance d'un composant entre differents clusters
     # Cela nous permetra donc de realiser un choix final
     '''check_heatmap(file)'''

     # clustering final
     # choix du nombre de clusters
     n_clusters = 7
     # preparation des données
     data, values_pca, labels = prepare_data(file, 'RobustScaler', 'KernelPCA', {'n_components': 5})
     # Selection de l'aliment souhaité
     selected_aliment_idx = select_food(data)

     # retourne les 10 voisins les plus proches
     search_cluster(data, n_clusters, values_pca, selected_aliment_idx)
     # Selection de l'aliment souhaite

main()