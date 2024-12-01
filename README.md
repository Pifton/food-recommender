# Clustering Alimentaire et recommandations d'aliments

## Table des Matières
- [Description](#description)
- [Structure du Projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Exemple d'Exécution](#exemple-dexécution)

## Description

Ce projet réalise un **clustering des aliments** basé sur leurs valeurs nutritionnelles. Le processus se déroule en plusieurs étapes clés :

1. **Identification d'un dataset** : Identification d'un dataset contenant toutes les données nécessaires
2. **Prétraitement des données** : Suppresion des données non nécessaires et nettoyage des données mal formatées
3. **Évaluation des Méthodes de Clustering** : Utilisation des méthodes du coude et de la silhouette pour déterminer le nombre optimal de clusters et évaluer différentes méthodes de prétraitement et de décomposition.
4. **Sélection des Meilleures Méthodes** : Sélection de **3 méthodes optimales** basées sur les scores de silhouette et d'inertie obtenus.
5. **Génération de Heatmaps** : Création de heatmaps pour évaluer la variance des composants entre les clusters des méthodes sélectionnées.
6. **Clustering Final** : Application de la méthode de clustering choisie avec un nombre défini de clusters.
7. **Recherche des Voisins les Plus Proches** : Permet à l'utilisateur de sélectionner un aliment et de trouver les **10 aliments les plus similaires**.

## Structure du Projet

```bash
tree
├── clusteringg.py #clustering final permettant de trouver les voisins les plus proches de l'aliment sélectionné
├── pre_evaluation.py #permet de créer les graphiques pour identifer les méthodes optimales ainsi que leurs nombres de clusters optimaux
├── pretraitement.py #prétraitement des données (scaler, decomposer, etc)
├── main.py #contient toutes les fonctions nécéssaires à l'execution du programme (en commentaire si besoin)
├── data_prep.py #permet de traiter les données (scaler, decomposer, parameter)
├── food_selection.py #sélectione un aliment et retourne son index
├── heatmap_evalution.py #permet de creer les heatmap pour évaluer la meilleure methode de clustering
├── data/
│   ├── ciqual.csv #dataset prétraité
│   └── Table Ciqual 2020_FR_2020 07 07.xls #dataset original
|
├── optimal_clusters/ #graphiques permettant d'évaluer le nombre optimal de clusters pour chaque méthodes (silhouette score et inertie)
|
├── heatmaps/ #images représentant les heatmaps des méthodes choisis
|
├── old_data/ #anciens dataset et code permettant l'exploitation du dataset actuel
|
└── sujet/ #sujet du projet actuel
```

## Prérequis

- **Python** : Version 3.7 ou supérieure
- **Pip** : Gestionnaire de paquets Python

## Installation

1. **Cloner le dépôt :**
    ```bash
    git clone https://github.com/votre-utilisateur/votre-repo.git
    cd votre-repo
    ```
2. **Installer les dépendances :**
   Dépendances nécessaires
  ```
  numpy // pandas // matplotlib // seaborn // scikit-learn
  ```
  Commande permettant d'installer les dépendances
  ```bash
  pip install 
  ```

## Utilisation

1. **Prétraitement des données, évaluation des méthodes de clustering et génération des heatmaps :**
       - Retourne uniquement les 10 voisins les plus proches actuellement.
       - Décommentez certaines lignes pour réitérer les différentes évaluations (le code est commenté pour les identifier facilement)
    ```bash
    python main.py
    ```

3. **Sélection d'un aliment et recherche des 10 voisins les plus proches :**
       - Lors de l'exécution de `main.py`, suivez les instructions à l'écran.

## Exemple d'Exécution

```bash
user@home:~/Directory/food-recommender$ python3 main.py 
Entrez le nom de l'aliment que vous souhaitez rechercher : 
sardine

Aliments trouvés :
1660: Sardine, grillée
1707: Sardine, crue
1872: Sardine, à l'huile, appertisée, égouttée
1873: Sardine, sauce tomate, appertisée, égouttée
1875: Sardine, à l'huile d'olive, appertisée, égouttée
1893: Sardine, filets sans arêtes à l'huile d'olive, appertisés, égouttés
2925: Huile de sardine

Entrez l'index de l'aliment recherché : 1707

Les 10 aliments les plus proches sont:
1. Hareng fumé, à l'huile - Distance: 2.9136
Valeurs nutritionnelles:
calories                       168.0
total_fat                       11.0
saturated_fat                   3.12
polyunsaturated_fatty_acids     5.48
monounsaturated_fatty_acids     1.95
protein                         15.1
carbohydrate                     2.0
sugars                           0.0
lactose                          0.0
fiber                            0.0
calcium                         59.2
vitamin_a                       12.0
vitamin_b5                      0.41
vitamin_b6                      0.28
vitamin_b12                      9.4
vitamin_d                       14.5
vitamin_c                        0.0
vitamin_e                       0.96
vitamin_k                        0.0
unsaturated_fat                 7.88
---------------------------
