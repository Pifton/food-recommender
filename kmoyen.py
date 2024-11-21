import csv
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import kmeans

# Fonction pour générer des couleurs aléatoires
def randomColor(nb):
    hex_colors = np.random.choice(list(range(256)), size=(nb, 3))
    colors = ['#%02x%02x%02x' % tuple(color) for color in hex_colors]
    return colors

def lire_csv(nom_fichier):
    aliments = []
    with open(nom_fichier, 'r', newline='', encoding='utf-8') as csvfile:
        lecteur = csv.DictReader(csvfile)
        for ligne in lecteur:
            nom_aliment = ligne['nom']
            type_aliment = ligne['type']
            calories = float(ligne['calories']) if ligne['calories'] else 0
            proteins = float(ligne['proteins']) if ligne['proteins'] else 0
            aliments.append((nom_aliment, type_aliment, calories, proteins))
    return aliments

def afficher_par_cluster(aliments, color, nb_ideal):
    centers = kmeans.definitive_centers(aliments, cluster_result[nb_ideal - 1], nb_ideal)
    clusters = cluster_result[nb_ideal - 1]
    for i in range(len(centers)):
        plt.scatter(centers[i][0], centers[i][1], marker='*', c=color[i], s=500, linewidths=5, edgecolor='k', zorder=10)
    for i in range(len(clusters)):
        plt.scatter(int(aliments[i][2]), int(aliments[i][3]), marker='o', c=color[clusters[i]], s=20)
    plt.title("Vue par cluster")

def afficher_par_type(aliments):
    couleur = {
        "fruit": "#e24091",
        "spice": "#800080",
        "sauce": "#40E0D0",
        "meat": "#C70039",
        "vegetable": "#32CD32",
        "nuts": "#964B00",
        "dairy": "#00FFFF",
        "egg": "#FFFF00",
        "seafood": "#225bb7",
        "cereal": "#e88b00",
        "drink": "#070021",
        "autre": "#cbcbcb"
    }
    legend_colors = {}
    for aliment in aliments:
        if aliment[1] not in legend_colors:
            plt.scatter(aliment[2], aliment[3], marker='o', c=couleur[aliment[1]], s=20, label=aliment[1])
            legend_colors[aliment[1]] = True
        else:
            plt.scatter(aliment[2], aliment[3], marker='o', c=couleur[aliment[1]], s=20)
    plt.title("Vue par type")
    plt.legend(loc='upper right')

def main():
    nom_fichier = 'nutrition.csv'
    aliments = lire_csv(nom_fichier)

    global cluster_result

    sse = []
    center_result = []
    cluster_result = []

    for i in range(1, 10):
        tmp_centre = kmeans.tmp_centre(i, list(aliments))
        cluster = kmeans.cluster_association(aliments, tmp_centre, i)
        center = kmeans.definitive_centers(aliments, cluster, i)
        sse.append(kmeans.calcul_sse(i, cluster, center, aliments))
        center_result.append(center)
        cluster_result.append(cluster)


    nb_ideal = int(input("Entrez le nombre de clusters ideal (entre 1 et 9): "))
    color = randomColor(nb_ideal)
    print("SSE", sse, len(sse))
    #plt.plot(range(1, 10), sse, marker='o', color='b')
    #plt.show()

    plt.figure(figsize=(11, 5))
    
    # Création d'une figure avec deux sous-graphiques
    plt.subplot(1, 2, 1)
    afficher_par_cluster(aliments, color, nb_ideal)
    
    plt.subplot(1, 2, 2)
    afficher_par_type(aliments)
    
    plt.show()

main()