import csv

# Nom du fichier CSV (à remplacer par le chemin de votre fichier)
csv_file = "nutrition.csv"
# Nom du fichier texte de sortie
txt_file = "new.txt"

try:
    # Lecture des noms de colonnes
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # Lire la première ligne contenant les titres
        headers = next(reader)

    # Écriture des noms de colonnes dans un fichier texte
    with open(txt_file, mode='w', encoding='utf-8') as file:
        for header in headers:
            file.write(header + '\n')

    print(f"Les noms des colonnes ont été enregistrés dans '{txt_file}'")
except FileNotFoundError:
    print(f"Le fichier '{csv_file}' est introuvable.")
except Exception as e:
    print(f"Une erreur s'est produite : {e}")
