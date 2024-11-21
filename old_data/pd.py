import pandas as pd
import re

csv_file = "nutrition.csv"

colonnes_interessees = [
    "name", "calories", "total_fat", "saturated_fat", "polyunsaturated_fatty_acids",
    "protein", "carbohydrate", "sugars", "lactose", "fiber", "calcium", 
    "vitamin_d", "vitamin_e", "vitamin_k", "vitamin_c", "vitamin_a", "vitamin_b12", "vitamin_b6",
]

# Fonction pour nettoyer les valeurs
def nettoyer_valeur(valeur):
    # Supprimer les lettres (unités de mesure) et garder que les chiffres et le point décimal
    if isinstance(valeur, str):
        valeur = re.sub(r'[a-zA-Z]', '', valeur)  # Retirer toutes les lettres
    # Convertir en float si possible
    try:
        return float(valeur) if valeur else 0
    except ValueError:
        return 0

try:
    # Charger le CSV dans un DataFrame pandas
    df = pd.read_csv(csv_file)

    # Filtrer pour ne conserver que les colonnes d'intérêt
    df_filtre = df[colonnes_interessees]
    print(df_filtre)

    for col in df_filtre.columns:
        if col != "name":  # Exclure "name" du nettoyage
            df_filtre[col] = df_filtre[col].apply(nettoyer_valeur)

    df_filtre.fillna(0, inplace=True)

    print("Aperçu des données filtrées :")
    print(df_filtre.head())

    df_filtre.to_csv("filtered_data.csv", index=False)
    print("Les colonnes filtrées ont été enregistrées dans 'donnees_filtrees.csv'.")
except KeyError as e:
    print(f"Erreur : Une ou plusieurs colonnes sont manquantes dans le fichier CSV.\n{e}")
except FileNotFoundError:
    print(f"Le fichier '{csv_file}' est introuvable.")
except Exception as e:
    print(f"Une erreur s'est produite : {e}")
