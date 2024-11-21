import pandas as pd

# Remplacez 'fichier.xlsx' et 'fichier.csv' par vos noms de fichiers
excel_file = "Table Ciqual 2020_FR_2020 07 07.xls"
csv_file = "test.csv"

filtered_colums = {
     "alim_nom_fr": "name",
    "alim_grp_nom_fr": "alim_grp_nom_fr",
    "alim_ssgrp_nom_fr": "alim_ssgrp_nom_fr",
    "alim_ssssgrp_nom_fr": "alim_ssssgrp_nom_fr",
    "Energie, Règlement UE N° 1169/2011 (kcal/100 g)": "calories",
    "Lipides (g/100 g)": "total_fat",
    "AG saturés (g/100 g)": "saturated_fat",
    "AG polyinsaturés (g/100 g)": "polyunsaturated_fatty_acids",
    "Protéines, N x facteur de Jones (g/100 g)": "protein",
    "Glucides (g/100 g)": "carbohydrate",
    "Sucres (g/100 g)": "sugars",
    "Lactose (g/100 g)": "lactose",
    "Fibres alimentaires (g/100 g)": "fiber",
    "Calcium (mg/100 g)": "calcium",
    "Vitamine D (µg/100 g)": "vitamin_d",
    "Vitamine E (mg/100 g)": "vitamin_e",
    "Vitamine K1 (µg/100 g)": "vitamin_k",
    "Vitamine C (mg/100 g)": "vitamin_c",
    "Rétinol (µg/100 g)": "vitamin_a",
    "Vitamine B12 (µg/100 g)": "vitamin_b12",
    "Vitamine B6 (mg/100 g)": "vitamin_b6",
    "Vitamine B5 ou Acide pantothénique (mg/100 g)": "vitamin_b5",
}

# Charger le fichier Excel
new_file = pd.read_excel(excel_file, engine="xlrd")

# Filtrer les colonnes spécifiées
new_file = new_file[list(filtered_colums.keys())]

# Renommer les colonnes
new_file.rename(columns=filtered_colums, inplace=True)

# Supprimer les lignes contenant des valeurs nulles
new_file.dropna(inplace=True)

# Nettoyer les chaînes de caractères (supprimer les espaces en début/fin)
new_file = new_file.applymap(
    lambda x: x.strip() if isinstance(x, str) else x
)

# Normaliser les chaînes pour assurer une cohérence
new_file = new_file.applymap(
    lambda x: x.lower() if isinstance(x, str) else x
)

# Convertir les valeurs à partir de "calories" en float
def convert_to_float(value):
    if isinstance(value, str):
        # Retirer les caractères non numériques et convertir en float
        value = value.replace(",", ".").replace("<", "").strip()
        try:
            return float(value)
        except ValueError:
            return None  # Si la conversion échoue, retourner None
    elif isinstance(value, (int, float)):
        return float(value)
    return value

# Appliquer la conversion sur les colonnes numériques (à partir de "calories")
columns_to_convert = [
    "calories", "total_fat", "saturated_fat", "polyunsaturated_fatty_acids",
    "protein", "carbohydrate", "sugars", "lactose", "fiber", "calcium",
    "vitamin_d", "vitamin_e", "vitamin_k", "vitamin_c", "vitamin_a",
    "vitamin_b12", "vitamin_b6", "vitamin_b5"
]

# Appliquer la conversion aux colonnes spécifiées
new_file[columns_to_convert] = new_file[columns_to_convert].applymap(convert_to_float)

# Ajouter la colonne unsaturated_fat comme somme des AG monoinsaturés et AG polyinsaturés
new_file['unsaturated_fat'] = new_file['AG monoinsaturés (g/100 g)'] + new_file['AG polyinsaturés (g/100 g)']

# Remplir les valeurs manquantes de "total_fat" si nécessaire
new_file['total_fat'] = new_file.apply(
    lambda row: row['unsaturated_fat'] + row['AG saturés (g/100 g)'] if pd.isnull(row['total_fat']) else row['total_fat'],
    axis=1
)

# Appliquer la logique pour les calories vides
new_file = new_file[~new_file['calories'].isnull() | (
    (new_file['protein'] > 0) |
    (new_file['total_fat'] > 0) |
    (new_file['saturated_fat'] > 0) |
    (new_file['polyunsaturated_fatty_acids'] > 0) |
    (new_file['carbohydrate'] > 0) |
    (new_file['sugars'] > 0) |
    (new_file['lactose'] > 0)
)]

# Exporter vers un fichier CSV
new_file.to_csv(csv_file, index=False, encoding="utf-8")

print(f"Conversion terminée : {csv_file}")
