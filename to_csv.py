import pandas as pd

# Chemin vers le fichier source
excel_file = "Table Ciqual 2020_FR_2020 07 07.xls"

# Charger le fichier XLS
df = pd.read_excel(excel_file)


# Exporter le fichier CSV
df.to_csv("non_filtered.csv", index=False, sep=";")

print("Fichier CSV créé avec succès !")
