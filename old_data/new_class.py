import pandas as pd

# Charger le fichier CSV
file_path = "filtered_data.csv"
df = pd.read_csv(file_path)

# Fonction pour extraire la partie principale du nom
def extraire_nom_principal(nom):
    return ",".join(nom.split(",")[:2]).strip()  # Garde les deux premières parties séparées par des virgules

# Ajouter une colonne pour le regroupement
df["nom_principal"] = df["name"].apply(extraire_nom_principal)

# Regrouper par le nom principal
df_grouped = df.groupby("nom_principal").mean(numeric_only=True)

# Ajouter la colonne `name` pour conserver le nom principal dans le résultat
df_grouped["name"] = df_grouped.index

# Réordonner les colonnes pour avoir "name" en premier
colonnes = ["name"] + [col for col in df_grouped.columns if col != "name"]
df_grouped = df_grouped[colonnes]

# Sauvegarder dans un nouveau fichier CSV
output_path = "filtered_data_grouped.csv"
df_grouped.to_csv(output_path, index=False)

print(f"Données regroupées et enregistrées dans : {output_path}")
