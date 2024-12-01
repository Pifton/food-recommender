import pandas as pd
import numpy as np
import csv

# Remplacez 'fichier.xlsx' et 'fichier.csv' par vos noms de fichiers
excel_file = "./data/Table Ciqual 2020_FR_2020 07 07.xls"
csv_file = "./data/ciqual.csv"
unsaturated_fat_list = []

filtered_colums = {
    "alim_nom_fr": "name",
    "alim_grp_nom_fr": "alim_grp_nom_fr",
    "alim_ssgrp_nom_fr": "alim_ssgrp_nom_fr",
    "alim_ssssgrp_nom_fr": "alim_ssssgrp_nom_fr",
    "Energie, Règlement UE N° 1169/2011 (kcal/100 g)": "calories",
    "Lipides (g/100 g)": "total_fat",
    "AG saturés (g/100 g)": "saturated_fat",
    "AG polyinsaturés (g/100 g)": "polyunsaturated_fatty_acids",
    "AG monoinsaturés (g/100 g)": "monounsaturated_fatty_acids",
    "Protéines, N x facteur de Jones (g/100 g)": "protein",
    "Glucides (g/100 g)": "carbohydrate",
    "Sucres (g/100 g)": "sugars",
    "Lactose (g/100 g)": "lactose",
    "Fibres alimentaires (g/100 g)": "fiber",
    "Calcium (mg/100 g)": "calcium",
    "Rétinol (µg/100 g)": "vitamin_a",
    "Vitamine B5 ou Acide pantothénique (mg/100 g)": "vitamin_b5",
    "Vitamine B6 (mg/100 g)": "vitamin_b6",
    "Vitamine B12 (µg/100 g)": "vitamin_b12",
    "Vitamine D (µg/100 g)": "vitamin_d",
    "Vitamine C (mg/100 g)": "vitamin_c",
    "Vitamine E (mg/100 g)": "vitamin_e",
    "Vitamine K1 (µg/100 g)": "vitamin_k",
}


new_file = pd.read_excel(excel_file, engine='xlrd')

new_file = new_file[list(filtered_colums.keys())]

# Renommer les colonnes
new_file.rename(columns=filtered_colums, inplace=True)

# new_file.replace('-', np.nan, inplace=True)
new_file.replace('-', 0.0, inplace=True)
new_file.replace(np.nan, 0.0, inplace=True)

for index,row in new_file.iterrows():
     for col in new_file.columns[4:]:
          value = row[col]
          if isinstance(value, str):
               value = value.replace(',', '.').replace('<', '').strip()
               value = value.replace(' < ', '').strip()
               if value == 'traces':
                    value = 0.0
               try:
                    value = float(value)
               except Exception as err:
                    print(value)
                    print(err)
               new_file.at[index, col] = value 
     unsaturated_fat = float(new_file.loc[index, "total_fat"]) - float(new_file.loc[index, "saturated_fat"])
     unsaturated_fat_list.append(unsaturated_fat)
     print(unsaturated_fat)

new_file["unsaturated_fat"] = unsaturated_fat_list



print(new_file.loc[1140, new_file.columns[4:]])
print(new_file.loc[2049, new_file.columns[4:]])

# Exporter vers un fichier CSV
# new_file.to_csv(csv_file, index=False)
new_file.to_csv(csv_file, index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

print(f"Conversion terminée : {csv_file}")