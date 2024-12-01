import sys

def select_food(data):
     # data, values_pca, labels = prepare_data(file, 'RobustScaler', 'PCA', {'n_components': 5})

     aliment = input("Entrez le nom de l'aliment que vous souhaitez rechercher : \n").strip()
     matching_aliment = data[data['name'].str.contains(aliment, case=False, na=False)]
     alim_grp_columns = ['alim_grp_nom_fr', 'alim_ssgrp_nom_fr', 'alim_ssssgrp_nom_fr']

     for col in alim_grp_columns:
          if len(matching_aliment) < 13:
               break
          alim_categories = matching_aliment[col].dropna().unique()
          print(f"\nCatégories disponibles pour {aliment} :")
          for i, cat in enumerate(alim_categories):
               print(f"{i+1}: {cat}")
          selected_option_num = int(input(f"\nEntrez le numéro correspondant à la catégorie de l'aliment recherché :  {col} : ").strip())
          selected_option = alim_categories[selected_option_num - 1]
          matching_aliment = matching_aliment[matching_aliment[col] == selected_option]
          print("     ###############     \n")

     print("\nAliments trouvés :")
     for idx, row in matching_aliment.iterrows():
          print(f"{idx}: {row['name']}")

     selected_aliment_idx = int(input("\nEntrez l'index de l'aliment recherché : ").strip())

     return selected_aliment_idx
