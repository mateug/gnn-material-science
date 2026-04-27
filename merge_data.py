import os
import json
import shutil
import pandas as pd
from litraj.data import download_dataset, load_data

# download the dataset to the selected folder
download_dataset(
    dataset_name='BVEL13k', 
    folder='datasets', 
    remove_zip=True) 

"""
# get train, val, test split of the dataset and the index dataframe
atoms_list_train, atoms_list_val, atoms_list_test, index = load_data('BVEL13k', 'datasets/BVEL13k') 

# the data is stored in the Ase's Atoms object
for atoms in atoms_list_train: 
    mp_id = atoms.info['material_id']
    e1d = atoms.info['E_1D']
    e2d = atoms.info['E_2D']
    e3d = atoms.info['E_3D']
"""

# 1. Cargar el índice de BVEL13k (asegúrate de que el CSV esté en la ruta correcta)
archivo_bvel = "datasets/BVEL13k/BVEL13k_index.csv" 
df_bvel = pd.read_csv(archivo_bvel)

# Crear diccionarios para búsquedas rápidas
e1d_dict = dict(zip(df_bvel['material_id'], df_bvel['E_1D']))
e2d_dict = dict(zip(df_bvel['material_id'], df_bvel['E_2D']))
e3d_dict = dict(zip(df_bvel['material_id'], df_bvel['E_3D']))

input_dir = "input/candidates"
matched_count = 0
removed_count = 0

print("Cruzando datos de Materials Project con BVEL13k_index.csv...")

for root, dirs, files in os.walk(input_dir):
    if "metadata.json" in files:
        json_path = os.path.join(root, "metadata.json")
        
        with open(json_path, 'r') as f:
            metadata = json.load(f)
            
        mat_id = metadata.get("material_id")
        
        # Verificar si existe en BVEL
        if mat_id in e3d_dict:
            # Guardamos las tres energías para tener el dataset completo
            metadata["E_1D"] = float(e1d_dict[mat_id])
            metadata["E_2D"] = float(e2d_dict[mat_id])
            metadata["E_3D"] = float(e3d_dict[mat_id])
            
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            matched_count += 1
        else:
            # No hay datos de difusión, borramos la carpeta
            shutil.rmtree(root)
            removed_count += 1

# Limpieza de carpetas vacías
for formula_folder in os.listdir(input_dir):
    formula_path = os.path.join(input_dir, formula_folder)
    if os.path.isdir(formula_path) and not os.listdir(formula_path):
        os.rmdir(formula_path)

print("-" * 30)
print(f"✅ Materiales listos (con E_1D, E_2D, E_3D): {matched_count}")
print(f"🗑️ Materiales borrados (sin datos en BVEL): {removed_count}")