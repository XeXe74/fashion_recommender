import os
from datasets import load_dataset

# 1. Crea la carpeta "data" si no existe en tu ordenador
os.makedirs("data", exist_ok=True)

print("Descargando el dataset desde Hugging Face... (puede tardar un poco)")
# 2. Descargamos el dataset (por defecto va a la caché temporal)
ds = load_dataset("Marqo/polyvore")

print("Guardando el dataset en tu carpeta /data/...")
# 3. Guardamos el dataset de forma permanente en tu carpeta local
ds.save_to_disk("data/polyvore_outfits")

print("¡Proceso completado! Puedes revisar la carpeta 'data'.")
