from datasets import load_from_disk
import json

print("Cargando datos locales...")
ds = load_from_disk("data/polyvore_outfits")
nombre_particion = list(ds.keys())[0]
datos = ds[nombre_particion]

# Diccionario para guardar los outfits
# Estructura: { "100002074": [ {"item_ID": "100002074_1", "category": "Day Dresses"}, ... ] }
armario_outfits = {}

print("Procesando y agrupando prendas en Outfits... (esto tomará unos segundos)")

# Vamos a procesar, por ejemplo, las primeras 10,000 prendas para que vaya súper rápido
for i in range(10000):
    prenda = datos[i]
    item_id_completo = prenda.get('item_ID', '')

    # Si por algún motivo no tiene ID, lo saltamos
    if not item_id_completo or '_' not in item_id_completo:
        continue

    # Separar el ID del outfit del ID de la prenda (ej: "100002074" y "1")
    outfit_id, prenda_id = item_id_completo.split('_', 1)

    # Si el outfit aún no existe en nuestro diccionario, lo creamos
    if outfit_id not in armario_outfits:
        armario_outfits[outfit_id] = []

    # Añadimos la categoría y el índice original (para saber dónde está la foto luego)
    armario_outfits[outfit_id].append({
        'item_id_completo': item_id_completo,
        'category': prenda.get('category', 'desconocido'),
        'indice_dataset': i  # Guardamos la 'i' para luego poder cargar su foto fácilmente
    })

print(f"\n¡Éxito! Hemos encontrado {len(armario_outfits)} outfits completos.")

# Vamos a imprimir un outfit al azar (el primero) para ver cómo quedó
primer_outfit = list(armario_outfits.keys())[0]
print(f"\nResumen del Outfit #{primer_outfit}:")
for articulo in armario_outfits[primer_outfit]:
    print(f" - {articulo['category']}")

# Guardamos este resumen en un JSON ligero para no tener que recalcularlo cada vez
with open("data/diccionario_outfits.json", "w") as f:
    json.dump(armario_outfits, f)
print("\nDiccionario guardado en data/diccionario_outfits.json")
