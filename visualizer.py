from datasets import load_from_disk

# Cargamos tus datos
ds = load_from_disk("data/polyvore_outfits")
nombre_particion = list(ds.keys())[0]
datos = ds[nombre_particion]

# Vamos a ver las primeras 10 prendas para ver si comparten Outfit ID
print(f"Total de prendas en el dataset: {len(datos)}\n")
for i in range(10):
    prenda = datos[i]
    print(f"Índice {i}: Categoría: {prenda.get('category', 'N/A')} | Item_ID: {prenda.get('item_ID', 'N/A')}")
