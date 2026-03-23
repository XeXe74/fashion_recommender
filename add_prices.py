import pickle
import numpy as np

# NO ES NECESARIO, ERA PARA EL RESNET

PRICE_RANGES = {
    "Pants": (20, 150), "Jeans": (30, 200), "Skinny Jeans": (30, 180),
    "Straight Leg Jeans": (30, 180), "Wide Leg Jeans": (30, 180),
    "Flared Jeans": (30, 180), "Bootcut Jeans": (30, 180),
    "Boyfriend Jeans": (30, 180), "Capri & Cropped Pants": (20, 120),
    "Shorts": (15, 100), "Activewear Shorts": (15, 80),
    "Skirts": (20, 150), "Mini Skirts": (20, 120), "Long Skirts": (25, 150),
    "Knee Length Skirts": (20, 130), "Activewear Skirts": (15, 80),
    "Tops": (10, 120), "Blouses": (20, 180), "Sweaters": (25, 300),
    "Cardigans": (25, 250), "Tunics": (20, 150), "Hoodies": (20, 180),
    "Sweatshirts": (20, 150), "T-Shirts": (10, 80), "Tank Tops": (10, 60),
    "Camisoles": (10, 80), "Jackets": (40, 500), "Coats": (60, 600),
    "Outerwear": (60, 600), "Blazers": (40, 400), "Vests": (20, 200),
    "Dresses": (30, 400), "Day Dresses": (30, 300),
    "Cocktail Dresses": (50, 500), "Gowns": (80, 800),
    "Wedding Dresses": (200, 2000),
}
DEFAULT_RANGE = (15, 200)

print("Loading catalog...")
with open("output/catalog_embeddings.pkl", "rb") as f:
    catalog = pickle.load(f)

# Add price to each item
for item in catalog["metadata"]:
    low, high = PRICE_RANGES.get(item["category"], DEFAULT_RANGE)
    item["price"] = round(float(np.random.uniform(low, high)), 2)

with open("output/catalog_embeddings_with_prices.pkl", "wb") as f:
    pickle.dump(catalog, f)

print(f"Done. Sample: {catalog['metadata'][0]}")
