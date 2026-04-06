import torch
import numpy as np
from datasets import load_from_disk
from PIL import Image
import pickle
import os
import open_clip

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CATEGORY_MAP = {
    "trousers": ["Pants", "Jeans", "Skinny Jeans", "Straight Leg Jeans",
                 "Wide Leg Jeans", "Flared Jeans", "Bootcut Jeans",
                 "Boyfriend Jeans", "Capri & Cropped Pants"],
    "shorts": ["Shorts", "Activewear Shorts"],
    "skirt": ["Skirts", "Mini Skirts", "Long Skirts", "Knee Length Skirts", "Activewear Skirts"],
    "long_sleeved_shirt": ["Tops", "Blouses", "Sweaters", "Cardigans", "Tunics", "Hoodies", "Sweatshirts"],
    "short_sleeved_shirt": ["Tops", "T-Shirts", "Tank Tops", "Camisoles"],
    "long_sleeved_outwear": ["Jackets", "Coats", "Outerwear", "Blazers"],
    "short_sleeved_outwear": ["Jackets", "Blazers"],
    "vest": ["Vests"],
    "vest_dress": ["Dresses", "Day Dresses"],
    "long_sleeved_dress": ["Dresses", "Gowns", "Wedding Dresses"],
    "short_sleeved_dress": ["Dresses", "Day Dresses", "Cocktail Dresses"],
    "sling_dress": ["Dresses", "Day Dresses"],
    "sling": ["Camisoles", "Tops"],
}


def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()
    model.to(DEVICE)
    return model, preprocess


def get_embedding(image: Image.Image, model, preprocess) -> np.ndarray:
    img_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_image(img_tensor).squeeze().cpu().numpy()
    return embedding / np.linalg.norm(embedding)


def build_catalog_embeddings(catalog_path: str, save_path: str = "output/catalog_embeddings.pkl", batch_size: int = 64):
    os.makedirs("output", exist_ok=True)

    print("Loading Polyvore dataset...")
    ds = load_from_disk(catalog_path)
    model, preprocess = load_model()

    embeddings = []
    metadata = []

    for i in range(0, len(ds), batch_size):
        batch = ds[i:i + batch_size]
        images = [img.convert("RGB") for img in batch["image"]]
        tensors = torch.stack([preprocess(img) for img in images]).to(DEVICE)

        with torch.no_grad():
            batch_emb = model.encode_image(tensors).cpu().numpy()

        norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)
        batch_emb = batch_emb / norms

        embeddings.append(batch_emb)
        for j in range(len(images)):
            metadata.append({
                "item_ID": batch["item_ID"][j],
                "category": batch["category"][j],
                "text": batch["text"][j],
                "price": round(float(np.random.uniform(*_get_price_range(batch["category"][j]))), 2)
            })

        if i % 5000 == 0:
            print(f"  Processed {i}/{len(ds)} items...")

    catalog = {
        "embeddings": np.vstack(embeddings),
        "metadata": metadata
    }
    with open(save_path, "wb") as f:
        pickle.dump(catalog, f)
    print(f"Catalog saved to {save_path}")
    return catalog


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


def _get_price_range(category: str) -> tuple:
    return PRICE_RANGES.get(category, DEFAULT_RANGE)


def recommend(crop_path: str, class_name: str = None,
              catalog_path: str = "output/catalog_embeddings.pkl", top_k: int = 20):
    """
    Recommend similar items from the catalog based on the embedding of the input crop image.
    """
    # Load the CLIP model and the catalog of embeddings
    model, preprocess = load_model()

    with open(catalog_path, "rb") as f:
        catalog = pickle.load(f)

    crop_img = Image.open(crop_path)
    crop_emb = get_embedding(crop_img, model, preprocess)

    if class_name and class_name in CATEGORY_MAP:
        valid_categories = CATEGORY_MAP[class_name]
        indices = [i for i, m in enumerate(catalog["metadata"])
                   if m["category"] in valid_categories]
        filtered_embeddings = catalog["embeddings"][indices]
    else:
        indices = list(range(len(catalog["metadata"])))
        filtered_embeddings = catalog["embeddings"]

    similarities = np.dot(filtered_embeddings, crop_emb)
    top_local = np.argsort(similarities)[::-1][:top_k]
    top_indices = [indices[i] for i in top_local]

    results = []
    for rank, idx in enumerate(top_indices):
        results.append({
            "score": float(similarities[top_local[rank]]),
            **catalog["metadata"][idx]
        })
    return results


if __name__ == "__main__":
    # Rebuild catalog with CLIP embeddings
    build_catalog_embeddings("data/polyvore_outfits/data")

    # Test with crops
    test_crops = [
        ("data/output/crops/crop_0_trousers.jpg", "trousers"),
        ("data/output/crops/crop_1_long_sleeved_shirt.jpg", "long_sleeved_shirt"),
    ]
    for crop_path, class_name in test_crops:
        print(f"\n--- Recommendations for {class_name} ---")
        results = recommend(crop_path, class_name=class_name, top_k=5)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['category']} - {r['text'][:60]} | {r['price']:.2f}€")
