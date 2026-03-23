import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from datasets import load_from_disk
from PIL import Image
import pickle
import os

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(DEVICE)
    return model


def get_embedding(image: Image.Image, model) -> np.ndarray:
    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().cpu().numpy()
    return embedding / np.linalg.norm(embedding)


def build_catalog_embeddings(catalog_path: str, save_path: str = "output/catalog_embeddings.pkl", batch_size: int = 64):
    os.makedirs("output", exist_ok=True)

    print("Loading Polyvore dataset...")
    ds = load_from_disk(catalog_path)
    model = load_model()

    embeddings = []
    metadata = []

    for i in range(0, len(ds), batch_size):
        batch = ds[i:i + batch_size]
        images = [img.convert("RGB") for img in batch["image"]]
        tensors = torch.stack([transform(img) for img in images]).to(DEVICE)

        with torch.no_grad():
            batch_emb = model(tensors).squeeze(-1).squeeze(-1).cpu().numpy()

        norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)
        batch_emb = batch_emb / norms

        embeddings.append(batch_emb)
        for j in range(len(images)):
            metadata.append({
                "item_ID": batch["item_ID"][j],
                "category": batch["category"][j],
                "text": batch["text"][j]
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


def recommend(crop_path: str, class_name: str = None,
              catalog_path: str = "output/catalog_embeddings_with_prices.pkl", top_k: int = 5):
    model = load_model()

    with open(catalog_path, "rb") as f:
        catalog = pickle.load(f)

    crop_img = Image.open(crop_path)
    crop_emb = get_embedding(crop_img, model)

    # Filter by category if class_name is provided
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
    for idx in top_indices:
        results.append({
            "score": float(similarities[top_local[results.__len__()]]),
            **catalog["metadata"][idx]
        })
    return results


if __name__ == "__main__":
    # Step 1: build catalog embeddings (run once, skip if pkl already exists)
    if not os.path.exists("output/catalog_embeddings.pkl"):
        build_catalog_embeddings("data/polyvore_outfits/data")

    # Step 2: test with crops using category filter
    test_crops = [
        ("data/output/crops/crop_0_trousers.jpg", "trousers"),
        ("data/output/crops/crop_1_long_sleeved_shirt.jpg", "long_sleeved_shirt"),
    ]
    for crop_path, class_name in test_crops:
        print(f"\n--- Recommendations for {class_name} ---")
        results = recommend(crop_path, class_name=class_name)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['category']} - {r['text'][:60]}")
