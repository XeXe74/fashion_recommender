import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from datasets import load_from_disk
from PIL import Image
import pickle
import os

# Set device for PyTorch (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Define category mapping for filtering recommendations
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

# Define image transformations for ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_model():
    """
    Load a pre-trained ResNet50 model and remove the final classification layer.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # Load pre-trained ResNet50 model with default weights
    model = nn.Sequential(*list(model.children())[:-1]) # Remove the final fully connected layer to get feature embeddings
    model.eval() # Set model to evaluation mode
    model.to(DEVICE)
    return model


def get_embedding(image: Image.Image, model) -> np.ndarray:
    """
    Get the normalized embedding vector for a given image using the provided model.
    """

    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE) # Preprocess the image and add batch dimension

    # Get the embedding from the model without computing gradients
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().cpu().numpy()

    return embedding / np.linalg.norm(embedding)


def build_catalog_embeddings(catalog_path: str, save_path: str = "output/catalog_embeddings.pkl", batch_size: int = 64):
    """
    Build and save the catalog embeddings for all items in the dataset.
    """

    os.makedirs("output", exist_ok=True) # Create output directory if it doesn't exist

    print("Loading Polyvore dataset...")
    ds = load_from_disk(catalog_path) # Load the dataset from the specified path
    model = load_model()

    embeddings = []
    metadata = []

    # Process the dataset in batches
    for i in range(0, len(ds), batch_size):
        batch = ds[i:i + batch_size]
        images = [img.convert("RGB") for img in batch["image"]]
        tensors = torch.stack([transform(img) for img in images]).to(DEVICE)

        with torch.no_grad():
            batch_emb = model(tensors).squeeze(-1).squeeze(-1).cpu().numpy() # Get embeddings and remove extra dimensions

        norms = np.linalg.norm(batch_emb, axis=1, keepdims=True) # Compute norms for normalization
        batch_emb = batch_emb / norms # Normalize embeddings

        embeddings.append(batch_emb) # Append batch embeddings to the list

        # Append metadata for each item in the batch
        for j in range(len(images)):
            metadata.append({
                "item_ID": batch["item_ID"][j],
                "category": batch["category"][j],
                "text": batch["text"][j]
            })

        # Print progress every 5000 items
        if i % 5000 == 0:
            print(f"  Processed {i}/{len(ds)} items...")

    # Combine all batch embeddings into a single array and save the catalog
    catalog = {
        "embeddings": np.vstack(embeddings),
        "metadata": metadata
    }

    # Save the catalog to a pickle file
    with open(save_path, "wb") as f:
        pickle.dump(catalog, f)
    print(f"Catalog saved to {save_path}")
    return catalog


def recommend(crop_path: str, class_name: str = None,
              catalog_path: str = "output/catalog_embeddings_with_prices.pkl", top_k: int = 5):

    """
    Recommend similar items from the catalog based on the embedding of the input crop image.
    """
    model = load_model() # Load the ResNet50 model for embedding extraction

    # Load the pre-computed catalog embeddings and metadata
    with open(catalog_path, "rb") as f:
        catalog = pickle.load(f)

    # Get the embedding for the input crop image
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

    # Compute cosine similarities and get top-k recommendations
    similarities = np.dot(filtered_embeddings, crop_emb)
    top_local = np.argsort(similarities)[::-1][:top_k]
    top_indices = [indices[i] for i in top_local]

    # Build the results with scores and metadata for the top recommendations
    results = []
    for idx in top_indices:
        results.append({
            "score": float(similarities[top_local[results.__len__()]]),
            **catalog["metadata"][idx]
        })
    return results

# TEST
if __name__ == "__main__":
    # Build catalog with CLIP embeddings if it doesn't exist
    if not os.path.exists("output/catalog_embeddings.pkl"):
        build_catalog_embeddings("data/polyvore_outfits/data")

    # Test with crops
    test_crops = [
        ("data/output/crops/crop_0_trousers.jpg", "trousers"),
        ("data/output/crops/crop_1_long_sleeved_shirt.jpg", "long_sleeved_shirt"),
    ]

    # Print recommendations for each test crop
    for crop_path, class_name in test_crops:
        print(f"\n--- Recommendations for {class_name} ---")
        results = recommend(crop_path, class_name=class_name)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['category']} - {r['text'][:60]}")
