from detector import detect_and_crop
from embedder import recommend as embed_recommend
from recommender import recommend_outfits
from visualizer import build_index, visualize_outfits
from datasets import load_from_disk

DATASET_PATH = "data/polyvore_outfits/data"
INPUT_IMAGE   = "data/input_outfits/outfit_2.jpg"
TOP_K_CANDIDATES = 20
TOP_K_OUTFITS    = 3

def main():
    print("FASHION RECOMMENDER SYSTEM")
    print("=" * 50)

    # Get user constraints
    user_input = input("\nEnter your constraints (e.g. 'menos de 150 euros knit oversized'): ").strip()

    # Default input if the user does not provide any constraints
    if not user_input:
        user_input = "menos de 150 euros"

    # Detect clothing items and crop them
    print("\n[1/4] Detecting clothes...")
    crops = detect_and_crop(INPUT_IMAGE)

    # Check if any crops were detected
    if not crops:
        print("No clothes detected. Check the input image.")
        return

    # Embed crops and get top candidates from the catalog
    print("\n[2/4] Extracting CLIP embeddings...")
    all_candidates = {}
    for crop in crops:
        class_name = crop["label"]
        crop_path  = crop["path"]
        print(f"  Embedding: {class_name} ({crop['confidence']:.0%})")
        candidates = embed_recommend(crop_path, class_name=class_name, top_k=TOP_K_CANDIDATES)
        if candidates:
            all_candidates[class_name] = candidates

    # Check if we have candidates for all detected items
    if not all_candidates:
        print("No valid candidates found.")
        return

    # Recommend outfits based on candidates and user constraints
    print("\n[3/4] Generating outfit recommendations...")
    outfits = recommend_outfits(all_candidates, user_input=user_input, top_k=TOP_K_OUTFITS)

    # Check if any outfits were generated
    if not outfits:
        print("No outfits found matching your constraints.")
        return

    # Print outfit summaries
    for i, outfit in enumerate(outfits):
        print(f"\n  Outfit {i+1} | Score: {outfit['outfit_score']:.3f} | Total: {outfit['total_price']:.2f}€")
        for class_name, item in outfit["items"].items():
            print(f"    {class_name}: {item['category']} - {item['text'][:50]} | {item['price']:.2f}€")

    # Use the dataset to visualize the recommended outfits
    print("\n[4/4] Loading dataset and visualizing...")
    ds = load_from_disk(DATASET_PATH)
    index = build_index(ds)
    visualize_outfits(outfits, ds, index)

    print("\nDone! Output saved to output/recommendations.png")

if __name__ == "__main__":
    main()