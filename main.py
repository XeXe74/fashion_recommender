from detector import detect_and_crop
from embedder import recommend as embed_recommend
from recommender import recommend_outfits
from visualizer import build_index, visualize_outfits
from datasets import load_from_disk
import recommender

DATASET_PATH = "data/polyvore_outfits/data"
INPUT_IMAGE   = "data/input_outfits/outfit_2.jpg"
TOP_K_OUTFITS    = 3

# Hyperparameter search space
ALPHAS = [0.3, 0.5, 0.7, 1.0]
TOP_KS = [10, 20, 30]

def hyperparameter_search(crops, user_input=""):
    """Try all (alpha, top_k) combinations and return the best config."""
    best_score = -1
    best_alpha = 0.7
    best_top_k = 20

    print(f"  Testing {len(ALPHAS) * len(TOP_KS)} combinations...")
    for alpha in ALPHAS:
        for top_k in TOP_KS:
            candidates = {}
            for crop in crops:
                c = embed_recommend(crop["path"], class_name=crop["label"], top_k=top_k)
                if c:
                    candidates[crop["label"]] = c

            recommender.ALPHA = alpha
            outfits = recommend_outfits(candidates, user_input=user_input, top_k=TOP_K_OUTFITS)

            scores = [item["final_score"]
                      for o in outfits
                      for item in o["items"].values()]
            score = sum(scores) / len(scores) if scores else 0

            print(f"    alpha={alpha}, top_k={top_k} -> score={score:.4f}")

            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_top_k = top_k

    print(f"\n  Best config: alpha={best_alpha}, top_k={best_top_k} (score={best_score:.4f})")
    return best_alpha, best_top_k

def main():
    print("FASHION RECOMMENDER SYSTEM")
    print("=" * 50)

    # Get user constraints
    user_input = input("\nEnter your constraints (e.g. 'menos de 150 euros knit oversized'): ").strip()

    # Default input if the user does not provide any constraints
    if not user_input:
        user_input = "menos de 150 euros"

    # Detect clothing items and crop them
    print("\n[1/5] Detecting clothes...")
    crops = detect_and_crop(INPUT_IMAGE)

    # Check if any crops were detected
    if not crops:
        print("No clothes detected. Check the input image.")
        return

    # Hyperparameter search
    print("\n[2/5] Running hyperparameter search...")
    best_alpha, best_top_k = hyperparameter_search(crops, user_input)
    recommender.ALPHA = best_alpha  # fix best alpha for the rest of the pipeline

    # Embed with best top_k
    print("\n[3/5] Extracting CLIP embeddings...")
    all_candidates = {}
    for crop in crops:
        class_name = crop["label"]
        crop_path = crop["path"]
        print(f"  Embedding: {class_name} ({crop['confidence']:.0%})")
        candidates = embed_recommend(crop_path, class_name=class_name, top_k=best_top_k)
        if candidates:
            all_candidates[class_name] = candidates

    # Check if we have candidates for all detected items
    if not all_candidates:
        print("No valid candidates found.")
        return

    # Recommend outfits based on candidates and user constraints
    print("\n[4/5] Generating outfit recommendations...")
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
    print("\n[5/5] Loading dataset and visualizing...")
    ds = load_from_disk(DATASET_PATH)
    index = build_index(ds)
    visualize_outfits(outfits, ds, index)

    print("\nDone! Output saved to output/recommendations.png")

if __name__ == "__main__":
    main()