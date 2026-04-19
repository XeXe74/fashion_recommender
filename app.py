import gradio as gr
import tempfile
import os
import recommender as recommender_module
from detector import detect_and_crop
from embedder import recommend as embed_recommend
from recommender import recommend_outfits, parse_user_input
from visualizer import build_index, get_image
from datasets import load_from_disk

# Parameters
DATASET_PATH = "data/polyvore_outfits/data"
ALPHAS = [0.3, 0.5, 0.7, 1.0]
TOP_KS = [10, 20, 30]
TOP_K_OUTFITS = 3

# Load dataset and build index once at startup
print("Loading dataset and building index...")
ds = load_from_disk(DATASET_PATH)
index = build_index(ds)
print("Ready!")

def score_bar(score, length=20):
    filled = int(round(score * length))
    return "█" * filled

def run_pipeline(pil_image, user_input):
    if pil_image is None:
        return [], "Please upload an image first."

    # Save PIL image to a temporary file for the detector
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil_image.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Detect and crop clothing items from the input image
        crops = detect_and_crop(tmp_path)
        if not crops:
            return [], "No clothing items detected. Try another photo."

        # Hyperparameter search
        best_score = -1
        best_alpha, best_top_k = 1.0, 20
        constraints   = parse_user_input(user_input)
        keywords_only = constraints.get("keywords", "")

        # Loop for all combinations of alpha and top_k to find the best config based on average final_score of recommended outfits
        for alpha in ALPHAS:
            for top_k in TOP_KS:
                candidates = {}
                for crop in crops:
                    c = embed_recommend(crop["path"], class_name=crop["label"], top_k=top_k)
                    if c:
                        candidates[crop["label"]] = c
                recommender_module.ALPHA = alpha

                # Get outfits with the current config and calculate the average final_score across all items in the recommended outfits
                outfits = recommend_outfits(candidates, user_input=keywords_only, top_k=TOP_K_OUTFITS)
                scores = [item["final_score"] for o in outfits for item in o["items"].values()]
                score = sum(scores) / len(scores) if scores else 0

                print(f"    alpha={alpha}, top_k={top_k} -> score={score:.4f}", flush=True)

                # If this config is better than the best one so far, update the best config
                if score > best_score:
                    best_score, best_alpha, best_top_k = score, alpha, top_k

        # With the best configuration, get the final candidates and outfits to display
        recommender_module.ALPHA = best_alpha
        all_candidates = {}
        for crop in crops:
            c = embed_recommend(crop["path"], class_name=crop["label"], top_k=best_top_k)
            if c:
                all_candidates[crop["label"]] = c

        # Recommend outfits
        outfits = recommend_outfits(all_candidates, user_input=user_input, top_k=TOP_K_OUTFITS)
        if not outfits:
            return [], "No outfits found matching your constraints."

        # Get the images and captions for the gallery, and prepare the summary text
        gallery_images = []
        medals = ["🥇", "🥈", "🥉"]
        summary_lines = [f"**Best config:** α={best_alpha}, top_k={best_top_k}, score={best_score:.3f}\n"]

        for i, outfit in enumerate(outfits):
            medal = medals[i] if i < 3 else f"#{i + 1}"
            bar = score_bar(outfit['outfit_score'])
            summary_lines.append(f"### {medal} Outfit {i + 1}")
            summary_lines.append(
                f"**Score:** `{bar}` {outfit['outfit_score']:.3f} &nbsp; **Total:** `{outfit['total_price']:.2f}€`")
            summary_lines.append("")
            summary_lines.append("| Garment | Item | Price |")
            summary_lines.append("|---------|------|-------|")
            for class_name, item in outfit["items"].items():
                img = get_image(ds, index, item["item_ID"])
                caption = f"{class_name} · {item['category']}\n{item['text'][:40]} | {item['price']:.2f}€"
                if img:
                    gallery_images.append((img, caption))
                name = item['text'][:50] + ("…" if len(item['text']) > 50 else "")
                summary_lines.append(f"| `{class_name}` | {name} | `{item['price']:.2f}€` |")
            summary_lines.append("\n---")

        return gallery_images, "\n".join(summary_lines)

    finally:
        os.unlink(tmp_path)  # Cleanup temporary file


# User Interface with Gradio
with gr.Blocks(title="Fashion Recommender", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👗 Visual Fashion Recommender")
    gr.Markdown("Upload an outfit photo and optionally add constraints. The system detects each garment and recommends similar items.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Outfit photo", sources=["upload", "clipboard"])
            user_input = gr.Textbox(
                label="Constraints (optional)",
                placeholder="e.g. under 100 euros casual"
            )
            submit_btn = gr.Button("🔍 Find similar items", variant="primary")

        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Recommended items", columns=3, height=520, object_fit="cover")
            summary_text = gr.Markdown()

    submit_btn.click(
        fn=run_pipeline,
        inputs=[input_image, user_input],
        outputs=[gallery, summary_text]
    )

if __name__ == "__main__":
    demo.launch()