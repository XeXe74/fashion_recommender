import matplotlib.pyplot as plt
from datasets import load_from_disk
import textwrap

DATASET_PATH = "data/polyvore_outfits/data"

def build_index(ds):
    print("Building item_ID index...")
    index = {ds[i]["item_ID"]: i for i in range(len(ds))}
    print(f"Index built: {len(index)} entries")
    return index

def get_image(ds, index, item_id):
    idx = index.get(item_id)
    if idx is None:
        return None
    return ds[idx]["image"]

def visualize_outfits(outfits, ds, index):
    n_outfits = len(outfits)
    n_items = max(len(o["items"]) for o in outfits)

    fig, axes = plt.subplots(n_outfits, n_items, figsize=(n_items * 4, n_outfits * 5))

    if n_outfits == 1:
        axes = [axes]
    if n_items == 1:
        axes = [[ax] for ax in axes]

    for i, outfit in enumerate(outfits):
        for j, (class_name, item) in enumerate(outfit["items"].items()):
            ax = axes[i][j]
            img = get_image(ds, index, item["item_ID"])

            if img:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "No image", ha="center", va="center", transform=ax.transAxes)

            name = textwrap.fill(item["text"], width=22)
            ax.set_title(
                f"{class_name}\n{item['category']} - {name}\n"
                f"{item['price']:.2f}€  |  sim: {item['final_score']:.3f}",
                fontsize=8
            )
            ax.axis("off")

        for j in range(len(outfit["items"]), n_items):
            axes[i][j].axis("off")

        axes[i][0].set_ylabel(
            f"Outfit {i+1}\nScore: {outfit['outfit_score']:.3f}\nTotal: {outfit['total_price']:.2f}€",
            fontsize=9, rotation=0, labelpad=130, va="center"
        )

    plt.suptitle("Fashion Recommendations", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("output/recommendations.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("Saved to output/recommendations.png")


if __name__ == "__main__":
    from embedder import recommend as embed_recommend
    from recommender import recommend_outfits

    ds = load_from_disk(DATASET_PATH)
    index = build_index(ds)

    user_input = "menos de 150 euros knit oversized"

    all_candidates = {
        "trousers": embed_recommend("data/output/crops/crop_0_trousers.jpg", class_name="trousers", top_k=20),
        "long_sleeved_shirt": embed_recommend("data/output/crops/crop_1_long_sleeved_shirt.jpg", class_name="long_sleeved_shirt", top_k=20),
    }

    outfits = recommend_outfits(all_candidates, user_input=user_input, top_k=3)
    visualize_outfits(outfits, ds, index)