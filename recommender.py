import re
import numpy as np
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ALPHA = 0.7  # Weight: visual vs textual (hyperparameter)

def parse_user_input(text: str) -> dict:
    """Extract structured constraints from free-form user input."""
    constraints = {}

    if not text:
        return constraints

    # Price range
    range_match = re.search(r"entre\s+(\d+)\s+y\s+(\d+)|between\s+(\d+)\s+and\s+(\d+)", text, re.IGNORECASE)
    max_match = re.search(r"menos de\s+(\d+)|máximo\s+(\d+)|under\s+(\d+)|max\s+(\d+)", text, re.IGNORECASE)
    min_match = re.search(r"más de\s+(\d+)|mínimo\s+(\d+)|over\s+\d+|min\s+(\d+)", text, re.IGNORECASE)

    if range_match:
        groups = range_match.groups()
        constraints["min_price"] = float(groups[0] or groups[2])
        constraints["max_price"] = float(groups[1] or groups[3])
    else:
        if max_match:
            constraints["max_price"] = float(next(g for g in max_match.groups() if g))
        if min_match:
            constraints["min_price"] = float(next(g for g in min_match.groups() if g))

    # Extract keywords
    keywords = re.sub(
        r"(menos de|más de|máximo|mínimo|entre|y|euros?|under|over\s+\d+|max\s+\d+|min\s+\d+|between|and|\d+)",
        "", text, flags=re.IGNORECASE
    ).strip()
    if keywords:
        constraints["keywords"] = keywords

    return constraints


def score_candidate(candidate: dict, keywords: str, vectorizer=None, tfidf_matrix=None, idx: int = 0) -> float:
    """Compute final score combining visual and textual similarity."""
    visual = candidate["score"]
    if keywords and vectorizer and tfidf_matrix is not None:
        text_score = float(cosine_similarity(tfidf_matrix[-1], tfidf_matrix[idx:idx+1]).flatten()[0])
    else:
        text_score = 0.0
    candidate["text_score"] = text_score
    return ALPHA * visual + (1 - ALPHA) * text_score


def recommend_outfits(all_candidates: dict, user_input: str = "", top_k: int = 3) -> list:
    """
    Generate complete outfit recommendations within budget.

    all_candidates: dict {class_name: [candidates from embedder]}
                    e.g. {"trousers": [...], "long_sleeved_shirt": [...]}
    user_input: free-form text with constraints
    top_k: number of outfits to return
    """
    constraints = parse_user_input(user_input)
    if constraints:
        print(f"  Parsed constraints: {constraints}")

    min_price = constraints.get("min_price", 0)
    max_price = constraints.get("max_price", float("inf"))
    keywords = constraints.get("keywords", "").strip()

    # Build TF-IDF matrix across all candidates if keywords provided
    tfidf_matrix = None
    vectorizer = None
    all_texts = []
    text_index_map = {}  # maps (class_name, i) -> index in corpus

    if keywords:
        idx = 0
        for class_name, candidates in all_candidates.items():
            for i, c in enumerate(candidates):
                all_texts.append(c["text"])
                text_index_map[(class_name, i)] = idx
                idx += 1
        all_texts.append(keywords)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Score each candidate
    scored = {}
    for class_name, candidates in all_candidates.items():
        scored[class_name] = []
        for i, c in enumerate(candidates):
            tidx = text_index_map.get((class_name, i), 0)
            final = score_candidate(c, keywords, vectorizer, tfidf_matrix, tidx)
            c["final_score"] = final
            scored[class_name].append(c)

    # Generate all outfit combinations
    class_names = list(scored.keys())
    candidate_lists = [scored[cn] for cn in class_names]

    valid_outfits = []
    for combo in product(*candidate_lists):
        total_price = sum(c["price"] for c in combo)
        if min_price <= total_price <= max_price:
            outfit_score = float(np.mean([c["final_score"] for c in combo]))
            valid_outfits.append({
                "items": {class_names[i]: combo[i] for i in range(len(class_names))},
                "total_price": round(total_price, 2),
                "outfit_score": outfit_score
            })

    # Sort by outfit score
    valid_outfits.sort(key=lambda x: x["outfit_score"], reverse=True)

    if not valid_outfits:
        print("  [Warning] No outfits found within budget. Returning best options ignoring price.")
        for combo in product(*candidate_lists):
            total_price = sum(c["price"] for c in combo)
            outfit_score = float(np.mean([c["final_score"] for c in combo]))
            valid_outfits.append({
                "items": {class_names[i]: combo[i] for i in range(len(class_names))},
                "total_price": round(total_price, 2),
                "outfit_score": outfit_score
            })
        valid_outfits.sort(key=lambda x: x["outfit_score"], reverse=True)

    seen_ids = set()
    diverse_outfits = []
    for outfit in valid_outfits:
        item_ids = {item["item_ID"] for item in outfit["items"].values()}
        if not item_ids & seen_ids:
            diverse_outfits.append(outfit)
            seen_ids.update(item_ids)
        if len(diverse_outfits) == top_k:
            break

    return diverse_outfits if diverse_outfits else valid_outfits[:top_k]


if __name__ == "__main__":
    from embedder import recommend as embed_recommend

    user_input = "menos de 150 euros knit oversized"

    # Get top-20 visual candidates per detected garment
    all_candidates = {
        "trousers": embed_recommend("data/output/crops/crop_0_trousers.jpg", class_name="trousers", top_k=20),
        "long_sleeved_shirt": embed_recommend("data/output/crops/crop_1_long_sleeved_shirt.jpg", class_name="long_sleeved_shirt", top_k=20),
    }

    outfits = recommend_outfits(all_candidates, user_input=user_input, top_k=3)

    for i, outfit in enumerate(outfits):
        print(f"\n--- Outfit {i+1} | Score: {outfit['outfit_score']:.3f} | Total: {outfit['total_price']:.2f}€ ---")
        for class_name, item in outfit["items"].items():
            print(f"  {class_name}: [{item['final_score']:.3f}] {item['category']} - {item['text'][:50]} | {item['price']:.2f}€")
