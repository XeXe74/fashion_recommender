from datasets import load_from_disk

# Script to build an index mapping item_ID to dataset index for quick lookup
ds = load_from_disk("data/polyvore_outfits/data")
print(f"Total items: {len(ds)}")

# Construct index
item_id_to_index = {ds[i]["item_ID"]: i for i in range(len(ds))}
print(f"Index built: {len(item_id_to_index)} entries")
print(f"Example: {list(item_id_to_index.items())[:3]}")