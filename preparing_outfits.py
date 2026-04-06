from datasets import load_from_disk
import json

print("Loading local data...")
ds = load_from_disk("data/polyvore_outfits")
partition_name = list(ds.keys())[0]
data = ds[partition_name]

# Dictionary to hold outfits grouped by their outfit ID
# Structure: { "100002074": [ {"item_ID": "100002074_1", "category": "Day Dresses"}, ... ] }
closet_clothes = {}

print("Processing items to group them into outfits...")

# Process each item in the dataset
for i in range(10000):
    cloth = data[i]
    complete_item_id = cloth.get('item_ID', '')

    # If we do not have a valid item_ID or it doesn't contain an underscore, we skip it
    if not complete_item_id or '_' not in complete_item_id:
        continue

    # Split the item_ID into outfit_id and cloth_id
    outfit_id, cloth_id = complete_item_id.split('_', 1)

    # If the outfit_id is not yet in our dictionary, we create a new entry for it
    if outfit_id not in closet_clothes:
        closet_clothes[outfit_id] = []

    # Add the current cloth to the corresponding outfit in the dictionary
    closet_clothes[outfit_id].append({
        'complete_item_id': complete_item_id,
        'category': cloth.get('category', 'unknown'),
        'dataset_index': i # We keep track of the original index in the dataset for later reference
    })

print(f"\n¡Done! Founded {len(closet_clothes)} outfits.")

# Print a summary of the first outfit to verify that we have grouped the items correctly
first_outfit = list(closet_clothes.keys())[0]
print(f"\nOutfit summary#{first_outfit}:")
for item in closet_clothes[first_outfit]:
    print(f" - {item['category']}")

# Save the resulting dictionary to a JSON file for later use in the recommendation system
with open("data/diccionario_outfits.json", "w") as f:
    json.dump(closet_clothes, f)
print("\nDictionary saved in data/diccionario_outfits.json")
