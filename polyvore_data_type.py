from datasets import load_from_disk

# Script to check the data type of the dataset loaded from disk
ds = load_from_disk("data/polyvore_outfits/data")
print(ds[0])
