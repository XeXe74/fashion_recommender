from datasets import load_from_disk
ds = load_from_disk("data/polyvore_outfits/data")
print(ds[0])
