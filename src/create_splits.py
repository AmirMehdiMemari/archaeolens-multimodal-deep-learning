import csv
import random
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/balanced_metadata.csv"
OUTPUT = ROOT / "data/manifests"
OUTPUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["vase", "bowl", "jar", "figurine", "amulet"]
SEED = 42

with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    records = list(reader)

assert len(records) == 1000, "Expected 1000 records"

groups = defaultdict(list)
seen_ids = set()
seen_hashes = set()

for row in records:
    object_id = row["object_id"]
    path = ROOT / row["image_path"]

    assert path.is_file(), f"Missing image: {path}"
    assert object_id not in seen_ids, f"Repeated object: {object_id}"

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest not in seen_hashes, f"Duplicate image: {object_id}"

    seen_ids.add(object_id)
    seen_hashes.add(digest)
    groups[row["label"]].append(row)

splits = {
    "train": [],
    "val": [],
    "test": []
}

rng = random.Random(SEED)

for label in CLASSES:
    items = sorted(groups[label], key=lambda x: x["object_id"])

    assert len(items) == 200, f"{label}: expected 200"

    rng.shuffle(items)

    splits["train"].extend(items[:140])
    splits["val"].extend(items[140:170])
    splits["test"].extend(items[170:200])

for name, items in splits.items():
    rng.shuffle(items)

    with (OUTPUT / f"{name}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)

print("\n=== DATASET SPLIT SUMMARY ===")
print(f'{"Category":<12} {"Train":>7} {"Val":>7} {"Test":>7}')
print("-" * 37)

for label in CLASSES:
    counts = [
        sum(r["label"] == label for r in splits[name])
        for name in ["train", "val", "test"]
    ]
    print(f"{label:<12} {counts[0]:>7} {counts[1]:>7} {counts[2]:>7}")

print("-" * 37)

for name, items in splits.items():
    print(f"{name}: {len(items)}")

print("Total:", sum(map(len, splits.values())))
print("Random seed:", SEED)
print("Exact duplicate leakage: NONE")
print("Saved manifests:", OUTPUT)
print("\nSPLIT COMPLETE")
