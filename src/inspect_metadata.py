import csv
import gzip
import re
from collections import Counter

# FIX: Allow large CSV fields
csv.field_size_limit(20_000_000)

path = "data/raw/met_openaccess.csv.gz"

categories = {
    "vase": r"\b(vase|amphora|hydria|krater)\b",
    "bowl": r"\bbowls?\b",
    "jar": r"\bjars?\b",
    "figurine": r"\b(figurine|statuette)\b",
    "amulet": r"\bamulets?\b"
}

all_counts = Counter()
historical_counts = Counter()
examples = {name: [] for name in categories}

total_records = 0

with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        total_records += 1

        if row["isPublicDomain"].lower() != "true":
            continue

        if not row["primaryImageSmall"].strip():
            continue

        object_name = row["objectName"] or ""

        for label, pattern in categories.items():
            if not re.search(pattern, object_name, re.I):
                continue

            all_counts[label] += 1

            try:
                year = int(row["objectBeginDate"])
            except (ValueError, TypeError):
                break

            if year != 0 and year <= 1500:
                historical_counts[label] += 1

                if len(examples[label]) < 2:
                    examples[label].append(
                        (row["objectID"], row["title"], year)
                    )
            break

print("\n=== MET DATASET INSPECTION ===")
print("Total museum records:", f"{total_records:,}")

print("\n=== DATASET AVAILABILITY ===")
print(f'{"Category":<12} {"All":>10} {"Historical":>12}')
print("-" * 36)

for label in categories:
    print(
        f"{label:<12} "
        f"{all_counts[label]:>10,} "
        f"{historical_counts[label]:>12,}"
    )

print("-" * 36)
print("TOTAL HISTORICAL:", sum(historical_counts.values()))

print("\n=== SAMPLE HISTORICAL OBJECTS ===")
for label, items in examples.items():
    print(f"\n{label.upper()}")
    for object_id, title, year in items:
        print(f"  {object_id} | {title} | {year}")
