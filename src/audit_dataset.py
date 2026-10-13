import csv
import hashlib
import random
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data/raw/historical_metadata.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

csv.field_size_limit(20_000_000)

with METADATA.open(encoding="utf-8-sig", newline="") as f:
    records = list(csv.DictReader(f))

classes = ["vase", "bowl", "jar", "figurine", "amulet"]

counts = Counter()
valid = defaultdict(list)
invalid = []
hashes = defaultdict(list)

for row in records:
    path = ROOT / row["image_path"]
    label = row["label"]

    try:
        with Image.open(path) as img:
            img.load()

            if min(img.size) < 128:
                raise ValueError("Image too small")

            image_hash = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

        hashes[image_hash].append(row["object_id"])
        counts[label] += 1
        valid[label].append(row)

    except (OSError, ValueError) as error:
        invalid.append((row["object_id"], str(error)))

duplicates = [
    ids for ids in hashes.values()
    if len(ids) > 1
]

print("\n=== DATASET QUALITY AUDIT ===")
print("Metadata records:", len(records))

for label in classes:
    print(f"{label}: {counts[label]} valid images")

print("\nValid images:", sum(counts.values()))
print("Invalid images:", len(invalid))
print("Exact duplicate groups:", len(duplicates))

if invalid:
    print("Invalid examples:", invalid[:10])

if duplicates:
    print("Duplicate examples:", duplicates[:10])

# Generate a reproducible contact sheet
rng = random.Random(42)

cell_w, cell_h = 190, 210
columns = 8
rows = len(classes)

sheet = Image.new(
    "RGB",
    (columns * cell_w, rows * cell_h),
    "white"
)

draw = ImageDraw.Draw(sheet)

for row_index, label in enumerate(classes):
    samples = rng.sample(
        valid[label],
        min(8, len(valid[label]))
    )

    for col_index, record in enumerate(samples):
        path = ROOT / record["image_path"]

        with Image.open(path) as img:
            img = img.convert("RGB")
            img = ImageOps.contain(img, (175, 175))

        x = col_index * cell_w
        y = row_index * cell_h

        sheet.paste(img, (x + (cell_w - img.width) // 2, y))

        draw.text(
            (x + 8, y + 178),
            label.upper(),
            fill="black"
        )

        draw.text(
            (x + 8, y + 193),
            f"ID: {record['object_id']}",
            fill="black"
        )

output = FIGURES / "dataset_contact_sheet.png"
sheet.save(output)

report = REPORTS / "dataset_audit.txt"
report.write_text(
    "\n".join([
        f"Metadata records: {len(records)}",
        f"Valid images: {sum(counts.values())}",
        f"Invalid images: {len(invalid)}",
        f"Exact duplicate groups: {len(duplicates)}",
        *[f"{label}: {counts[label]}" for label in classes],
        f"Invalid records: {invalid}",
        f"Duplicate groups: {duplicates}"
    ]),
    encoding="utf-8"
)

print("\nContact sheet:", output)
print("Audit report:", report)
print("\nAUDIT COMPLETE")
