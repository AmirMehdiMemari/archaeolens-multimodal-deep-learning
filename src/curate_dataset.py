import csv
import re
import hashlib
import random
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/historical_metadata.csv"
OUTPUT = ROOT / "data/processed/curated_metadata.csv"
REJECTED = ROOT / "reports/curation_rejections.csv"
SHEET = ROOT / "reports/figures/curated_contact_sheet.png"

csv.field_size_limit(20_000_000)

# Visually identified problematic examples from our contact sheet.
MANUAL_REJECT = {
    "571503", "733298", "250834", "762897",
    "250197", "764300", "242133", "456882",
    "322811", "322434"
}

FRAGMENT_PATTERN = re.compile(
    r"\b(fragments?|sherds?|shards?|potsherds?)\b"
    r"|\b(?:piece|part)\s+of\b"
    r"|\b(?:rim|neck|handle|spout|base|lid)\s+"
    r"(?:of|from)\b"
    r"|\b(?:model|mold|mould|replica)\s+(?:of|for)\b",
    re.I
)

CLASSES = ["vase", "bowl", "jar", "figurine", "amulet"]

with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    records = list(csv.DictReader(f))

kept = []
rejected = []
seen_hashes = {}
reasons = Counter()

for row in records:
    object_id = row["object_id"]
    label = row["label"]
    title = row.get("title", "")
    name = row.get("object_name", "")
    path = ROOT / row["image_path"]

    reason = None

    if object_id in MANUAL_REJECT:
        reason = "manual_visual_review"

    elif FRAGMENT_PATTERN.search(title + " " + name):
        reason = "fragment_or_noncomplete_object"

    elif label == "bowl" and re.search(
        r"\b(helmet|sallet|armour|armor)\b",
        title + " " + name, re.I
    ):
        reason = "non_vessel_bowl"

    elif not path.is_file():
        reason = "missing_image"

    if reason is None:
        try:
            with Image.open(path) as image:
                image.verify()

            image_hash = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

            if image_hash in seen_hashes:
                reason = "exact_duplicate"
            else:
                seen_hashes[image_hash] = object_id

        except (OSError, ValueError):
            reason = "invalid_image"

    if reason:
        rejected.append({
            **row,
            "rejection_reason": reason
        })
        reasons[reason] += 1
    else:
        kept.append(row)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
REJECTED.parent.mkdir(parents=True, exist_ok=True)
SHEET.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(kept)

with REJECTED.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(records[0].keys()) + ["rejection_reason"]
    )
    writer.writeheader()
    writer.writerows(rejected)

# Visual verification of remaining samples
rng = random.Random(42)
sheet = Image.new("RGB", (1520, 1050), "white")
draw = ImageDraw.Draw(sheet)

for row_index, label in enumerate(CLASSES):
    pool = [r for r in kept if r["label"] == label]
    samples = rng.sample(pool, min(8, len(pool)))

    for column, record in enumerate(samples):
        with Image.open(ROOT / record["image_path"]) as img:
            image = ImageOps.contain(
                img.convert("RGB"),
                (175, 175)
            )

        x = column * 190
        y = row_index * 210

        sheet.paste(image, (x + (190 - image.width)//2, y))
        draw.text((x + 8, y + 178), label.upper(), fill="black")
        draw.text(
            (x + 8, y + 193),
            f"ID: {record['object_id']}",
            fill="black"
        )

sheet.save(SHEET)

counts = Counter(row["label"] for row in kept)

print("\n=== DATASET CURATION SUMMARY ===")
print("Original records:", len(records))
print("Rejected records:", len(rejected))
print("Remaining records:", len(kept))

print("\nREJECTION REASONS")
for reason, count in reasons.items():
    print(f"{reason}: {count}")

print("\nCURATED CLASS DISTRIBUTION")
for label in CLASSES:
    print(f"{label}: {counts[label]}")

print("\nCurated metadata:", OUTPUT)
print("Rejection log:", REJECTED)
print("Contact sheet:", SHEET)
print("\nCURATION COMPLETE")
