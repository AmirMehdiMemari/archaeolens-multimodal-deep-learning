import csv
import gzip
import io
import random
import re
import time
import hashlib
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
CURATED = ROOT / "data/processed/curated_metadata.csv"
OUTPUT = ROOT / "data/processed/balanced_metadata.csv"
REJECTED = ROOT / "reports/curation_rejections.csv"
SOURCE = RAW / "met_openaccess.csv.gz"

TARGET = 200
csv.field_size_limit(20_000_000)

CATEGORIES = {
    "vase": r"\b(vase|amphora|hydria|krater)\b",
    "bowl": r"\bbowls?\b",
    "jar": r"\bjars?\b",
    "figurine": r"\b(figurine|statuette)\b",
    "amulet": r"\bamulets?\b"
}

FRAGMENTS = re.compile(
    r"\b(fragments?|sherds?|shards?|potsherds?)\b"
    r"|\b(?:piece|part)\s+of\b"
    r"|\b(?:rim|neck|handle|spout|base|lid)\s+(?:of|from)\b"
    r"|\b(?:model|mold|mould|replica)\s+(?:of|for)\b",
    re.I
)

def image_hash(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            if min(img.size) < 128:
                return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError):
        return None

# Preserve previous curation and allow resuming.
input_path = OUTPUT if OUTPUT.exists() else CURATED

with input_path.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    previous = list(reader)

records = []
seen_ids = set()
seen_hashes = set()

for row in previous:
    path = ROOT / row["image_path"]
    digest = image_hash(path)

    if digest and digest not in seen_hashes:
        records.append(row)
        seen_ids.add(row["object_id"])
        seen_hashes.add(digest)

counts = Counter(r["label"] for r in records)

rejected_ids = set()
with REJECTED.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        rejected_ids.add(row["object_id"])

candidates = {label: [] for label in CATEGORIES}

print("Scanning eligible historical objects...")

with gzip.open(SOURCE, "rt", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):

        object_id = row.get("objectID", "")

        if object_id in seen_ids or object_id in rejected_ids:
            continue

        if row.get("isPublicDomain", "").lower() != "true":
            continue

        url = row.get("primaryImageSmall", "").strip()
        if urlparse(url).hostname != "images.metmuseum.org":
            continue

        try:
            year = int(row["objectBeginDate"])
        except (ValueError, TypeError):
            continue

        if not (-10000 <= year <= 1500) or year == 0:
            continue

        name = row.get("objectName") or ""
        title = row.get("title") or ""

        if FRAGMENTS.search(name + " " + title):
            continue

        matches = [
            label for label, pattern in CATEGORIES.items()
            if re.search(pattern, name, re.I)
        ]

        if len(matches) != 1:
            continue

        label = matches[0]

        if label == "bowl" and re.search(
            r"\b(helmet|sallet|armor|armour)\b",
            name + " " + title, re.I
        ):
            continue

        candidates[label].append({
            "object_id": object_id,
            "label": label,
            "title": title,
            "object_name": name,
            "culture": row.get("culture", ""),
            "period": row.get("period", ""),
            "object_date": row.get("objectDate", ""),
            "object_begin_date": year,
            "department": row.get("department", ""),
            "medium": row.get("medium", ""),
            "image_url": url,
            "image_path": f"data/raw/images/{label}/{object_id}.jpg",
            "object_url": row.get("objectURL", "")
        })

rng = random.Random(42)
for label in CATEGORIES:
    rng.shuffle(candidates[label])

session = requests.Session()
session.headers.update({
    "User-Agent": "ArchaeoLens research project"
})

retry = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
session.mount("https://", HTTPAdapter(max_retries=retry))

new_downloads = 0
failures = 0
duplicates = 0

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    f.flush()

    for label in CATEGORIES:

        print(
            f"\n[{label.upper()}] "
            f"{counts[label]}/{TARGET} "
            f"| candidates: {len(candidates[label])}",
            flush=True
        )

        for obj in candidates[label]:
            if counts[label] >= TARGET:
                break

            object_id = obj["object_id"]
            path = ROOT / obj["image_path"]
            path.parent.mkdir(parents=True, exist_ok=True)

            digest = image_hash(path)

            if digest is None:
                try:
                    response = session.get(
                        obj["image_url"],
                        timeout=(10, 30)
                    )
                    response.raise_for_status()

                    if len(response.content) > 20_000_000:
                        raise ValueError("Image too large")

                    with Image.open(
                        io.BytesIO(response.content)
                    ) as source:
                        image = ImageOps.exif_transpose(
                            source
                        ).convert("RGB")

                    if min(image.size) < 128:
                        raise ValueError("Image too small")

                    image.thumbnail((768, 768))

                    temporary = path.with_suffix(".part")
                    image.save(
                        temporary,
                        format="JPEG",
                        quality=90
                    )
                    temporary.replace(path)

                    digest = image_hash(path)
                    if digest is None:
                        raise ValueError("Invalid saved image")

                    new_downloads += 1
                    time.sleep(0.15)

                except (
                    requests.RequestException,
                    OSError,
                    ValueError
                ):
                    failures += 1
                    continue

            if digest in seen_hashes:
                duplicates += 1
                continue

            writer.writerow(obj)
            f.flush()

            records.append(obj)
            seen_ids.add(object_id)
            seen_hashes.add(digest)
            counts[label] += 1

            if counts[label] % 25 == 0:
                print(
                    f"  Verified: {counts[label]}/{TARGET}",
                    flush=True
                )

        print(
            f"Completed {label}: {counts[label]}/{TARGET}",
            flush=True
        )

print("\n=== BALANCED DATASET SUMMARY ===")

for label in CATEGORIES:
    print(f"{label}: {counts[label]}")

print("TOTAL:", sum(counts.values()))
print("New downloads:", new_downloads)
print("Download failures:", failures)
print("Duplicate candidates skipped:", duplicates)
print("Manifest:", OUTPUT)

if all(counts[label] == TARGET for label in CATEGORIES):
    print("\nSUCCESS: BALANCED DATASET READY")
else:
    print("\nWARNING: Some categories remain below target")
