import csv
import gzip
import io
import random
import re
import time
import argparse
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SOURCE = RAW / "met_openaccess.csv.gz"
OUTPUT = RAW / "historical_metadata.csv"
PILOT = RAW / "metadata.csv"
IMAGES = RAW / "images"

csv.field_size_limit(20_000_000)

CATEGORIES = {
    "vase": r"\b(vase|amphora|hydria|krater)\b",
    "bowl": r"\bbowls?\b",
    "jar": r"\bjars?\b",
    "figurine": r"\b(figurine|statuette)\b",
    "amulet": r"\bamulets?\b",
}

FIELDS = [
    "object_id", "label", "title", "object_name",
    "culture", "period", "object_date",
    "object_begin_date", "department", "medium",
    "image_url", "image_path", "object_url"
]

def usable(path):
    try:
        with Image.open(path) as image:
            return min(image.size) >= 128
    except (OSError, ValueError):
        return False

def main(target):
    rng = random.Random(42)
    candidates = {label: [] for label in CATEGORIES}

    print("Reading historical museum metadata...")

    with gzip.open(SOURCE, "rt", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
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

            name = (row.get("objectName") or "").strip()
            title = row.get("title") or ""

            matches = [
                label for label, pattern in CATEGORIES.items()
                if re.search(pattern, name, re.I)
            ]

            if len(matches) != 1:
                continue

            label = matches[0]

            # Reject obvious non-vessel bowl matches.
            if label == "bowl" and re.search(
                r"\b(helmet|sallet|armor|armour)\b",
                name + " " + title, re.I
            ):
                continue

            candidates[label].append({
                "object_id": row["objectID"],
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
                "object_url": row.get("objectURL", "")
            })

    # Reuse previously downloaded pilot images.
    pilot = {}
    if PILOT.exists():
        with PILOT.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                pilot[row["object_id"]] = row

    # Resume previously verified historical downloads.
    existing = []
    seen = set()

    if OUTPUT.exists():
        with OUTPUT.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                object_id = row["object_id"]
                path = ROOT / row["image_path"]

                if object_id not in seen and usable(path):
                    existing.append(row)
                    seen.add(object_id)

    counts = Counter(row["label"] for row in existing)
    failures = Counter()
    reused = 0
    downloaded = 0

    retry = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )

    session = requests.Session()
    session.headers.update({
        "User-Agent": "ArchaeoLens academic research dataset downloader"
    })
    session.mount("https://", HTTPAdapter(max_retries=retry))

    IMAGES.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        f.flush()

        for label in CATEGORIES:
            pool = candidates[label]
            rng.shuffle(pool)

            # Prefer existing pilot images to avoid downloading twice.
            pool.sort(
                key=lambda x: x["object_id"] not in pilot
            )

            (IMAGES / label).mkdir(parents=True, exist_ok=True)

            print(
                f"\n[{label.upper()}] "
                f"{counts[label]}/{target} "
                f"| candidates: {len(pool)}",
                flush=True
            )

            for obj in pool:
                if counts[label] >= target:
                    break

                object_id = obj["object_id"]

                if object_id in seen:
                    continue

                relative = (
                    f"data/raw/images/{label}/{object_id}.jpg"
                )
                image_path = ROOT / relative

                # Check for an existing usable image.
                previous = pilot.get(object_id)

                if previous and previous["label"] == label:
                    old_relative = previous["image_path"]
                    old_path = ROOT / old_relative

                    if usable(old_path):
                        relative = old_relative
                        image_path = old_path

                if usable(image_path):
                    reused += 1
                else:
                    try:
                        response = session.get(
                            obj["image_url"],
                            timeout=(10, 30)
                        )
                        response.raise_for_status()

                        if len(response.content) > 20_000_000:
                            raise ValueError("Image exceeds size limit")

                        with Image.open(
                            io.BytesIO(response.content)
                        ) as source_image:
                            image = ImageOps.exif_transpose(
                                source_image
                            ).convert("RGB")

                        if min(image.size) < 128:
                            raise ValueError("Image too small")

                        image.thumbnail((768, 768))

                        temp_path = image_path.with_suffix(".tmp")
                        image.save(
                            temp_path,
                            format="JPEG",
                            quality=90
                        )
                        temp_path.replace(image_path)

                        downloaded += 1
                        time.sleep(0.15)

                    except (requests.RequestException,
                            OSError, ValueError) as error:
                        failures[type(error).__name__] += 1
                        continue

                obj["image_path"] = relative

                writer.writerow(obj)
                f.flush()

                seen.add(object_id)
                counts[label] += 1

                if counts[label] % 25 == 0:
                    print(
                        f"  Verified: {counts[label]}/{target}",
                        flush=True
                    )

            print(
                f"Completed {label}: {counts[label]}/{target}",
                flush=True
            )

    print("\n=== HISTORICAL DATASET SUMMARY ===")
    for label in CATEGORIES:
        print(f"{label}: {counts[label]}")

    print("TOTAL:", sum(counts.values()))
    print("New downloads:", downloaded)
    print("Images reused:", reused)
    print("Download failures:", dict(failures))
    print("Manifest:", OUTPUT)

    if any(counts[label] < target for label in CATEGORIES):
        print("WARNING: Some classes are below target.")
    else:
        print("SUCCESS: All class targets reached.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=200)
    args = parser.parse_args()

    main(args.per_class)
