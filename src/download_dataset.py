import csv
import io
import re
import time
import argparse
from pathlib import Path

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMAGES = RAW / "images"
METADATA = RAW / "metadata.csv"

API = "https://collectionapi.metmuseum.org/public/collection"

CATEGORIES = {
    "vase": (["vase", "amphora"], r"\b(vase|amphora|hydria|krater)\b"),
    "bowl": (["bowl"], r"\bbowls?\b"),
    "jar": (["jar"], r"\bjars?\b"),
    "figurine": (["figurine", "statuette"], r"\b(figurine|statuette)\b"),
    "amulet": (["amulet"], r"\bamulets?\b"),
}

FIELDS = [
    "object_id", "label", "title", "object_name",
    "culture", "period", "object_date", "department",
    "medium", "image_url", "image_path"
]

def main(per_class):
    RAW.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "ArchaeoLens-Research/1.0"})
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))

    existing = []
    if METADATA.exists():
        with METADATA.open(encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))

    valid_rows = [
        row for row in existing
        if (ROOT / row["image_path"]).is_file()
    ]
    seen = {str(row["object_id"]) for row in valid_rows}

    with METADATA.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(valid_rows)

        for label, (terms, pattern) in CATEGORIES.items():
            folder = IMAGES / label
            folder.mkdir(parents=True, exist_ok=True)
            count = sum(row["label"] == label for row in valid_rows)

            print(f"\n[{label.upper()}] {count}/{per_class}")

            for term in terms:
                if count >= per_class:
                    break

                offset = 0

                while count < per_class and offset < 10000:
                    try:
                        response = session.get(
                            f"{API}/v1.1/search",
                            params={
                                "q": term,
                                "hasImages": "true",
                                "offset": offset,
                                "limit": 500
                            },
                            timeout=30
                        )
                        response.raise_for_status()
                        result = response.json()
                    except requests.RequestException as error:
                        print("Search error:", error)
                        break

                    ids = result.get("objectIDs") or []
                    if not ids:
                        break

                    for object_id in ids:
                        if count >= per_class:
                            break
                        if str(object_id) in seen:
                            continue

                        try:
                            response = session.get(
                                f"{API}/v1/objects/{object_id}",
                                timeout=30
                            )
                            response.raise_for_status()
                            obj = response.json()

                            name = obj.get("objectName") or ""
                            image_url = obj.get("primaryImageSmall") or ""

                            if not obj.get("isPublicDomain"):
                                continue
                            if not image_url:
                                continue
                            if not re.search(pattern, name, re.I):
                                continue

                            image_response = session.get(
                                image_url, timeout=40
                            )
                            image_response.raise_for_status()

                            image = Image.open(
                                io.BytesIO(image_response.content)
                            ).convert("RGB")

                            if min(image.size) < 128:
                                continue

                            image.thumbnail((768, 768))
                            relative_path = (
                                f"data/raw/images/{label}/{object_id}.jpg"
                            )
                            image.save(
                                ROOT / relative_path,
                                format="JPEG",
                                quality=90
                            )

                            row = {
                                "object_id": object_id,
                                "label": label,
                                "title": obj.get("title", ""),
                                "object_name": name,
                                "culture": obj.get("culture", ""),
                                "period": obj.get("period", ""),
                                "object_date": obj.get("objectDate", ""),
                                "department": obj.get("department", ""),
                                "medium": obj.get("medium", ""),
                                "image_url": image_url,
                                "image_path": relative_path
                            }

                            writer.writerow(row)
                            f.flush()
                            seen.add(str(object_id))
                            count += 1

                            if count % 10 == 0:
                                print(f"  Downloaded: {count}/{per_class}")

                        except (
                            requests.RequestException,
                            OSError,
                            ValueError
                        ) as error:
                            print(f"  Skipped {object_id}: {error}")

                        time.sleep(0.2)

                    offset += len(ids)
                    if offset >= result.get("total", 0):
                        break

            print(f"Completed {label}: {count}/{per_class}")

    print("\n=== DATASET SUMMARY ===")
    with METADATA.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for label in CATEGORIES:
        n = sum(row["label"] == label for row in rows)
        print(f"{label}: {n}")

    print(f"Total images: {len(rows)}")
    print(f"Metadata: {METADATA}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=20)
    args = parser.parse_args()
    main(args.per_class)
