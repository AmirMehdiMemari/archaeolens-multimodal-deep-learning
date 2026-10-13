from pathlib import Path
from PIL import Image, ImageOps
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
METADATA = APP / "gallery_metadata.csv"
THUMBNAILS = APP / "assets" / "thumbnails"
BACKUP = ROOT / "data/processed/gallery_metadata_before_packaging.csv"

THUMBNAILS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    METADATA,
    dtype={"object_id": str}
).fillna("")

assert len(df) == 700, "Unexpected gallery size"
assert df["object_id"].is_unique, "Duplicate object IDs"

# Keep original metadata for reproducibility
if not BACKUP.exists():
    shutil.copy2(METADATA, BACKUP)

created = 0
missing = []

for i, row in df.iterrows():
    object_id = str(row["object_id"]).strip()

    relative = Path(
        "app/assets/thumbnails"
    ) / f"{object_id}.jpg"

    destination = ROOT / relative
    source = ROOT / str(row["image_path"])

    if not destination.is_file():
        if not source.is_file():
            missing.append(object_id)
            continue

        with Image.open(source) as original:
            image = ImageOps.exif_transpose(
                original
            ).convert("RGB")

            image.thumbnail((320, 320))

            image.save(
                destination,
                format="JPEG",
                quality=80,
                optimize=True
            )

        created += 1

    with Image.open(destination) as check:
        check.verify()

    df.at[i, "image_path"] = relative.as_posix()

if missing:
    raise RuntimeError(
        f"Missing gallery images: {missing[:20]}"
    )

# Update gallery metadata only after all images verify
temporary = APP / "gallery_metadata.tmp.csv"
df.to_csv(temporary, index=False)
temporary.replace(METADATA)

total_mb = sum(
    file.stat().st_size
    for file in THUMBNAILS.glob("*.jpg")
) / (1024 ** 2)

required = [
    APP / "gallery_index.pt",
    APP / "clip_linear_probe_best.pt",
    APP / "gallery_metadata.csv",
    APP / "streamlit_app.py"
]

assert all(path.is_file() for path in required)
assert all(
    (ROOT / path).is_file()
    for path in df["image_path"]
)

print("\n=== ARCHAEOLENS DEPLOYMENT PACKAGE ===")
print("Gallery records:", len(df))
print("Thumbnails created:", created)
print("Available thumbnails:", len(list(THUMBNAILS.glob("*.jpg"))))
print("Thumbnail storage (MB):", round(total_mb, 2))
print("Trained classifier:", "READY")
print("Retrieval embeddings:", "READY")
print("Streamlit application:", "READY")
print("Missing images:", len(missing))
print("\nGALLERY PACKAGING COMPLETE")
