from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"

st.set_page_config(
    page_title="ArchaeoLens",
    page_icon="🏺",
    layout="wide"
)

torch.set_num_threads(4)

@st.cache_resource(show_spinner="Loading ArchaeoLens AI models...")
def load_engine():
    index = torch.load(
        APP / "gallery_index.pt",
        map_location="cpu",
        weights_only=True
    )

    checkpoint = torch.load(
        APP / "clip_linear_probe_best.pt",
        map_location="cpu",
        weights_only=True
    )

    metadata = pd.read_csv(
        APP / "gallery_metadata.csv"
    ).fillna("")

    features = F.normalize(
        index["embeddings"].float(),
        dim=1
    )

    assert len(metadata) == len(features), (
        "Gallery embeddings and metadata do not match."
    )

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            checkpoint["architecture"],
            pretrained=checkpoint["pretrained"],
            device="cpu"
        )
    )

    model.eval()

    tokenizer = open_clip.get_tokenizer(
        checkpoint["architecture"]
    )

    classes = checkpoint["classes"]

    head = nn.Linear(
        features.shape[1],
        len(classes)
    )

    head.load_state_dict(
        checkpoint["head_state_dict"]
    )
    head.eval()

    return (
        model, preprocess, tokenizer,
        head, features, metadata,
        classes, checkpoint["feature_scale"]
    )

def find_similar(query_embedding, features, k=5):
    query_embedding = F.normalize(
        query_embedding.float(),
        dim=-1
    )

    similarities = (
        query_embedding @ features.T
    ).squeeze(0)

    scores, indices = torch.topk(
        similarities,
        k=min(k, len(features))
    )

    return list(zip(
        indices.tolist(),
        scores.tolist()
    ))

def display_results(matches, metadata):
    st.subheader("Related Museum Artifacts")

    columns = st.columns(len(matches))

    for column, (index, score) in zip(columns, matches):
        row = metadata.iloc[index]

        with column:
            local_image = ROOT / str(row["image_path"])

            if local_image.is_file():
                st.image(
                    str(local_image),
                    use_container_width=True
                )
            elif row["image_url"]:
                st.image(
                    row["image_url"],
                    use_container_width=True
                )

            st.markdown(f"**{row['title']}**")

            st.caption(
                f"Category: {row['label']}"
            )

            st.caption(
                f"Culture: {row['culture'] or 'Unknown'}"
            )

            st.caption(
                f"Period: {row['period'] or row['object_date'] or 'Unknown'}"
            )

            st.caption(
                f"Cosine similarity: {score:.3f}"
            )

            if row["object_url"]:
                st.markdown(
                    f"[View museum record]({row['object_url']})"
                )

st.title("🏺 ArchaeoLens")
st.markdown(
    "**Multimodal Deep Learning for Archaeological Artifact Intelligence**"
)

st.caption(
    "Explore historical objects through computer vision, "
    "vision-language embeddings, and museum collection data."
)

required_files = [
    "gallery_index.pt",
    "gallery_metadata.csv",
    "clip_linear_probe_best.pt"
]

missing = [
    name for name in required_files
    if not (APP / name).is_file()
]

if missing:
    st.error(f"Missing application assets: {missing}")
    st.stop()

(
    model, preprocess, tokenizer,
    head, gallery_features, metadata,
    classes, feature_scale
) = load_engine()

with st.sidebar:
    st.header("Model Information")
    st.write("Vision encoder: CLIP ViT-B/32")
    st.write("Classifier: Trained linear probe")
    st.write(f"Gallery: {len(metadata)} museum images")
    st.write("Validation accuracy: 87.33%")
    st.write("Image retrieval Hit@5: 96.67%")
    st.caption(
        "Results are experimental. Model predictions are "
        "not verified archaeological identifications."
    )

image_tab, text_tab = st.tabs([
    "📷 Search by Image",
    "📝 Search by Text"
])

with image_tab:
    st.subheader("Upload an Artifact Photograph")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        uploaded_image = ImageOps.exif_transpose(
            Image.open(uploaded_file)
        ).convert("RGB")

        st.image(
            uploaded_image,
            caption="Uploaded artifact",
            width=300
        )

        with st.spinner("Analyzing artifact..."):
            with torch.inference_mode():
                image_tensor = preprocess(
                    uploaded_image
                ).unsqueeze(0)

                image_embedding = F.normalize(
                    model.encode_image(
                        image_tensor
                    ).float(),
                    dim=-1
                )

                logits = head(
                    image_embedding * feature_scale
                )

                probabilities = torch.softmax(
                    logits, dim=-1
                )[0]

                best_class = probabilities.argmax().item()

                matches = find_similar(
                    image_embedding,
                    gallery_features
                )

        st.success(
            f"Predicted artifact category: "
            f"{classes[best_class].title()}"
        )

        st.caption(
            f"Model score: {probabilities[best_class].item():.1%} "
            "(not a calibrated confidence estimate)"
        )

        display_results(matches, metadata)

with text_tab:
    st.subheader("Natural Language Artifact Search")

    query = st.text_input(
        "Describe the artifact",
        placeholder=(
            "Ancient Greek amphora with black figures"
        )
    )

    if st.button("Search Museum Collection"):
        if not query.strip():
            st.warning("Enter an artifact description.")
        else:
            with st.spinner("Searching artifacts..."):
                with torch.inference_mode():
                    tokens = tokenizer([query])

                    text_embedding = F.normalize(
                        model.encode_text(
                            tokens
                        ).float(),
                        dim=-1
                    )

                    matches = find_similar(
                        text_embedding,
                        gallery_features
                    )

            display_results(matches, metadata)

st.divider()
st.caption(
    "ArchaeoLens Research Prototype | "
    "Museum data: The Metropolitan Museum of Art Open Access"
)
