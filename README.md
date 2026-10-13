# ArchaeoLens

### Multimodal Deep Learning for Archaeological Artifact Intelligence

ArchaeoLens is an end-to-end research prototype combining computer vision,
vision-language models, and an interactive web application to explore
archaeological artifacts from museum collections.

Users can upload artifact photographs, predict object categories, find
visually similar historical objects, and search the collection using
natural-language descriptions.

## Features

- Image classification across five artifact categories
- Image-to-image similarity retrieval
- Natural-language artifact search
- Museum object metadata and source links
- Interactive Streamlit application
- Reproducible training and evaluation notebooks

## Dataset

**Source:** The Metropolitan Museum of Art Open Access Collection

https://github.com/metmuseum/openaccess

Five artifact categories:

| Category | Images |
|---|---:|
| Vase | 200 |
| Bowl | 200 |
| Jar | 200 |
| Figurine | 200 |
| Amulet | 200 |
| **Total** | **1,000** |

Dataset split:

- Training: 700 images
- Validation: 150 images
- Testing: 150 images
- Random seed: 42

Images were filtered using museum public-domain status, historical
date metadata, object categories, and image-quality checks.

The original research dataset is excluded from Git.

## Deep Learning Architecture

### ResNet50 Baseline

- ImageNet-pretrained ResNet50
- Frozen convolutional feature extractor
- Trained five-class classification head
- Early stopping based on validation Macro F1

### CLIP ViT-B/32

- Pretrained OpenCLIP vision-language encoder
- LAION-2B pretrained weights
- 512-dimensional normalized visual embeddings
- Five-class trained linear probe
- Cosine similarity for artifact retrieval
- Text embeddings for natural-language search

## Experimental Results

### Classification — Validation

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| ResNet50 frozen backbone | 81.33% | 0.8116 |
| CLIP zero-shot | 81.33% | 0.8072 |
| CLIP linear probe | **87.33%** | **0.8723** |

### Final Held-Out Test Results

The selected CLIP linear probe was evaluated on 150 test images
without additional training.

| Metric | Result |
|---|---:|
| Classification accuracy | **84.00%** |
| Classification Macro F1 | **0.8402** |
| Correct predictions | 126 / 150 |
| Retrieval Top-1 category accuracy | **82.00%** |
| Retrieval Hit@5 | **96.67%** |
| Retrieval Precision@5 | **75.87%** |

Retrieval evaluation uses 150 test queries against a 700-image
training gallery.

Retrieval relevance is approximated using matching museum object
categories, rather than expert archaeological relevance judgments.

## Application

The Streamlit application provides two search modes.

**Image search**
1. Upload an artifact photograph.
2. Predict its artifact category.
3. Retrieve the five most visually similar museum objects.
4. View museum descriptions and original collection links.

**Text search**
1. Describe an artifact in natural language.
2. Encode the description using CLIP.
3. Retrieve related museum artifacts using cosine similarity.

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501

The pretrained CLIP weights are downloaded on the first run,
so an internet connection is required.

For research notebooks and experiments, also install:

```bash
pip install -r requirements-research.txt
```

The original research images are not included in the repository.
The data preparation scripts, museum metadata source, and split
manifests document the research workflow. Regenerating the dataset
may produce different files if the museum collection changes.

## Repository Structure

```text
archaeolens-multimodal-deep-learning/
├── app/
│   ├── streamlit_app.py
│   ├── gallery_index.pt
│   ├── gallery_metadata.csv
│   ├── clip_linear_probe_best.pt
│   └── assets/thumbnails/
├── data/manifests/
├── notebooks/
├── src/
├── reports/
│   └── figures/
├── models/
├── README.md
├── requirements.txt
└── requirements-research.txt
```

## Limitations

- Relatively small curated dataset of 1,000 museum images
- Museum category labels can be visually ambiguous
- Limited generalization evidence for user-captured photographs
- Similar objects may belong to related museum collections
- One identical image URL occurred across training and validation
- Removing that overlapping validation image changed Macro F1
  from 0.8723 to 0.8718
- The original checkpoint was selected using the full validation set
- Perceptual-hash screening does not exclude every near-duplicate
- Model scores are not calibrated probabilities

The system is a research prototype, not a substitute for
professional archaeological identification.

## Acknowledgments

Artifact metadata and public-domain images:
The Metropolitan Museum of Art Open Access Collection.

Vision-language model: OpenCLIP.

Development: PyTorch, Torchvision, and Streamlit.

## Project Status

Functional local research prototype with completed baseline
experiments, held-out test evaluation, and interactive retrieval.

Cloud deployment and independent external-image evaluation remain
future validation steps.
