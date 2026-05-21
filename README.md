<div align="center">

# DeepFake Detection using CSWin Transformer

**Transformer-based DeepFake detection with Explainable AI visualization**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.13%2B-red?logo=keras)](https://keras.io)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-blueviolet?logo=gradio)](https://gradio.app)

</div>

---

## Overview

This project presents an advanced **DeepFake Detection System** that classifies face images as **REAL** or **FAKE** using the **CSWin Transformer** — a vision transformer architecture with Cross-Shaped Window Self-Attention that reduces computational complexity from **O(N²)** to **O(N√N)** compared to standard ViTs.

The system integrates **Explainable AI (XAI)** to visually explain every prediction, highlighting the exact facial regions responsible for the decision — improving interpretability and trust.

**Pipeline:** Raw Image → MTCNN Face Detection → CSWin Transformer → REAL / FAKE + XAI Explanation

---

## Project Structure

```
deepfake-detection-cswin-transformer/
│
├── src/
│   ├── gradio_app.py           ← Web UI 
│   ├── predict.py              ← CLI single-image prediction + XAI
│   ├── train.py                ← Training entry point
│   ├── preprocess.py           ← MTCNN face extraction entry point
│   ├── cswin_transformer.py    ← Model architecture + dataset pipeline
│   ├── explainability.py       ← IG / SmoothGrad / Occlusion methods
│   ├── mtcnn_face_detection.py ← MTCNN face extractor class
│   ├── visualization.py        ← Evaluation plots
│   └── utils.py                ← Shared path and image helpers
│
├── data/
│   ├── raw/                    ← Place your raw dataset here
│   │   ├── train/real/
│   │   ├── train/fake/
│   │   ├── validation/real/
│   │   ├── validation/fake/
│   │   ├── test/real/
│   │   └── test/fake/
│   └── mtcnn_output/           ← Auto-created after preprocessing
│
├── outputs/                    ← Saved model, plots, predictions (auto-created)
├── requirements.txt
└── README.md
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/deepfake-detection-cswin-transformer.git
cd deepfake-detection-cswin-transformer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Pretrained Model

The trained model file (`cswin_best.keras`) is not included in this repository.

You can either:

### Option 1 — Train the model from scratch

Run:

```bash
python src/train.py
```

The trained model will automatically be saved to:

```bash
outputs/cswin_best.keras
```

---

### Option 2 — Download the pretrained model

Download the pretrained CSWin Transformer model from:

https://drive.google.com/file/d/1i7SdDouhiganJLIntYYHwdH9DDhuq9CH/view?usp=sharing

After downloading, place the model file inside:

```bash
outputs/
```

Final structure:

```bash
outputs/
└── cswin_best.keras
```

---

## Quick Start

### Run the Web App

**Step 1 — Set the model path** (choose one):

```bash
# Option A — environment variable (recommended)
# Windows
set MODEL_PATH=C:\full\path\to\cswin_best.keras

# Linux / macOS
export MODEL_PATH=/full/path/to/cswin_best.keras

# Option B — place the model file at:
outputs/cswin_best.keras
```

**Step 2 — Launch**
```bash
cd src
python gradio_app.py
```

Open **http://127.0.0.1:7860** in your browser. Upload a face image — the verdict appears instantly and the full XAI explanation follows in ~30 seconds.

---

## Training From Scratch

**Step 1 — Preprocess (MTCNN face extraction)**
```bash
cd src
export DATA_DIR=../data/raw          # Linux/macOS
set DATA_DIR=..\data\raw             # Windows

python preprocess.py
```
Extracted faces are saved to `data/mtcnn_output/`.

**Step 2 — Train**
```bash
python train.py
```
Best checkpoint saves automatically to `outputs/cswin_best.keras`. Training resumes from the last checkpoint if interrupted.

**Step 3 — Evaluate**
```bash
python visualization.py
```
Saves a 6-panel evaluation figure to `outputs/eval_plots/cswin_evaluation.png`.

**Step 4 — Predict (CLI)**
```bash
python predict.py --image path/to/face.jpg
```

---

## Model Architecture

### MTCNN — Face Detection & Preprocessing

**MTCNN (Multi-Task Cascaded Convolutional Networks)** extracts accurate face crops before classification. It handles varying lighting, orientations, scales, and partial occlusions through three sequential stages:

| Stage | Network | Function |
|-------|---------|----------|
| 1 | **P-Net** (Proposal) | Fully convolutional; scans 12×12 regions across an image pyramid; generates candidate bounding boxes via regression + NMS |
| 2 | **R-Net** (Refinement) | Refines P-Net candidates; eliminates false positives; improves bounding box accuracy |
| 3 | **O-Net** (Output) | Final face verification; facial landmark localization; precise face alignment |

### CSWin Transformer — Classification

**CSWin Transformer** uses **Cross-Shaped Window Self-Attention (CSWSA)** — each attention layer splits the feature map into horizontal and vertical stripes and attends within each stripe in parallel. This captures both local and global spatial context efficiently.

```
Input Image (128×128)
      │
  PatchEmbedding (4×4 patches → 32×32 tokens, dim=64)
      │
  Stage 1: 2× CSWinBlock (heads=2,  split=2)  → PatchMerging (dim→128)
  Stage 2: 2× CSWinBlock (heads=4,  split=2)  → PatchMerging (dim→256)
  Stage 3: 6× CSWinBlock (heads=8,  split=4)  → PatchMerging (dim→512)
  Stage 4: 2× CSWinBlock (heads=16, split=2)
      │
  GlobalAveragePooling → Dropout → Dense(1, sigmoid)
      │
   REAL / FAKE
```

**Complexity advantage:**

| Method | Complexity |
|--------|-----------|
| Standard ViT (global attention) | O(N²) |
| **CSWin Transformer (stripe attention)** | **O(N√N)** |

---

## Explainable AI (XAI)

Every prediction is accompanied by three complementary attribution methods:

| Method | What it shows |
|--------|--------------|
| **Integrated Gradients** | Pixel-level evidence — red = pushes toward REAL, blue = pushes toward FAKE |
| **SmoothGrad** | Noise-averaged gradients for cleaner, less spiky saliency maps |
| **Occlusion Sensitivity** | Model-agnostic: which image patches change the prediction most when hidden |

Results are aggregated into an **anatomical region importance score** across five face zones: Forehead, Eyes/Brows, Nose, Mouth, Cheeks/Jaw.

The XAI figure includes:
- Input image with face bounding box
- Verdict card (label + confidence + raw score)
- 4 attribution overlay maps
- IG positive / negative decomposition
- Combined attribution with top-5 critical pixels marked
- Region importance bar chart

---

## Dataset

**Deep Fake Face Detection Dataset**
🔗 [kaggle.com/datasets/vasubhut/deep-fake-face-detection](https://www.kaggle.com/datasets/vasubhut/deep-fake-face-detection)

| Split | Real | Fake | Total |
|-------|------|------|-------|
| Train | 70,001 | 70,001 | 140,002 |
| Validation | 19,787 | 19,641 | 39,428 |
| Test | 5,413 | 5,492 | 10,905 |

- **Input image size:** 256×256 (resized to 128×128 after MTCNN extraction)
- **Classes:** Balanced (50% Real / 50% Fake)

---

## Experimental Environment

| Component | Details |
|-----------|---------|
| Platform | Kaggle Notebooks |
| GPU | NVIDIA Tesla T4 |
| CUDA | 11.2 |
| Framework | TensorFlow 2.x + Keras |
| Training epochs | 50 (with warmup cosine LR decay) |
| Optimizer | AdamW + gradient clipping |

---

## Applications

- Social media content verification
- Digital forensics and cybersecurity
- Identity verification systems
- News and media authenticity checking
- Academic research in AI safety

---

## Limitations

- Performance may degrade on out-of-distribution datasets not seen during training
- Heavily occluded or low-resolution faces reduce detection accuracy
- XAI computation (IG + SmoothGrad + Occlusion) takes some seconds per image on CPU

---

## Future Work

- Extend to **video deepfake detection** (temporal consistency analysis)
- Lightweight transformer optimization for **mobile deployment**
- **Federated learning** for privacy-preserving training
- Improved robustness across heterogeneous and cross-dataset scenarios

---

## Environment Variables

All paths can be overridden without editing code:

| Variable | Default | Used by |
|----------|---------|---------|
| `MODEL_PATH` | `outputs/cswin_best.keras` | gradio_app, predict, visualization |
| `DATA_DIR` | `data/mtcnn_output` | train, visualization |
| `PROCESSED_DIR` | `data/mtcnn_output` | preprocess |
| `OUT_DIR` | `outputs/` (subfolder varies) | visualization, predict, gradio_app |
| `THRESHOLD` | `0.5` | predict, gradio_app |
