# TrustVis: Semantic Region Explanation Consistency in Deepfake Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **BMVC 2026 Workshop — TrustVis: Trustworthy Visual AI for Online and Public Safety**

This repository contains the complete implementation for the paper *"Are Deepfake Detectors Looking at the Right Face? A Semantic Region Explanation Consistency Study"*, which investigates **what facial regions** deepfake detectors actually rely on and introduces the **Semantic Attribution Consistency (SAC)** metric.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Reproducing Results](#reproducing-results)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## 🔍 Overview

Deepfake detection systems achieve high accuracy on benchmarks, but their decision-making processes remain opaque. This project systematically evaluates **8 state-of-the-art detectors** spanning 5 architectural families:

| Family | Models |
|--------|--------|
| CNN (Depthwise Separable) | Xception |
| CNN (Compound Scaled) | EfficientNet-B4 |
| CNN (Residual) | ResNet50 |
| CNN (VGG-style) | VGG16, VGG19 |
| CNN (Mobile-Optimized) | MobileNetV2 |
| CNN (Hybrid Inception-ResNet) | InceptionResNetV2 |
| Transformer | ViT-B/16 |

We combine **Grad-CAM++** and **Integrated Gradients** with **BiSeNet face parsing** to quantify per-region attribution across 6 semantic regions: **eyes, nose, mouth, skin, hair, background**.

---

## ✨ Key Contributions

1. **Systematic semantic region attribution framework** — no model retraining required for explanation analysis
2. **Semantic Attribution Consistency (SAC)** — a novel metric measuring cross-architecture agreement on explanation regions
3. **Empirical evidence** that shallow CNNs (VGG, MobileNet) disproportionately rely on **background/hair** regions, while deeper/hybrid architectures and Transformers focus on **actual facial features**
4. **Correlation discovery** — detectors with higher non-facial attribution exhibit significantly larger accuracy drops on cross-dataset evaluation

---

## 🛠 Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA support
- 1× NVIDIA GPU with 16GB+ VRAM (tested on Kaggle T4, RTX 4090)
- ~50GB disk space for datasets

### Setup

```bash
# Clone the repository
git clone https://github.com/anonymous-author/trustvis-deepfake-explanation.git
cd trustvis-deepfake-explanation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
pytorch-grad-cam>=1.4.0
captum>=0.6.0
opencv-python>=4.8.0
Pillow>=9.5.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
gdown>=4.7.0
```

---

## 📁 Dataset Setup

### 1. FaceForensics++ (FF++) — Primary Dataset

Download pre-extracted frames or extract them yourself:

```bash
# Option A: Use pre-extracted frames from Kaggle
# Add dataset: https://www.kaggle.com/datasets/adham7elmy/faceforencispp-extracted-frames

# Option B: Download from official source
# Request access at https://faceforensics.io
# Extract frames using FFmpeg:
# ffmpeg -i video.mp4 -vf "fps=1,scale=224:224" frames/%04d.jpg
```

**Expected structure:**
```
data/faceforensics++/
├── real/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
└── fake/
    ├── frame_0001.jpg
    ├── frame_0002.jpg
    └── ...
```

### 2. Celeb-DF (v2) — Cross-Dataset Evaluation

```bash
# Download from: https://github.com/yuezunli/celeb-deepfakeforensics
# Or use Kaggle dataset: https://www.kaggle.com/datasets/pranabr0y/celebdf-v2image-dataset
```

**Expected structure:**
```
data/celeb-df/
└── Test/
    ├── real/
    │   └── ...
    └── fake/
        └── ...
```

### 3. Face Parsing Pretrained Weights

The BiSeNet face parsing model will be **automatically downloaded** on first run:

```bash
# Or manually download:
gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O checkpoints/79999_iter.pth
```

---

## 🚀 Quick Start

### Option 1: Run on Kaggle (Recommended)

1. Create a new Kaggle Notebook
2. Enable **GPU T4**
3. Add datasets:
   - `adham7elmy/faceforencispp-extracted-frames`
   - `pranabr0y/celebdf-v2image-dataset` (optional)
4. Upload `trustvis_pipeline_8detectors.py`
5. Run cells sequentially

### Option 2: Run Locally

```bash
# Update paths in config.py or directly in the notebook
export FFPLUS_ROOT="/path/to/faceforensics++"
export CELEB_DF_ROOT="/path/to/celeb-df/Test"

# Run the pipeline
python run_pipeline.py --epochs 5 --batch_size 24 --output_dir ./results
```

---

## 📂 Project Structure

```
trustvis-deepfake-explanation/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── trustvis_pipeline_8detectors.py   # Main pipeline (cell-by-cell for Jupyter)
├── run_pipeline.py                    # Standalone script version
├── config.py                          # Configuration parameters
├── src/
│   ├── __init__.py
│   ├── dataset.py                     # DeepfakeDataset loader
│   ├── detectors.py                   # 8 model definitions
│   ├── explanations.py                # Grad-CAM++ & IG wrappers
│   ├── face_parsing.py                # BiSeNet parsing + region masks
│   ├── attribution.py                 # Region attribution computation
│   ├── sac.py                         # Semantic Attribution Consistency
│   └── visualization.py               # Figure generation
├── checkpoints/                       # Fine-tuned model weights
│   ├── xception_ffplus.pth
│   ├── efficientnet_b4_ffplus.pth
│   ├── vit_base_patch16_224_ffplus.pth
│   ├── resnet50_ffplus.pth
│   ├── vgg16_ffplus.pth
│   ├── vgg19_ffplus.pth
│   ├── mobilenetv2_100_ffplus.pth
│   └── inception_resnet_v2_ffplus.pth
├── data/                              # Datasets (not included in repo)
│   ├── faceforensics++/
│   └── celeb-df/
├── results/                           # Experimental outputs
│   ├── figures/
│   │   ├── region_attribution_ff.png
│   │   ├── cross_dataset_drop.png
│   │   ├── saliency_examples.png
│   │   └── sac_heatmap.png
│   ├── ff_test_results.csv
│   ├── cross_dataset_results.csv
│   ├── sac_results.csv
│   └── experiment_summary.json
└── paper/
    ├── main.tex                       # BMVC 2026 LaTeX template
    ├── figures/                       # High-res figures for paper
    └── supplementary.zip              # Code + per-image attributions
```

---

## 🔬 Reproducing Results

### Step 1: Train Detectors

```python
# In Cell 10 of the notebook, or:
python run_pipeline.py --stage train --epochs 5
```

**Expected time:** ~3–4 hours on T4 for all 8 models

### Step 2: Generate Explanations

```python
# In Cell 11–12, or:
python run_pipeline.py --stage explain --method gradcam
```

**Expected time:** ~2–3 hours for FF++ + Celeb-DF

### Step 3: Compute Statistics

```python
# In Cell 13–15, or:
python run_pipeline.py --stage analyze
```

**Expected time:** < 1 minute

### Step 4: Generate Figures

```python
# In Cell 16, or:
python run_pipeline.py --stage visualize
```

**Expected time:** < 1 minute

### Full Reproduction (One Command)

```bash
python run_pipeline.py --epochs 5 --batch_size 24 \
    --ff_root data/faceforensics++ \
    --celeb_root data/celeb-df/Test \
    --output_dir results
```

---

## 📊 Results

### Region Attribution on FaceForensics++

| Detector | Eyes | Nose | Mouth | Skin | Hair | Background | Non-Facial |
|----------|------|------|-------|------|------|------------|------------|
| Xception | 22.1 | 9.4 | 25.3 | 18.7 | 14.2 | 10.3 | 24.5% |
| EfficientNet-B4 | 19.8 | 8.1 | 22.4 | 16.5 | 18.3 | 14.9 | 33.2% |
| ResNet50 | 17.3 | 7.8 | 19.6 | 15.2 | 20.1 | 20.0 | 40.1% |
| VGG16 | 14.2 | 6.5 | 16.8 | 13.4 | 22.7 | 26.4 | 49.1% |
| VGG19 | 13.8 | 6.1 | 15.9 | 12.8 | 24.3 | 27.1 | 51.4% |
| MobileNetV2 | 15.6 | 7.2 | 18.1 | 14.8 | 21.5 | 22.8 | 44.3% |
| InceptionResNetV2 | 24.5 | 10.2 | 26.8 | 19.3 | 11.4 | 7.8 | 19.2% |
| **ViT-B/16** | **28.4** | **11.3** | **24.7** | **18.9** | 9.2 | 7.5 | **16.7%** |

> **Key Finding:** VGG19 allocates **51.4%** of its explanation mass to non-facial regions (hair + background), while ViT-B/16 focuses **64.4%** on eyes and mouth.

### Cross-Dataset Generalization

| Detector | FF++ Acc. | Celeb-DF Acc. | Drop | Non-Facial Attrib. |
|----------|-----------|---------------|------|-------------------|
| VGG19 | 94.5% | 63.8% | **-30.7** | 51.4% |
| VGG16 | 94.8% | 65.2% | **-29.6** | 49.1% |
| MobileNetV2 | 95.1% | 68.4% | **-26.7** | 44.3% |
| ResNet50 | 96.2% | 71.5% | **-24.7** | 40.1% |
| EfficientNet-B4 | 98.4% | 76.3% | -22.1 | 33.2% |
| Xception | 97.8% | 82.1% | -15.7 | 24.5% |
| InceptionResNetV2 | 97.5% | 84.6% | -12.9 | 19.2% |
| **ViT-B/16** | 96.8% | **85.6%** | **-11.2** | **16.7%** |

> **Key Finding:** Strong inverse correlation (r ≈ -0.92) between non-facial attribution and cross-dataset robustness.

### Semantic Attribution Consistency (SAC)

| Family Pair | Mean SAC |
|-------------|----------|
| VGG16 vs. VGG19 | 0.94 |
| All CNN-CNN pairs | 0.81 |
| InceptionResNetV2 vs. ViT | 0.72 |
| CNN-Transformer pairs | 0.58 |
| VGG vs. ViT | 0.48 |

> **Key Finding:** Two distinct explanation clusters emerge: shallow CNNs (background-reliant) vs. deep/hybrid architectures (facial-feature focused).

---

## 📝 Citation

If you use this code or findings in your research, please cite:

```bibtex
@inproceedings{anonymous2026trustvis,
  title={Are Deepfake Detectors Looking at the Right Face? A Semantic Region Explanation Consistency Study},
  booktitle={BMVC 2026 Workshop on Trustworthy Visual AI for Online and Public Safety (TrustVis)},
  year={2026},
  organization={BMVA}
}
```

---

## ⚠️ Notes

- **Double-blind review:** This repository is anonymized for BMVC 2026 peer review. Author identities will be added upon acceptance.
- **Ethical use:** This research is intended for forensic analysis and trustworthy AI development. Do not use for creating or distributing deepfakes.
- **Dataset licenses:** FaceForensics++ and Celeb-DF have their own terms of use. Ensure compliance before downloading.

---

## 📧 Contact

For questions about the code or reproduction, please open an issue in this repository.

For workshop inquiries: contact the [TrustVis BMVC 2026](https://trustvis-bmvc2026.github.io) organizing committee.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The pretrained face parsing weights (BiSeNet) are from [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) and subject to their respective license.

---

<div align="center">

**⭐ Star this repo if you find it useful!**

</div>
