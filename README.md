# TrustVis: Semantic Region Explanation Consistency in Deepfake Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **BMVC 2026 Workshop — TrustVis: Trustworthy Visual AI for Online and Public Safety** (poster)

Code and results for *"Are Deepfake Detectors Looking at the Right Face? A Semantic Region Explanation Consistency Study"*
by **Kazi Rifat Ahmed** (Daffodil International University) and **Abu Nokib Kamran** (University of South Wales).
We measure **which facial regions** deepfake detectors rely on, separate attention from region size with an
**area-normalised** score, and introduce **Semantic Attribution Consistency (SAC)** to compare detectors image by image.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Project Structure](#-project-structure)
- [Reproducing Results](#-reproducing-results)
- [Results](#-results)
- [Limitations](#-limitations)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Overview

Deepfake detectors reach 96–99% accuracy on FaceForensics++ but lose 30–43 accuracy points on Celeb-DF, and their decisions
remain opaque. We fine-tune **8 ImageNet-pretrained backbones from 7 architectural families** under one protocol:

| Family | Models |
|--------|--------|
| CNN (depthwise separable) | Xception |
| CNN (compound scaled) | EfficientNet-B4 |
| CNN (residual) | ResNet50 |
| CNN (plain VGG stacks) | VGG16, VGG19 |
| CNN (mobile, inverted residuals) | MobileNetV2 |
| CNN (Inception + residual) | Inception-ResNet-v2 |
| Transformer | ViT-B/16 |

For every test image we compute a **Grad-CAM++** map for the predicted class, parse the face with **BiSeNet**
(CelebAMask-HQ, 19 classes merged into 6 regions: **eyes, nose, mouth, skin, hair, background**) and measure:

- **Relative attribution** `R_{d,k}(x) = Σ S_d(x)·M_k / Σ S_d(x)`: share of explanation mass in region *k* (sums to 100%).
- **Area-normalised attribution** `E_{d,k} = mean R_{d,k} / mean area_k`: `E > 1` means more saliency per pixel than average.
- **SAC(d1, d2)** = mean over images of `cos(R_{d1}(x), R_{d2}(x))`, validated against pixel-level map correlation and
  shared Celeb-DF errors.

Grad-CAM++ maps that are all zero carry no spatial information. They are excluded and their rate is reported.

---

## ✨ Key Contributions

1. **A semantic region attribution framework** that works with any architecture without changing the detectors, plus an
   **area-normalised score** that separates attention from region size.
2. **Semantic Attribution Consistency (SAC)**, a per-image measure of whether two detectors allocate explanation mass to
   the same regions. It is checked against pixel-level agreement and error overlap.
3. **Architecture-dependent explanations.** Relative to their area, all detectors favour eyes + nose + mouth as a group.
   Xception, ResNet50 and Inception-ResNet-v2 form a tight cluster (SAC > 0.94, pixel-level r ≥ 0.73), while ViT-B/16
   diverges from every CNN at the pixel level.
4. **An exploratory observation:** detectors with *more* non-facial attribution on FF++ showed *smaller* FF++→Celeb-DF
   accuracy drops (r = −0.71, n = 8). The association does not hold for the AUC drop.

---

## 🛠 Installation

Tested on Kaggle (Python 3.12, one NVIDIA T4).

```bash
git clone https://github.com/kazirifatahmed/TrustVis-SemanticRegionAttribution.git
cd TrustVis-SemanticRegionAttribution
pip install -r requirements.txt
```

`requirements.txt`:

```
torch>=2.1
torchvision>=0.16
timm>=1.0
grad-cam>=1.5        # PyPI name; imported as pytorch_grad_cam
captum>=0.7          # Integrated Gradients (notebook 03, optional)
gdown
opencv-python-headless
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
pillow
tqdm
```

> Note: `pip install pytorch-grad-cam` fails. The package is published as **`grad-cam`**.

---

## 📁 Dataset Setup

| Dataset | Use | Source |
|---|---|---|
| FaceForensics++ (pre-extracted frames) | fine-tuning + in-distribution test | Kaggle `adham7elmy/faceforencispp-extracted-frames` (`real/`, `fake/<method>/`) |
| Celeb-DF v2 (image dataset) | cross-dataset test | Kaggle `pranabr0y/celebdf-v2image-dataset` (`Celeb_V2/Test/real`, `.../fake`) |
| BiSeNet face-parsing weights | region masks | downloaded automatically (`gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812`) |

Frames are resized to 224×224 without further cropping or alignment. Please follow the original FaceForensics++ and
Celeb-DF terms of use.

---

## 📂 Project Structure

```
TrustVis-SemanticRegionAttribution/
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/
│   ├── 01_original_experiment.ipynb          # fine-tunes the 8 detectors, writes per-image Grad-CAM++ results (paper run)
│   ├── 02_camera_ready_analysis.ipynb        # all paper tables/figures from the saved per-image results (CPU only)
│   └── 03_trustvis_pipeline_corrected.ipynb  # corrected end-to-end pipeline (target layers, empty maps, optional video split, IG)
├── results/                                  # exact outputs of notebook 02
│   ├── diagnosis_empty_maps.csv
│   ├── table1_ff_all.csv / .tex              # + table1_ff_{fake,real,correct_fake}.csv
│   ├── table2_generalisation.csv
│   ├── table3_pairs_ff.csv / .tex
│   ├── pairs_celebdf.csv
│   └── correlations.csv
├── figures/
│   ├── submitted_run/                        # figures used in the paper
│   └── video_split_run/                      # notebook 03 with SPLIT_BY_VIDEO=True (different models; not in the paper)
└── paper/                                    # camera-ready PDF
```

---

## 🔬 Reproducing Results

1. **Train and explain** with `notebooks/01_original_experiment.ipynb` on a GPU. Each detector is fine-tuned for
   **6 epochs** on 4,000 FF++ frames (AdamW, lr 1e-4, weight decay 1e-4, cosine schedule, batch 24, normalisation layers
   frozen). The notebook then writes per-image Grad-CAM++ results to `trustvis_results/attributions/`.
   About 2 h 15 min on a Kaggle T4.
2. **Analyse** with `notebooks/02_camera_ready_analysis.ipynb`. Attach the output of step 1 as an input; the pickles are
   found automatically. This produces every table in `results/` in a few minutes and needs no GPU.
3. **Optional:** `notebooks/03_trustvis_pipeline_corrected.ipynb` runs the corrected pipeline:
   - post-activation target layers;
   - explicit empty-map handling;
   - batched explanations;
   - Integrated Gradients;
   - an optional video-disjoint split (`SPLIT_BY_VIDEO=True`).

   To reuse the paper's checkpoints, copy `*_ffplus.pth` into `trustvis_results/checkpoints/` and keep
   `SPLIT_BY_VIDEO=False`.

---

## 📊 Results

All numbers are from the paper run (`results/*.csv`). FF++ test set: 2,000 frames (991 real / 1,009 fake). Celeb-DF
v2 test set: 1,000 frames (500 / 500).

### 1. Region attribution on FaceForensics++

Mean relative attribution **R (%)** over images with a non-empty map. Area-normalised **E** is in brackets
(E > 1 = more saliency per pixel than the image average). Features = eyes + nose + mouth; non-facial = hair + background.

| Detector | Eyes | Nose | Mouth | Skin | Hair | Background | Non-facial (E) | Features (E) | Empty maps |
|---|---|---|---|---|---|---|---|---|---|
| Xception | 2.5 | 5.3 | 3.0 | 48.3 | 15.3 | 25.6 | 40.9 (0.59) | 10.8 (2.42) | 0.00% |
| EfficientNet-B4 | 1.3 | 1.9 | 2.5 | 27.7 | 12.4 | 54.1 | 66.5 (0.95) | 5.7 (1.29) | 0.00% |
| ViT-B/16 | 6.0 | 4.1 | 3.9 | 47.9 | 12.3 | 25.9 | 38.2 (0.55) | 13.9 (3.15) | 54.65% |
| ResNet50 | 2.8 | 6.5 | 3.9 | 54.9 | 11.6 | 20.2 | **31.8 (0.46)** | 13.2 (2.98) | 0.00% |
| VGG16 | 2.2 | 4.7 | 2.9 | 39.8 | 16.0 | 34.4 | 50.4 (0.72) | 9.8 (2.21) | 0.45% |
| VGG19 | 1.2 | 2.7 | 1.8 | 27.3 | 18.8 | 48.3 | **67.1 (0.96)** | 5.6 (1.27) | 0.60% |
| MobileNetV2 | 2.3 | 6.9 | 2.4 | 49.2 | 11.9 | 27.2 | 39.1 (0.56) | 11.7 (2.64) | 45.20% |
| Inception-ResNet-v2 | 1.1 | 4.1 | 4.7 | 43.4 | 13.1 | 33.6 | 46.7 (0.67) | 9.8 (2.21) | 0.00% |
| *Area share* | *1.25* | *1.97* | *1.23* | *25.8* | *21.1* | *48.6* | *69.7* | *4.44* | |

> **Key findings**
> - Raw shares are dominated by region size: skin and background get the most mass for every detector.
> - Relative to area, **every detector gives eyes + nose + mouth 1.3–3.2× their pixel share**, and hair + background
>   less than their share (E = 0.46–0.96).
> - EfficientNet-B4 and VGG19 have E ≈ 1 almost everywhere, meaning their maps are diffuse. Their high non-facial share
>   does not reflect a preference for background.
> - Restricting to correctly classified fakes gives a nearly identical ranking (Spearman ρ = 0.93).

### 2. Cross-dataset generalisation (FF++ → Celeb-DF)

| Detector | FF++ Acc. | Celeb-DF Acc. | Acc. drop (pp) | FF++ AUC | Celeb-DF AUC | AUC drop (pp) | Non-facial R |
|---|---|---|---|---|---|---|---|
| Xception | 98.6 | 60.5 | 38.1 | 99.8 | 72.6 | 27.3 | 40.9 |
| EfficientNet-B4 | 96.2 | 66.1 | 30.1 | 99.4 | 74.8 | 24.6 | 66.5 |
| ViT-B/16 | 96.3 | 53.8 | 42.5 | 99.2 | 57.0 | 42.2 | 38.2 |
| ResNet50 | 97.5 | 60.9 | 36.6 | 99.7 | 67.1 | 32.6 | 31.8 |
| VGG16 | 97.9 | 62.7 | 35.2 | 99.7 | 66.9 | 32.9 | 50.4 |
| VGG19 | 96.2 | 60.8 | 35.4 | 99.5 | 64.2 | 35.4 | 67.1 |
| MobileNetV2 | 96.6 | 55.4 | 41.2 | 99.4 | 65.1 | 34.3 | 39.1 |
| Inception-ResNet-v2 | 98.5 | 63.5 | 35.0 | 99.8 | 69.5 | 30.3 | 46.7 |

> **Key finding (exploratory):** detectors with more non-facial attribution on FF++ lost *less* accuracy on Celeb-DF.
> - Accuracy drop: Pearson r = −0.71 (p = 0.049), Spearman ρ = −0.69 (p = 0.058), bootstrap 95% CI [−0.96, −0.27],
>   leave-one-out range −0.82 to −0.52.
> - The association is **not present for the AUC drop** (r = −0.34, p = 0.42), so it may partly reflect calibration
>   shift.
> - With n = 8 and one transfer pair, treat this as a hypothesis, not a robustness marker.

![cross-dataset drop](figures/submitted_run/cross_dataset_drop.png)

### 3. Semantic Attribution Consistency (SAC)

Selected pairs. SAC is the mean per-image cosine on FF++. SAC_mean is the cosine of the dataset-mean region vectors,
shown for contrast. Pixel r is the mean per-image Pearson correlation between saliency maps. Error Jaccard is the
overlap of misclassified Celeb-DF images.

| Detector pair | SAC | SAC_mean | Pixel r | Top-20% IoU | Error Jaccard |
|---|---|---|---|---|---|
| Xception vs. ResNet50 | 0.985 | 0.988 | 0.944 | 0.782 | 0.544 |
| Xception vs. Inception-ResNet-v2 | 0.977 | 0.985 | 0.732 | 0.457 | 0.450 |
| ResNet50 vs. Inception-ResNet-v2 | 0.949 | 0.955 | 0.761 | 0.478 | 0.494 |
| Xception vs. VGG16 | 0.874 | 0.977 | 0.374 | 0.315 | 0.326 |
| EfficientNet-B4 vs. Inception-ResNet-v2 | 0.872 | 0.908 | 0.082 | 0.238 | 0.428 |
| Xception vs. MobileNetV2 | 0.854 | 0.997 | 0.404 | 0.367 | 0.537 |
| ViT-B/16 vs. Xception | 0.790 | 0.996 | 0.161 | 0.209 | 0.405 |
| VGG16 vs. VGG19 | 0.726 | 0.946 | 0.110 | 0.178 | 0.512 |
| ViT-B/16 vs. EfficientNet-B4 | 0.662 | 0.829 | 0.002 | 0.143 | 0.362 |
| ViT-B/16 vs. VGG19 | 0.626 | 0.850 | 0.024 | 0.143 | 0.362 |

All 28 pairs are in `results/table3_pairs_ff.csv` and `results/pairs_celebdf.csv`.

> **Key findings**
> - **The skip-connection cluster is real.** Xception, ResNet50 and Inception-ResNet-v2 have SAC > 0.94 *and* agree at
>   the pixel level (r = 0.73–0.94).
> - **ViT-B/16 diverges image by image.** Its maps are nearly uncorrelated with every CNN's (pixel r ≤ 0.16), although
>   its *average* region allocation is close to theirs (SAC_mean 0.83–0.996).
> - **SAC tracks pixel agreement more than shared failures.** Across the 28 pairs, SAC correlates with pixel agreement
>   at r = 0.83 and with Celeb-DF error overlap at r = 0.42 (error Jaccard 0.31–0.54).

![SAC heatmaps](figures/submitted_run/sac_heatmaps.png)

### 4. Qualitative examples

![saliency examples](figures/submitted_run/saliency_examples.png)

Xception and ResNet50 produce broad, face-centred maps. EfficientNet-B4 highlights the top-right image corner on all
three images, a content-independent artifact. ViT-B/16 produces sparse maps that land either on the background or on
small facial areas.

---

## ⚠️ Limitations

- **Frame-level FF++ split.** Frames of the same source video can appear in both train and test, so FF++ accuracies are
  optimistic. In a video-disjoint re-split under the same 6-epoch budget, six detectors reached 61–78% and two failed to
  converge (`figures/video_split_run/`). Celeb-DF evaluation is unaffected.
- **Fake training pool.** It consists of the first 31,949 fake frames in sorted file order (of 159,601), so it may be
  dominated by a single manipulation method.
- **Empty maps.** Grad-CAM++ returned empty maps for 54.65% (ViT-B/16) and 45.20% (MobileNetV2) of FF++ test images.
  Their statistics use the remaining images.
- **Small sample.** Correlations rest on n = 8 detectors and a single transfer pair, and all analyses are correlational.

---

## 📝 Citation

```bibtex
@inproceedings{ahmed2026deepfakeregions,
  title     = {Are Deepfake Detectors Looking at the Right Face? A Semantic Region Explanation Consistency Study},
  author    = {Ahmed, Kazi Rifat and Kamran, Abu Nokib},
  booktitle = {BMVC 2026 Workshop on Trustworthy Visual AI for Online and Public Safety (TrustVis)},
  year      = {2026}
}
```

---

## 📧 Contact

Questions about the code: open an issue, or email rifat.swe@diu.edu.bd.

**Ethical use:** this research supports forensic analysis and trustworthy AI. Do not use it to create or distribute
deepfakes.

---

## 📄 License

MIT. See [LICENSE](LICENSE). The BiSeNet face-parsing weights come from
[zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) and are subject to their licence.
FaceForensics++ and Celeb-DF are subject to their own terms of use.
