# [CVPR2025] LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging

<img src="documentation/assets/LesionLocatorLogo.png" />

This repository contains the official implementation of our CVPR 2025 paper:

### 🎯 **LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging**


The paper introduces a novel framework for **zero-shot lesion segmentation** and **longitudinal tumor tracking** in 3D full-body imaging. By combining a large-scale lesion dataset, promptable segmentation, and deep-learning-based image registration, our framework achieves state-of-the-art results for both tasks.

> **Authors**: Maximilian Rokuss, Yannick Kirchhoff, Seval Akbal, Balint Kovacs, Saikat Roy, Constantin Ulrich, Tassilo Wald, Lukas T. Rotkopf, Heinz-Peter Schlemmer and Klaus Maier-Hein  
> **Paper**: [![CVPR](https://img.shields.io/badge/%20CVPR%202025%20-open%20access-blue.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Rokuss_LesionLocator_Zero-Shot_Universal_Tumor_Segmentation_and_Tracking_in_3D_Whole-Body_CVPR_2025_paper.html)

## News/Updates:
- 🎖️ **9/25**: Featured in the winning solution of the **[MICCAI autoPET IV Challenge](https://autopet-iv.grand-challenge.org/leaderboard/)** on *Promptable Longitudinal Lesion Tracking*
- 🎤 **6/25**: LesionLocator was featured in the [**Voxel51 Visual AI in Healthcare Event**](https://voxel51.com/events/visual-ai-in-healthcare-june-27-2025) — Watch the talk [here](https://youtu.be/Bh8tqpHFQF0)
- 💻 **4/25**: LesionLocator **code released**! 🥳 The checkpoint can be found [here](https://zenodo.org/records/15174217)
- 🗃️ **3/25**: Lesion **Dataset** with Synthetic Follow-Ups Released 👉 [here](https://doi.dkfz.de/10.6097/DKFZ/IR/E230/20250324_1.zip)
- 📄 **2/25**: CVPR Acceptance 🎉

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/MIC-DKFZ/LesionLocator
cd LesionLocator
```

### 2. Create a conda environment (recommended)

```bash
conda create -n lesionlocator python=3.12 -y
conda activate lesionlocator
```

### 3. Install the package

```bash
pip install -e .
```

If you want to use the napari viewer for visual inspection:

```bash
pip install "napari[all]"
```

---

## 🚀 Features & Usage

LesionLocator has **two main modes**:

### 1️⃣ Zero-Shot Lesion Segmentation (Single Timepoint)

Perform universal lesion segmentation using **point** or **3D bounding box** prompts. No lesion type–specific fine-tuning required.

#### Usage

```bash
LesionLocator_segment
    -i /path/to/image(s)
    -p /path/to/prompt(s) 
    -t box 
    -o /path/to/output/folder
    -m /path/to/LesionLocatorCheckpoint 
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `-i` | **Input image(s)**: Path to an image file (`.nii.gz`) or a folder containing multiple images. |
| `-p` | **Prompt(s)**: Can be either `.json` files containing 3D point or box coordinates, or `.nii.gz` instance segmentation maps. Prompts must have the same filename as their corresponding input image. Binary masks will be automatically converted to instance maps.  More info [here](/documentation/prompting.md). |
| `-t` | **Prompt type**: Choose between `point` or `box`. Determines how the model interprets the prompts. Default is `box`. |
| `-o` | **Output folder**: Path where the predicted segmentation masks will be saved. Created automatically if it doesn't exist. |
| `-m` | **Model folder**: Path to the downloaded `LesionLocatorCheckpoint` directory containing trained model weights. |
| `-f` | **Model folds**: Specify one or more folds. Defaults to all 5 folds for ensemble prediction. |
| `--disable_tta` | Disables test-time augmentation (TTA) using mirroring. Speeds up inference at the cost of accuracy. |
| `--continue_prediction`, `--c` | Continues a previously interrupted run by skipping existing output files. |
| `--visualize` | Opens results in a `napari` viewer for inspection. Requires `napari[all]` installed. |


🧠 If you provide a 3D (instance) segmentation mask as a prompt, LesionLocator will internally extract the bounding boxes or points automatically, depeding on the specified prompt type `-t`. Details on how to handle promting and the .json format can be found [here](/documentation/prompting.md). You can also run `LesionLocator_segment -h` for help. 

**Examples:**

```bash
LesionLocator_segment -i image.nii.gz -p label.nii.gz -t box -o /output/folder -m /path/to/LesionLocatorCheckpoint 
```
```bash
LesionLocator_segment -i image.nii.gz -p points.json -t point -o /output/folder -m /path/to/LesionLocatorCheckpoint 
```
```bash
LesionLocator_segment -i /image/folder -p /prompts/folder -t box -o /output/folder -m /path/to/LesionLocatorCheckpoint 
```

---

### 2️⃣ Lesion Segmentation & Tracking Across Timepoints (Longitudinal)

Includes the lesion segmentation above but also tracks lesions across multiple timepoints using previous labels or prompts.

#### Usage

```bash
LesionLocator_track
    -bl /path/to/baseline.nii.gz
    -fu /path/to/followup1.nii.gz /path/to/followup2.nii.gz
    -p /path/to/baseline_prompt_or_mask(s)
    -t prev_mask
    -o /path/to/output
    -m /path/to/LesionLocatorCheckpoint
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `-bl` | **Baseline scan**: Path to the `.nii.gz` image used as the baseline. Must match format expected by the model (or dataset config). |
| `-fu` | **Follow-up scan(s)**: One or more `.nii.gz` files representing follow-up timepoints. Provide multiple paths separated by space for autoregressive tracking. |
| `-p` | **Prompt**: Path to the baseline prompt. Can be one `.nii.gz` instance segmentation mask or list of several semantic segmentation masks. Can also be a `.json` file with a point/box prompt. Used to initialize tracking. More info [here](/documentation/prompting.md). |
| `-o` | **Output folder**: Where tracked lesion segmentations will be saved. Created automatically if it doesn't exist. |
| `-t` | **Prompt type**: Options: `point`, `box`, or `prev_mask`. Use `point`/`box` to trigger segmentation of baseline lesions before tracking. Use `prev_mask` to track from an existing baseline label mask directly. Default is `prev_mask`. |
| `-m` | **Model folder**: Path to the `LesionLocatorCheckpoint` directory containing tracking model weights. |
| `-f` | **Model folds**: Specify one or more folds. Defaults to all 5 folds for ensemble prediction. |
| `-device` | **Device selection**: Options are `cuda`, `cpu`, or `mps` (Apple Silicon). Controls where the inference runs. Default is `cuda`. |

You can also run `LesionLocator_track -h` for help.

**Examples**

This will do both the segmentation of the baseline scan using box prompts generated from `baseline_label.nii.gz` and then track each lesion:
```bash
LesionLocator_track -bl baseline.nii.gz -fu followup.nii.gz -p baseline_label.nii.gz -t box -o output/folder -m /path/to/LesionLocatorCheckpoint
```

Similarly works with `.json` prompts:
```bash
LesionLocator_track -bl baseline.nii.gz -fu followup.nii.gz -p points.json -t point -o output/folder -m /path/to/LesionLocatorCheckpoint
```

This will trigger *just the tracking* (`-t prev_mask`) of all provided semantic masks autoregressively for all follow-ups:
```bash
LesionLocator_track -bl baseline.nii.gz -fu followup1.nii.gz followup2.nii.gz -p mask1.nii.gz mask2.nii.gz mask3.nii.gz -t prev_mask -o output/folder -m /path/to/LesionLocatorCheckpoint
```

 To run inference with LesionLocator, you need to download the pretrained model checkpoint: 

#### [🔗 Download LesionLocator Checkpoint](https://zenodo.org/records/15174217)

Once downloaded, extract the contents and use the `-m` argument in the CLI tools to point to the directory named `LesionLocatorCheckpoint`. Let us know if you run into issues downloading or loading checkpoints!

---


## 🗃️ Lesion Dataset with Synthetic Follow-Ups

We have released the **Lesion Dataset with Synthetic Follow-Ups** [here](https://doi.dkfz.de/10.6097/DKFZ/IR/E230/20250324_1.zip) (ca. 700 GB), which includes simulated follow-up scans with consistent instance labels. Due to image size, quality, or licensing constraints, not all images were used to synthesize a second timepoint and were excluded from this dataset. We reccomend downloading e.g. with `wget` since our servers can sometimes be a bit unstable.

```bash
wget -c --no-check-certificate -O LesionLocator_SynteticLongitudinalDataset.zip https://doi.dkfz.de/10.6097/DKFZ/IR/E230/20250324_1.zip
```

#### **⚠ Important Notes**  
- **Instance-based labels (not semantic):** Lesions in a single patient scan are labeled consecutively, and the same lesion retains the same label across both timepoints.
- **Synthetic deformations & segmentation masks:** Some images contain **unrealistic deformations** or **challenging segmentation masks**, which may serve as useful test cases for **improving automated analysis methods**.  
- **Recommended use:** This dataset is **ideal for pretraining** or for use **alongside real longitudinal data** to enhance model robustness and generalization.
- **Longitudinal tracking tip:** When using this dataset for **longitudinal tracking**, we highly recommend **cropping one image during data augmentation** to **increase (translational) misalignment** and better simulate real-world conditions. 

#### **Included Datasets**  
This dataset incorporates lesion data from various publicly available sources:  

- **[COVID-19 CT Lung](https://zenodo.org/records/3757476)**  
- **[KiTS](https://kits-challenge.org/kits23/)**  
- **[LIDC](https://www.cancerimagingarchive.net/collection/lidc-idri/)**  
- **[LNDb](https://lndb.grand-challenge.org/)**  
- **[MSD Colon](http://medicaldecathlon.com/)**  
- **[MSD Hepatic Vessels](http://medicaldecathlon.com/)**  
- **[MSD Liver](http://medicaldecathlon.com/)**  
- **[MSD Lung](http://medicaldecathlon.com/)**  
- **[MSD Pancreas](http://medicaldecathlon.com/)**  
- **[NIH Lymph](https://www.cancerimagingarchive.net/collection/ct-lymph-nodes/)**  
- **[NSCLC Radiomics](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/)**  

#### 📜 This dataset is released under the **CC BY-NC-SA 4.0** license. 

---

## More Related Code Releases

If you are interested check out [**nnInteractive**](https://github.com/MIC-DKFZ/nnInteractive) or [**LongiSeg**](https://github.com/MIC-DKFZ/LongiSeg) which we both just released:

- [**nnInteractive**](https://github.com/MIC-DKFZ/nnInteractive) is our recent model for 3D interactive segmentation across all modalities, all target structures using various types of prompts! Check out the [python backend](https://github.com/MIC-DKFZ/nnInteractive) or the [napari plugin](https://github.com/MIC-DKFZ/napari-nninteractive). &nbsp; &nbsp; [![arXiv](https://img.shields.io/badge/arXiv-2503.08373-b31b1b.svg)](https://arxiv.org/abs/2503.08373)

- [**LongiSeg**](https://github.com/MIC-DKFZ/LongiSeg) is an extension of the popular [nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet), designed specifically for **longitudinal medical image segmentation**. By incorporating temporal information across multiple timepoints, LongiSeg enhances segmentation accuracy and consistency, making it a robust tool for analyzing medical imaging over time. LongiSeg includes several methods for temporal feature merging, including the newly introduced [Difference Weighting Block](https://github.com/MIC-DKFZ/Longitudinal-Difference-Weighting). &nbsp; &nbsp;   [![arXiv](https://img.shields.io/badge/arXiv-2404.03010-B31B1B.svg)](https://arxiv.org/abs/2409.13416)

---

## 📚 Citation

If you find our work useful, please consider citing:

```bibtex
@InProceedings{Rokuss_2025_CVPR,
    author    = {Rokuss, Maximilian and Kirchhoff, Yannick and Akbal, Seval and Kovacs, Balint and Roy, Saikat and Ulrich, Constantin and Wald, Tassilo and Rotkopf, Lukas T. and Schlemmer, Heinz-Peter and Maier-Hein, Klaus},
    title     = {LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {30872-30885}
}
```

---

## 📬 Contact

For questions, issues, or collaborations, please contact:

📧 maximilian.rokuss@dkfz-heidelberg.de
