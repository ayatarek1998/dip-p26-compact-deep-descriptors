# Compact Deep Feature Descriptors vs. Classical SIFT/ORB

**CSE 681 – Digital Image Processing**  
**Ain Shams University | Fall 2026**  
**Team Project – Catalog No. 26**

## Project Overview

This project investigates whether a compact learned local-feature descriptor can provide a good balance between matching accuracy and computational efficiency compared with classical feature descriptors such as SIFT and ORB.

We will implement and evaluate both approaches on the HPatches image-matching benchmark. The comparison will focus on matching accuracy, execution time, and memory usage, using RANSAC for geometric verification as covered in Week 9 of the course.

## Project Milestones

| Milestone | Week | Status |
|---|---|---|
| M1 — Project proposal, literature review, and SIFT/ORB baseline evaluation | 3 | In progress |
| M2 — Complete baseline results, first trained descriptor, and preliminary comparison | 6 | Not started |
| M3 — Final evaluation, speed ablation, image registration demo, and final report | 9 | Not started |

## Repository Structure

```text
p26/
├── notebooks/
│   ├── M1_A_...ipynb       # Student A – Milestone 1
│   ├── M1_B_...ipynb       # Student B – Milestone 1
│   ├── M1_C_...ipynb       # Student C – Milestone 1
│   ├── M2_A_...ipynb       # Student A – Milestone 2
│   ├── M2_B_...ipynb       # Student B – Milestone 2
│   └── ...
├── report/
│   └── m1/
│       ├── proposal.tex
│       ├── references.bib
│       └── figures/        # Figures included in the report
├── results/
│   ├── m1/                 # Milestone 1 experimental results
│   ├── m2/                 # Milestone 2 experimental results
│   └── m3/                 # Milestone 3 experimental results
├── tables/                 # Generated result tables
├── figures/                # Generated evaluation figures
└── requirements.txt        # Python dependencies and pinned versions
```

### Notebook Organization

Each notebook contains its own implementation of the code required for its experiments. Notebooks should be self-contained and should not depend on a shared Python library or another student's notebook.

Depending on the assigned task, a notebook may include:

- Dataset loading and preprocessing.
- Feature detection and extraction (SIFT, ORB, Harris, etc.).
- CNN-based descriptor architecture and training.
- Feature matching and correspondence filtering.
- Evaluation metrics and geometric verification.
- Runtime and memory measurements.
- Image matching and registration visualizations.

Common approaches can be implemented independently in different notebooks when needed. This keeps each experiment reproducible and makes individual contributions easier to review.

## Team Members and Responsibilities

| Role | Student | M1 Responsibility |
|---|---|---|
| A – Data & Baseline Lead | Aya Samy Ahmed Salem | HPatches data pipeline, SIFT/ORB baseline evaluation, and literature review table |
| B – Model/Algorithm Lead | Aya Tarek Mohamed Ali | Design and selection of the compact descriptor architecture |
| C – Evaluation & Integration Lead | Laila Yehia Elgindi | Matching accuracy and runtime evaluation protocol |

## Running the M1 Experiments

The experiments are designed to run on Google Colab, with datasets and large artifacts stored in Google Drive rather than committed to the repository.

### 1. Prepare Google Drive

Create the following directory structure in Google Drive:

```text
MyDrive/
└── DIP_P26/
    └── data/
        └── HPatches/
```

Place the HPatches sequences dataset inside the `data/HPatches/` directory.

If the dataset is not available locally, the corresponding notebook can download it from the official HPatches source, provided the download is accessible.

### 2. Run the Notebooks

Open the corresponding notebooks from the `notebooks/` directory in Google Colab:

- `M1_A_...` – Student A
- `M1_B_...` – Student B
- `M1_C_...` – Student C

Each notebook contains all the code needed to execute its assigned experiments. Run all cells in order.

Experimental outputs, including raw results, generated tables, and figures, will be saved under:

```text
DIP_P26/
├── results/
├── tables/
└── figures/
```

## Evaluation Protocol

The main experiments will use the HPatches image-matching benchmark with the following configuration:

| Parameter | Configuration |
|---|---|
| Dataset | HPatches sequences |
| Subset | D2-Net subset |
| Number of sequences | 108 (52 illumination, 56 viewpoint) |
| Number of image pairs | 540 |
| Image resolution | Full resolution |
| Maximum keypoints per image | 2,000 |
| Matching strategy | Mutual nearest-neighbour matching |
| Matching metric | MMA at thresholds 1–10 pixels |
| Geometric evaluation | Homography accuracy at 1, 3, and 5 pixels |
| Geometric verification | RANSAC with a 3-pixel threshold |
| Timing | Single CPU thread |
| Timing measurements | Median of 10 runs after 2 warm-up runs |
| Random seed | 42 |

For comparison with the results reported by DeTone et al. (2018), we will also reproduce their evaluation setup using a resolution of 480 × 640, 1,000 keypoints, and 116 sequences.

The reproduction experiment will be reported separately from the main HPatches evaluation to keep the different configurations distinguishable.

## Datasets and Licensing

The project uses the following datasets:

- **HPatches:** Introduced by Balntas et al. (CVPR 2017), used for feature matching and evaluation.  
  Official repository: https://github.com/hpatches/hpatches-dataset

- **PhotoTour (Brown/UBC patches):** Planned training dataset for the learned descriptor starting from Milestone 2.

- **COCO val2017:** Alternative source of images if the planned training data pipeline requires a fallback.

HPatches will only be used for evaluation and will not be included in the descriptor training process.

All datasets and large model artifacts are stored separately in Google Drive or Colab storage. They are not committed to this Git repository.