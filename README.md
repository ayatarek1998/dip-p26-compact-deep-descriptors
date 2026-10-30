# Compact Deep Feature Descriptors vs. Classical SIFT/ORB
**CSE 681 Digital Image Processing, Ain Shams University (Fall 2026). Team project, catalog Project 26.**

We train a small learned local-feature descriptor and benchmark its matching accuracy, speed and memory against the classical
SIFT/ORB + RANSAC pipeline (course Week 9) on the HPatches image-matching benchmark.

| Milestone | Week | Status |
|---|---|---|
| M1 — Proposal & literature review + SIFT/ORB baseline numbers | 3 | in progress |
| M2 — Baseline table + first trained descriptor + preliminary comparison | 6 | — |
| M3 — Final comparison, speed ablation, registration demo, final report | 9 | — |

## Repository layout
```
p26lib.py               shared library: data loading, SIFT/ORB/Harris extractors, CNN descriptor,
                        training, matching, metrics (MMA, homography accuracy, FPR95), timing, demo
notebooks/              one Colab notebook per student per milestone (M1_A, M1_B, M1_C, ...)
report/m1/              M1 proposal (LaTeX template), references.bib, generated tables/ and figures/
results/m1/             raw results (CSV / JSON) behind every number in the report
tools/sync_to_github.ipynb   copies results from the shared Drive folder into this repo and pushes
requirements.txt        pinned package versions (written from the Colab environment)
```

## Team and roles
| Role | Student | M1 task |
|---|---|---|
| A — Data & Baseline Lead | *name* | HPatches pipeline; SIFT/ORB baseline numbers; literature table |
| B — Model/Algorithm Lead | *name* | Design/select the compact descriptor architecture |
| C — Evaluation & Integration Lead | *name* | Matching-accuracy + speed protocol |

## Reproducing the M1 results
1. Put `p26lib.py` in a Google Drive folder `MyDrive/DIP_P26/`, and the HPatches **sequences** archive in `MyDrive/DIP_P26/data/`
   (any `.zip`/`.tar.gz` whose name contains "hpatches"; it is downloaded from the official server if missing).
2. Open `notebooks/M1_A_…`, `M1_B_…`, `M1_C_…` in Colab and *Run all*. Outputs go to `DIP_P26/{results,tables,figures}`.
3. Run `tools/sync_to_github.ipynb` to copy them into `results/m1/` and `report/m1/{tables,figures}/` and commit.

**Protocol:** HPatches sequences, D2-Net subset (108 sequences = 52 illumination + 56 viewpoint, 540 image pairs), full resolution,
up to 2000 keypoints per image, mutual-nearest-neighbour matching, MMA@1–10 px, homography accuracy @1/3/5 px (RANSAC 3 px),
timing on 1 CPU thread (median of 10 runs after 2 warm-ups). Random seed **42** everywhere.
Reproduction check against DeTone et al. (2018) uses their setup (480×640, 1000 keypoints, 116 sequences): `P.PROTOCOLS["superpoint"]`.

## Data and licenses
- HPatches (Balntas et al., CVPR 2017): https://github.com/hpatches/hpatches-dataset (see its license; research use)
- Training (from M2): Brown/UBC PhotoTour patches, with a fallback to COCO val2017 images. HPatches is never used for training.
Datasets are **not** committed to this repository.

## AI-use disclosure
Code scaffolding for the notebooks and the shared library was AI-assisted; all experiments were run and all results verified by the team.
