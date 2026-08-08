# XenoScan — analysis code

Source code for **"XenoScan: a deep-learning-based large-scale organismal phenomics analysis
platform for aquatic model organism embryos"**.

This repository contains the **core analysis pipeline only**. Figure-generation scripts,
statistical-analysis scripts, exploratory notebooks and video-rendering code are deliberately
excluded; the statistical tables are distributed with the paper as Dataset S1 and Dataset S2.

| | |
|---|---|
| **Image dataset** | [10.5281/zenodo.21845686](https://doi.org/10.5281/zenodo.21845686) — 186,313 images, 1,159 embryos |
| **Trained weights** | in the same Zenodo record, fetched by `download_models.py` (see below) |
| **Licence** | MIT (code) · CC BY 4.0 (images) |

---

## Pipeline

```
raw plate scans
      │
      ▼
1_preprocessing   ── per-embryo ROI crops, indexed by embryo id and time point
      │
      ▼
2_segmentation    ── U-Net body masks  ──►  masked embryo images
      │
      ├──────────────┬──────────────────┬─────────────────────┐
      ▼              ▼                  ▼                     │
3_quantification  4_hpf_prediction   5_classification         │
 area, perimeter,  developmental      drug-response           │
 length,           time (hpf)         probability +           │
 circularity       regression         Grad-CAM                │
```

Every stage reads a flat CSV manifest (one row per embryo × time point) and writes another,
so stages can be run independently once the preceding manifest exists.

---

## Quick start

```bash
git clone https://github.com/hmyang-UNIST/XenoScan.git
cd XenoScan

pip install -r requirements.txt
python download_models.py          # fetch the trained weights from Zenodo

cd 5_classification
python __predict_and_gradcam.py    # smoke test on the bundled example images
```

---

## Contents

| Directory | Entry point | Modules | Trained model |
|---|---|---|---|
| `1_preprocessing/` | `__run_preprocessing.py`, `__get_sample_images.py` | `_ImgOps.py`, `_get_plates.py`, `_get_samples_submodules.py`, `_align_crop_masking.py` | — |
| `2_segmentation/` | `__augmentation.py` → `_train_unet.py` → `__predict_mask.py` → `__postprocessing.py` → `_segment_roi.py` | `_train_modules.py` | `models/UNET_XENOPUS_832x512.h5`, `models/UNET_XENOPUS_208x128.h5` |
| `3_quantification/` | `__run_quantification.py` | `__quantification_sub.py` | — |
| `4_hpf_prediction/` | `_train_hpf.py` | `_model.py`, `_ds_loader.py`, `_augmentation.py` | `models/predict_hpf.h5` |
| `5_classification/` | `__run_train.py` → `_train.py`; `__predict_and_gradcam.py` | `_resnet.py`, `_model.py`, `_ds_gen.py`, `_ds_loader.py`, `_augmentation.py`, `_gradCam_sub.py`, `dict_noUse.py` | `models/resnet18_TB-01.h5` … `resnet18_TB-04.h5` |
| `example_data/` | — | `example_manifest.csv` | 12 test-set images |

**Naming.** A single leading underscore marks an importable module; a double underscore marks a
runnable entry point. Every script expects the working directory to be its own stage directory
and imports its siblings flat (`from _train_modules import *`), so `cd` into the stage
directory before running.

**Developmental windows.** `TB-01` … `TB-04` are the four developmental windows named in the
manuscript. They partition the time axis, and one model was trained per window:

| Model | Window | Frames | hpf |
|---|---|---|---|
| `resnet18_TB-01.h5` | cleavage | 0–48 | 2 – 22.4 |
| `resnet18_TB-02.h5` | neural | 49–67 | 22.4 – 30.3 |
| `resnet18_TB-03.h5` | tailbud | 68–128 | 30.3 – 55.3 |
| `resnet18_TB-04.h5` | free-swimming | 129–199 | 55.3 – 85 |

Hours post fertilization are `(frame index × 25/60) + 2`, the 2 h offset being the interval
between fertilization and transfer of the embryos into the imaging plate after first cleavage.

**Data directory.** Scripts read and write under a relative `./data` path (the original
absolute paths have been removed). Point that at your own copy of the image tree, or edit the
path constants near the top of each entry point.

---

## Trained models

The seven `.h5` weight files total **229 MB** and are **not tracked in git** — they are
archived as a single `XenoScan-models.tar` in the same Zenodo record as the images
([10.5281/zenodo.21845686](https://doi.org/10.5281/zenodo.21845686)) and fetched on demand:

```bash
python download_models.py            # fetch every missing weight file
python download_models.py --check    # report what is present
python download_models.py --force    # re-download
```

The archive is verified against the MD5 that Zenodo publishes for it before anything is
unpacked, and each file is installed into its stage directory. These are the exact weights
behind every result in the paper.

`predict_hpf.h5` was trained with a **mean-squared-error** loss and reports mean absolute error
as its evaluation metric. Its regression target is `frame × 25/60`, which is **2 h below** the
`hpf = frame × 25/60 + 2` convention used elsewhere in the dataset and the manuscript.

| File | Size | Stage |
|---|---|---|
| `resnet18_TB-01.h5` … `TB-04.h5` | 44.9 MB each | drug-response classification |
| `UNET_XENOPUS_832x512.h5`, `UNET_XENOPUS_208x128.h5` | 23.7 MB each | body segmentation |
| `predict_hpf.h5` | 2.5 MB | developmental-time regression |

---

## Running

```bash
# 2 — segmentation
cd 2_segmentation
python __augmentation.py                                    # bake the augmented training pairs
python _train_unet.py                                       # edit lines 24-25 to switch scale
for g in 0 1 2 3 4; do python __predict_mask.py $g & done   # manual 5-way sharding
python __postprocessing.py                                  # fuse the two scales (per-pixel max)
python _segment_roi.py                                      # threshold, largest component, apply

# 3 — quantification
cd 3_quantification
python __run_quantification.py            # __qinfo-input.csv -> __qinfo-results.csv

# 4 — developmental-time regression
cd 4_hpf_prediction
python _train_hpf.py <gpu_id>

# 5 — drug-response classification
cd 5_classification
python _train.py <gpu_id> <window 1-4> <resnet|dense> <3|5> <model_id>
python __run_train.py                     # launcher for the full grid
python __predict_and_gradcam.py           # inference + Grad-CAM using the shipped models
```

Configuration is **edit-in-place**, not command-line: only `_train.py`,
`__predict_and_gradcam.py` and `__predict_mask.py` read `sys.argv`, and those arguments are
positional. Each entry point pins its GPU with `os.environ["CUDA_VISIBLE_DEVICES"]` near the
top — change that line to match your machine. There is no `tf.distribute`; multi-GPU runs are
achieved by launching the same script once per shard.

---

## Environment

```
python 3.9
tensorflow 2.11
opencv-python, scikit-image, albumentations (<2.0)
numpy, pandas, matplotlib, tqdm
```

The published models were trained under TensorFlow/Keras 2.11.0 on NVIDIA GeForce RTX 3090
(24 GB) GPUs. See `requirements.txt`. Note that `albumentations` must be pinned below 2.0:
version 2.x removed `A.Flip`, which `_augmentation.py` uses.

---

## Relationship to the Methods

The hyperparameters in `5_classification/_train.py` are set to the values reported in the
Methods and used for the released weights: **batch size 32**, Adam at learning rate 1 × 10⁻³,
up to 5,000 epochs of 250 steps, early stopping on `val_accuracy` with patience 375 and
best-epoch weight restoration.

Two points of transparency about the released weights:

- **Random seeds were not fixed.** Retraining will not reproduce the shipped weights
  bit-for-bit. Three independent replicates were trained for the cleavage and neural windows
  to assess run-to-run variability.
- `2_segmentation/_train_unet.py` does **not** set `restore_best_weights`, so a fresh training
  run writes the *last* epoch to `models/`, not the best one. The published
  `UNET_XENOPUS_*.h5` files are the weights used for every result in the paper.

`MODIFICATIONS.md` lists every difference between these files and the working copies used
during the study, with the reason for each.

---

## Example data

`example_data/classification_test_images/` holds 12 images — one embryo per treatment
(control, BIO, Wnt-C59), sampled at one time point in each of the four developmental windows —
so that `5_classification/__predict_and_gradcam.py` can be smoke-tested without the full
dataset. `example_manifest.csv` lists embryo id, frame, window, stage name, and treatment for
each. These images are drawn from the independent test set and were never used in training.

The complete image dataset — **186,313 images from 1,159 embryos** — is deposited at
[10.5281/zenodo.21845686](https://doi.org/10.5281/zenodo.21845686) under CC BY 4.0.

---

## Citation

If you use this code, please cite the paper and this repository.

Code is released under the **MIT Licence** (see `LICENSE`). The image dataset is released
separately under **CC BY 4.0**.
