# Modifications relative to the working copies

Scripts were copied verbatim except where a change was required to (a) match the Methods
text, (b) remove a version date from a filename or path, or (c) let the file import or run at
all. Nothing else was touched — no reformatting, no refactoring.

## To match the Methods text

| File | Change | Reason |
|---|---|---|
| `5_classification/_train.py` | `BATCH_SIZE` 16 → **32** | Methods reports batch size 32, which is the setting used for the released weights |
| `5_classification/_train.py` | `patience_earlyStop` 75*3 (=225) → **375** | Methods reports patience 375 |
| `5_classification/_train.py` | EarlyStopping `monitor` `val_loss` → **`val_accuracy`** | Methods reports early stopping on validation accuracy |

## To remove version dates

| File | Change |
|---|---|
| `1_preprocessing/__get_sample_images.py` | renamed from `__get_sampleImgs-2023-10.py` |
| `2_segmentation/__predict_mask.py` | model filenames `UNET_XENOPUS_*_v2023-0412.h5` → `UNET_XENOPUS_*.h5`; directory `./xx_models/` → `./models/` |
| `2_segmentation/_train_unet.py` | output name `UNET_XENOPUS_{h}x{w}_v2023-0412` → `UNET_XENOPUS_{h}x{w}`; output dir `./xx_models` → `./models` |
| `4_hpf_prediction/_train_hpf.py` | output filename `{model_info}_v{date_of_train}.h5` → `{model_info}.h5` |
| `5_classification/_train.py` | removed `date_of_work = '230328'`; model tag no longer carries the date |
| `5_classification/__run_train.py` | removed `date_of_work`; model tag de-dated |
| `5_classification/__predict_and_gradcam.py` | renamed from `__prediction_TB12.py`; removed `date_of_work`; model tag → `resnet18_TB-{n:02d}`; model path → `./models/`; test-set directory and input manifest de-dated; removed the now-unused `M_CODE` |
| model files | `UNET_XENOPUS_832x512_v2023-0412.h5` → `UNET_XENOPUS_832x512.h5`; `UNET_XENOPUS_208x128_v2023-0412.h5` → `UNET_XENOPUS_208x128.h5`; `__predict_HPF_v221012.h5` → `predict_hpf.h5`; `M_Xenopus_230328_832x512_20K_resnet_NCLS-003_TB-0n[_ID-003].h5` → `resnet18_TB-0n.h5` |

## To make the files runnable

| File | Change | Reason |
|---|---|---|
| all scripts | absolute data roots `/home/hmyang/__Tia/__data/__Xenopus[-working]` and `/home/hmyang/01_TBL_Tia/01_Xenopus` replaced with the relative `./data` | those directories exist only on the original machine; left as-is every script fails on its first path, and the package would publish an unrelated user's directory layout |
| `4_hpf_prediction/_train_hpf.py` | `import tensorflow_addons as tfa` commented out | the import is unused in this file, and `tensorflow_addons` is no longer maintained or installable against current TensorFlow. Without this the file cannot be imported at all |
| `5_classification/__predict_and_gradcam.py` | output CSV path no longer interpolates `date_of_work` | that reference survived the constant's removal and would have raised `NameError` at the end of a full inference run |

## Dependency re-organization

| File | Change | Reason |
|---|---|---|
| `5_classification/__predict_and_gradcam.py` | `timeblocks` `['TB-01','TB-02']` → all four | the working copy had been narrowed to two windows for a partial re-run; the shipped script covers the whole pipeline |
| `5_classification/__predict_and_gradcam.py` | `IDs` `[1,2,3]` → `[0]` | the working copy looped over three training replicates; the package ships one released model per window, so the loop runs once |
| `5_classification/__run_train.py` | `timeblock` loop `[1,2]` → `[1,2,3,4]` | same reason; restores full coverage and keeps the launcher in sync with `_train.py` |

## Model provenance

`resnet18_TB-01.h5` is the `_ID-003` replicate; `resnet18_TB-02/03/04.h5` are the base models.
These are exactly the four sets of weights used to produce Figure 4 and the screening results.

## Verification

All 28 Python files compile (`python -m py_compile`). No absolute paths, no version dates, and
no references to removed constants remain in any shipped script.

**Correction, 2026-08-08.** The developmental-time model was initially packaged as
`__predict_HPF_MAE_v221013.h5`. That file does **not** reproduce the published predictions.
The model behind the reported MAE 3.22 h / RMSE 4.74 h / R² 0.958 is
`__predict_HPF_v221012.h5` (verified: it reproduces the stored `hpf_pred` column of both
published prediction files to within float32 rounding, while the MAE-named file deviates by
up to 21.9 h). The correct file is now shipped, renamed `predict_hpf.h5` because it was
trained with a mean-squared-error loss, not MAE — the `MAE` in the original filename was
hard-coded by `_train_hpf.py:30` regardless of the loss actually selected.
