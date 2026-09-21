## ⚡ Speeding Up Inference & Reducing Memory Usage

LesionLocator resamples every image to a spacing of **1.0 × 0.8 × 0.8 mm**. For scans with thick slices this can produce very large volumes — e.g. a CT of shape `512 × 512 × 196` at `0.98 × 0.98 × 5.0 mm` becomes roughly `980 × 625 × 625` voxels (~380 M). The default (skimage-based) resampling is accurate, but on such images it is slow and memory hungry, so on machines with limited RAM inference can slow to a crawl or appear to **hang right after `Loading segmentation model.`** (see issue [#11](https://github.com/MIC-DKFZ/LesionLocator/issues/11)).

Good news: there is a **much faster and far more memory-efficient PyTorch resampling backend** built in. Enabling it is a one-time patch of the `plans.json` in your checkpoint.

In our test on a synthetic `512 × 512 × 196` CT at `0.98 × 0.98 × 5.0 mm` (single fold, RTX 3090), it cut the runtime from **107 s to 40 s** and peak RAM from **~29 GB to ~15 GB**. Resampling quality can be marginally worse in rare corner cases, but for typical CT this is not noticeable.

---

### 🔧 How to enable it

Set the following keys in the `3d_fullres` configuration of the `plans.json` inside your checkpoint:

```json
"resampling_fn_data": "resample_torch_fornnunet",
"resampling_fn_seg": "resample_torch_fornnunet",
"resampling_fn_data_kwargs": {"is_seg": false, "memefficient_seg_resampling": true, "force_separate_z": false},
"resampling_fn_seg_kwargs": {"is_seg": true, "memefficient_seg_resampling": true, "force_separate_z": false},
"resampling_fn_probabilities": "resample_torch_fornnunet",
"resampling_fn_probabilities_kwargs": {"is_seg": false, "memefficient_seg_resampling": true, "force_separate_z": false}
```

Or simply run this snippet, once per checkpoint you use:

```python
import json

# bbox_optimized (for -t box), point_optimized (for -t point) and/or LesionLocatorTrack
path = "/path/to/LesionLocatorCheckpoint/LesionLocatorSeg/bbox_optimized/plans.json"

plans = json.load(open(path))
cfg = plans["configurations"]["3d_fullres"]
for key in ("data", "seg", "probabilities"):
    cfg[f"resampling_fn_{key}"] = "resample_torch_fornnunet"
    cfg[f"resampling_fn_{key}_kwargs"] = {"is_seg": key == "seg",
                                          "memefficient_seg_resampling": True,
                                          "force_separate_z": False}
json.dump(plans, open(path, "w"), indent=4)
```

A reference file with these settings applied is provided in [`assets/plans_torch_resampling.json`](assets/plans_torch_resampling.json).

> ⚠️ **Patch these keys, don't overwrite the whole `plans.json`.** The released checkpoints ship a `plans.json` containing several configurations, and the segmentation models run `3d_fullres_bs3` (which inherits from `3d_fullres`). Replacing the file with one that only contains `3d_fullres` makes startup fail with `RuntimeError: Requested configuration 3d_fullres_bs3 not found in plans`. The tracking model also uses a different spacing and patch size than the segmentation model.

---

### 🚀 Further options

| Option | Effect |
|--------|--------|
| `-f 0` | Use a single fold instead of the 5-fold ensemble (~5× fewer forward passes) |
| `--disable_tta` | Disable mirroring TTA (8× fewer forward passes, `LesionLocator_segment` only) |
| `-npp` / `-nps` | Lower these (e.g. to `1`) to reduce peak RAM during preprocessing and export |
| `LesionLocator_compile=1` | Environment variable to enable `torch.compile` for the segmentation network |

---

### 💻 Hardware

Memory requirements scale strongly with image size, so there is no single number. For reference, we run all default settings comfortably on **64 GB RAM** and an **RTX 3090 (24 GB VRAM)**.
