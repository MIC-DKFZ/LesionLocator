# AGENTS.md

Orientation for coding agents working in the **LesionLocator** repository. Read this before
changing code: it explains what the package does, what it deliberately does *not* contain, how
inference flows through the modules, and which invariants break silently if you touch them.

---

## 1. What this project is

Official implementation of the CVPR 2025 paper *LesionLocator: Zero-Shot Universal Tumor
Segmentation and Tracking in 3D Whole-Body Imaging*
([CVPR open access](https://openaccess.thecvf.com/content/CVPR2025/html/Rokuss_LesionLocator_Zero-Shot_Universal_Tumor_Segmentation_and_Tracking_in_3D_Whole-Body_CVPR_2025_paper.html),
[arXiv:2502.20985](https://arxiv.org/abs/2502.20985)). Two capabilities:

1. **Promptable zero-shot lesion segmentation** (single timepoint). A 3D point or bounding box
   marks a lesion; a Residual-Encoder U-Net segments it. The prompt is rendered into a *dense*
   binary volume and concatenated to the image as an extra input channel — there is no separate
   prompt encoder.
2. **Longitudinal lesion tracking** (multiple timepoints). `TrackNet` registers baseline to
   follow-up with a UniGradICON network, warps the baseline lesion mask through the deformation
   field, then refines the warped mask with a U-Net on a crop centred on it. Multi-timepoint
   tracking is autoregressive: the prediction for timepoint *t* becomes the prompt for *t+1*.

Public mirror: <https://github.com/MIC-DKFZ/LesionLocator> (issues live there). `origin` in this
working copy points at the internal DKFZ GitLab.

---

## 2. Ground rules — read these first

- **This repo is inference-only.** `lesionlocator/training/` contains *only*
  `build_network_architecture` static methods used to reconstruct the networks at load time. There
  is no training loop, no optimizer, no dataloader, no dataset-preprocessing entry point. Questions
  about *how the model was trained* are answered by the paper, not by this code.
- **It is a trimmed fork of nnU-Net v2.** `preprocessing/`, `imageio/`, `utilities/plans_handling/`,
  `utilities/label_handling/`, `inference/export_prediction.py` and `inference/data_iterators.py`
  are nnU-Net files with `nnunet` renamed to `lesionlocator`. Follow nnU-Net conventions there and
  check upstream before "fixing" something that looks odd.
- **There are no tests and no CI.** Nothing will catch a regression for you. See §10.
- **Model weights are not in the repo.** Everything end-to-end needs the
  [Zenodo checkpoint](https://zenodo.org/records/15174217) (`LesionLocatorCheckpoint/`, see §7) and,
  realistically, a CUDA GPU.
- **Dead nnU-Net leftovers** (imported by nothing — do not treat them as live code):
  `utilities/collate_outputs.py`, `utilities/json_export.py`, `utilities/network_initialization.py`.

---

## 3. Repository map

```
lesionlocator/
  inference/
    lesionlocator_segment.py           # LesionLocatorSegmenter + `LesionLocator_segment` CLI  (main file, ~600 loc)
    lesionlocator_segment_and_track.py # LesionLocatorSegTracker + `LesionLocator_track` CLI
    data_iterators.py                  # spawn-based preprocessing workers feeding the predictor
    export_prediction.py               # logits -> original geometry -> .nii.gz on disk
    sliding_window_prediction.py       # gaussian window + step computation
  modules/tracknet.py                  # TrackNet: registration + warp + crop + refinement U-Net
  training/LesionLocatorTrainer/       # ONLY build_network_architecture (Segment / Track variants)
  utilities/prompt_handling/
    prompt_handler.py                  # prompt extraction (mask/json -> sparse) + sparse -> dense
    create_prompt_jsons_from_seg.py    # `LesionLocator_create_prompt_json` CLI
  preprocessing/                       # nnU-Net: cropping, normalization, resampling, DefaultPreprocessor
  utilities/plans_handling/            # PlansManager / ConfigurationManager (reads plans.json)
  imageio/                             # SimpleITKIO (default), Nibabel, Tiff, natural images
documentation/
  prompting.md                         # user-facing prompt format docs
  inference_speed.md                   # user-facing speed/memory guide — see §8
  assets/plans_torch_resampling.json   # drop-in faster/leaner plans.json — see §8
readme.md                              # user-facing docs
```

---

## 4. Environment & install

```bash
conda create -n lesionlocator python=3.12 -y && conda activate lesionlocator
pip install -e .            # pyproject requires-python >= 3.10
pip install "napari[all]"   # only for --visualize
```

Pin notes: `acvl-utils>=0.2.3,<0.3` and `dynamic-network-architectures>=0.3.1,<0.4` are deliberate —
newer majors break. `cc3d` (`connected-components-3d`) is imported directly by the prompt handling
code but is **not** declared in `pyproject.toml`; it currently arrives transitively via `acvl-utils`.
`icon_registration` (used by `TrackNet`) arrives via `unigradicon`.

---

## 5. CLI entry points

Declared in `pyproject.toml` under `[project.scripts]`:

| Command | Function | Purpose |
|---|---|---|
| `LesionLocator_segment` | `inference.lesionlocator_segment:predict_seg_from_prompt` | single-timepoint promptable segmentation |
| `LesionLocator_track` | `inference.lesionlocator_segment_and_track:segment_and_track` | segmentation + longitudinal tracking |
| `LesionLocator_create_prompt_json` | `utilities.prompt_handling.create_prompt_jsons_from_seg:prompt_jsons` | turn label maps into point/box JSONs |

**`LesionLocator_segment`**: `-i` image or folder, `-p` prompt (`.json` or mask) or folder,
`-t {point,box}` (**required**, despite the documented default), `-o` output folder, `-m` checkpoint
folder, `-f` folds (default all five), `-step_size` (0.5), `--disable_tta`, `-npp`/`-nps`
(preprocessing / export processes, default 3), `-device {cuda,cpu,mps}`, `--continue_prediction`,
`--visualize`, `--verbose`, `--disable_progress_bar`.

**`LesionLocator_track`**: `-bl` baseline, `-fu` one or more follow-ups (ordered → autoregressive),
`-p` prompt(s), `-t {point,box,prev_mask}` (default `prev_mask`), `-o`, `-m`, `-f`, `-device`.
Note it has **no** `--disable_tta` / `-step_size` / `-npp` knobs.

`LesionLocator_create_prompt_json -i <labels> -o <out> -label_type {instance,semantic}` writes
*both* `<out>/points/` and `<out>/boxes/`.

---

## 6. How inference actually works

### 6.1 Segmentation (`LesionLocatorSegmenter`)

1. `initialize_from_trained_model_folder` loads `dataset.json` + `plans.json`, rebuilds the network
   via the trainer named in the checkpoint (`recursive_find_python_class` over
   `training/LesionLocatorTrainer`), and keeps one state dict per fold in `self.list_of_parameters`.
2. `predict_from_files` pairs images with prompts **by filename**, then hands them to
   `preprocessing_iterator_fromfiles` (`data_iterators.py`), which runs `DefaultPreprocessor.run_case`
   in `spawn`ed daemon workers: transpose → crop to nonzero → `CTNormalization` → **resample to
   1.0 × 0.8 × 0.8 mm**.
3. Prompts become a sparse list, one entry per lesion ID:
   `get_prompt_from_json` (JSON coordinates mapped through the crop + resample) or
   `get_prompt_from_inst_or_bin_seg` (cc3d statistics on the preprocessed mask).
4. `predict_from_data_iterator` loops over lesions. `sparse_to_dense_prompt` renders each prompt into
   a binary volume (box → filled box; point → `skimage.morphology.ball(5)`), which
   `predict_logits_from_preprocessed_data` concatenates to the image as channel 1.
5. `_internal_get_sliding_window_slicers` is **prompt-restricted**: if the prompt's bounding box fits
   inside `patch_size` (`[160, 224, 224]`), exactly *one* patch centred on the prompt is predicted;
   otherwise only patches overlapping the prompt run. This is why box prompts are cheap.
6. Per patch: TTA mirroring over 3 axes = 8 forward passes, per fold. `export_prediction_from_logits`
   (in an export process pool) resamples logits back, undoes cropping/transpose and writes
   **one file per lesion**: `<image_stem>_lesion_<id>.nii.gz`. Nothing merges them back into an
   instance map.

### 6.2 Tracking (`LesionLocatorSegTracker` + `TrackNet`)

- With `-t point|box`, the baseline is first segmented by a full `LesionLocatorSegmenter` run; the
  resulting per-lesion files become the tracking prompt list. With `-t prev_mask` the given mask(s)
  are used directly (one instance map, or a list of semantic masks — matched to lesion IDs
  **positionally**, 1-based).
- `track()` slides a 2-image window over `[baseline] + follow_ups` and loops lesions inside it.
- `TrackNet.forward(x0, x1, prompt, is_inference=True)`: optional z-translation alignment when the
  z-extents differ by > 20 slices → trilinear resample all three to `[175, 175, 175]` → UniGradICON
  registration → warp the prompt with `compute_warped_image_multiNC` (nearest / spline order 0) →
  resample back → crop a `patch_size` patch centred on the warped mask's centre of mass → U-Net
  refinement on `[image, warped_mask]` → paste logits back into full-volume shape.
- There is **no sliding window** here: one crop per lesion, but TTA mirroring is always on
  (8 passes × folds per lesion). `-f 0` is the only lever.
- `prompt = predicted_files` at the end of each timepoint makes it autoregressive; a lesion skipped
  at timepoint *t* (empty prompt) propagates as `None` to all later timepoints.

---

## 7. Checkpoint layout (not in this repo)

`-m` must point at the extracted `LesionLocatorCheckpoint/`:

```
LesionLocatorCheckpoint/
  LesionLocatorSeg/
    bbox_optimized/    plans.json, dataset.json, fold_{0..4}/checkpoint_final.pth   # used when -t box
    point_optimized/   plans.json, dataset.json, fold_{0..4}/checkpoint_final.pth   # used when -t point
  LesionLocatorTrack/  plans.json, dataset.json, fold_{0..4}/checkpoint_final.pth
```

The sub-folder is chosen by `-t` in code (`"bbox_optimized" if args.t == 'box' else "point_optimized"`).
Key `plans.json` values for the segmentation configuration (`3d_fullres`): spacing `[1.0, 0.8, 0.8]`,
`patch_size [160, 224, 224]`, `ResidualEncoderUNet`, `CTNormalization`, `SimpleITKIO`,
`file_ending` `.nii.gz` (changeable in `dataset.json`).

---

## 8. Performance: why it can look like a hang, and the fix (issue #11)

**Symptom** (reported in [issue #11](https://github.com/MIC-DKFZ/LesionLocator/issues/11)): the run
prints `Loading segmentation model.` and then appears to freeze at
`for preprocessed in data_iterator:` in `predict_from_data_iterator`.

**Cause**: it is not the iterator, it is preprocessing. LesionLocator *always* resamples to
`1.0 × 0.8 × 0.8 mm`. A typical clinical CT at `0.98 × 0.98 × 5.0 mm`, shape `512×512×196`, blows up
to roughly `980 × 625 × 625` ≈ 380 M voxels. The default nnU-Net resampling backend
(`resample_data_or_seg_to_shape`, skimage/scipy, in `preprocessing/resampling/default_resampling.py`)
is accurate but slow and memory-hungry, so on a small-RAM machine the process goes into swap/OOM and
looks hung.

**Fix — switch the checkpoint's plans to the torch resampling backend.** In the `3d_fullres`
configuration of the checkpoint's `plans.json`, set `resampling_fn_data` / `resampling_fn_seg` /
`resampling_fn_probabilities` to `resample_torch_fornnunet`
(`preprocessing/resampling/resample_torch.py`) with kwargs
`{"is_seg": …, "memefficient_seg_resampling": true, "force_separate_z": false}`. See [documentation/inference_speed.md](documentation/inference_speed.md)
for a copy-paste patch script.

**Patch those keys in place — do NOT overwrite the whole `plans.json`** with
[`documentation/assets/plans_torch_resampling.json`](documentation/assets/plans_torch_resampling.json)
(added in commit `225b419`, "faster torch resampling plans, resolves #11"), even though issue #11
suggests copying it. Verified against the released checkpoint: that asset contains **only** the
`3d_fullres` configuration, while the released `plans.json` carries five
(`2d`, `3d_lowres`, `3d_fullres`, `3d_fullres_bs3`, `3d_cascade_fullres`) and both segmentation
checkpoints declare `configuration: 3d_fullres_bs3` (which `inherits_from: 3d_fullres`). Copying the
file therefore aborts at startup with
`RuntimeError: Requested configuration 3d_fullres_bs3 not found in plans`. `LesionLocatorTrack` uses
`3d_fullres` but with a different spacing (`[3.0, 0.8418, 0.8418]`) and patch size (`[96, 256, 256]`),
so the file is not a drop-in there either. Apart from the missing configurations the asset is
byte-identical to the released seg plans except for the six resampling keys, and the bbox/point plans
are identical to each other.

**Measured** (synthetic `512×512×196` @ `0.98×0.98×5.0 mm`, `-t box -f 0 -npp 1 -nps 1`, RTX 3090,
62 GB RAM): default skimage backend **107 s / 29.2 GB peak RSS**; torch backend **40 s / 14.5 GB**.
Slight quality loss is possible in corner cases.

**Other speed/memory levers**, in rough order of impact:

| Lever | Effect |
|---|---|
| torch resampling plans (above) | large drop in preprocessing time and peak RAM |
| `-f 0` | 1 fold instead of a 5-fold ensemble → ~5× fewer forward passes (both CLIs) |
| `--disable_tta` | drops mirroring → 8× fewer forward passes (`LesionLocator_segment` only) |
| `-step_size` > 0.5 | fewer sliding-window patches; only matters when the prompt exceeds `patch_size` |
| lower `-npp` / `-nps` | fewer concurrent preprocessing/export processes → less peak RAM |
| `LesionLocator_compile=1` (env var) | `torch.compile` on the segmentation network |

**Reference hardware** (maintainer, issue #11): 64 GB RAM + RTX 3090 (24 GB VRAM) runs defaults
fine; 8 GB RAM / 1 CPU fails even with `-npp 1 -nps 1 -f 0`. Requirements scale with image size, so
there is no single number. `perform_everything_on_device=True` is hardcoded in both CLIs; on CUDA
OOM the sliding-window code falls back to CPU result arrays automatically.

---

## 9. Conventions, invariants and traps

- **Coordinates are `z, y, x`** everywhere in prompt JSONs. Boxes are
  `[z_min, z_max, y_min, y_max, x_min, x_max]` with **half-open** ends (Python slicing). Values are
  stored as *strings* in the JSON. See [documentation/prompting.md](documentation/prompting.md).
- **JSON coordinates are in original image voxel space.** `get_centroids_from_json` /
  `get_bboxes_from_json` map them through `bbox_used_for_cropping` and
  `shape_after_cropping_and_before_resampling`. Their third parameter is named `patch_size` but
  `data_iterators.py` passes the *preprocessed image shape* — do not "fix" it to the model patch size.
- **Lesion IDs are 1-based and iterated densely** (`range(1, max(id)+1)`). Missing IDs yield empty
  entries that are skipped, which keeps output suffixes aligned with the input IDs. Prompts derived
  from masks behave the same way (`[]` for absent labels).
- **Prompt files must share the base filename of their image**; in folder mode the prompt folder must
  contain either JSONs or masks, never both.
- **Network input channels = image channels + 1.** Both trainers pass `num_input_channels + 1` to
  `get_network_from_plans`; the extra channel is the dense prompt concatenated in
  `predict_logits_from_preprocessed_data` / `TrackNet.forward`. Change one side, change the other.
- **Boxes from masks are dilated by 1 voxel** per side (clipped to bounds) in
  `get_bboxes_from_inst_or_bin_seg`; point prompts become a radius-5 ball in *preprocessed* space.
- **Multiprocessing uses `spawn`**, workers are daemons, and the queues have `maxsize=1`. A worker
  killed by the OOM killer surfaces as `Background workers died...` — usually a RAM problem, see §8.
- **`--continue_prediction`** infers where to resume via `np.max(np.where(existing_files)[0])`, so it
  needs a non-empty output folder (the CLI unsets the flag if the folder is empty).
- **Outputs are per lesion**, never a merged instance map — both for segmentation
  (`<stem>_lesion_<id>`) and tracking (`<followup_stem>_lesion_<id>`).

---

## 10. How to verify a change

There is no test suite, so verify deliberately:

```bash
pip install -e .
python -c "import lesionlocator"                 # import graph intact
LesionLocator_segment -h && LesionLocator_track -h && LesionLocator_create_prompt_json -h
```

- **Prompt/geometry changes** are the highest-risk area and can be checked without a GPU: build a
  small synthetic `.nii.gz` label map with a few cubes, run `LesionLocator_create_prompt_json`, and
  confirm that `get_prompt_from_json(...)` and `get_prompt_from_inst_or_bin_seg(...)` land on the same
  voxels after `DefaultPreprocessor.run_case`. Round-tripping through
  `export_prediction_from_logits` should return the original geometry (shape, spacing, origin).
- **Inference changes** need the real checkpoint and a GPU; sanity-check one small case per prompt
  type (`point`, `box`) and one two-timepoint tracking run before claiming anything works.
- Keep the user-facing docs in sync: [readme.md](readme.md) (CLI tables),
  [documentation/prompting.md](documentation/prompting.md) (prompt formats) and
  [documentation/inference_speed.md](documentation/inference_speed.md) (speed/memory).
