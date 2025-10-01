## 🧠 Prompt Types for LesionLocator

LesionLocator supports various types of prompts to guide lesion **segmentation** and **tracking**.

👉 **Point and Box prompts** can be provided in **two formats**:
1. **Directly** as `.json` files containing coordinates or bounding boxes
2. **Indirectly** as `.nii.gz` segmentation mask(s) — the model will automatically convert them into point or box prompts via connected component analysis

Additionally, instance segmentation labels can be used **as prompts directly** in tracking mode.

### 🔹 1. Point Prompts (`point`)
A *point prompt* is a single 3D coordinate indicating e.g. the center of a lesion. It's provided via a `.json` file.

#### 📄 Format:
```json
{
    "1": {
        "point": ["166.01", "361.74", "162.64"]
    },
    "2": {
        "point": ["176.87", "255.07", "70.27"]
    },
}
```

- The `point` list contains the Z, Y, X coordinates.
- Multiple points are listed under different IDs if needed.

### 🔹 2. Box Prompts (`box`)
A *box prompt* defines a 3D bounding box around a lesion, provided in a `.json` file.

#### 📄 Format:
```json
{
    "1": {
        "box": ["146", "189", "316", "409", "114", "234"]
    },
    "2": {
        "box": ["88", "123", "274", "354", "102", "179"]
    }
}
```

- This format uses `[z_min, z_max, y_min, y_max, x_min, x_max]` with **half-open intervals** (Python-style indexing).
- Boxes typically help the model to segment lesions more precisely than points.

### 🔹 3. (Instance) Segmentation Labels (`prev_mask`)
For tracking tasks or automatic prompt generation, **segmentation labels** can be used directly.

- Labels must be 3D volumetric `.nii.gz` files (or specify a different file ending in the dataset.json files of the checkpoint)
- For `prev_mask`, this serves as the source lesion location and shape for tracking over time.
- When used during segmentation, the model can auto-generate point or box prompts from the label.

---

## ⚙️ How to Generate Prompt JSONs

You can generate prompt `.json` files (for both `point` and `box` prompts) using the script provided below.

### ▶️ Example CLI Usage:
```bash
LesionLocator_create_prompt_json
  -i path/to/label/images
  -o path/to/output/jsons
  -label_type instance
```

### 🧩 Options:
| Argument | Description |
|----------|-------------|
| `-i` | Folder containing 3D label images |
| `-o` | Output directory (will be created) |
| `-label_type` | Type of input labels: `semantic` (binary) or `instance` (per-lesion IDs) |

#### Label Types

- **`instance`** → Use if labels already have **per-lesion IDs** (1, 2, 3, …). Prompts are generated for each ID.  
- **`semantic`** → Use if labels are **binary** (all lesions = 1). LesionLocator splits them with **connected components** before generating prompts.  

👉 In both cases, prompts are generated per lesion. JSONs are named after their source images for compatibility.

