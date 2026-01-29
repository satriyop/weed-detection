# Kaggle Step-by-Step Execution Guide

Reference guide for running project notebooks on Kaggle.

---

## Notebook 01: Data Exploration

### Step 1: Attach a Dataset

In the Kaggle notebook editor:

**Default (Crop & Weed Detection — ready immediately):**

1. Click **"Add Data"** (right sidebar, or `+ Add Input`)
2. Search: `crop and weed detection`
3. Select **"Crop And Weed Detection Data with Bounding Boxes"** by `ravirajsinh45`
4. Click **"Add"** — dataset appears at `/kaggle/input/crop-and-weed-detection-data-with-bounding-boxes/`

**Recommended upgrade (Bangladesh Rice Field Weed):**

This dataset is NOT on Kaggle. You must upload it manually:

1. Download from Mendeley Data: https://data.mendeley.com/datasets/mt72bmxz73/4
2. Go to kaggle.com > **Datasets** > **New Dataset**
3. Upload the extracted folder, name it `bangladesh-rice-field-weed`
4. Set visibility to **Private**
5. Back in the notebook: **Add Data** > search your username > attach the uploaded dataset

### Step 2: Set Runtime to CPU

This is an exploration notebook — no GPU needed.

1. Click **Settings** (right sidebar gear icon)
2. **Accelerator** → set to **None** (CPU)
3. **Internet** → set to **On** (needed for `pip install`)
4. **Persistence** → **Files only** (default is fine)

> **Important:** Don't waste GPU hours here. Save them for training notebooks.

### Step 3: Configure the Dataset Switcher

In the notebook's configuration cell (early in Section 2):

```python
# Default — works with Crop & Weed Detection (already on Kaggle)
ACTIVE_DATASET = 'crop_weed_yolo'

# Or switch to Bangladesh Rice Weed (if uploaded)
ACTIVE_DATASET = 'bangladesh_rice_weed'
```

All analysis cells adapt automatically to whichever dataset you choose.

### Step 4: Run All Cells

1. Click **"Run All"** (top menu → Run → Run All, or `Ctrl+Shift+Enter`)
2. Expected runtime: ~5-10 minutes on CPU
3. Watch for errors — if the dataset structure differs, the structure discovery cell will reveal the actual layout

Common issues and fixes:

| Error | Cause | Fix |
|-------|-------|-----|
| `DATASET NOT FOUND` | Dataset not attached | Redo Step 1 |
| `No pairs found` (YOLO) | Images/labels in unexpected subdirectory | Check structure discovery output, adjust DATA_ROOT |
| `load_classes_txt` returns None | No classes.txt file | The notebook falls back to default class names |
| `find_image_path` returns None | Images in unexpected path | Check structure discovery output for actual layout |

### Step 5: Record Key Observations

After all cells run, note these outputs (needed for later notebooks):

| What to Record | Which Section | Why It Matters |
|----------------|---------------|----------------|
| Class distribution chart | Section 4 | Determines if you need weighted loss |
| Imbalance ratio | Section 4 output text | >3x means special handling needed |
| Image dimensions | Section 6 | Confirms if resize is needed |
| Visual inspection | Section 5 | Verify annotation quality / class separability |
| Data quality issues | Section 8 | Missing annotations, coordinate errors |
| YOLO bbox quality | Section 5 (YOLO only) | Are bounding boxes tight and accurate? |

### Step 6: Save the Notebook

1. Click **"Save Version"** (top right)
2. Choose **"Save & Run All (Commit)"** — saves a versioned snapshot with outputs
3. Set **Environment**: Quick Save (no need to re-run if outputs are already visible)
4. The saved version becomes your reference for all later notebooks

### Step 7: Decision Point — What's Next

Based on notebook 01 results, two parallel paths:

```
Notebook 01 done
     |
     +---> PATH A: Notebook 04 — Classification
     |    Train EfficientNetV2-S on Crop & Weed Detection or Bangladesh Rice Weed.
     |    Runtime: GPU (P100). Default: crop_weed_yolo (2 classes).
     |    Upgrade: bangladesh_rice_weed (11 species, better relevance).
     |
     +---> PATH B: Notebook 02 — Segmentation Exploration
          Explore RiceSEG for pixel-level weed masks (DeepLabV3+ training).
          Requires: Upload RiceSEG from HuggingFace as a private Kaggle Dataset.
```

**Recommended next step:** If you want quick results, go to notebook 04. If you want to explore segmentation data first, go to notebook 02.

---

## Notebook 02: Segmentation Exploration (RiceSEG)

### Step 1: Upload RiceSEG to Kaggle

RiceSEG is hosted on HuggingFace, NOT Kaggle. One-time setup:

1. Download RiceSEG from HuggingFace (search: "RiceSEG" or find via the paper)
2. Go to kaggle.com > **Datasets** > **New Dataset**
3. Upload the extracted folder, name it `riceseg`
4. Set visibility to **Private**
5. Wait for the upload to complete (may take a few minutes)

### Step 2: Create the Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Upload or paste `02-segmentation-exploration.ipynb`
4. **Add Data** → search your username → attach `riceseg`

### Step 3: Set Runtime to CPU

No GPU needed for exploration.

1. **Accelerator** → **None** (CPU)
2. **Internet** → **On**

### Step 4: Run All Cells

1. **Run All** (`Ctrl+Shift+Enter`)
2. Expected runtime: ~10-15 minutes (mask analysis scans all 3,078 images)
3. The mask analysis cell will print progress every 500 images

### Step 5: Record Key Observations

| What to Record | Which Section | Why It Matters |
|----------------|---------------|----------------|
| Weed pixel percentage | Section 3 | Expect ~1.6% — determines class weight for training |
| Class weights | Section 3 output | Use these in DeepLabV3+ loss function |
| Images with weed pixels | Section 3 | How many images are actually useful for weed training |
| Philippines subset size | Section 3 (per-country) | Most relevant subset for Indonesia |
| Mask value range | Section 5 | Verify mask encoding matches class definitions |
| Image dimensions | Section 5 | Confirm 512x512 native resolution |

### Step 6: Save the Notebook

Same as notebook 01 — **Save & Run All (Commit)**.

### Step 7: Decision Point

```
Notebook 02 done
     |
     +---> Notebook 03 or 05 — Train DeepLabV3+ on RiceSEG
          Use class weights and training config from this notebook.
          Runtime: GPU required.
```

---

## Notebook 04: Classification Baseline

### Step 1: Create a New Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Upload or paste `04-classification-baseline.ipynb`

### Step 2: Attach Dataset + Enable GPU

**For Crop & Weed Detection (default):**

1. **Add Data** → search `crop and weed detection` → add by `ravirajsinh45`
2. Set `ACTIVE_DATASET = 'crop_weed_yolo'` in the config cell

**For Bangladesh Rice Field Weed (recommended upgrade):**

1. Upload from Mendeley Data first (see notebook 01 Step 1 instructions)
2. **Add Data** → search your username → attach `bangladesh-rice-field-weed`
3. Set `ACTIVE_DATASET = 'bangladesh_rice_weed'` in the config cell

**Settings:**

- **Accelerator** → **GPU P100** (needed for training)
- **Internet** → **On** (needed for `pip install`, pretrained weights, W&B)
- **Persistence** → **Files only**

### Step 3: Configure W&B (Optional but Recommended)

1. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)
2. When the notebook prompts for the key, paste it
3. If you skip W&B, training still works — metrics print to stdout instead

### Step 4: Run All Cells

1. **Run All** (`Ctrl+Shift+Enter`)
2. Expected runtime: ~30-45 minutes with GPU
   - Phase 1 (frozen backbone, 10 epochs): ~15 min
   - Phase 2 (unfrozen, 10 more epochs): ~15 min
   - Evaluation: ~5 min

### Step 5: Check Results

| What to Check | Crop & Weed (2-class) | Bangladesh (11-class) | Action if Bad |
|---------------|----------------------|----------------------|---------------|
| Val accuracy (frozen) | 70-90% | 50-70% | Normal — backbone features are generic |
| Val accuracy (unfrozen) | 85-95% | 80-90% | If much lower, check data pipeline |
| Confusion matrix | Diagonal-heavy | Some off-diagonal expected | Off-diagonal reveals hard pairs |
| Per-class F1 | >0.8 each | >0.6 each | Low F1 classes need more augmentation |

### Step 6: Save Outputs

1. **Save Version** → **Save & Run All (Commit)**
2. The trained model (`efficientnet_v2s_{dataset}_v1.pt`) is saved to `/kaggle/working/`
3. Download the model file from the notebook output tab if needed locally
