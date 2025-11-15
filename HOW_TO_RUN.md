# How to Run the Code

**Student:** Mayank Dahotre  
**Date:** November 15, 2025  
**Assignment:** Mesh Normalization, Quantization, and Error Analysis

---

## 📋 Prerequisites

### Required Software
- Python 3.8 or higher
- Jupyter Notebook

### Required Libraries
Install all dependencies using:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy matplotlib pandas scipy jupyter
```

---

## 🚀 Running the Code

### Option 1: Jupyter Notebooks (Recommended)

All notebooks are **already executed** with outputs saved. You can view them directly or re-run them.

#### Main Assignment
```bash
cd code
jupyter notebook mesh_ml_assignment.ipynb
```

**What it does:**
- Loads all 8 mesh files from `../meshes/`
- Applies 4 normalization methods (Min-Max [0,1], [-1,1], Z-Score, Unit Sphere)
- Performs quantization with multiple bin sizes (128, 512, 1024, 2048)
- Calculates error metrics (MSE, MAE, RMSE, Max Error)
- Displays results and comparisons

**Expected runtime:** 2-3 minutes  
**Output:** Results displayed in notebook cells

---

#### Bonus Task 1: Seam Tokenization
```bash
cd code
jupyter notebook bonus_task1_seam_tokenization.ipynb
```

**What it does:**
- Detects seams (boundary edges) in all 8 meshes
- Generates token sequences for seam representation
- Verifies encoding/decoding (lossless reconstruction)
- Creates 8 visualization images (4-panel analysis)
- Saves visualizations to `visualizations/bonus1_seam_analysis_{mesh}.png`

**Expected runtime:** 2-3 minutes  
**Output:** 8 PNG images + results in notebook

---

#### Bonus Task 2: Invariance + Adaptive Quantization
```bash
cd code
jupyter notebook bonus_task2_adaptive_quantization.ipynb
```

**What it does:**
- Tests transformation invariance with 10 random rotations/translations per mesh
- Implements PCA-based invariant normalization
- Computes local vertex density using k-NN (k=10)
- Assigns adaptive bins (64-2048) based on density
- Compares adaptive vs uniform quantization
- Creates 8 visualization images (6-panel analysis)
- Saves visualizations to `visualizations/bonus2_analysis_{mesh}.png`

**Expected runtime:** 3-4 minutes  
**Output:** 8 PNG images + results in notebook

---

### Option 2: Python Scripts (Optional)

#### Generate All Main Assignment Visualizations
```bash
cd code
python generate_all_mesh_visualizations.py
```

**Output:**
- 33 visualization images saved to `visualizations/`
- Includes: normalization comparisons, error plots, 3D visualizations, distributions

**Expected runtime:** 5-7 minutes

---

#### Save All Output Meshes
```bash
cd code
python save_output_meshes.py
```

**Output:**
- 72 OBJ files saved to `../output_meshes/`
- 9 files per mesh (original + 4 normalized + 4 reconstructed)

**Expected runtime:** 1-2 minutes

---

## 📁 File Structure

```
Mesh ML Assignment/
├── README.md                    # Main documentation
├── HOW_TO_RUN.md               # This file
├── requirements.txt            # Dependencies
├── meshes/                     # Input (8 OBJ files)
├── output_meshes/              # Output (72 OBJ files)
└── code/
    ├── mesh_ml_assignment.ipynb              # Main (executed)
    ├── bonus_task1_seam_tokenization.ipynb   # Bonus 1 (executed)
    ├── bonus_task2_adaptive_quantization.ipynb # Bonus 2 (executed)
    ├── mesh_processing.py                    # Core modules
    ├── mesh_normalizers.py
    ├── mesh_metrics.py
    ├── mesh_pipeline.py
    ├── generate_all_mesh_visualizations.py   # Utility scripts
    ├── save_output_meshes.py
    └── visualizations/                       # 49 PNG images
```

---

## 📊 Expected Outputs

### Notebooks
- All cells executed with visible outputs
- Results tables displayed
- Summary statistics shown

### Visualizations (49 PNG images)
- **Main:** 33 images (normalization, error, 3D, distribution)
- **Bonus 1:** 8 images (seam analysis)
- **Bonus 2:** 8 images (invariance + adaptive)

### Output Meshes (72 OBJ files)
- 9 files per mesh × 8 meshes
- Organized in folders by mesh name

---

## ⚠️ Troubleshooting

### Issue: Module not found
```bash
pip install numpy matplotlib pandas scipy jupyter
```

### Issue: Mesh files not found
Ensure you're in the correct directory:
```bash
cd "Mesh ML Assignment"
cd code
```

### Issue: Jupyter won't start
```bash
pip install --upgrade jupyter notebook
jupyter notebook
```

---

## ✅ Verification

All outputs are already generated. To verify:

1. **Check notebooks have outputs:**
   ```bash
   cd code
   jupyter notebook mesh_ml_assignment.ipynb
   ```
   You should see results in cells without re-running.

2. **Check visualizations:**
   ```bash
   ls code/visualizations/*.png
   ```
   Should show 49 PNG files.

3. **Check output meshes:**
   ```bash
   ls output_meshes/
   ```
   Should show 8 folders with 9 files each.

---

**Status:** ✅ All code ready to run!

