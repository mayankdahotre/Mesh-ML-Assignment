# Visualizations Directory

This folder contains all generated visualizations for the main assignment and bonus tasks.

---

## 📊 Main Assignment Visualizations

### Normalization Comparison (3 meshes)
Compares all normalization methods with distribution plots and statistics.

- `normalization_comparison_branch.png` - Branch mesh normalization analysis
- `normalization_comparison_cylinder.png` - Cylinder mesh normalization analysis
- `normalization_comparison_explosive.png` - Explosive mesh normalization analysis

**Contents**: 6-panel figure showing:
1. Original distribution
2. Min-Max [0, 1] distribution
3. Min-Max [-1, 1] distribution
4. Z-Score distribution
5. Unit Sphere distribution
6. Statistics comparison (mean & std dev)

---

### Quantization Error Analysis (3 meshes)
Shows how reconstruction error varies with bin size for each normalization method.

- `quantization_error_branch.png` - Branch mesh quantization analysis
- `quantization_error_cylinder.png` - Cylinder mesh quantization analysis
- `quantization_error_explosive.png` - Explosive mesh quantization analysis

**Contents**: 4-panel figure showing:
- MSE and MAE vs bin size for each normalization method
- Log-log plots for clear visualization
- Bin sizes: 64, 128, 256, 512, 1024, 2048, 4096

---

### Overall Comparison
Compares all methods across all 8 meshes.

- `overall_comparison.png` - MSE and MAE comparison across all meshes

**Contents**: 2-panel figure showing:
1. MSE by method for all meshes (bar chart)
2. MAE by method for all meshes (bar chart)

---

### 3D Mesh Visualizations (3 meshes)
Shows 3D scatter plots of meshes before and after normalization/quantization.

- `3d_visualization_branch.png` - Branch mesh 3D views
- `3d_visualization_cylinder.png` - Cylinder mesh 3D views
- `3d_visualization_explosive.png` - Explosive mesh 3D views

**Contents**: 6-panel figure showing:
1. Original mesh
2. Min-Max [0, 1] normalized
3. Min-Max [-1, 1] normalized
4. Z-Score normalized
5. Unit Sphere normalized
6. Quantized (512 bins)

---

### Distribution Plots (8 meshes)
Individual mesh coordinate distribution analysis.

- `branch_distribution.png`
- `cylinder_distribution.png`
- `explosive_distribution.png`
- `fence_distribution.png`
- `girl_distribution.png`
- `person_distribution.png`
- `table_distribution.png`
- `talwar_distribution.png`

**Contents**: 4-panel figure showing:
1. X-axis distribution histogram
2. Y-axis distribution histogram
3. Z-axis distribution histogram
4. Overall mesh statistics (vertices, faces, ranges)

---

### Method Comparison
- `method_comparison.png` - Overall method comparison
- `bin_size_comparison.png` - Bin size impact analysis

---

## 🌟 Bonus Option 1: Seam Tokenization Visualizations

### Seam Analysis
- `bonus1_seam_analysis_branch.png` - Comprehensive seam tokenization analysis

**Contents**: 4-panel figure showing:
1. Seam chain length distribution
2. Token type distribution
3. Top 10 longest seam chains
4. Statistics summary

---

### 3D Seam Visualization
- `bonus1_seam_3d_branch.png` - 3D visualization of seams

**Contents**: 3-panel figure showing:
1. All vertices
2. Seam vertices highlighted in red
3. Top 5 longest chains colored individually

---

## 🌟 Bonus Option 2: Invariance + Adaptive Quantization Visualizations

### Comprehensive Analysis
- `bonus2_comprehensive_analysis_branch.png` - 9-panel comprehensive analysis
- `bonus_option2_analysis.png` - Alternative 9-panel analysis

**Contents**: 9-panel figure showing:
1. **Original (Normalized)** - 3D scatter of invariant normalized mesh
2. **Rotated (Normalized)** - 3D scatter after rotation (should match original)
3. **Normalization Difference** - Histogram of differences (should be near zero)
4. **Local Vertex Density** - 3D scatter colored by density
5. **Density Distribution** - Histogram of density values
6. **Adaptive Bin Assignment** - Histogram of assigned bins (64-2048)
7. **Error vs Bins** - Log-log plot of reconstruction error
8. **Per-Axis Error** - Box plot showing error distribution per axis
9. **Invariance Test** - Bar chart of errors across 10 random transformations

---

## 📈 Summary Statistics

### Total Visualizations: 36 PNG images + 2 text files

#### Main Assignment: 33 images
- Normalization comparisons: 8 (one per mesh)
- Quantization error plots: 8 (one per mesh)
- 3D visualizations: 8 (one per mesh)
- Distribution plots: 8 (one per mesh)
- Overall comparison: 1

#### Bonus Option 1: 2 images
- Seam analysis: 1
- 3D seam visualization: 1

#### Bonus Option 2: 1 image
- Comprehensive analysis: 1

#### Text Files: 2
- Analysis report: 1 (analysis_report.txt)
- README: 1 (this file)

---

## 🎨 Visualization Features

All visualizations include:
- ✅ High resolution (150 DPI)
- ✅ Clear titles and labels
- ✅ Grid lines for readability
- ✅ Color-coded for clarity
- ✅ Legends where applicable
- ✅ Professional formatting

---

## 🔍 How to View

### Option 1: File Explorer
Navigate to `code/visualizations/` and open any PNG file.

### Option 2: Python
```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('visualizations/bonus2_comprehensive_analysis_branch.png')
plt.figure(figsize=(18, 14))
plt.imshow(img)
plt.axis('off')
plt.show()
```

### Option 3: Jupyter Notebook
```python
from IPython.display import Image, display

display(Image('visualizations/bonus2_comprehensive_analysis_branch.png'))
```

---

## 📝 Notes

- All visualizations are generated from `generate_all_visualizations.py`
- Images are saved in PNG format for maximum compatibility
- File names follow the pattern: `{category}_{mesh_name}.png`
- Bonus visualizations are prefixed with `bonus1_` or `bonus2_`

---

## ✅ Verification

All visualizations have been successfully generated and saved. You can verify by checking:
1. File count: 25 PNG files + 1 TXT file
2. File sizes: All images should be > 100 KB
3. Image dimensions: Vary by visualization type

---

**Generated on**: 2025-11-15
**Script**: `generate_all_visualizations.py`
**Total Size**: ~15-20 MB

