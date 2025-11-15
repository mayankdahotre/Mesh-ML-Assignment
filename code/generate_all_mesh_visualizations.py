"""
Generate complete visualizations for ALL 8 meshes
- 3D visualizations
- Normalization comparisons
- Quantization error plots
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mesh_processing import MeshLoader, MeshNormalizer
from mesh_normalizers import MeshQuantizer, ZScoreNormalizer, UnitSphereNormalizer
from mesh_metrics import ErrorMetrics

# Create visualizations directory
vis_dir = Path('visualizations')
vis_dir.mkdir(exist_ok=True)

print("=" * 80)
print("GENERATING COMPLETE VISUALIZATIONS FOR ALL MESHES")
print("=" * 80)

# Get all mesh files
mesh_dir = Path('../meshes')
mesh_files = sorted(mesh_dir.glob('*.obj'))

print(f"\nFound {len(mesh_files)} mesh files")
print(f"Output directory: {vis_dir.absolute()}\n")

total_images = 0

for mesh_idx, mesh_file in enumerate(mesh_files, 1):
    mesh_name = mesh_file.stem
    
    print(f"\n{'='*80}")
    print(f"Processing [{mesh_idx}/{len(mesh_files)}]: {mesh_name}")
    print(f"{'='*80}")
    
    # Load mesh
    vertices, faces = MeshLoader.load_obj(str(mesh_file))
    print(f"  Vertices: {len(vertices):,}")
    print(f"  Faces: {len(faces):,}")
    
    # ========================================================================
    # 1. NORMALIZATION COMPARISON
    # ========================================================================
    print(f"\n  [1/3] Generating normalization comparison...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Normalization Methods - {mesh_name}', fontsize=16, fontweight='bold')
    
    # Original
    ax = axes[0, 0]
    ax.hist(vertices.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title('Original', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    stats_text = f'Mean: {vertices.mean():.3f}\nStd: {vertices.std():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Min-Max [0, 1]
    norm_01, params_01 = MeshNormalizer.minmax_normalize(vertices, (0, 1))
    ax = axes[0, 1]
    ax.hist(norm_01.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.set_title('Min-Max [0, 1]', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    stats_text = f'Mean: {norm_01.mean():.3f}\nStd: {norm_01.std():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Min-Max [-1, 1]
    norm_11, params_11 = MeshNormalizer.minmax_normalize(vertices, (-1, 1))
    ax = axes[0, 2]
    ax.hist(norm_11.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.set_title('Min-Max [-1, 1]', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    stats_text = f'Mean: {norm_11.mean():.3f}\nStd: {norm_11.std():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Z-Score
    norm_z, params_z = ZScoreNormalizer.normalize(vertices)
    ax = axes[1, 0]
    ax.hist(norm_z.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    ax.set_title('Z-Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    stats_text = f'Mean: {norm_z.mean():.3f}\nStd: {norm_z.std():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Unit Sphere
    norm_s, params_s = UnitSphereNormalizer.normalize(vertices)
    ax = axes[1, 1]
    ax.hist(norm_s.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.set_title('Unit Sphere', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    stats_text = f'Mean: {norm_s.mean():.3f}\nStd: {norm_s.std():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Statistics comparison
    ax = axes[1, 2]
    methods = ['Original', 'MinMax\n[0,1]', 'MinMax\n[-1,1]', 'Z-Score', 'Unit\nSphere']
    means = [vertices.mean(), norm_01.mean(), norm_11.mean(), norm_z.mean(), norm_s.mean()]
    stds = [vertices.std(), norm_01.std(), norm_11.std(), norm_z.std(), norm_s.std()]
    
    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax.bar(x - width/2, means, width, label='Mean', alpha=0.8, color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, stds, width, label='Std Dev', alpha=0.8, color='coral', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Statistics Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = vis_dir / f'normalization_comparison_{mesh_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved: normalization_comparison_{mesh_name}.png")
    total_images += 1
    
    # ========================================================================
    # 2. QUANTIZATION ERROR ANALYSIS
    # ========================================================================
    print(f"  [2/3] Generating quantization error analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'Quantization Error Analysis - {mesh_name}', fontsize=16, fontweight='bold')
    
    bin_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    # Test each normalization method
    methods = {
        'MinMax [0,1]': (norm_01, params_01, 'green'),
        'MinMax [-1,1]': (norm_11, params_11, 'orange'),
        'Z-Score': (norm_z, params_z, 'red'),
        'Unit Sphere': (norm_s, params_s, 'purple')
    }
    
    for idx, (method_name, (norm_v, params, color)) in enumerate(methods.items()):
        ax = axes[idx // 2, idx % 2]
        
        # Scale to [0, 1] for quantization
        norm_01_scaled = (norm_v - norm_v.min()) / (norm_v.max() - norm_v.min() + 1e-8)
        
        mse_values = []
        mae_values = []
        
        for n_bins in bin_sizes:
            quant = MeshQuantizer.quantize(norm_01_scaled, n_bins)
            dequant = MeshQuantizer.dequantize(quant, n_bins)
            
            # Scale back
            dequant_scaled = dequant * (norm_v.max() - norm_v.min()) + norm_v.min()
            
            # Denormalize
            if method_name in ['MinMax [0,1]', 'MinMax [-1,1]']:
                recon = MeshNormalizer.minmax_denormalize(dequant_scaled, params)
            elif method_name == 'Z-Score':
                recon = ZScoreNormalizer.denormalize(dequant_scaled, params)
            else:
                recon = UnitSphereNormalizer.denormalize(dequant_scaled, params)
            
            mse = ErrorMetrics.mean_squared_error(vertices, recon)
            mae = ErrorMetrics.mean_absolute_error(vertices, recon)
            
            mse_values.append(mse)
            mae_values.append(mae)
        
        ax.plot(bin_sizes, mse_values, 'o-', linewidth=2.5, markersize=8, 
                label='MSE', color=color, alpha=0.8)
        ax.plot(bin_sizes, mae_values, 's--', linewidth=2.5, markersize=8, 
                label='MAE', color=color, alpha=0.5)
        ax.set_xlabel('Number of Bins', fontweight='bold', fontsize=11)
        ax.set_ylabel('Error', fontweight='bold', fontsize=11)
        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10)
        
        # Add text with best error
        best_mse = min(mse_values)
        best_mae = min(mae_values)
        text = f'Best MSE: {best_mse:.2e}\nBest MAE: {best_mae:.6f}'
        ax.text(0.98, 0.98, text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    output_file = vis_dir / f'quantization_error_{mesh_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved: quantization_error_{mesh_name}.png")
    total_images += 1

    # ========================================================================
    # 3. 3D VISUALIZATION
    # ========================================================================
    print(f"  [3/3] Generating 3D visualization...")

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'3D Mesh Visualization - {mesh_name}', fontsize=16, fontweight='bold')

    # Sample vertices for visualization (to avoid overcrowding)
    max_points = 5000
    if len(vertices) > max_points:
        indices = np.random.choice(len(vertices), max_points, replace=False)
    else:
        indices = np.arange(len(vertices))

    # 1. Original
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.scatter(vertices[indices, 0], vertices[indices, 1], vertices[indices, 2],
               c='blue', s=1, alpha=0.6)
    ax.set_title('Original', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 2. Min-Max [0, 1]
    ax = fig.add_subplot(2, 3, 2, projection='3d')
    ax.scatter(norm_01[indices, 0], norm_01[indices, 1], norm_01[indices, 2],
               c='green', s=1, alpha=0.6)
    ax.set_title('Min-Max [0, 1]', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 3. Min-Max [-1, 1]
    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.scatter(norm_11[indices, 0], norm_11[indices, 1], norm_11[indices, 2],
               c='orange', s=1, alpha=0.6)
    ax.set_title('Min-Max [-1, 1]', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 4. Z-Score
    ax = fig.add_subplot(2, 3, 4, projection='3d')
    ax.scatter(norm_z[indices, 0], norm_z[indices, 1], norm_z[indices, 2],
               c='red', s=1, alpha=0.6)
    ax.set_title('Z-Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 5. Unit Sphere
    ax = fig.add_subplot(2, 3, 5, projection='3d')
    ax.scatter(norm_s[indices, 0], norm_s[indices, 1], norm_s[indices, 2],
               c='purple', s=1, alpha=0.6)
    ax.set_title('Unit Sphere', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 6. Quantized (512 bins)
    norm_01_scaled = (norm_01 - norm_01.min()) / (norm_01.max() - norm_01.min() + 1e-8)
    quant_512 = MeshQuantizer.quantize(norm_01_scaled, 512)
    dequant_512 = MeshQuantizer.dequantize(quant_512, 512)
    dequant_512_scaled = dequant_512 * (norm_01.max() - norm_01.min()) + norm_01.min()
    recon_512 = MeshNormalizer.minmax_denormalize(dequant_512_scaled, params_01)

    ax = fig.add_subplot(2, 3, 6, projection='3d')
    ax.scatter(recon_512[indices, 0], recon_512[indices, 1], recon_512[indices, 2],
               c='brown', s=1, alpha=0.6)
    ax.set_title('Quantized (512 bins)', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.tight_layout()
    output_file = vis_dir / f'3d_visualization_{mesh_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved: 3d_visualization_{mesh_name}.png")
    total_images += 1

    # ========================================================================
    # 4. DISTRIBUTION PLOT (Individual)
    # ========================================================================
    print(f"  [4/4] Generating distribution plot...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Vertex Coordinate Distribution - {mesh_name}', fontsize=16, fontweight='bold')

    # X, Y, Z distributions
    for idx, (axis_name, axis_idx) in enumerate([('X', 0), ('Y', 1), ('Z', 2)]):
        ax = axes[idx // 2, idx % 2]

        values = vertices[:, axis_idx]
        ax.hist(values, bins=50, alpha=0.7, color=['red', 'green', 'blue'][axis_idx], edgecolor='black')
        ax.set_title(f'{axis_name}-Axis Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f'Mean: {values.mean():.3f}\nStd: {values.std():.3f}\nMin: {values.min():.3f}\nMax: {values.max():.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Overall statistics
    ax = axes[1, 1]
    ax.axis('off')

    stats_text = f"""
MESH STATISTICS
{'='*40}

Vertices: {len(vertices):,}
Faces: {len(faces):,}

Coordinate Ranges:
  X: [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]
  Y: [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]
  Z: [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]

Overall Statistics:
  Mean: {vertices.mean():.3f}
  Std Dev: {vertices.std():.3f}
  Min: {vertices.min():.3f}
  Max: {vertices.max():.3f}
"""

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    output_file = vis_dir / f'{mesh_name}_distribution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved: {mesh_name}_distribution.png")
    total_images += 1

print("\n" + "=" * 80)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("=" * 80)
print(f"\nTotal images generated: {total_images}")
print(f"Images per mesh: 4 (normalization, quantization, 3D, distribution)")
print(f"Total meshes: {len(mesh_files)}")
print(f"Expected total: {len(mesh_files) * 4}")
print(f"\nLocation: {vis_dir.absolute()}")
print("\n" + "=" * 80)

