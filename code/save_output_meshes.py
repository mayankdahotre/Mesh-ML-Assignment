"""
Save processed output meshes for submission
"""

import numpy as np
from pathlib import Path
from mesh_processing import MeshLoader, MeshNormalizer
from mesh_normalizers import ZScoreNormalizer, UnitSphereNormalizer, MeshQuantizer

def save_obj(vertices, faces, filepath):
    """Save mesh to OBJ file"""
    with open(filepath, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def main():
    print("=" * 80)
    print("SAVING OUTPUT MESHES")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path('../output_meshes')
    output_dir.mkdir(exist_ok=True)
    
    # Get mesh files
    mesh_dir = Path('../meshes')
    mesh_files = sorted(mesh_dir.glob('*.obj'))
    
    print(f"\nFound {len(mesh_files)} mesh files")
    print(f"Output directory: {output_dir.absolute()}\n")
    
    for mesh_file in mesh_files:
        mesh_name = mesh_file.stem
        print(f"Processing: {mesh_name}")
        
        # Load original mesh
        vertices, faces = MeshLoader.load_obj(str(mesh_file))
        
        # Create subdirectory for this mesh
        mesh_output_dir = output_dir / mesh_name
        mesh_output_dir.mkdir(exist_ok=True)
        
        # Save original (for reference)
        save_obj(vertices, faces, mesh_output_dir / f"{mesh_name}_original.obj")
        print(f"  ✓ Saved: {mesh_name}_original.obj")
        
        # 1. Min-Max [0, 1] normalized
        norm_01, params_01 = MeshNormalizer.minmax_normalize(vertices, (0, 1))
        save_obj(norm_01, faces, mesh_output_dir / f"{mesh_name}_minmax_0_1.obj")
        print(f"  ✓ Saved: {mesh_name}_minmax_0_1.obj")
        
        # 2. Min-Max [-1, 1] normalized
        norm_11, params_11 = MeshNormalizer.minmax_normalize(vertices, (-1, 1))
        save_obj(norm_11, faces, mesh_output_dir / f"{mesh_name}_minmax_-1_1.obj")
        print(f"  ✓ Saved: {mesh_name}_minmax_-1_1.obj")
        
        # 3. Z-Score normalized
        norm_z, params_z = ZScoreNormalizer.normalize(vertices)
        save_obj(norm_z, faces, mesh_output_dir / f"{mesh_name}_zscore.obj")
        print(f"  ✓ Saved: {mesh_name}_zscore.obj")
        
        # 4. Unit Sphere normalized
        norm_s, params_s = UnitSphereNormalizer.normalize(vertices)
        save_obj(norm_s, faces, mesh_output_dir / f"{mesh_name}_unit_sphere.obj")
        print(f"  ✓ Saved: {mesh_name}_unit_sphere.obj")
        
        # 5. Quantized and reconstructed (Min-Max [0,1] with 1024 bins)
        # Scale to [0, 1] for quantization
        norm_01_scaled = (norm_01 - norm_01.min()) / (norm_01.max() - norm_01.min() + 1e-8)
        
        # Quantize
        quantized = MeshQuantizer.quantize(norm_01_scaled, 1024)
        
        # Dequantize
        dequantized = MeshQuantizer.dequantize(quantized, 1024)
        
        # Scale back
        dequantized_scaled = dequantized * (norm_01.max() - norm_01.min()) + norm_01.min()
        
        # Denormalize to original space
        reconstructed = MeshNormalizer.minmax_denormalize(dequantized_scaled, params_01)
        
        save_obj(reconstructed, faces, mesh_output_dir / f"{mesh_name}_reconstructed_1024bins.obj")
        print(f"  ✓ Saved: {mesh_name}_reconstructed_1024bins.obj")
        
        # 6. Quantized (different bin sizes)
        for n_bins in [128, 512, 2048]:
            quant = MeshQuantizer.quantize(norm_01_scaled, n_bins)
            dequant = MeshQuantizer.dequantize(quant, n_bins)
            dequant_scaled = dequant * (norm_01.max() - norm_01.min()) + norm_01.min()
            recon = MeshNormalizer.minmax_denormalize(dequant_scaled, params_01)
            
            save_obj(recon, faces, mesh_output_dir / f"{mesh_name}_reconstructed_{n_bins}bins.obj")
            print(f"  ✓ Saved: {mesh_name}_reconstructed_{n_bins}bins.obj")
        
        print(f"  Total: 9 output files for {mesh_name}\n")
    
    # Create summary file
    summary_file = output_dir / "README.txt"
    with open(summary_file, 'w') as f:
        f.write("OUTPUT MESHES - Mesh ML Assignment\n")
        f.write("=" * 80 + "\n\n")
        f.write("This folder contains processed mesh outputs.\n\n")
        f.write("STRUCTURE:\n")
        f.write("-" * 80 + "\n")
        f.write("Each mesh has its own subfolder with the following files:\n\n")
        f.write("1. {mesh}_original.obj - Original mesh (for reference)\n")
        f.write("2. {mesh}_minmax_0_1.obj - Min-Max normalized to [0, 1]\n")
        f.write("3. {mesh}_minmax_-1_1.obj - Min-Max normalized to [-1, 1]\n")
        f.write("4. {mesh}_zscore.obj - Z-Score normalized\n")
        f.write("5. {mesh}_unit_sphere.obj - Unit Sphere normalized\n")
        f.write("6. {mesh}_reconstructed_128bins.obj - Quantized with 128 bins\n")
        f.write("7. {mesh}_reconstructed_512bins.obj - Quantized with 512 bins\n")
        f.write("8. {mesh}_reconstructed_1024bins.obj - Quantized with 1024 bins\n")
        f.write("9. {mesh}_reconstructed_2048bins.obj - Quantized with 2048 bins\n\n")
        f.write("TOTAL FILES:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Meshes processed: {len(mesh_files)}\n")
        f.write(f"Files per mesh: 9\n")
        f.write(f"Total output files: {len(mesh_files) * 9}\n\n")
        f.write("USAGE:\n")
        f.write("-" * 80 + "\n")
        f.write("These OBJ files can be:\n")
        f.write("- Opened in Blender, MeshLab, or any 3D viewer\n")
        f.write("- Used for further processing\n")
        f.write("- Compared visually to see normalization effects\n")
        f.write("- Analyzed for quality assessment\n\n")
        f.write("NOTES:\n")
        f.write("-" * 80 + "\n")
        f.write("- All meshes preserve original topology (same faces)\n")
        f.write("- Only vertex positions are modified\n")
        f.write("- Reconstructed meshes show quantization effects\n")
        f.write("- Higher bin counts = better reconstruction quality\n\n")
        f.write(f"Generated: 2025-11-15\n")
        f.write(f"Script: save_output_meshes.py\n")
    
    print("=" * 80)
    print("OUTPUT MESHES SAVED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nLocation: {output_dir.absolute()}")
    print(f"Total meshes: {len(mesh_files)}")
    print(f"Files per mesh: 9")
    print(f"Total files: {len(mesh_files) * 9}")
    print(f"\nSummary: {summary_file.absolute()}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

