"""
Complete pipeline for mesh normalization, quantization, and error analysis
"""

import numpy as np
from typing import Dict, Tuple
from mesh_processing import MeshLoader, MeshNormalizer
from mesh_normalizers import ZScoreNormalizer, UnitSphereNormalizer, MeshQuantizer
from mesh_metrics import ErrorMetrics


class MeshProcessingPipeline:
    """Complete pipeline for mesh preprocessing"""
    
    def __init__(self, vertices: np.ndarray):
        """
        Initialize pipeline with mesh vertices.
        
        Args:
            vertices: Original mesh vertices (N, 3)
        """
        self.original_vertices = vertices.copy()
        self.results = {}
    
    def run_normalization_comparison(self, n_bins: int = 1024) -> Dict:
        """
        Run all normalization methods and compare results.
        
        Args:
            n_bins: Number of quantization bins
            
        Returns:
            Dictionary containing results for all methods
        """
        methods = {
            'minmax_0_1': self._process_minmax((0, 1), n_bins),
            'minmax_-1_1': self._process_minmax((-1, 1), n_bins),
            'zscore': self._process_zscore(n_bins),
            'unit_sphere': self._process_unit_sphere(n_bins)
        }
        
        self.results = methods
        return methods
    
    def _process_minmax(self, target_range: Tuple[float, float], n_bins: int) -> Dict:
        """Process with Min-Max normalization"""
        # Normalize
        normalized, norm_params = MeshNormalizer.minmax_normalize(
            self.original_vertices, target_range
        )
        
        # For quantization, we need to convert to [0, 1] range
        if target_range == (0, 1):
            quantize_input = normalized
        else:
            # Convert from [-1, 1] to [0, 1] for quantization
            quantize_input = (normalized + 1) / 2
        
        # Quantize
        quantized = MeshQuantizer.quantize(quantize_input, n_bins)
        
        # Dequantize
        dequantized = MeshQuantizer.dequantize(quantized, n_bins)
        
        # Convert back to original range if needed
        if target_range != (0, 1):
            dequantized = dequantized * 2 - 1
        
        # Denormalize
        reconstructed = MeshNormalizer.minmax_denormalize(dequantized, norm_params)
        
        # Calculate errors
        errors = ErrorMetrics.compute_all_metrics(self.original_vertices, reconstructed)
        
        return {
            'normalized': normalized,
            'quantized': quantized,
            'dequantized': dequantized,
            'reconstructed': reconstructed,
            'norm_params': norm_params,
            'errors': errors,
            'n_bins': n_bins
        }
    
    def _process_zscore(self, n_bins: int) -> Dict:
        """Process with Z-Score normalization"""
        # Normalize
        normalized, norm_params = ZScoreNormalizer.normalize(self.original_vertices)
        
        # Convert to [0, 1] for quantization
        # Z-score typically gives values in roughly [-3, 3], so we'll clip and scale
        quantize_input = np.clip(normalized, -3, 3)
        quantize_input = (quantize_input + 3) / 6  # Map [-3, 3] to [0, 1]
        
        # Quantize
        quantized = MeshQuantizer.quantize(quantize_input, n_bins)
        
        # Dequantize
        dequantized = MeshQuantizer.dequantize(quantized, n_bins)
        
        # Convert back from [0, 1] to [-3, 3]
        dequantized = dequantized * 6 - 3
        
        # Denormalize
        reconstructed = ZScoreNormalizer.denormalize(dequantized, norm_params)
        
        # Calculate errors
        errors = ErrorMetrics.compute_all_metrics(self.original_vertices, reconstructed)
        
        return {
            'normalized': normalized,
            'quantized': quantized,
            'dequantized': dequantized,
            'reconstructed': reconstructed,
            'norm_params': norm_params,
            'errors': errors,
            'n_bins': n_bins
        }
    
    def _process_unit_sphere(self, n_bins: int) -> Dict:
        """Process with Unit Sphere normalization"""
        # Normalize
        normalized, norm_params = UnitSphereNormalizer.normalize(self.original_vertices)
        
        # Convert to [0, 1] for quantization
        # Unit sphere gives values in roughly [-1, 1]
        quantize_input = (normalized + 1) / 2
        
        # Quantize
        quantized = MeshQuantizer.quantize(quantize_input, n_bins)
        
        # Dequantize
        dequantized = MeshQuantizer.dequantize(quantized, n_bins)
        
        # Convert back to [-1, 1]
        dequantized = dequantized * 2 - 1
        
        # Denormalize
        reconstructed = UnitSphereNormalizer.denormalize(dequantized, norm_params)
        
        # Calculate errors
        errors = ErrorMetrics.compute_all_metrics(self.original_vertices, reconstructed)
        
        return {
            'normalized': normalized,
            'quantized': quantized,
            'dequantized': dequantized,
            'reconstructed': reconstructed,
            'norm_params': norm_params,
            'errors': errors,
            'n_bins': n_bins
        }
    
    def get_summary_table(self) -> Dict:
        """
        Get summary table of errors for all methods.
        
        Returns:
            Dictionary suitable for pandas DataFrame
        """
        summary = []
        
        for method_name, result in self.results.items():
            errors = result['errors']
            summary.append({
                'Method': method_name,
                'MSE': errors['mse'],
                'MAE': errors['mae'],
                'RMSE': errors['rmse'],
                'Max Error': errors['max_error'],
                'N_bins': result['n_bins']
            })
        
        return summary

