"""
Additional normalization methods: Z-Score and Unit Sphere
"""

import numpy as np
from typing import Tuple, Dict


class ZScoreNormalizer:
    """Z-Score (Standardization) Normalization"""
    
    @staticmethod
    def normalize(vertices: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Z-Score Normalization: Center and scale by standard deviation.
        Formula: x' = (x - μ) / σ
        
        Args:
            vertices: Original vertices (N, 3)
            
        Returns:
            normalized_vertices: Normalized vertices
            params: Dictionary containing normalization parameters
        """
        mean = vertices.mean(axis=0)
        std = vertices.std(axis=0)
        
        # Avoid division by zero
        std[std == 0] = 1.0
        
        normalized = (vertices - mean) / std
        
        params = {
            'method': 'zscore',
            'mean': mean,
            'std': std
        }
        
        return normalized, params
    
    @staticmethod
    def denormalize(normalized_vertices: np.ndarray, params: Dict) -> np.ndarray:
        """
        Reverse Z-Score normalization.
        
        Args:
            normalized_vertices: Normalized vertices
            params: Parameters from normalization
            
        Returns:
            Original scale vertices
        """
        mean = params['mean']
        std = params['std']
        
        vertices = normalized_vertices * std + mean
        
        return vertices


class UnitSphereNormalizer:
    """Unit Sphere Normalization"""
    
    @staticmethod
    def normalize(vertices: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Unit Sphere Normalization: Scale mesh to fit in a sphere of radius 1.
        
        Args:
            vertices: Original vertices (N, 3)
            
        Returns:
            normalized_vertices: Normalized vertices
            params: Dictionary containing normalization parameters
        """
        # Center the mesh at origin
        centroid = vertices.mean(axis=0)
        centered = vertices - centroid
        
        # Find maximum distance from origin
        max_distance = np.max(np.linalg.norm(centered, axis=1))
        
        # Avoid division by zero
        if max_distance == 0:
            max_distance = 1.0
        
        # Scale to unit sphere
        normalized = centered / max_distance
        
        params = {
            'method': 'unit_sphere',
            'centroid': centroid,
            'max_distance': max_distance
        }
        
        return normalized, params
    
    @staticmethod
    def denormalize(normalized_vertices: np.ndarray, params: Dict) -> np.ndarray:
        """
        Reverse Unit Sphere normalization.
        
        Args:
            normalized_vertices: Normalized vertices
            params: Parameters from normalization
            
        Returns:
            Original scale vertices
        """
        centroid = params['centroid']
        max_distance = params['max_distance']
        
        # Scale back and translate
        vertices = normalized_vertices * max_distance + centroid
        
        return vertices


class MeshQuantizer:
    """Quantization and Dequantization for mesh vertices"""
    
    @staticmethod
    def quantize(normalized_vertices: np.ndarray, n_bins: int = 1024) -> np.ndarray:
        """
        Quantize normalized vertices (assumed to be in [0, 1] range).
        Formula: q = int(x' × (n_bins - 1))
        
        Args:
            normalized_vertices: Normalized vertices in [0, 1] range
            n_bins: Number of quantization bins (default: 1024)
            
        Returns:
            quantized_vertices: Integer quantized vertices
        """
        # Clip to [0, 1] range to be safe
        clipped = np.clip(normalized_vertices, 0, 1)
        
        # Quantize
        quantized = np.round(clipped * (n_bins - 1)).astype(np.int32)
        
        return quantized
    
    @staticmethod
    def dequantize(quantized_vertices: np.ndarray, n_bins: int = 1024) -> np.ndarray:
        """
        Dequantize vertices back to continuous values.
        Formula: x' = q / (n_bins - 1)
        
        Args:
            quantized_vertices: Integer quantized vertices
            n_bins: Number of quantization bins (must match quantization)
            
        Returns:
            dequantized_vertices: Continuous vertices in [0, 1] range
        """
        dequantized = quantized_vertices.astype(np.float32) / (n_bins - 1)
        
        return dequantized

