"""
Error measurement and visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import pandas as pd


class ErrorMetrics:
    """Calculate error metrics between original and reconstructed meshes"""
    
    @staticmethod
    def mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate Mean Squared Error (MSE).
        Formula: MSE = (1/N) * Σ(original - reconstructed)²
        
        Args:
            original: Original vertices (N, 3)
            reconstructed: Reconstructed vertices (N, 3)
            
        Returns:
            MSE value
        """
        mse = np.mean((original - reconstructed) ** 2)
        return float(mse)
    
    @staticmethod
    def mean_absolute_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error (MAE).
        Formula: MAE = (1/N) * Σ|original - reconstructed|
        
        Args:
            original: Original vertices (N, 3)
            reconstructed: Reconstructed vertices (N, 3)
            
        Returns:
            MAE value
        """
        mae = np.mean(np.abs(original - reconstructed))
        return float(mae)
    
    @staticmethod
    def root_mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error (RMSE).
        
        Args:
            original: Original vertices (N, 3)
            reconstructed: Reconstructed vertices (N, 3)
            
        Returns:
            RMSE value
        """
        mse = ErrorMetrics.mean_squared_error(original, reconstructed)
        return float(np.sqrt(mse))
    
    @staticmethod
    def max_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate maximum absolute error.
        
        Args:
            original: Original vertices (N, 3)
            reconstructed: Reconstructed vertices (N, 3)
            
        Returns:
            Maximum error value
        """
        max_err = np.max(np.abs(original - reconstructed))
        return float(max_err)
    
    @staticmethod
    def per_axis_error(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate error metrics per axis (x, y, z).
        
        Args:
            original: Original vertices (N, 3)
            reconstructed: Reconstructed vertices (N, 3)
            
        Returns:
            Dictionary with per-axis MSE and MAE
        """
        errors = {}
        axes = ['x', 'y', 'z']
        
        for i, axis in enumerate(axes):
            errors[axis] = {
                'mse': float(np.mean((original[:, i] - reconstructed[:, i]) ** 2)),
                'mae': float(np.mean(np.abs(original[:, i] - reconstructed[:, i]))),
                'max': float(np.max(np.abs(original[:, i] - reconstructed[:, i])))
            }
        
        return errors
    
    @staticmethod
    def compute_all_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict:
        """
        Compute all error metrics.
        
        Args:
            original: Original vertices (N, 3)
            reconstructed: Reconstructed vertices (N, 3)
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            'mse': ErrorMetrics.mean_squared_error(original, reconstructed),
            'mae': ErrorMetrics.mean_absolute_error(original, reconstructed),
            'rmse': ErrorMetrics.root_mean_squared_error(original, reconstructed),
            'max_error': ErrorMetrics.max_error(original, reconstructed),
            'per_axis': ErrorMetrics.per_axis_error(original, reconstructed)
        }
        
        return metrics


class MeshVisualizer:
    """Visualization utilities for meshes and errors"""
    
    @staticmethod
    def plot_vertex_distribution(vertices: np.ndarray, title: str = "Vertex Distribution"):
        """
        Plot 3D scatter of vertices and histograms per axis.
        
        Args:
            vertices: Vertex array (N, 3)
            title: Plot title
        """
        fig = plt.figure(figsize=(15, 5))
        
        # 3D scatter plot
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                   c=vertices[:, 2], cmap='viridis', s=1, alpha=0.6)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'{title} - 3D View')
        
        # Histograms per axis
        axes_names = ['X', 'Y', 'Z']
        for i in range(3):
            ax = fig.add_subplot(1, 3, i + 2)
            ax.hist(vertices[:, i], bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel(f'{axes_names[i]} coordinate')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{axes_names[i]}-axis Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

