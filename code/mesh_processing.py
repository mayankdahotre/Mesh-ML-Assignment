"""
Mesh Normalization, Quantization, and Error Analysis
Assignment for Mesh ML - Data Preprocessing Pipeline
"""

import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path


class MeshLoader:
    """Load and parse .obj mesh files"""
    
    @staticmethod
    def load_obj(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load vertices and faces from an OBJ file.
        
        Args:
            filepath: Path to the .obj file
            
        Returns:
            vertices: numpy array of shape (N, 3) containing vertex coordinates
            faces: numpy array of shape (M, 3) containing face indices
        """
        vertices = []
        faces = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                
                # Parse vertex (v x y z)
                if parts[0] == 'v':
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                
                # Parse face (f v1 v2 v3 or f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3)
                elif parts[0] == 'f':
                    # Handle different face formats
                    face_vertices = []
                    for i in range(1, len(parts)):
                        # Split by '/' and take first index (vertex index)
                        vertex_idx = int(parts[i].split('/')[0]) - 1  # OBJ indices start at 1
                        face_vertices.append(vertex_idx)
                    if len(face_vertices) >= 3:
                        faces.append(face_vertices[:3])  # Take first 3 vertices for triangle
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32) if faces else np.array([], dtype=np.int32)
        
        return vertices, faces
    
    @staticmethod
    def get_mesh_info(vertices: np.ndarray, faces: np.ndarray) -> Dict:
        """
        Get basic information about the mesh.
        
        Args:
            vertices: Vertex array
            faces: Face array
            
        Returns:
            Dictionary containing mesh statistics
        """
        info = {
            'num_vertices': len(vertices),
            'num_faces': len(faces),
            'min_coords': vertices.min(axis=0),
            'max_coords': vertices.max(axis=0),
            'mean_coords': vertices.mean(axis=0),
            'std_coords': vertices.std(axis=0),
            'bounding_box_size': vertices.max(axis=0) - vertices.min(axis=0)
        }
        return info


class MeshNormalizer:
    """Implement various normalization techniques for mesh vertices"""
    
    @staticmethod
    def minmax_normalize(vertices: np.ndarray, 
                         target_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, Dict]:
        """
        Min-Max Normalization: Scale vertices to a target range.
        Formula: x' = (x - x_min) / (x_max - x_min) * (range_max - range_min) + range_min
        
        Args:
            vertices: Original vertices (N, 3)
            target_range: Tuple of (min, max) for target range
            
        Returns:
            normalized_vertices: Normalized vertices
            params: Dictionary containing normalization parameters for reversal
        """
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        
        # Avoid division by zero
        range_vals = v_max - v_min
        range_vals[range_vals == 0] = 1.0
        
        # Normalize to [0, 1] first
        normalized = (vertices - v_min) / range_vals
        
        # Scale to target range
        target_min, target_max = target_range
        normalized = normalized * (target_max - target_min) + target_min
        
        params = {
            'method': 'minmax',
            'v_min': v_min,
            'v_max': v_max,
            'target_range': target_range
        }
        
        return normalized, params
    
    @staticmethod
    def minmax_denormalize(normalized_vertices: np.ndarray, params: Dict) -> np.ndarray:
        """
        Reverse Min-Max normalization.
        
        Args:
            normalized_vertices: Normalized vertices
            params: Parameters from normalization
            
        Returns:
            Original scale vertices
        """
        target_min, target_max = params['target_range']
        v_min = params['v_min']
        v_max = params['v_max']
        
        # Scale back from target range to [0, 1]
        vertices = (normalized_vertices - target_min) / (target_max - target_min)
        
        # Scale back to original range
        range_vals = v_max - v_min
        range_vals[range_vals == 0] = 1.0
        vertices = vertices * range_vals + v_min
        
        return vertices

