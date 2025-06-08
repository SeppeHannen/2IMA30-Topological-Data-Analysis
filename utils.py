import numpy as np
import open3d as o3d


def load_point_cloud(path: str) -> np.ndarray:
    """Load a point cloud file (.pcd/.ply) and return Nx3 numpy array."""
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points)
