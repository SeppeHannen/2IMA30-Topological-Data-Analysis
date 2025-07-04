import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG (Edit parameters here) ---
FILE = "data/bag_0/cloud_final.ply" # Which point cloud to load
VOXEL = 0.002                       # Side length used for voxel downsampling

# --- LOAD ---
pcd = o3d.io.read_point_cloud(FILE)
print(pcd)                          # point count

# --- DOWNSAMPLE & COLOR BY HEIGHT ---
pcd = pcd.voxel_down_sample(VOXEL)
pts = np.asarray(pcd.points)
z = pts[:, 2]
h_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)   # +eps to avoid /0
pcd.colors = o3d.utility.Vector3dVector(plt.cm.viridis(h_norm)[:, :3])

# --- VIEW ---
o3d.visualization.draw_geometries(
    [pcd],
    window_name="Cherry-tree point cloud",
    width=960, height=720
)
