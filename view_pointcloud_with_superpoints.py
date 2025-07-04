import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG (Edit parameters here) ---
FILE = "data/bag_0/cloud_final.ply" # Which point cloud to load
VOXEL = 0.002                       # Side length used for voxel downsampling

# --- LOAD ---
pcd = o3d.io.read_point_cloud(FILE)
print(pcd)

pcd = pcd.voxel_down_sample(VOXEL)
pts = np.asarray(pcd.points)

z = pts[:, 2]
h_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)   # +eps to avoid /0
pcd.colors = o3d.utility.Vector3dVector(plt.cm.viridis(h_norm)[:, :3])

def superpoint_selection(pts, r_super=0.1):
    bool_pts = np.zeros(pts.shape[0], dtype=bool)
    super_points = []
    while not np.all(bool_pts):
        remaining_indices = np.where(~bool_pts)[0] # Remaining Indices
        pts_remain = pts[~bool_pts] # Uncovered Points
        rand_super_pt = pts_remain[np.random.choice(pts_remain.shape[0])] # Randomly Chosen Super Point

        bool_pts_super_pt = np.sum(np.abs(pts_remain - rand_super_pt), axis=1) < r_super # Subset of Uncovered Points
        super_pt = np.mean(pts_remain[bool_pts_super_pt], axis=0) # Mean of Subset of Uncovered Points
        super_points.append(super_pt)

        bool_pts[remaining_indices[bool_pts_super_pt]] = True # Change set of Covered Points accordingly

    return np.array(super_points)

r_super = 0.05
super_points = superpoint_selection(pts, r_super)

# Create List of Sphere Geometries at each Superpoint
super_spheres = []
for pt in super_points:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r_super)  # adjust size here
    sphere.translate(pt)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])  # red
    super_spheres.append(sphere)

# Convert super_points to Open3D PointCloud
pcd_super = o3d.geometry.PointCloud()
pcd_super.points = o3d.utility.Vector3dVector(super_points)

# Visualization of Point Clouds and Superpoint Cover
o3d.visualization.draw_geometries(
    [pcd] + super_spheres,
    window_name="Cherry-tree Point Cloud with Superpoints",
    width=960, height=720
)