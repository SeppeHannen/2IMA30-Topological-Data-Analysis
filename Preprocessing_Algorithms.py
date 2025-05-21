import open3d as o3d
import copy
import numpy as np
import matplotlib.pyplot as plt

FILE = "data/bag_0/bottom_cloud.ply"
pcd_original = o3d.io.read_point_cloud(FILE)
np_original_pcd = len(pcd_original.points)
print("Original num points:", np_original_pcd)

# Color the base cloud for consistency
pcd_original.paint_uniform_color([1, 0.706, 0])  # orange

# ---- 1. Statistical Outlier Removal ----
def statistical_outlier_removal(pcd, nb_neighbors, std_ratio):
    pcd_stat = copy.deepcopy(pcd)
    pcd_stat, _ = pcd_stat.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    num_points = len(pcd_stat.points)
    return pcd_stat, num_points

# Translate for visualization
pcd_stat, num_stat_points = statistical_outlier_removal(pcd_original, nb_neighbors=50, std_ratio=0.3)
pcd_stat.paint_uniform_color([0, 0.8, 0.2])  # green
pcd_stat.translate((0.02, 0, 0))

# print("After Statistical Outlier Removal:", num_stat_points)
o3d.visualization.draw_geometries(
    [pcd_original, pcd_stat],
    window_name="Voxel Downsampled vs Statistical Outlier Removal",
    width=960, height=720
)

std_ratios = np.linspace(0.05, 0.5, 10)
num_voxel_points_radius = np.full(10, np.nan)
for i, std_ratio in enumerate(std_ratios):
    _, num_radius_points = statistical_outlier_removal(pcd_original, nb_neighbors=50, std_ratio=std_ratio)
    num_voxel_points_radius[i] = num_radius_points 

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(std_ratios, num_voxel_points_radius, label="Num. Points")
ax.set_yscale('log')
ax.set_ylim([0, np_original_pcd])
ax.set_xlabel("Voxel Radius")
ax.set_ylabel("Number of points")
plt.show()

# ---- 2. Radius Outlier Removal ----
def radius_outlier_removal(pcd, nb_points, radius):
    pcd_radius = copy.deepcopy(pcd)
    pcd_radius, _ = pcd_radius.remove_radius_outlier(nb_points=nb_points, radius=radius)
    num_points = len(pcd_radius.points)
    return pcd_radius, num_points

pcd_radius, num_radius_points = radius_outlier_removal(pcd_original, nb_points=50, radius=0.02)
# print("After Radius Outlier Removal:", num_radius_points)
pcd_radius.paint_uniform_color([0, 0.8, 0.2])  # green
pcd_radius.translate((0.02, 0, 0))

o3d.visualization.draw_geometries(
    [pcd_original, pcd_radius],
    window_name="Voxel Downsampled vs Radius Outlier Removal",
    width=960, height=720
)

radius_sizes = np.linspace(0.005, 0.05, 10)
num_voxel_points_radius = np.full(10, np.nan)
for i, voxel_radius in enumerate(radius_sizes):
    _, num_radius_points = radius_outlier_removal(pcd_original, nb_points=50, radius=voxel_radius)
    num_voxel_points_radius[i] = num_radius_points 

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(radius_sizes, num_voxel_points_radius, label="Num. Points")
ax.set_yscale('log')
ax.set_ylim([0, np_original_pcd])
ax.set_xlabel("Voxel Radius")
ax.set_ylabel("Number of points")
plt.show()

# ---- 3. Voxel ----
def voxel_downsampling(pcd, voxel_radius):
    pcd_voxel = copy.deepcopy(pcd)
    pcd_voxel = pcd_voxel.voxel_down_sample(voxel_radius)
    num_points = len(pcd_voxel.points)
    return pcd_voxel, num_points

pcd_voxel, num_voxel_points = voxel_downsampling(pcd_original, 0.02)
# print(f"After Voxel Downsampling for radius {0.02}:", num_voxel_points)
pcd_voxel.paint_uniform_color([0, 0.8, 0.2])  # green
pcd_voxel.translate((0.02, 0, 0))

o3d.visualization.draw_geometries(
    [pcd_original, pcd_voxel],
    window_name="Orginal vs Voxel Downsampled Point clouds",
    width=960, height=720
)

voxel_sizes = np.linspace(0.01, 0.2, 20)
num_voxel_points_radius = np.full(20, np.nan)
for i, voxel_radius in enumerate(voxel_sizes):
    _, num_voxel_points = voxel_downsampling(pcd_original, voxel_radius)
    num_voxel_points_radius[i] = num_voxel_points 

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(voxel_sizes, num_voxel_points_radius, label="Num. Points")
ax.set_yscale('log')
ax.set_ylim([0, np_original_pcd])
ax.set_xlabel("Voxel Radius")
ax.set_ylabel("Number of points")
plt.show()

# ---- 4. Normal Estimation ----
# Estimate normals
pcd_normal = copy.deepcopy(pcd_original)
pcd_normal.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

# Normalize normals (optional)
pcd_normal.normalize_normals()

# Parameters for arrow size
arrow_length = 0.005

# Create lines representing normal vectors
points = np.asarray(pcd_normal.points)
normals = np.asarray(pcd_normal.normals)

# Start and end points of arrows
line_points = []
line_indices = []

for i, (p, n) in enumerate(zip(points, normals)):
    line_points.append(p)
    line_points.append(p + arrow_length * n)
    line_indices.append([2 * i, 2 * i + 1])

# Create LineSet for arrows
arrow_lines = o3d.geometry.LineSet()
arrow_lines.points = o3d.utility.Vector3dVector(line_points)
arrow_lines.lines = o3d.utility.Vector2iVector(line_indices)

# Optional: set color of arrows (red)
arrow_lines.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in line_indices])

# Visualize
o3d.visualization.draw_geometries(
    [pcd_normal, arrow_lines],
    window_name="Point Normals as Arrows",
    width=960, height=720
)