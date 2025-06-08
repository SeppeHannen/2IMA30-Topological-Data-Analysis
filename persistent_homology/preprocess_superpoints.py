import os
import numpy as np
import open3d as o3d

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_processed')

os.makedirs(OUTPUT_DIR, exist_ok=True)


def statistical_outlier_removal(pcd, nb_neighbors=50, std_ratio=0.3):
    cleaned, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                std_ratio=std_ratio)
    return cleaned


def radius_outlier_removal(pcd, nb_points=50, radius=0.02):
    cleaned, _ = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return cleaned


def voxel_downsample(pcd, voxel_size=0.02):
    return pcd.voxel_down_sample(voxel_size)


def estimate_normals(pcd, k=30):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    pcd.normalize_normals()
    return pcd


def superpoint_selection_with_normals(pts, normals, r_super=0.05):
    """Select superpoints and aggregate normals."""
    used = np.zeros(len(pts), dtype=bool)
    super_pts = []
    super_norms = []
    while not np.all(used):
        remaining = np.where(~used)[0]
        pts_remain = pts[remaining]
        normals_remain = normals[remaining]
        rand_idx = np.random.choice(len(pts_remain))
        center = pts_remain[rand_idx]
        mask = np.sum(np.abs(pts_remain - center), axis=1) < r_super
        cluster_pts = pts_remain[mask]
        cluster_normals = normals_remain[mask]
        super_pt = cluster_pts.mean(axis=0)
        mean_normal = cluster_normals.mean(axis=0)
        norm = np.linalg.norm(mean_normal)
        if norm > 0:
            mean_normal = mean_normal / norm
        super_pts.append(super_pt)
        super_norms.append(mean_normal)
        used[remaining[mask]] = True
    return np.array(super_pts), np.array(super_norms)


def preprocess_file(input_path, output_path,
                     voxel_size=0.02, r_super=0.05,
                     nb_neighbors=50, std_ratio=0.3,
                     nb_points=50, radius=0.02, knn=30):
    pcd = o3d.io.read_point_cloud(input_path)
    pcd = statistical_outlier_removal(pcd, nb_neighbors, std_ratio)
    pcd = radius_outlier_removal(pcd, nb_points, radius)
    pcd = voxel_downsample(pcd, voxel_size)
    pcd = estimate_normals(pcd, knn)
    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    super_pts, super_norms = superpoint_selection_with_normals(pts, normals, r_super)
    pcd_super = o3d.geometry.PointCloud()
    pcd_super.points = o3d.utility.Vector3dVector(super_pts)
    pcd_super.normals = o3d.utility.Vector3dVector(super_norms)
    o3d.io.write_point_cloud(output_path, pcd_super)
    print(f"Saved {output_path} with {len(super_pts)} superpoints")


def main():
    files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith('.pcd')]
    for fname in files:
        inp = os.path.join(DATASET_DIR, fname)
        outname = os.path.splitext(fname)[0] + '_superpoints.pcd'
        out = os.path.join(OUTPUT_DIR, outname)
        preprocess_file(inp, out)


if __name__ == '__main__':
    main()
