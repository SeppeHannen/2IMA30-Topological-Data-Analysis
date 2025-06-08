import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
PROC_DIR = os.path.join(os.path.dirname(__file__), 'dataset_processed')


def load_raw(index):
    path = os.path.join(DATA_DIR, f'tree_{index}.pcd')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Raw point cloud {path} not found')
    return o3d.io.read_point_cloud(path)


def load_superpoints(index):
    path = os.path.join(PROC_DIR, f'tree_{index}_superpoints.pcd')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Superpoint file {path} not found')
    return o3d.io.read_point_cloud(path)


def color_by_height(pcd):
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    z = pts[:, 2]
    h_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
    pcd.colors = o3d.utility.Vector3dVector(plt.cm.viridis(h_norm)[:, :3])
    return pcd


def visualize(index, show_raw=False, radius=0.05):
    pcd_super = load_superpoints(index)
    super_points = np.asarray(pcd_super.points)

    super_geoms = []
    for pt in super_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(pt)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        super_geoms.append(sphere)

    geoms = super_geoms

    if show_raw:
        pcd_raw = load_raw(index)
        pcd_raw = color_by_height(pcd_raw)
        geoms = [pcd_raw] + geoms

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f'Tree {index} Superpoints',
        width=960,
        height=720,
    )


def main():
    parser = argparse.ArgumentParser(description='Visualize super points for a tree')
    parser.add_argument('index', type=int, help='tree index to load')
    parser.add_argument('--show-raw', action='store_true', help='display original point cloud as well')
    parser.add_argument('--radius', type=float, default=0.05, help='radius for the superpoint spheres')
    args = parser.parse_args()
    visualize(args.index, show_raw=args.show_raw, radius=args.radius)


if __name__ == '__main__':
    main()

