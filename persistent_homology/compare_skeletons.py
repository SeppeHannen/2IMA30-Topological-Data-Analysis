import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

def load_npz(path):
    z = np.load(path)
    return np.array(z["pts"], copy=True), np.array(z["edges"], copy=True)

def create_geometry(points, edges, color=None, n_layers=None):
    points = np.array(points, copy=True)
    edges = np.asarray(edges)
    
    if edges.ndim != 2 or edges.shape[1] not in [2, 3]:
        raise ValueError(f"[ERROR] edges must be (N, 2) or (N, 3), got {edges.shape}")
    if edges.dtype not in [np.int32, np.int64]:
        print(f"[INFO] Converting edges from {edges.dtype} to int32")
        edges = edges.astype(np.int32)

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Create line set
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(edges[:, :2])

    # Coloring
    if edges.shape[1] == 3 and n_layers is not None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(layer / max(n_layers - 1, 1))[:3] for layer in edges[:, 2]]
    else:
        colors = [color or [0, 0, 0]] * len(edges)

    lines.colors = o3d.utility.Vector3dVector(colors)
    return pcd, lines

def run_side_by_side_viewer(pts_a, edges_a, pts_b, edges_b, n_layers_b=None):
    pcd_a, lines_a = create_geometry(pts_a, edges_a, color=[1, 0, 0])
    pcd_b, lines_b = create_geometry(pts_b, edges_b, n_layers=n_layers_b)

    vis_a = o3d.visualization.Visualizer()
    vis_b = o3d.visualization.Visualizer()
    vis_a.create_window("Plain MST", width=960, height=720, left=0, top=0)
    vis_b.create_window("Layered MST", width=960, height=720, left=960, top=0)

    for vis, pcd, lines in [(vis_a, pcd_a, lines_a), (vis_b, pcd_b, lines_b)]:
        vis.add_geometry(pcd)
        vis.add_geometry(lines)

    # Sync camera from A to B
    vis_a.poll_events()
    vis_a.update_renderer()
    cam_params = vis_a.get_view_control().convert_to_pinhole_camera_parameters()
    vis_b.get_view_control().convert_from_pinhole_camera_parameters(cam_params)

    try:
        while True:
            cam_params = vis_a.get_view_control().convert_to_pinhole_camera_parameters()
            vis_b.get_view_control().convert_from_pinhole_camera_parameters(cam_params)
            vis_a.poll_events()
            vis_a.update_renderer()
            vis_b.poll_events()
            vis_b.update_renderer()
    except KeyboardInterrupt:
        vis_a.destroy_window()
        vis_b.destroy_window()


if __name__ == "__main__":
    # Replace these with your actual paths
    path_a = "persistent_homology/skeletons/tree_0_layers_1.npz"
    path_b = "persistent_homology/skeletons/tree_0_layers_2.npz"

    pts_a, edges_a = load_npz(path_a)
    pts_b, edges_b = load_npz(path_b)

    run_side_by_side_viewer(pts_a, edges_a, pts_b, edges_b, n_layers_b=2)
