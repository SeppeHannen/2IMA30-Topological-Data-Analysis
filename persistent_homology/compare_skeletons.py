import open3d as o3d
import numpy as np

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

def load_npz(path):
    z = np.load(path)
    pts = z["pts"]
    edges = z["edges"]
    if edges.shape[1] == 3:
        print(f"[INFO] Loaded edges with 3 columns from {path}, dropping layer column")
        edges = edges[:, :2]
    return pts, edges

def create_geometry(points, edges, color):
    # Defensive reshape and type check
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"[ERROR] edges should be shape (N, 2), got {edges.shape}")
    if edges.dtype not in [np.int32, np.int64]:
        print(f"[INFO] Converting edges from {edges.dtype} to int32")
        edges = edges.astype(np.int32)

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Create line set
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(edges)
    lines.colors = o3d.utility.Vector3dVector([color] * len(edges))

    return pcd, lines

def run_side_by_side_viewer(pts, edges_a, edges_b):
    # Create two geometries from same points, different edges
    pcd_a, lines_a = create_geometry(pts, edges_a, color=[1, 0, 0])  # red
    pcd_b, lines_b = create_geometry(pts, edges_b, color=[0, 0, 1])  # blue

    # Create visualizers
    vis_a = o3d.visualization.Visualizer()
    vis_b = o3d.visualization.Visualizer()
    vis_a.create_window("Plain MST", width=960, height=720, left=0, top=0)
    vis_b.create_window("Layered MST", width=960, height=720, left=960, top=0)

    for vis, pcd, lines in [(vis_a, pcd_a, lines_a), (vis_b, pcd_b, lines_b)]:
        vis.add_geometry(pcd)
        vis.add_geometry(lines)
        
    # Sync initial camera
    vis_a.poll_events()
    vis_a.update_renderer()
    cam_params = vis_a.get_view_control().convert_to_pinhole_camera_parameters()
    vis_b.get_view_control().convert_from_pinhole_camera_parameters(cam_params)

    try:
        while True:
            # Pull camera from A, apply to B
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
    # Replace with your actual paths
    path_a = "persistent_homology/skeletons/tree_0_layers_1.npz"
    path_b = "persistent_homology/skeletons/tree_0_layers_2.npz"

    pts, edges_a = load_npz(path_a)
    _, edges_b = load_npz(path_b)

    run_side_by_side_viewer(pts, edges_a, edges_b)
