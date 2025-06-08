import argparse
import numpy as np
import open3d as o3d
from ripser import ripser
import gudhi
import datetime
import os

def create_log_file():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f"log_{now}.txt")
    return open(log_path, "w")

def log(log_file, message):
    print(message)
    log_file.write(message + "\n")

def load_points(file_path, sample=None, voxel=None):
    pcd = o3d.io.read_point_cloud(file_path)
    if voxel:
        pcd = pcd.voxel_down_sample(voxel)
    pts = np.asarray(pcd.points)
    if sample and pts.shape[0] > sample:
        idx = np.random.choice(pts.shape[0], sample, replace=False)
        pts = pts[idx]
        pcd.points = o3d.utility.Vector3dVector(pts)
    return pts, pcd

def compute_connected_component_edges(points, max_dim=1, max_edge=None, threshold=None, log_file=None):
    if max_edge is None:
        max_edge = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))

    rips = gudhi.RipsComplex(points=points, max_edge_length=max_edge)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    st.persistence()
    dim0_pairs, _, _, _ = st.flag_persistence_generators()

    edges = []
    kept = 0
    skipped = 0
    for v_birth, v1_death, v2_death in dim0_pairs:
        edge = [min(v1_death, v2_death), max(v1_death, v2_death)]
        death = st.filtration(edge)
        if threshold is None or death >= threshold:
            edges.append((v1_death, v2_death))
            kept += 1
            if log_file:
                log(log_file, f"KEPT   Edge ({v1_death}, {v2_death}) has length {death:.8f}")
        else:
            skipped += 1
            if log_file:
                log(log_file, f"SKIPPED Edge ({v1_death}, {v2_death}) has length {death:.8f}")
    if log_file:
        log(log_file, f"---\nKept {kept} edges, skipped {skipped} below threshold {threshold}")
    return edges

def compute_persistent_cycles(points, threshold, maxdim=1):
    res = ripser(points, maxdim=maxdim, do_cocycles=True)
    cycles = []
    for (birth, death), cocycle in zip(res['dgms'][1], res['cocycles'][1]):
        pers = death - birth if np.isfinite(death) else np.inf
        if threshold is not None and pers < threshold:
            continue
        edges = []
        for i, j, coeff in cocycle:
            if int(round(coeff)) % 2 == 1:
                edges.append((int(i), int(j)))
        cycles.append(edges)
    return cycles

def build_line_set(points, edges, color):
    if not edges:
        return None
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in edges])
    return line_set

def main():
    parser = argparse.ArgumentParser(description="Visualize persistent features on a point cloud")
    parser.add_argument("pcd", help="Path to point cloud file (.pcd/.ply)")
    parser.add_argument("--sample", type=int, default=2000, help="Random sample of points for PH computation")
    parser.add_argument("--voxel", type=float, help="Voxel size for downsampling before sampling")
    parser.add_argument("--cycle_thresh", type=float, default=None, help="Only show cycles with persistence >= thresh")
    parser.add_argument("--component_thresh", type=float, default=None, help="Only show component merges occurring at scale >= thresh")
    args = parser.parse_args()

    log_file = create_log_file()

    log(log_file, f"Loading point cloud: {args.pcd}")
    points, pcd = load_points(args.pcd, sample=args.sample, voxel=args.voxel)
    log(log_file, f"Point cloud loaded with {points.shape[0]} points.")

    log(log_file, f"\nComputing connected component edges (threshold = {args.component_thresh})...")
    cc_edges = compute_connected_component_edges(points, threshold=args.component_thresh, log_file=log_file)

    # log(log_file, f"\nComputing persistent 1D cycles (threshold = {args.cycle_thresh})...")
    # cycles = compute_persistent_cycles(points, threshold=args.cycle_thresh)

    log(log_file, f"\nBuilding geometry for visualization...")
    geometries = [pcd]
    ls = build_line_set(points, cc_edges, [1.0, 0.0, 0.0])
    if ls:
        geometries.append(ls)

    # for cyc in cycles:
    #     cyc_ls = build_line_set(points, cyc, [0.0, 0.0, 1.0])
    #     if cyc_ls:
    #         geometries.append(cyc_ls)

    log(log_file, "Launching Open3D visualizer...")
    o3d.visualization.draw_geometries(geometries, window_name="Persistent features")
    log_file.close()

if __name__ == "__main__":
    main()
