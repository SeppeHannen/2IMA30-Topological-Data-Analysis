import argparse
import datetime
import os
from itertools import combinations
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from networkx.utils import UnionFind
from tqdm import tqdm


def create_log_file():
    """Create a timestamped log file inside ``logs``."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(log_dir, f"mst_{now}.txt")
    return open(path, "w")


def log(log_file, message: str):
    print(message)
    log_file.write(message + "\n")

def load_points(path: str, sample: int | None = None, voxel: float | None = None):
    """Load a point cloud and optionally downsample and sample points."""
    pcd = o3d.io.read_point_cloud(str(path))
    if voxel:
        pcd = pcd.voxel_down_sample(voxel)
    pts = np.asarray(pcd.points)
    if sample and pts.shape[0] > sample:
        idx = np.random.choice(len(pts), sample, replace=False)
        pts = pts[idx]
        pcd.points = o3d.utility.Vector3dVector(pts)
    return pts, pcd

def layered_constrained_mst(
    points: np.ndarray,
    num_layers: int = 5,
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "tab10",
    log_file=None,
    show: bool = True,
) -> List[Tuple[int, int]]:
    """Build a height-stratified incremental MST and optionally visualize it."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be of shape (n,3)")

    idx_sorted = np.argsort(points[:, 1])
    slices = np.array_split(idx_sorted, num_layers)
    if log_file:
        log(log_file, f"Building MST over {len(points)} points in {num_layers} layers")

    uf = UnionFind(range(len(points)))
    processed: set[int] = set()
    edges_by_layer: List[List[Tuple[int, int]]] = []

    for layer_idx, layer in enumerate(tqdm(slices, desc="Layers")):
        layer = list(map(int, layer))
        if log_file:
            log(log_file, f"Layer {layer_idx}: {len(layer)} points")
        candidates: List[Tuple[float, int, int]] = []
        if layer_idx == 0:
            total = len(layer) * (len(layer) - 1) // 2
            for u, v in tqdm(combinations(layer, 2), total=total, desc="intra", leave=False):
                w = float(np.linalg.norm(points[u] - points[v]))
                candidates.append((w, u, v))
        else:
            for u, v in tqdm(combinations(layer, 2), total=len(layer)*(len(layer)-1)//2, desc="intra", leave=False):
                w = float(np.linalg.norm(points[u] - points[v]))
                candidates.append((w, u, v))
            for u in tqdm(layer, desc="cross", leave=False):
                for v in processed:
                    w = float(np.linalg.norm(points[u] - points[v]))
                    candidates.append((w, u, v))
        candidates.sort()
        new_edges: List[Tuple[int, int]] = []
        for w, u, v in tqdm(candidates, desc="mst", leave=False):
            if uf[u] != uf[v]:
                uf.union(u, v)
                new_edges.append((u, v))
        edges_by_layer.append(new_edges)
        processed.update(layer)

    # Visualisation ----------------------------------------------------
    if show:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4, alpha=0.3,
                   c=points[:, 1], cmap=cmap)

        for l_idx, e_list in enumerate(edges_by_layer):
            color = plt.get_cmap(cmap)(l_idx / max(1, num_layers))
            for u, v in e_list:
                ax.plot([points[u, 0], points[v, 0]],
                        [points[u, 1], points[v, 1]],
                        [points[u, 2], points[v, 2]],
                        color=color, linewidth=1)
            if e_list:
                ax.plot([], [], [], color=color, label=f"Layer {l_idx}")

        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
        mid = points.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        fig.savefig("layered_mst.png", bbox_inches="tight")
        plt.close(fig)

    all_edges = [e for layer in edges_by_layer for e in layer]
    return all_edges


def main():
    parser = argparse.ArgumentParser(description="Build layered constrained MST")
    parser.add_argument("pcd", help="Path to point cloud (.pcd/.ply)")
    parser.add_argument("--layers", type=int, default=5, help="Number of layers")
    parser.add_argument("--sample", type=int, help="Randomly sample this many points")
    parser.add_argument("--voxel", type=float, help="Voxel size before sampling")
    parser.add_argument("--no-vis", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    log_file = create_log_file()
    log(log_file, f"Loading point cloud: {args.pcd}")
    points, _ = load_points(args.pcd, sample=args.sample, voxel=args.voxel)
    log(log_file, f"Loaded {points.shape[0]} points")

    log(log_file, "Computing MST...")
    layered_constrained_mst(points, num_layers=args.layers, log_file=log_file, show=not args.no_vis)
    log_file.close()


if __name__ == "__main__":
    main()
