import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from typing import List, Tuple
from networkx.utils import UnionFind


def layered_constrained_mst(points: np.ndarray,
                            num_layers: int = 5,
                            figsize: tuple[int, int] = (8, 6),
                            cmap: str = "tab10") -> List[Tuple[int, int]]:
    """Build a height-stratified incremental MST and visualize it."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be of shape (n,3)")

    idx_sorted = np.argsort(points[:, 1])
    slices = np.array_split(idx_sorted, num_layers)

    uf = UnionFind(range(len(points)))
    processed: set[int] = set()
    edges_by_layer: List[List[Tuple[int, int]]] = []

    for layer_idx, layer in enumerate(slices):
        layer = list(map(int, layer))
        candidates: List[Tuple[float, int, int]] = []
        if layer_idx == 0:
            for u, v in combinations(layer, 2):
                w = float(np.linalg.norm(points[u] - points[v]))
                candidates.append((w, u, v))
        else:
            current = processed | set(layer)
            for u, v in combinations(layer, 2):
                w = float(np.linalg.norm(points[u] - points[v]))
                candidates.append((w, u, v))
            for u in layer:
                for v in processed:
                    w = float(np.linalg.norm(points[u] - points[v]))
                    candidates.append((w, u, v))
        candidates.sort()
        new_edges: List[Tuple[int, int]] = []
        for w, u, v in candidates:
            if uf[u] != uf[v]:
                uf.union(u, v)
                new_edges.append((u, v))
        edges_by_layer.append(new_edges)
        processed.update(layer)

    # Visualisation ----------------------------------------------------
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

    plt.show()

    all_edges = [e for layer in edges_by_layer for e in layer]
    return all_edges


if __name__ == "__main__":
    import sys
    import pathlib
    from utils import load_point_cloud

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python layered_constrained_mst.py <pointcloud.pcd> [num_layers]")

    pc_path = pathlib.Path(sys.argv[1])
    pc = load_point_cloud(pc_path)
    layers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    layered_constrained_mst(pc, num_layers=layers)
