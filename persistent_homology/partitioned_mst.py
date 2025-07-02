#!/usr/bin/env python3
"""
Partitioned (layered) MST skeletonisation, with
• optional k-NN pruning,
• optional random sampling,
• orientation arrow,
• and a plain MST mode when --layers 1.
"""

import argparse
from collections import defaultdict
from itertools import combinations
from typing import List, Tuple, Optional

import numpy as np
import open3d as o3d


# ───────────  Union–Find  ───────────
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        else:
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
        return True


# ───────────  I/O  ───────────
def load_points(path: str, *, sample: Optional[int] = None,
                voxel: Optional[float] = None):
    pcd = o3d.io.read_point_cloud(path)
    if voxel:
        pcd = pcd.voxel_down_sample(voxel)
    pts = np.asarray(pcd.points)

    if sample is not None and pts.shape[0] > sample:
        idx = np.random.choice(pts.shape[0], sample, replace=False)
        pts = pts[idx]
        pcd.points = o3d.utility.Vector3dVector(pts)
    return pts, pcd


# ───────────  Partition helper  ───────────
def partition_by_scalar(values: np.ndarray, k: int) -> List[np.ndarray]:
    vmin, vmax = float(values.min()), float(values.max())
    bins = np.linspace(vmin, vmax, k + 1)
    layers = defaultdict(list)
    for idx, v in enumerate(values):
        layer = np.searchsorted(bins, v, side="right") - 1
        layers[min(max(layer, 0), k - 1)].append(idx)
    return [np.asarray(layers[i], dtype=int) for i in range(k)]


# ───────────  Plain MST  ───────────
def build_plain_mst(pts: np.ndarray, *, knn: Optional[int] = None,
                    verbose: bool = False) -> List[Tuple[int, int, int]]:
    """Classic Kruskal on the whole point set."""
    n = pts.shape[0]
    uf = UnionFind(n)
    edges: List[Tuple[int, int, int]] = []

    if knn is None:
        # Exhaustive edge list
        cand = [(np.linalg.norm(pts[u] - pts[v]), u, v)
                for u, v in combinations(range(n), 2)]
    else:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        k = min(knn + 1, n)
        dists, nbrs = tree.query(pts, k=k)
        cand = []
        for u in range(n):
            for dist, j in zip(dists[u, 1:], nbrs[u, 1:]):
                cand.append((dist, u, j))

    cand.sort(key=lambda x: x[0])

    for w, u, v in cand:
        if uf.union(u, v):
            edges.append((u, v, 0))          # layer tag 0 for colouring
            if len(edges) == n - 1:
                break

    assert len(edges) == n - 1, "MST failed (wrong edge count)"
    if verbose:
        print(f"Plain MST: {len(edges)} edges (n-1={n-1})")
    return edges


# ───────────  Layered MST  ───────────
def build_partitioned_mst(pts: np.ndarray, layers: List[np.ndarray],
                          *, knn: Optional[int] = None,
                          verbose: bool = False):
    n = pts.shape[0]
    uf = UnionFind(n)
    edges: List[Tuple[int, int, int]] = []
    processed: set[int] = set()

    if knn is not None:
        from scipy.spatial import cKDTree

    for lid, idxs in enumerate(layers):
        if verbose:
            print(f"Layer {lid}: {len(idxs)} vertices")

        cand: List[Tuple[float, int, int]] = []

        # internal edges
        if len(idxs) > 1:
            if knn is None:
                for u, v in combinations(idxs, 2):
                    cand.append((np.linalg.norm(pts[u] - pts[v]), u, v))
            else:
                k = min(knn + 1, len(idxs))
                tree = cKDTree(pts[idxs])
                dists, nbrs = tree.query(pts[idxs], k=k)
                for row_i, u in enumerate(idxs):
                    for dist, j in zip(dists[row_i, 1:], nbrs[row_i, 1:]):
                        cand.append((dist, u, idxs[j]))

        # cross-layer edges
        if processed:
            prev = np.fromiter(processed, dtype=int)
            if knn is None:
                for u in idxs:
                    for v in prev:
                        cand.append((np.linalg.norm(pts[u] - pts[v]), u, v))
            else:
                tree_prev = cKDTree(pts[prev])
                k = min(knn, len(prev))
                dists, nbrs = tree_prev.query(pts[idxs], k=k)
                for row_i, u in enumerate(idxs):
                    for dist, j in zip(np.atleast_1d(dists[row_i]),
                                       np.atleast_1d(nbrs[row_i])):
                        cand.append((dist, u, prev[j]))

        if not cand and len(idxs) <= 1 and not processed:
            processed.update(idxs.tolist())
            continue

        cand.sort(key=lambda x: x[0])
        root_target = uf.find(next(iter(processed)) if processed else idxs[0])

        for w, u, v in cand:
            if uf.union(u, v):
                edges.append((u, v, lid))
            if all(uf.find(x) == root_target for x in idxs):
                break

        processed.update(idxs.tolist())

    assert len(edges) == n - 1, (
        f"Expected {n - 1} edges, got {len(edges)}")
    return edges


# ───────────  Visuals  ───────────
def create_arrow(dir_vec, center, length, color=(1, 0, 0)):
    d = dir_vec / np.linalg.norm(dir_vec)
    cyl_h, cone_h = 0.8 * length, 0.2 * length
    arrow = o3d.geometry.TriangleMesh.create_arrow(0.015 * length, 0.03 * length,
                                                   cyl_h, cone_h)
    arrow.paint_uniform_color(color)

    z = np.array([0, 0, 1])
    if np.allclose(d, z):
        R = np.eye(3)
    elif np.allclose(d, -z):
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.pi * np.array([0, 1, 0]))
    else:
        axis = np.cross(z, d)
        axis /= np.linalg.norm(axis)
        angle = float(np.arccos(np.clip(np.dot(z, d), -1, 1)))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(center)
    return arrow


def visualize(pts, pcd, edges, n_layers, dir_vec):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector([(u, v) for u, v, _ in edges])
    line_set.colors = o3d.utility.Vector3dVector(
        [cmap(l / max(1, n_layers - 1))[:3] for _, _, l in edges])

    diag = np.linalg.norm(pts.max(0) - pts.min(0))
    arrow = create_arrow(dir_vec, pts.mean(0), 0.25 * diag)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Partitioned MST")
    vis.add_geometry(pcd.paint_uniform_color([0.05, 0.05, 0.05]))
    vis.add_geometry(line_set)
    vis.add_geometry(arrow)

    def print_front(_):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        front = -cam.extrinsic[:3, 2]
        print("\n[P] camera front direction:", np.round(front, 6))
        return False

    vis.register_key_callback(ord("P"), print_front)
    vis.run()
    vis.destroy_window()


# ───────────  CLI  ───────────
def vec3(text: str) -> np.ndarray:
    try:
        v = np.fromstring(text, sep=",", dtype=float)
        if v.size != 3 or np.linalg.norm(v) == 0:
            raise ValueError
        return v
    except ValueError:
        raise argparse.ArgumentTypeError("Direction must be 'dx,dy,dz'")


def main():
    ap = argparse.ArgumentParser("Partitioned MST (layered) – with plain MST mode")
    ap.add_argument("pcd")
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--sample", type=int, help="Randomly sample N points")
    ap.add_argument("--voxel", type=float, help="Voxel size before sampling")
    ap.add_argument("--direction", type=vec3, default=np.array([0, -1, 0]),
                    help="Height/projection direction")
    ap.add_argument("--knn", type=int,
                    help="Prune candidate edges to k nearest neighbours "
                         "(omit for exact algorithm)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--export", type=str,
                  help="Save pts+edges to NPZ (for scripted comparisons)")
    args = ap.parse_args()

    dir_vec = args.direction / np.linalg.norm(args.direction)
    pts, pcd = load_points(args.pcd, sample=args.sample, voxel=args.voxel)

    if args.layers == 1:
        edges = build_plain_mst(pts, knn=args.knn, verbose=args.verbose)
        n_layers_for_color = 1          # colouring all edges the same
    else:
        scalar = pts @ dir_vec
        layers = partition_by_scalar(scalar, args.layers)
        edges = build_partitioned_mst(pts, layers,
                                      knn=args.knn, verbose=args.verbose)
        n_layers_for_color = args.layers

    visualize(pts, pcd, edges, n_layers_for_color, dir_vec)

    # ---------- export ----------
    if args.export:
        np.savez(args.export, pts=pts,
                 edges=np.asarray(edges, dtype=np.int32))
        if args.verbose:
            print(f"NPZ saved → {args.export}")


if __name__ == "__main__":
    main()
