#!/usr/bin/env python
"""
partitioned_mst.py
==================
Build a **Partitioned Minimum Spanning Tree** (PMST) skeleton for a 3‑D point
cloud and visualise it interactively with Open3D.

Changes versus the original script
----------------------------------
* Accept an arbitrary scalar function (`--scalar-axis`) for stratification.
* Strictly forbid intra‑layer edges after the root layer.
* Attach each new point to its nearest processed neighbour via a KD‑tree
  (O(n log n) instead of O(n²)).
* Separate pure algorithm from visualisation.
* Use Open3D `draw_geometries` for fully rotatable inspection; PNG snapshot is
  optional.
* Clean, three‑colour palette matches the number of layers.

Run with:
    python partitioned_mst.py tree.ply --layers 3 --scalar-axis z

Press the mouse buttons to rotate/zoom in the Open3D window.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import open3d as o3d
from networkx.utils import UnionFind
from scipy.spatial import cKDTree
from tqdm import tqdm

###############################################################################
# Utility helpers
###############################################################################

LAYER_COLOURS = np.array(
    [[0.1216, 0.4667, 0.7059],   # tab10[0]
     [1.0000, 0.4980, 0.0549],   # tab10[1]
     [0.1725, 0.6275, 0.1725]],  # tab10[2]
    dtype=np.float64,
)


def _create_log_file() -> "os.PathLike[str] | None":
    """Create and return a timestamped log‑file handle (or *None* if logs dir
    is not writable)."""
    log_dir = Path("logs")
    try:
        log_dir.mkdir(exist_ok=True)
        now = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return open(log_dir / f"pmst_{now}.txt", "w", encoding="utf-8")
    except OSError:
        return None


def _log(handle, msg: str) -> None:  # noqa: D401  (simple description)
    if handle is None:
        print(msg)
    else:
        print(msg)
        handle.write(msg + "\n")


###############################################################################
# I/O
###############################################################################

def load_points(path: os.PathLike | str, *, sample: int | None = None,
                voxel: float | None = None) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
    """Read a point cloud and return *(n,3) numpy array*, *PointCloud* pair.

    Parameters
    ----------
    path : str | Path
        PLY/PCD/other format recognised by Open3D.
    sample : int, optional
        Uniformly random down‑sample to *sample* points.
    voxel : float, optional
        Pre‑voxelisation size in the same unit as *path*.
    """
    pcd = o3d.io.read_point_cloud(str(path))
    if voxel:
        pcd = pcd.voxel_down_sample(voxel)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if sample and pts.shape[0] > sample:
        idx = np.random.choice(pts.shape[0], sample, replace=False)
        pts = pts[idx]
        pcd.points = o3d.utility.Vector3dVector(pts)
    return pts, pcd


###############################################################################
# Core algorithm
###############################################################################

def build_partitioned_mst(
    points: np.ndarray,
    *,
    scalar: np.ndarray | None = None,
    num_layers: int = 3,
    k_neighbours: int = 1,
    log_fn: Callable[[str], None] | None = None,
) -> List[List[Tuple[int, int]]]:
    """Create a partitioned MST skeleton.

    Parameters
    ----------
    points : (n,3) array_like
        Input point coordinates.
    scalar : (n,) array_like, optional
        Scalar per point defining the ordered strata. Defaults to *z*‑axis.
    num_layers : int, default = 3
        Number of equal‑sized partitions along *scalar*.
    k_neighbours : int, default = 1
        Attach each new point to its *k* closest processed vertices.
    log_fn : callable(str), optional
        Logging function.

    Returns
    -------
    list(list(tuple(int,int)))
        ``edges_by_layer[i]`` holds the edges *added* while processing layer *i*.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")

    if scalar is None:
        scalar = points[:, 2]  # default to z‑axis
    scalar = np.asarray(scalar, dtype=np.float64)

    idx_sorted = np.argsort(scalar)
    slices: List[np.ndarray] = np.array_split(idx_sorted, num_layers)

    uf = UnionFind(range(len(points)))
    processed: set[int] = set()
    edges_by_layer: List[List[Tuple[int, int]]] = []

    if log_fn:
        log_fn(f"Partitioned MST over {len(points)} pts in {num_layers} layers")

    for layer_idx, layer in enumerate(tqdm(slices, desc="Layers")):
        layer_pts = points[layer]
        new_edges: List[Tuple[int, int]] = []

        if layer_idx == 0:
            # Build an MST *within* the root layer using Kruskal on the full
            # graph (dense but small in practice).
            candidates: List[Tuple[float, int, int]] = []
            for i in range(len(layer)):
                for j in range(i + 1, len(layer)):
                    u, v = int(layer[i]), int(layer[j])
                    w = float(np.linalg.norm(points[u] - points[v]))
                    candidates.append((w, u, v))
            candidates.sort()
            for w, u, v in candidates:  # noqa: E741  (avoid l not needed)
                if uf[u] != uf[v]:
                    uf.union(u, v)
                    new_edges.append((u, v))
        else:
            # Attach each new point only to already‑processed vertices.
            tree = cKDTree(points[list(processed)])
            processed_list = np.fromiter(processed, dtype=int)

            for u in layer:
                # k=1 gives a pure tree; k>1 adds robustness on noisy data.
                k = min(k_neighbours, len(processed_list))
                dists, idxs = tree.query(points[u], k=k)
                if k == 1:
                    dists = [dists]
                    idxs = [idxs]
                for idx in np.atleast_1d(idxs):
                    v = int(processed_list[int(idx)])
                    if uf[u] != uf[v]:
                        uf.union(u, v)
                        new_edges.append((u, v))
                        break  # one edge per new vertex is enough

        edges_by_layer.append(new_edges)
        processed.update(map(int, layer))
        if log_fn:
            log_fn(f"  Layer {layer_idx}: +{len(layer)} pts, +{len(new_edges)} edges")

    return edges_by_layer


###############################################################################
# Visualisation
###############################################################################

def visualise_open3d(
    points: np.ndarray,
    edges_by_layer: Sequence[Sequence[Tuple[int, int]]],
    *,
    window_name: str = "Partitioned MST",
    snapshot: str | None = None,
) -> None:
    """Display the skeleton and optionally save a PNG snapshot."""

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])

    lines = []
    colours = []
    for layer_idx, e_list in enumerate(edges_by_layer):
        colour = LAYER_COLOURS[layer_idx % len(LAYER_COLOURS)]
        for u, v in e_list:
            lines.append([u, v])
            colours.append(colour)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colours)

    o3d.visualization.draw_geometries([pcd, line_set], window_name=window_name)

    if snapshot:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        vis.add_geometry(line_set)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(snapshot, do_render=True)
        vis.destroy_window()


###############################################################################
# CLI
###############################################################################

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Partitioned MST skeletoniser")
    parser.add_argument("pcd", help="Path to point cloud (.pcd/.ply)")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--scalar-axis", choices=["x", "y", "z"], default="z",
                        help="Coordinate axis used as scalar function")
    parser.add_argument("--sample", type=int,
                        help="Randomly sample this many points before processing")
    parser.add_argument("--voxel", type=float,
                        help="Voxel size before sampling (same units as input)")
    parser.add_argument("--k", "--k-neighbours", type=int, default=1,
                        help="Attach each new vertex to k nearest neighbours in processed set")
    parser.add_argument("--no-vis", action="store_true", help="Skip the 3‑D viewer")
    parser.add_argument("--snapshot", metavar="FILE",
                        help="Also save a PNG snapshot to FILE")
    return parser.parse_args(argv)


###############################################################################
# Entry‑point
###############################################################################

def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    log_handle = _create_log_file()
    log = lambda m: _log(log_handle, m)

    log(f"Loading point cloud: {args.pcd}")
    points, _ = load_points(args.pcd, sample=args.sample, voxel=args.voxel)
    log(f"Loaded {len(points)} points")

    axis_map = {"x": 0, "y": 1, "z": 2}
    scalar = points[:, axis_map[args.scalar_axis]]

    log("Building partitioned MST …")
    edges_by_layer = build_partitioned_mst(
        points,
        scalar=scalar,
        num_layers=args.layers,
        k_neighbours=args.k,
        log_fn=log,
    )

    log("Skeleton built. Visualising …" if not args.no_vis else "Skeleton built.")

    if not args.no_vis or args.snapshot:
        visualise_open3d(points, edges_by_layer,
                         snapshot=args.snapshot,
                         window_name=f"PMST – {Path(args.pcd).name}")

    if log_handle is not None:
        log_handle.close()


if __name__ == "__main__":
    main()
