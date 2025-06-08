"""New visualizer: visualize 0-D persistent homology of a point cloud.

This script is a standalone alternative to visualize_persistence.py that uses only numpy, scipy and matplotlib.

This script reads a 3-D point cloud (``.txt`` or ``.npy``), computes the
Vietoris--Rips 0-D persistence using a size-based elder rule and visualises
both the barcode and the robust connected components.  A "cut" radius can be
selected manually via ``--min-persistence`` or automatically using the largest
non-trivial gap in merge radii (``--auto-cut``).

The surviving components at the chosen scale are shown as a coloured scatter
plot with a legend indicating the persistence of each component.

Dependencies: ``numpy``, ``scipy`` and ``matplotlib`` only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import distance


# ---------------------------------------------------------------------------
# Basic union--find with size-based elder rule
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)
        self.size = np.ones(n, dtype=int)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> int:
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return x
        if self.size[x] < self.size[y] or (self.size[x] == self.size[y] and x > y):
            x, y = y, x
        self.parent[y] = x
        self.size[x] += self.size[y]
        return x


@dataclass
class Edge:
    length: float
    u: int
    v: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_points(path: str) -> np.ndarray:
    """Load Nx3 points from ``path`` (.txt or .npy)."""
    if path.lower().endswith(".npy"):
        pts = np.load(path)
    else:
        pts = np.loadtxt(path)
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("point cloud must be of shape (n,3)")
    return pts


def pairwise_edges(points: np.ndarray) -> List[Edge]:
    """Return all edges sorted by Euclidean length."""
    dist = distance.squareform(distance.pdist(points))
    i, j = np.triu_indices(len(points), k=1)
    d = dist[i, j]
    order = np.argsort(d)
    edges = [Edge(float(d[k]), int(i[k]), int(j[k])) for k in order]
    return edges


def zero_dim_persistence(points: np.ndarray) -> Tuple[np.ndarray, List[Edge]]:
    """Compute 0-D persistence intervals and return them with sorted edges."""
    n = len(points)
    edges = pairwise_edges(points)
    uf = UnionFind(n)
    death = np.full(n, np.inf)

    for e in edges:
        ru, rv = uf.find(e.u), uf.find(e.v)
        if ru == rv:
            continue
        if uf.size[ru] > uf.size[rv] or (uf.size[ru] == uf.size[rv] and ru < rv):
            winner, loser = ru, rv
        else:
            winner, loser = rv, ru
        uf.parent[loser] = winner
        uf.size[winner] += uf.size[loser]
        death[loser] = e.length

    intervals = np.column_stack((np.zeros(n), death))
    return intervals, edges


def components_at_threshold(n: int, edges: Iterable[Edge], tau: float) -> np.ndarray:
    """Label points by the components surviving up to ``tau``."""
    uf = UnionFind(n)
    for e in edges:
        if e.length > tau:
            break
        uf.union(e.u, e.v)
    labels = np.array([uf.find(i) for i in range(n)])
    _, inv = np.unique(labels, return_inverse=True)
    return inv


def plot_barcode(intervals: np.ndarray, cut: float | None, path: str | None, show: bool) -> None:
    finite = intervals[np.isfinite(intervals[:, 1])]
    order = np.argsort(finite[:, 1])
    fig, ax = plt.subplots(figsize=(6, 3))
    for idx, bar_idx in enumerate(order):
        b, d = finite[bar_idx]
        ax.hlines(idx, b, d, color="tab:blue")
    if cut is not None:
        ax.axvline(cut, color="red", linestyle="--")
    ax.set_xlabel("radius")
    ax.set_ylabel("component")
    ax.set_ylim(-1, len(order) + 1)
    ax.invert_yaxis()
    if path:
        fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_clusters(points: np.ndarray, labels: np.ndarray, intervals: np.ndarray, tau: float,
                  path: str | None, show: bool) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    uniq = np.unique(labels)
    handles: List[Line2D] = []
    for idx, u in enumerate(uniq):
        mask = labels == u
        color = cmap(idx % 20)
        ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], color=color, s=10)
        pers = intervals[u, 1]
        label = f"id {idx}, pers = {pers:.3f}" if np.isfinite(pers) else f"id {idx}, pers = ∞"
        handles.append(Line2D([0], [0], marker="o", linestyle="", color=color, label=label))
    ax.legend(handles=handles, loc="upper right")
    ax.set_title(f"Cut radius τ = {tau:.3f}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if path:
        fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize persistent connected components (0-D)")
    parser.add_argument("cloud", help="Path to point cloud (.txt or .npy)")
    parser.add_argument("--min-persistence", type=float, help="Manual persistence threshold")
    parser.add_argument("--auto-cut", action="store_true", help="Use largest gap heuristic for the cut radius")
    parser.add_argument("--top-gaps", type=int, default=0, metavar="k", help="List the k largest gaps")
    parser.add_argument("--save-barcode", help="Path to save the barcode figure")
    parser.add_argument("--save-3d", help="Path to save the 3-D cluster figure")
    parser.add_argument("--show", action="store_true", help="Display the figures interactively")
    args = parser.parse_args()

    pts = load_points(args.cloud)
    intervals, edges = zero_dim_persistence(pts)

    deaths = np.sort(intervals[np.isfinite(intervals[:, 1]), 1])
    gaps = np.diff(deaths) if len(deaths) > 0 else np.array([])

    if args.top_gaps:
        order = np.argsort(gaps)[::-1]
        print("Largest gaps (gap -> suggested cut):")
        for rank, idx in enumerate(order[: args.top_gaps], 1):
            mid = 0.5 * (deaths[idx] + deaths[idx + 1])
            print(f"{rank}. {gaps[idx]:.6f} -> {mid:.6f}")

    auto = args.auto_cut or (args.min_persistence is None)
    tau = args.min_persistence
    if auto and len(gaps) > 0:
        idx = np.argmax(gaps[:-1]) if len(gaps) > 1 else 0
        tau = 0.5 * (deaths[idx] + deaths[idx + 1])
    elif auto and len(deaths) > 0 and tau is None:
        tau = deaths[-1]

    plot_barcode(intervals, tau if auto else args.min_persistence, args.save_barcode, args.show)

    if tau is not None:
        labels = components_at_threshold(len(pts), edges, tau)
        pers_plot_needed = args.save_3d or args.show
        if pers_plot_needed:
            plot_clusters(pts, labels, intervals, tau, args.save_3d, args.show)


if __name__ == "__main__":  # pragma: no cover
    main()
