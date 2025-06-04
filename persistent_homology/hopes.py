# -----------------------------------------------------------------------------
# HoPeS main class
# -----------------------------------------------------------------------------

# -*- coding: utf-8 -*-
"""hopes.py – Reference implementation of HoPeS(d)

This module implements the Higher‑Dimensional **H**omologically **P**ersistent **S**keleton
(HoPeS) exactly as described in *Kališnik‑Verovšek, Kurlin & Lešnik –
"A Higher‑Dimensional Homologically Persistent Skeleton" (2017).*  

Dependencies
------------
- numpy           ≥ 1.20
- gudhi           ≥ 3.8.0   (pip install gudhi)
- scipy           ≥ 1.9      (only for pairwise distances if you disable sklearn)
- scikit‑learn    ≥ 1.3      (only used for fast pairwise distances)
- tqdm            ≥ 4.64     (for progress bars)

Key public classes
------------------
- **WeightedComplex** – holds a simplex tree with filtration values (=weights).
- **MinimalSpanningDTree** – constructs the minimal spanning *d*-tree (MST(d)).
- **HoPeS** – orchestrates: builds filtration, MST(d), critical faces with birth &
  death, and exposes skeletons at arbitrary scales.

This implementation purposely limits itself to Vietoris–Rips filtrations – the
paper is agnostic about filtration as long as it is monotone.  Extend
`_build_filtration()` to plug Čech or Alpha complexes.

Author : OpenAI – ChatGPT‑4o reference implementation (May 2025)
License: MIT
"""
from __future__ import annotations

import itertools
import math
import warnings
import matplotlib.pyplot as plt
import open3d as o3d
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

import numpy as np
from sklearn.metrics import pairwise_distances
import gudhi

import time

# -----------------------------------------------------------------------------
# Helper dataclasses
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Simplex:
    """A (vertex‑set, filtration‑value) pair."""

    vertices: Tuple[int, ...]
    weight: float

    def dim(self) -> int:
        return len(self.vertices) - 1


@dataclass
class CriticalFace:
    simplex: Simplex  # the d‑simplex itself
    birth: float
    death: float | float("inf")

    @property
    def alive_interval(self) -> Tuple[float, float | float("inf")]:
        return self.birth, self.death


# -----------------------------------------------------------------------------
# WeightedComplex – wraps a Gudhi simplex tree but keeps an explicit registry
# -----------------------------------------------------------------------------

class WeightedComplex:
    """A finite weighted simplex (Definition 2.9 in the paper).

    Internally uses a Gudhi SimplexTree so we inherit efficient filtration &
    persistence calculations for free.
    """

    def __init__(self, simplices: Iterable[Simplex]):
        self.tree = gudhi.SimplexTree()
        for s in simplices:
            self.tree.insert(s.vertices, filtration=s.weight)
        self.tree.initialize_filtration()

    # ------------------------------------------------------------------
    # Exposed helpers
    # ------------------------------------------------------------------
    def vertices(self) -> List[int]:
        return [v for v, _ in self.tree.get_skeleton(0)]

    def max_weight(self) -> float:
        return max(w for _, w in self.tree.get_filtration())

    def all_simplices(self, dim: int | None = None) -> List[Simplex]:
        if dim is None:
            return [Simplex(tuple(sig[0]), sig[1]) for sig in self.tree.get_filtration()]
        return [
            Simplex(tuple(sig[0]), sig[1])
            for sig in self.tree.get_skeleton(dim)
            if len(sig[0]) - 1 == dim
        ]

    # --------------------------------------------------------------
    # Conversion helpers used by HoPeS
    # --------------------------------------------------------------
    def restricted_tree(self, alpha: float) -> "gudhi.SimplexTree":
        """Return a Gudhi SimplexTree containing only simplices w ≤ α."""
        st = gudhi.SimplexTree()
        for verts, weight in self.tree.get_filtration():
            if weight <= alpha:
                st.insert(verts, filtration=weight)
        st.initialize_filtration()
        return st


# -----------------------------------------------------------------------------
# Minimal Spanning d‑Tree (Algorithm 3.2)
# -----------------------------------------------------------------------------

class MinimalSpanningDTree:
    """Construct and hold MST(d) according to Algorithm 3.2."""

    def __init__(self, wcplx: WeightedComplex, d: int):
        if d < 0:
            raise ValueError("d must be ≥ 0")
        self.wcplx = wcplx
        self.d = d
        self._tree_simplices: Dict[Tuple[int, ...], float] = {}
        self._critical_faces: List[Simplex] = []
        self._build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def simplices(self) -> List[Simplex]:
        return [Simplex(v, w) for v, w in self._tree_simplices.items()]

    def critical_faces(self) -> List[Simplex]:
        return list(self._critical_faces)

    def tree_as_simplex_tree(self) -> gudhi.SimplexTree:
        st = gudhi.SimplexTree()
        for verts, weight in self._tree_simplices.items():
            st.insert(verts, filtration=weight)
        st.initialize_filtration()
        return st

    # ------------------------------------------------------------------
    # Internal implementation – Algorithm 3.2 literally
    # ------------------------------------------------------------------
    def _build(self):
        # Pre‑group simplices by weight
        by_weight: Dict[float, List[Simplex]] = defaultdict(list)
        for s in self.wcplx.all_simplices():
            by_weight[s.weight].append(s)
        weight_levels = sorted(by_weight)

        # Start with empty tree
        current_tree = gudhi.SimplexTree()

        # Always include the full (d‑1) skeleton up to current α.
        def add_skeleton(level: float):
            for s in by_weight[level]:
                if s.dim() <= self.d - 1:
                    current_tree.insert(s.vertices, filtration=s.weight)

        # Helpers to check d‑acyclic condition quickly via homology dim.
        def is_d_acyclic(tmp_tree: gudhi.SimplexTree) -> bool:
            """
            True  → no H_d classes present.
            False → at least one H_d class has just been born.
            """
            tmp_tree.initialize_filtration()
            tmp_tree.persistence(homology_coeff_field=2, persistence_dim_max=self.d)
            return len(tmp_tree.persistence_intervals_in_dimension(self.d)) == 0

        for w in weight_levels:
            # 1. Add (d-1) faces of this level unconditionally.
            add_skeleton(w)

            # 2. Consider d‑faces of this weight in **deterministic order**.
            d_faces = [s for s in by_weight[w] if s.dim() == self.d]
            d_faces.sort(key=lambda s: s.vertices)  # elder rule tie‑breaker

            for simplex in d_faces:
                # Try adding and check if still d‑acyclic
                current_tree_copy = current_tree.copy()
                current_tree_copy.insert(simplex.vertices, filtration=simplex.weight)
                if is_d_acyclic(current_tree_copy):
                    # safe to add – part of MST(d)
                    current_tree = current_tree_copy
                    self._tree_simplices[simplex.vertices] = simplex.weight
                else:
                    # becomes critical face
                    self._critical_faces.append(simplex)

        # store also all lower‑dim faces (needed for spanning)
        for s in self.wcplx.all_simplices():
            if s.dim() < self.d and s.vertices not in self._tree_simplices:
                self._tree_simplices[s.vertices] = s.weight





class HoPeS:
    """High‑level driver that exposes skeletons at any scale α."""

    def __init__(
        self,
        points: np.ndarray,
        d: int,
        max_edge_length: float | None = None,
        filtration: str = "rips",
        rips_max_dim: int | None = None,
        sparse: bool = False,
        k_neighbors: int = 20,
    ):
        print("[HoPeS] Initializing with {} points in dim {}...".format(len(points), points.shape[1]))
        t0 = time.time()

        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2:
            raise ValueError("`points` must be an (N, m) array")
        self.d = int(d)
        if self.d < 0:
            raise ValueError("d must be ≥ 0")
        self.filtration = filtration
        self.rips_max_dim = rips_max_dim or (d + 1)
        self.sparse = sparse
        self.k_neighbors = k_neighbors
        self.max_edge_length = (
            float(max_edge_length)
            if max_edge_length is not None
            else float(np.max(pairwise_distances(self.points)))
        )

        print("[HoPeS] Building filtration...")
        t1 = time.time()
        self._weighted_complex: WeightedComplex = self._build_filtration()
        print(f"[HoPeS] Filtration built in {time.time() - t1:.2f}s")

        print("[HoPeS] Constructing minimal spanning d-tree...")
        t2 = time.time()
        self._mst_d = MinimalSpanningDTree(self._weighted_complex, self.d)
        print(f"[HoPeS] MST(d) built in {time.time() - t2:.2f}s")

        print("[HoPeS] Assigning birth and death to critical faces...")
        t3 = time.time()
        self._critical_faces: List[CriticalFace] = self._assign_birth_death()
        print(f"[HoPeS] Birth/death assignment completed in {time.time() - t3:.2f}s")
        print(f"[HoPeS] Total initialization completed in {time.time() - t0:.2f}s")

    def _build_filtration(self) -> WeightedComplex:
        if self.filtration != "rips":
            raise NotImplementedError("Only `rips` filtration is supported for now")

        print("[Filtration] Computing pairwise distances...")
        t0 = time.time()
        dists = pairwise_distances(self.points)
        print(f"[Filtration] Pairwise distances computed in {time.time() - t0:.2f}s")

        n = dists.shape[0]
        max_dim = self.rips_max_dim
        simplices: List[Simplex] = []

        for v in range(n):
            simplices.append(Simplex((v,), 0.0))

        print("[Filtration] Building Rips complex...")
        t1 = time.time()
        rips = gudhi.RipsComplex(distance_matrix=dists, max_edge_length=self.max_edge_length)
        st = rips.create_simplex_tree(max_dimension=max_dim)

        allowed_edges = set()
        if self.sparse:
            print("[Filtration] Computing k-nearest neighbors for sparsification...")
            nbrs = NearestNeighbors(n_neighbors=self.k_neighbors).fit(self.points)
            _, indices = nbrs.kneighbors(self.points)
            for i in range(n):
                for j in indices[i]:
                    if i != j:
                        allowed_edges.add((min(i, j), max(i, j)))

        for verts, filt in st.get_filtration():
            if len(verts) == 1:
                continue
            if self.sparse:
                if any((min(i, j), max(i, j)) not in allowed_edges for i, j in itertools.combinations(verts, 2)):
                    continue
            simplices.append(Simplex(tuple(verts), float(filt)))

        print(f"[Filtration] Rips complex built in {time.time() - t1:.2f}s")
        return WeightedComplex(simplices)

    def _assign_birth_death(self) -> List[CriticalFace]:
        print("[Persistence] Running persistent homology for dimension {}...".format(self.d))
        t0 = time.time()
        st = self._weighted_complex.tree
        st.persistence(homology_coeff_field=2, persistence_dim_max=self.d)

        bd_map: Dict[Tuple[int, ...], Tuple[float, float | float("inf")]] = {}
        for dim, pair in st.persistence_pairs():
            if dim != self.d:
                continue
            creator, destroyer = pair
            birth = st.filtration(creator)
            death = st.filtration(destroyer) if destroyer else math.inf
            bd_map[tuple(creator)] = (birth, death)

        critical_faces: List[CriticalFace] = []
        for simplex in self._mst_d.critical_faces():
            verts = simplex.vertices
            if verts not in bd_map:
                warnings.warn(
                    f"No persistence pair found for critical face {verts}; assigning death=inf",
                    RuntimeWarning,
                )
                birth, death = simplex.weight, math.inf
            else:
                birth, death = bd_map[verts]
            critical_faces.append(CriticalFace(simplex, birth, death))

        print(f"[Persistence] Finished in {time.time() - t0:.2f}s")
        return critical_faces
    
    def export_simplex_tree(self, alpha: float) -> gudhi.SimplexTree:
        """Return a GUDHI SimplexTree containing only critical faces alive at scale α."""

        st = gudhi.SimplexTree()
        for cf in self._critical_faces:
            if cf.birth <= alpha < cf.death:
                st.insert(cf.simplex.vertices, filtration=cf.simplex.weight)
        return st

    def to_open3d_geometry(self, alpha: float) -> List[o3d.geometry.Geometry]:
        """Return Open3D geometries for 3D visualization of the HoPeS skeleton."""

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        st = self.export_simplex_tree(alpha)
        edges = []
        triangles = []

        for simplex, _ in st.get_filtration():
            if len(simplex) == 2:
                edges.append(simplex)
            elif len(simplex) == 3:
                triangles.append(simplex)

        geometries = [pcd]

        if edges:
            line_set = o3d.geometry.LineSet()
            line_set.points = pcd.points
            line_set.lines = o3d.utility.Vector2iVector(edges)
            line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(edges))
            geometries.append(line_set)
        else:
            print(f"[viz] No edges in skeleton at α = {alpha}")

        if triangles:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = pcd.points
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.4, 0.6, 1.0])
            geometries.append(mesh)
        else:
            print(f"[viz] No triangles in skeleton at α = {alpha}")

        return geometries



# -----------------------------------------------------------------------------
# Example usage (run `python hopes.py` to test quickly)
# -----------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    def generate_torus(n: int = 80, R: float = 2.5, r: float = 1.0) -> np.ndarray:
        theta = np.random.uniform(0, 2 * math.pi, n)
        phi = np.random.uniform(0, 2 * math.pi, n)
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)
        return np.vstack((x, y, z)).T

    pts = generate_torus() + 0.05 * np.random.randn(80, 3)

    hopes = HoPeS(pts, d=2, sparse=True, max_edge_length=100.0, rips_max_dim=2)
    alpha_view = 15  # choose a scale

    print("Critical faces (birth, death):")
    for cf in hopes._critical_faces:
        print(cf.simplex.vertices, cf.birth, cf.death)

    try:
        vis_geometries = hopes.to_open3d_geometry(alpha_view)
        o3d.visualization.draw_geometries(vis_geometries)
    except ImportError:
        print("Open3D not available. Install it via `pip install open3d`.")
