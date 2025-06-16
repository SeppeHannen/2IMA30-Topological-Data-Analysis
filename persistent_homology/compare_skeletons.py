#!/usr/bin/env python3
"""
compare_layers.py  –  Interactive, side-by-side comparison of
                      • plain MST  (left)
                      • layered MST with K strata (right)
Drag/zoom in either pane and the other follows.
Requires Open3D ≥ 0.15 (for gui.SceneWidget camera callbacks).
"""

import argparse
from pathlib import Path
from itertools import combinations
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# ────────── build skeletons (import your existing helpers) ──────────
# Here we *import* the two build_… functions from the script you already have.
# from partitioned_mst import (
#     load_points, partition_by_scalar,
#     build_plain_mst, build_partitioned_mst,
# )
# For brevity I'll inline stubs that expect the same signatures.

# --------------  tiny stubs – replace with real ones  --------------
def load_points(p, sample=None, voxel=None):
    pcd = o3d.io.read_point_cloud(p)
    return np.asarray(pcd.points), pcd

def build_plain_mst(pts, knn=None, verbose=False):
    # toy: connect sequentially
    return [(i, i + 1, 0) for i in range(len(pts) - 1)]

def partition_by_scalar(vals, k):
    idx = np.arange(len(vals))
    return [idx]   # single layer stub

def build_partitioned_mst(pts, layers, knn=None, verbose=False):
    return build_plain_mst(pts)
# --------------------------------------------------------------------

# ────────────────────── helper ──────────────────────
def make_scene_widget(name, pcd, edges, n_layers, renderer):  # ← add arg
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")

    pcd_local = o3d.geometry.PointCloud(pcd)           # copy for this view

    ls = o3d.geometry.LineSet()
    ls.points = pcd_local.points
    ls.lines  = o3d.utility.Vector2iVector([(u, v) for u, v, _ in edges])
    ls.colors = o3d.utility.Vector3dVector(
        [cmap(l / max(1, n_layers - 1))[:3] for _, _, l in edges])

    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(renderer)     # ← use *window* renderer
    widget.scene.add_geometry(f"{name}_pcd",   pcd_local,
                              rendering.MaterialRecord())
    mat = rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = 2.0
    widget.scene.add_geometry(f"{name}_lines", ls, mat)

    bbox = pcd_local.get_axis_aligned_bounding_box()
    widget.setup_camera(60, bbox, bbox.get_center())
    widget.scene.show_axes(False)
    return widget


# ────────────────────── main viewer ──────────────────────
def run_viewer(pcd, edges_left, edges_right, n_layers_left, n_layers_right):
    app = gui.Application.instance
    app.initialize()

    w = app.create_window("MST comparison (linked cameras)", 1280, 720)

    left  = make_scene_widget("plain",   pcd, edges_left,   n_layers_left,
                              w.renderer)
    right = make_scene_widget("layered", pcd, edges_right, n_layers_right,
                              w.renderer)

    row = gui.Horiz(0)
    row.add_child(left)
    row.add_child(right)
    w.add_child(row)

    _busy = {"flag": False}

    def copy_cam(src, dst):
        cam = src.scene.camera
        dst.scene.camera.set_projection(cam.get_field_of_view(),
                                        cam.get_near(), cam.get_far(),
                                        cam.get_aspect())
        dst.scene.camera.look_at(cam.get_look_at(),
                                 cam.get_position(),
                                 cam.get_up())

    def on_cam_change(src, dst):
        def _cb(ev):
            if _busy["flag"]:
                return
            _busy["flag"] = True
            copy_cam(src, dst)
            _busy["flag"] = False
        return _cb

    left.set_on_camera_changed(on_cam_change(left, right))
    right.set_on_camera_changed(on_cam_change(right, left))

    app.run()

# ──────────────────────────  CLI / main  ──────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pcd")
    ap.add_argument("--layers", type=int, default=6,
                    help="K strata for the *right* pane (left pane is always 1).")
    ap.add_argument("--sample", type=int,
                    help="Optional random sample for speed")
    args = ap.parse_args()

    pts, pcd = load_points(args.pcd, sample=args.sample)

    def load_npz(path):
        z = np.load(path)
        return z["pts"], z["edges"]

    pts, edges_plain  = load_npz("persistent_homology/skeletons/tree_0_layers_1.npz")
    _,   edges_layered = load_npz("persistent_homology/skeletons/tree_0_layers_2.npz")

    run_viewer(pcd, edges_plain, edges_layered, 1, args.layers)
