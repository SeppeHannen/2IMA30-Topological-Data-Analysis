import argparse
import os
import numpy as np
import open3d as o3d
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt


def load_points(path: str) -> np.ndarray:
    """Load points from a PCD file using Open3D."""
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)


def sample_points(points: np.ndarray, num: int) -> np.ndarray:
    """Randomly sample ``num`` points without replacement."""
    if len(points) <= num:
        return points
    idx = np.random.choice(len(points), num, replace=False)
    return points[idx]


def compute_barcodes(points: np.ndarray, max_dim: int = 1):
    """Compute persistence diagrams up to ``max_dim``."""
    result = ripser(points, maxdim=max_dim)
    return result['dgms']


def save_barcode(diagrams, out_path: str):
    """Save barcode plot for the given diagrams."""
    plt.figure()
    plot_diagrams(diagrams, show=False)
    plt.savefig(out_path)
    plt.close()


def process_file(path: str, out_dir: str, sample_size: int, max_dim: int):
    name = os.path.splitext(os.path.basename(path))[0]
    points = load_points(path)
    if sample_size:
        points = sample_points(points, sample_size)
    dgms = compute_barcodes(points, max_dim=max_dim)
    out_png = os.path.join(out_dir, f"{name}_barcode.png")
    save_barcode(dgms, out_png)
    print(f"Saved {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute Vietoris-Rips persistence barcodes for .pcd point clouds"
    )
    parser.add_argument(
        "--data-dir", default="dataset", help="Folder containing point cloud .pcd files"
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("persistent_homology", "barcodes"),
        help="Directory to store barcode images",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of points to sample from each cloud (default 1000)",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1,
        help="Maximum homology dimension to compute (default 1)",
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for fname in os.listdir(args.data_dir):
        if fname.lower().endswith(".pcd"):
            path = os.path.join(args.data_dir, fname)
            print(f"Processing {path} ...")
            process_file(path, args.out_dir, args.sample_size, args.max_dim)


if __name__ == "__main__":  # pragma: no cover
    main()
