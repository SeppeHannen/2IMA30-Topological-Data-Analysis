# Analyzing UFO Cherry trees with Mapper and Partitioned Minimum Spanning Trees

This repository accompanies Group 2's report for the TU/e course 2IMA30 Topological Data Analysis. It provides implementations of the Mapper and PMST algorithms used for skeletonizing LiDAR scans of upright fruiting offshoot cherry trees; see the report for details.

Note that the actual dataset is not stored in this repository as it is too large; use the helper script `google_drive_helper.py` to retrieve the cherry tree dataset from the shared Google Drive location. 
Alternatively, the dataset (or a desired subset of it) can be downloaded from [this website](https://paperswithcode.com/dataset/ufo-cherry-tree-point-clouds) and manually placed in the `data` folder, such that `data/bag_X/cloud_final.pcd` is a valid path (with `X` replaced by whichever bag you wish to analyze). 
Some example point clouds before and after preprocessing are stored in the `dataset` and `dataset_processed` folders.

## Downloading the dataset from Google Drive

1. Obtain a Google service-account JSON key and set the environment variable `GOOGLE_APPLICATION_CREDENTIALS` to its path.
2. Install the required packages:
   ```bash
   pip install google-api-python-client google-auth google-auth-httplib2 \
               google-auth-oauthlib
   ```
3. List available files in the dataset folder:
   ```bash
   python google_drive_helper.py list <FOLDER_ID>
   ```
4. Download a file or an entire folder:
   ```bash
   python google_drive_helper.py download <FILE_ID> --dest data/filename.ply
   python google_drive_helper.py download-folder <FOLDER_ID> --dest data/
   ```

## Data visualization
Run `view_tree.py` to view an interactive 3D visualization of one tree's LiDAR scan, without further processing. 
Alternatively, run `view_pointcloud_with_superpoints.py` to view the pointcloud downsampled using superpoint selection.
Additional preprocessing steps with corresponding visualizations can be found in `Preprocessing_Algorithms.py`.
Note: Mapper and persistent homology use their own separate preprocessing scripts (integrated into the notebook for Mapper), as they may require different parameters to function properly.

## Mapper
All information pertaining to our implementation of Mapper is found in the `Mapper.ipynb` notebook. Run the notebook in full to produce a graph-over-pointcloud overlay with default hyperparameter settings.

## Persistent homology / Partitioned MST
The scripts related to the PMST algorithm can be found in the `persistent_homology` folder. Run `partitioned_mst.py` with a `.pcd` file to obtain a PMST for a single tree; we recommend using a preprocessed point cloud from `dataset_processed`, as otherwise the algorithm takes a long time to run. 
Alternatively, run `compare_skeletons.py` to get a side-by-side comparison of the standard and partitioned MST algorithms.

