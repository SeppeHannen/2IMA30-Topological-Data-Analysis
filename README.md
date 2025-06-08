# 2IMA30-Topological-Data-Analysis

This project contains experiments with point-cloud based algorithms. The actual
point-cloud data is not stored in the repository. Use the helper script
`google_drive_helper.py` to retrieve the cherry tree dataset from the shared
Google Drive location.

## Downloading the dataset

1. Obtain a Google service-account JSON key and set the environment variable
   `GOOGLE_APPLICATION_CREDENTIALS` to its path.
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

## Visualizing Persistent Features

Run `visualize_persistence.py` to display persistent connected components and cycles on top of a point cloud.
Example:
```bash
python visualize_persistence.py dataset/tree_0.pcd --sample 2000 --voxel 0.01 --cycle_thresh 0.05
```
This downsamples the cloud, computes persistence up to dimension 1 and overlays the longest cycles in blue and the merging edges in red.

