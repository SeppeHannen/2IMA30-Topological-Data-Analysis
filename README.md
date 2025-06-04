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
