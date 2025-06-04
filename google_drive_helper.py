#!/usr/bin/env python3
"""Utility for downloading the cherry tree dataset from Google Drive.

This script exposes simple commands to list files in a public Google Drive
folder and download individual files or whole folders. Authentication is
performed using a Google service account whose JSON key must be supplied via
the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable.

Examples
--------
List available files within a folder::

  python google_drive_helper.py list <FOLDER_ID>

Download a specific file::

  python google_drive_helper.py download <FILE_ID> --dest data/bag_0.ply

Download an entire folder recursively::

  python google_drive_helper.py download-folder <FOLDER_ID> --dest data/

Required packages:
  google-api-python-client
  google-auth
  google-auth-httplib2
  google-auth-oauthlib
"""

from __future__ import annotations

import argparse
import io
import os
from typing import List, Dict

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _get_service():
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
        )
    creds = service_account.Credentials.from_service_account_file(
        cred_path, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


def list_folder(folder_id: str) -> List[Dict[str, str]]:
    """Return metadata of files directly within ``folder_id``."""
    service = _get_service()
    results = []
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, mimeType, size)",
                pageToken=page_token,
            )
            .execute()
        )
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results


def download_file(file_id: str, dest_path: str) -> None:
    """Download a single file from Drive to ``dest_path``."""
    service = _get_service()
    request = service.files().get_media(fileId=file_id)
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def download_folder(folder_id: str, dest_dir: str) -> None:
    """Recursively download a Drive folder into ``dest_dir``."""
    files = list_folder(folder_id)
    os.makedirs(dest_dir, exist_ok=True)
    for f in files:
        path = os.path.join(dest_dir, f["name"])
        if f["mimeType"] == "application/vnd.google-apps.folder":
            download_folder(f["id"], path)
        else:
            download_file(f["id"], path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Google Drive dataset helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ls_p = sub.add_parser("list", help="List files in a folder")
    ls_p.add_argument("folder_id", help="ID of the Google Drive folder")

    dl_p = sub.add_parser("download", help="Download a single file")
    dl_p.add_argument("file_id", help="ID of the Google Drive file")
    dl_p.add_argument("--dest", default=".", help="Destination path")

    dlf_p = sub.add_parser("download-folder", help="Download an entire folder")
    dlf_p.add_argument("folder_id", help="ID of the Google Drive folder")
    dlf_p.add_argument("--dest", default=".", help="Destination directory")

    args = parser.parse_args()

    if args.cmd == "list":
        for f in list_folder(args.folder_id):
            size = f.get("size", "folder")
            print(f"{f['id']}\t{size}\t{f['name']}")
    elif args.cmd == "download":
        download_file(args.file_id, args.dest)
    elif args.cmd == "download-folder":
        download_folder(args.folder_id, args.dest)


if __name__ == "__main__":
    main()
