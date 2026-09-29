"""
Download the UCI Student Performance Mathematics dataset.

The script downloads the official UCI archive and extracts:
    data/student-mat.csv

Run:
    python download_dataset.py
"""

from pathlib import Path
import io
import zipfile
import urllib.request

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "student-mat.csv"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student.zip"

if CSV_PATH.exists():
    print(f"Dataset already exists: {CSV_PATH}")
else:
    print("Downloading official UCI Student Performance dataset...")
    with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        member = next(
            name for name in zf.namelist()
            if name.endswith("student-mat.csv")
        )
        CSV_PATH.write_bytes(zf.read(member))

    print(f"Dataset saved to: {CSV_PATH}")
    print("Rows: 395")
    print("Columns: 33")
