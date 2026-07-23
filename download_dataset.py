import urllib.request
import zipfile
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
zip_path = "data/student.zip"
extract_dir = "data/"

print("Downloading dataset from UCI Machine Learning Repository...")
try:
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete.")

    print("Extracting zip file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")

    # Clean up zip
    os.remove(zip_path)
    print("Cleanup complete.")
except Exception as e:
    print(f"Error downloading or extracting: {e}")
