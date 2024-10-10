import requests
import gzip
import shutil
import os

url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline-2024-sample/sample-0001.xml.gz"
destination_folder = "./dataset"
destination_file = os.path.join(destination_folder, "sample-0001.xml")

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Download the file
response = requests.get(url, stream=True)
gz_file_path = os.path.join(destination_folder, "sample-0001.xml.gz")
with open(gz_file_path, 'wb') as f:
    shutil.copyfileobj(response.raw, f)

# Extract the .gz file
with gzip.open(gz_file_path, 'rb') as f_in:
    with open(destination_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Optionally, remove the .gz file after extraction
os.remove(gz_file_path)

print(f"File downloaded and extracted to {destination_file}")
