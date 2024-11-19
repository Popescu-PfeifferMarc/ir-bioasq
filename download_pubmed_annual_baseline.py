import os
import requests
import shutil
import time
from datetime import timedelta

base_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
file_template = "pubmed24n{:04d}.xml.gz"
dataset_folder = "./dataset/pubmed_baseline"
file_count = 1219

# Create dataset folder if it doesn't exist
os.makedirs(dataset_folder, exist_ok=True)

start_time = time.time()

for i in range(1, file_count + 1):
    current_time = time.time()
    elapsed_time = current_time - start_time
    if i > 1:
        avg_time_per_file = elapsed_time / (i - 1)
        remaining_files = file_count - i + 1
        estimated_remaining_time = avg_time_per_file * remaining_files
        eta = str(timedelta(seconds=int(estimated_remaining_time)))
    else:
        eta = "calculating..."
    
    file_name = file_template.format(i)
    url = base_url + file_name
    
    print(f"[{i}/{file_count}] Downloading {file_name} - Elapsed: {str(timedelta(seconds=int(elapsed_time)))} - ETA: {eta}")

    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(os.path.join(dataset_folder, file_name), 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"Failed to download {file_name}. HTTP Status Code: {response.status_code}")
        continue

print("All files downloaded successfully 🚀")
total_elapsed_time = time.time() - start_time
print(f"Total elapsed time: {str(timedelta(seconds=int(total_elapsed_time)))}")
