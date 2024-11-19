import os
import requests
import shutil
import tarfile

# Base URL and file range
base_url = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/xml/"
file_template = "oa_noncomm_xml.PMC{:03d}xxxxxx.baseline.2024-06-18.tar.gz"
temp_folder = "temp"
dataset_folder = "./dataset/pubmed"

# Remove the dataset/pubmed folder if it exists
if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
if os.path.exists(dataset_folder):
    shutil.rmtree(dataset_folder)

# Ensure the temp and dataset/pubmed folders exist
os.makedirs(temp_folder, exist_ok=True)
os.makedirs(dataset_folder, exist_ok=True)


# Download and unpack files 001 to 011
for i in range(1, 12):
    file_name = file_template.format(i)
    url = base_url + file_name
    local_path = os.path.join(temp_folder, file_name)
    
    print(f"Downloading\t{url}")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        # Unpack the contents of the .tar.gz file to dataset/pubmed
        print(f"Unpacking\t{local_path}")
        with tarfile.open(local_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.isreg():  # skip if the TarInfo is not files
                    member.name = os.path.basename(member.name) # remove the path by reset it
                    tar.extract(member, dataset_folder)
    else:
        print(f"Failed to download {file_name}. HTTP Status Code: {response.status_code}")

print("All files downloaded and unpacked successfully 🚀")

if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
