import os
import requests
import shutil
import gzip

# Base URL and file range
base_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
file_template = "pubmed24n{:04d}.xml.gz"
dataset_folder = "./dataset/pubmed_annual_baseline"

# Ensure the dataset folder exists
os.makedirs(dataset_folder, exist_ok=True)

# Download and unpack files 0001 to 1219
def main():
    for i in range(1, 1220):
        in_file_name = file_template.format(i)
        out_file_name = in_file_name[:-3]  # Remove the .gz extension
        url = base_url + in_file_name
        out_file_path = os.path.join(dataset_folder, out_file_name)
        
        print(f"Downloading\t{url}")        
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Raise an error for HTTP issues
            with gzip.GzipFile(fileobj=response.raw) as gz:
                with open(out_file_path, "wb") as output_file:
                    shutil.copyfileobj(gz, output_file)
        print(f"File successfully downloaded and unpacked to {out_file_path}")
    print("All files downloaded and unpacked successfully 🚀")

if __name__ == "__main__":
    main()
