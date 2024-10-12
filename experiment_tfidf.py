import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from extract_xml import extract_xml
import numpy as np
import time
dataset_folder = './dataset/pubmed/'

def pretty_print_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours == 0:
        return f"{int(minutes)}m {int(seconds)}s"
    return f"{int(hours)}h {int(minutes)}m"

# Batch processing function to load documents
def load_documents_in_batches(folder_path, batch_size):
    batch = []
    total_files = len(os.listdir(folder_path))
    start_time = time.time()
    for idx, filename in enumerate(os.listdir(folder_path)):
        try:
            content = extract_xml(os.path.join(folder_path, filename))
            if content["pmid"] is None or  content["title"] is None or content["body"] is None:
                continue
            batch.append((content["pmid"], content["title"] + " " + content["body"]))
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue
        
        # Yield a batch of documents when batch_size is reached
        if len(batch) == batch_size:
            yield batch
            batch = []
            elapsed_time = time.time() - start_time
            time_per_file = elapsed_time / (idx + 1)
            time_left = time_per_file * (total_files - (idx + 1))
            print(f"Loaded {idx + 1} / {total_files} documents, Elapsed time: {pretty_print_duration(elapsed_time)}, Time left: {pretty_print_duration(time_left)}")
    if batch:
        print(f"Loaded {total_files} / {total_files} documents in {pretty_print_duration(elapsed_time)}")
        yield batch  # Return the remaining documents in the last batch

# Function to process documents in batches
def process_documents_in_batches(folder_path, batch_size=100):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10_000)
    partial_results = []
    batch_count = 0
    pmid_list = []
    
    for batch in load_documents_in_batches(folder_path, batch_size):
        # Incrementally fit and transform the batches
        pmids = [x[0] for x in batch]
        pmid_list.extend(pmids)

        texts = [x[1] for x in batch]
        tfidf_matrix = vectorizer.fit_transform(texts)
        partial_results.append(tfidf_matrix.toarray())
        batch_count += 1
        print(f"Processed {batch_count * batch_size} documents")
    
    # Concatenate results from all batches
    tfidf_full_matrix = np.vstack(partial_results)
    return pmid_list, tfidf_full_matrix, vectorizer.get_feature_names_out()

# Process in batches (adjust batch_size based on available memory)
pmid_list, tfidf_matrix, feature_names = process_documents_in_batches(dataset_folder, batch_size=100)

# Convert to DataFrame and save to CSV
df_tfidf = pd.DataFrame(tfidf_matrix, columns=feature_names, index=pmid_list)
df_tfidf.to_csv('tfidf_results_large.csv')

print("TF-IDF results saved to 'tfidf_results_large.csv'")
