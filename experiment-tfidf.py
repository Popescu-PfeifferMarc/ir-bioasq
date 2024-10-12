import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from extract_xml import extract_xml
import numpy as np
import time

dataset_folder = './dataset/pubmed/'

# Batch processing function to load documents
def load_documents_in_batches(folder_path, batch_size):
    batch = []
    total_files = len(os.listdir(folder_path))
    start_time = time.time()
    for idx, filename in enumerate(os.listdir(folder_path)):
        content = extract_xml(os.path.join(folder_path, filename))
        if content["title"] is None or content["body"] is None:
            continue
        batch.append(content["title"] + " " + content["body"])
        
        # Yield a batch of documents when batch_size is reached
        if len(batch) == batch_size:
            yield batch
            batch = []
            elapsed_time = time.time() - start_time
            time_per_file = elapsed_time / (idx + 1)
            time_left = time_per_file * (total_files - (idx + 1))
            print(f"Loaded {idx + 1} / {total_files} documents, Elapsed time: {elapsed_time:.2f}s, Time left: {time_left:.2f}s")
    if batch:
        yield batch  # Return the remaining documents in the last batch
        print(f"Loaded {total_files} / {total_files} documents in {time.time() - start_time:.2f}s")

# Function to process documents in batches
def process_documents_in_batches(folder_path, batch_size=100):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10_000)  # Use max_features to limit vocabulary size
    partial_results = []
    batch_count = 0
    
    for batch in load_documents_in_batches(folder_path, batch_size):
        # Incrementally fit and transform the batches
        tfidf_matrix = vectorizer.fit_transform(batch)
        partial_results.append(tfidf_matrix.toarray())
        batch_count += 1
        print(f"Processed {batch_count * batch_size} documents")
    
    # Concatenate results from all batches
    tfidf_full_matrix = np.vstack(partial_results)
    return tfidf_full_matrix, vectorizer.get_feature_names_out()

# Process in batches (adjust batch_size based on available memory)
tfidf_matrix, feature_names = process_documents_in_batches(dataset_folder, batch_size=100)

# Convert to DataFrame and save to CSV
df_tfidf = pd.DataFrame(tfidf_matrix, columns=feature_names)
df_tfidf.to_csv('tfidf_results_large.csv')

print("TF-IDF results saved to 'tfidf_results_large.csv'")
