import logging
import os
import sys
import csv
import json

from bm25s import tokenize, BM25

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

# Config
golden_file = "./dataset/12B_golden_combined.json"
articles_file = "./dataset/pubmed_annual_baseline.csv"
output_base_folder = "./out/taskB_bm25"
include_abstract = True
output_folder = output_base_folder + "_title/" if not include_abstract else output_base_folder + "_title_abstract/"

# Setup
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s", ' '.join(sys.argv))
os.makedirs(os.path.dirname(output_folder), exist_ok=True)

# Corpus
class DocumentCorpus:
    def __iter__(self):
        with open(articles_file, mode='r', newline='', errors='ignore') as inp_file:
            reader = csv.reader(inp_file)
            for row in reader:
                # doc_pmid = row[0]
                doc_title = row[1]
                doc_abstract = row[2]
                if include_abstract:
                    yield str(doc_title) + '\n' + str(doc_abstract)
                else:
                    yield str(doc_title)

corpus = DocumentCorpus()
logger.info("Created corpus")

# Pre-process
processed_corpus = tokenize([entry for entry in corpus])
logger.info("Processed corpus")

# Index
retriever = BM25()
retriever.index(processed_corpus)
logger.info("Indexed corpus")

class GoldenDataLoader:
    def __init__(self, golden_file_path):
        """
        Initialize the GoldenDataLoader with the path to the golden data JSON file.
        """
        self.golden_file_path = golden_file_path
    def __iter__(self):
        """
        Lazily load queries, golden documents, and snippets from the JSON file.
        """
        with open(self.golden_file_path, "r") as file:
            data = json.load(file).get("questions", [])
            for question in data:
                query = question.get("body", "")
                golden_docs = question.get("documents", [])
                golden_snippets = [snippet.get("text", "") for snippet in question.get("snippets", [])]
                yield query, golden_docs, golden_snippets
def process_query_and_retrieve_documents_and_snippets(query,model=retriever,dataset=inp):
    query_tokens = tokenize(query)

    # Compute similarities
    doc, scores= model.retrieve(query_tokens, k=10)
    top_docs_indices = sorted(doc[0])
    relevant_documents = []
    snippets = []

    logger.info("Processing top documents for query: %s", query)

    for doc_id in top_docs_indices:
        if doc_id in lookup:
            pmid = lookup[doc_id]["pmid"]
            abstract = lookup[doc_id]["doc_title"]
            
            relevant_documents.append(f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}")

            # Process abstract for snippets
            sentences = sent_tokenize(abstract)
            for sentence in sentences:
                snippets.append({
                    "text": sentence,
                    "source": pmid,
                })
    snippets_tokenized=tokenize([snippet["text"] for snippet in snippets])

    # Re-index the BM25 retriever with snippets
    retriever=BM25()
    retriever.index(snippets_tokenized)
    # Retrieve the top k most relevant snippets globally
    snippet_docs, snippet_scores = retriever.retrieve(query_tokens, k=10)

    # Extract the top k snippets with scores and source document IDs
    top_snippets = [
        {
            "text": snippets[idx]["text"],
            "score": snippet_scores[0, i],
            "source": snippets[idx]["source"],
        }
        for i, idx in enumerate(snippet_docs[0])
    ]

    # Sort snippets by score

    return relevant_documents, top_snippets
def calculate_macro_metrics(retrieved_docs, golden_docs):
    """
    Calculate Macro Precision, Recall, and F1-Score for a single query.

    Parameters:
    retrieved_docs (list): List of retrieved document IDs.
    golden_docs (list): List of golden document IDs.

    Returns:
    dict: Precision, recall, and F1-score for this query.
    """
    retrieved_set = set(retrieved_docs)
    golden_set = set(golden_docs)

    intersection = retrieved_set & golden_set

    precision = len(intersection) / len(retrieved_set) if retrieved_set else 0
    recall = len(intersection) / len(golden_set) if golden_set else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}
def calculate_precision_at_k(retrieved_snippets, golden_snippets, k=10):
    """
    Calculate Precision at k for a single query.
    Parameters:
    retrieved_snippets (list): List of retrieved snippet texts.
    golden_snippets (list): List of golden (relevant) snippet texts.
    k (int): The cutoff rank.

    Returns:
    float: Precision at k.
    """
    retrieved_at_k = retrieved_snippets[:k]
    relevant_retrieved = sum(1 for snippet in retrieved_at_k if snippet in golden_snippets)
    precision_at_k = relevant_retrieved / k
    return precision_at_k
def evaluate_model():
    logger.info("Starting query processing and evaluation...")
    all_metrics = {"precision": [], "recall": [], "f1": []}
    all_snippet_precision=[]
    for query, golden_docs, golden_snippets in GoldenDataLoader(golden_file_path):
        # Retrieve relevant documents and snippets
        retrieved_docs, retrieved_snippets = process_query_and_retrieve_documents_and_snippets(query)

        # Extract snippet texts from retrieved_snippets
        retrieved_snippet_texts = [snippet["text"] for snippet in retrieved_snippets]

        # Calculate Precision at 10
        precision_at_10 = calculate_precision_at_k(retrieved_snippet_texts, golden_snippets, k=10)
        logger.info("Query Precision@10: %f", precision_at_10)

        # Append precision
        all_snippet_precision.append(precision_at_10)
        # Evaluate this query
        metrics = calculate_macro_metrics(retrieved_docs, golden_docs)
        logger.info("Query Metrics - Precision: %f, Recall: %f, F1: %f", metrics["precision"], metrics["recall"], metrics["f1"])

        # Append metrics
        all_metrics["precision"].append(metrics["precision"])
        all_metrics["recall"].append(metrics["recall"])
        all_metrics["f1"].append(metrics["f1"])

    # Calculate aggregated macro-averaged metrics
    macro_avg_precision = sum(all_metrics["precision"]) / len(all_metrics["precision"]) if all_metrics["precision"] else 0
    macro_avg_recall = sum(all_metrics["recall"]) / len(all_metrics["recall"]) if all_metrics["recall"] else 0
    macro_avg_f1 = sum(all_metrics["f1"]) / len(all_metrics["f1"]) if all_metrics["f1"] else 0
    # Calculate Mean Precision at 10 across all queries
    mean_precision_at_10 = sum(all_snippet_precision) / len(all_snippet_precision) if all_snippet_precision else 0
    logger.info("Mean Precision@10: %f", mean_precision_at_10)
    logger.info("Macro-Averaged Metrics - Precision: %f, Recall: %f, F1: %f", macro_avg_precision, macro_avg_recall, macro_avg_f1)

    return {
        "mean_precision_at_10": mean_precision_at_10,
        "macro_avg_precision": macro_avg_precision,
        "macro_avg_recall": macro_avg_recall,
        "macro_avg_f1": macro_avg_f1
    }

results=evaluate_model()

results
with open(os.path.join(outp, "evaluation_results.json"), "w") as outfile:
    json.dump(results, outfile, indent=4)


"""
TODO

includeAbstract flag needs to be implemented

evalution move to evalute task A (can also be a notebook)


output format should be
{
  "questions": [
    {
      "query": "Concizumab is used for which diseases?",
      "documents": [
        {
          "pmid": "http://www.ncbi.nlm.nih.gov/pubmed/37341887",
          "score": 0.9
        }
      ],
      "snippets": [
        {
          "text": "Concizumab is being developed by Novo Nordisk for the treatment of hemophilia A and B with and without inhibitors. In March 2023, concizumab was approved in Canada for the treatment of adolescent and adult patients (12 years of age or older) with hemophilia B who have FIX inhibitors and require routine prophylaxis to prevent or reduce the frequency of bleeding episodes. ",
          "pmid": "37341887",
          "score": 0.9
        }
      ]
    },
    {
        // more here
    }
  ]
}



"""