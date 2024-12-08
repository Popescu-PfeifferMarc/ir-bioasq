import logging
import os.path
import sys
import json

# SETUP
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s", ' '.join(sys.argv))

file_path = "./results.json"  # Replace with your results JSON file path
golden_file_path = "./dataset/12B_golden_combined.json" # Replace with your golden JSON file path
output_file_path = "./evaluation_results.json"

def load_results_json(file_path):
    try:
        # Load the JSON file
        with open(file_path, "r", encoding="utf-8") as golden_file_data:
            data = json.load(golden_file_data).get("questions")
        
        results = []

        # Loop through each query
        for item in data:
            query_data = {
                "query": item.get("query"),
                "output_docs": item.get("documents"),# Contains both score and document url,
                "output_snippets": item.get("snippets") # Contains score,pmid,snippet text
            }
            results.append(query_data)
        
        return results

    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def load_golden_json(golden_file_path):
    try:
        # Load the JSON file
        with open(golden_file_path, "r", encoding="utf-8") as golden_file_data:
            data = json.load(golden_file_data).get("questions")
        
            results = []

            # Loop through each query
            for item in data:
                query_data = {
                    "query": item.get("body"),
                    "golden_docs": item.get("documents"),
                    "golden_snippets": [
                        {"text":doc.get("text"),"pmid": doc.get("document")} 
                        for doc in item.get("snippets", [])
                    ]
                }
                results.append(query_data)
        
        return results

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def merge_dictionaries(golden_data, results):
    merged_dict = {}

    # Add items from the first list
    for item in golden_data:
        merged_dict[item['query']] = item

    # Add or update items from the second list
    for item in results:
        if item['query'] in merged_dict:
            # If the query exists, you can merge the dictionaries as needed
            merged_dict[item['query']].update(item)
        else:
            # Otherwise, just add the new item
            merged_dict[item['query']] = item

    # Convert the merged dictionary back to a list
    return list(merged_dict.values())

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
  
def process_queries_and_evaluate(file_path, golden_file_path):
    logger.info("Starting query processing and evaluation...")
    all_metrics = {"precision": [], "recall": [], "f1": []}
    all_snippet_precision=[]
    results = load_results_json(file_path)
    golden_data = load_golden_json(golden_file_path)
    final_dictionary=merge_dictionaries(golden_data,results)
    for  x in final_dictionary:
        query = x["query"]
        golden_docs = x["golden_docs"]
        golden_snippets = x["golden_snippets"]
        retrieved_docs = [x["pmid"] for x in x["output_docs"]]
        # We only extract document ids from output_docs
        retrieved_snippet_texts = [x["text"] for x in x["output_snippets"]]
        logger.info("Processing query: %s", query)

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
        "Mean Precision@10": mean_precision_at_10,
        "macro_avg_precision": macro_avg_precision,
        "macro_avg_recall": macro_avg_recall,
        "macro_avg_f1": macro_avg_f1
    }


evaluation=process_queries_and_evaluate(file_path,golden_file_path)



# Save the evaluation results to a JSON file
with open(output_file_path, "w", encoding="utf-8") as output_file:
    json.dump(evaluation, output_file, ensure_ascii=False, indent=4)

logger.info(f"Evaluation results saved to {output_file_path}")
