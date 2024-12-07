import logging
import os.path
import sys
import json
from rouge_score import rouge_scorer
# SETUP
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s", ' '.join(sys.argv))

file_path = "./results.json"  # Replace with your results JSON file path
golden_file_path = "./golden_data.json" # Replace with your golden JSON file path
output_file_path = "./macro_averages.json"
def load_results_task_2(file_path):
    try:
        # Load the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file).get("questions")
        
        results = []

        # Loop through each query
        for item in data:
            query_data = {
                "query": item.get("query"),
                "answer": item.get("answer")
            }
            results.append(query_data)
        
        return results

    except Exception as e:
        print(f"An error occurred: {e}")
        return []
def load_golden_task_2(golden_file_path):
    try:
        # Load the JSON file
        with open(golden_file_path, "r", encoding="utf-8") as golden_file_data:
            data = json.load(golden_file_data).get("questions")
        
            results = []

            # Loop through each query
            for item in data:
                query_data = {
                    "query": item.get("body"),
                    "ideal_answer": item.get("ideal_answer")
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

task_2_merged_dictionary=merge_dictionaries(load_golden_task_2(golden_file_path),
                                            load_results_task_2(file_path))
# This will create a dictionary with query, our answer and ideal answer

def calculate_rouge_scores_for_queries(queries):
    """
    Calculates ROUGE-2 and ROUGE-SU4 scores for all answers in each query against the ideal answer.
    
    Args:
    queries (list): List of dictionaries where each dictionary contains a query, answers, and an ideal answer.
    
    Returns:
    list: A list of dictionaries containing the ROUGE scores for each answer in every query.
    """
    results = []
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2", "rougeLsum"], use_stemmer=True)  # Use ROUGE-Lsum as SU4 approximation

    for query in queries:
        ideal_answer = " ".join(query["ideal_answer"])  # Combine list into a single string
        answer=query["answer"]
        query_scores = {"query": query["query"], "scores": {}}
        scores = scorer.score(ideal_answer, answer)
        query_scores["scores"] = {
            "ROUGE-1": scores["rouge1"].fmeasure,
            "ROUGE-2": scores["rouge2"].fmeasure,
            "ROUGE-SU4": scores["rougeLsum"].fmeasure,
        }
        
        results.append(query_scores)

    return results
def calculate_macro_averages(results):
    """
    Calculate macro averages for ROUGE-1, ROUGE-2, and ROUGE-SU4 for all answers.

    Args:
    results (list): List of dictionaries containing ROUGE scores for each query.

    Returns:
    dict: Macro-averaged ROUGE-1, ROUGE-2, and ROUGE-SU4 scores.
    """
    # Initialize variables to aggregate scores
    total_rouge1, total_rouge2, total_rouge_su4, total_count = 0, 0, 0, 0

    # Iterate through each query result
    for result in results:
        if "scores" in result and result["scores"]:
            total_rouge1 += result["scores"]["ROUGE-1"]
            total_rouge2 += result["scores"]["ROUGE-2"]
            total_rouge_su4 += result["scores"]["ROUGE-SU4"]
            total_count += 1

    # Calculate macro averages
    macro_averages = {
        "Macro ROUGE-1": total_rouge1 / total_count if total_count > 0 else 0,
        "Macro ROUGE-2": total_rouge2 / total_count if total_count > 0 else 0,
        "Macro ROUGE-SU4": total_rouge_su4 / total_count if total_count > 0 else 0,
    }

    return macro_averages

rouge_results = calculate_rouge_scores_for_queries(task_2_merged_dictionary)
macro_averages = calculate_macro_averages(rouge_results)

print("Macro-Averaged ROUGE Scores:")
print(f"  Macro ROUGE-1: {macro_averages['Macro ROUGE-1']:.4f}")
print(f"  Macro ROUGE-2: {macro_averages['Macro ROUGE-2']:.4f}")
print(f"  Macro ROUGE-SU4: {macro_averages['Macro ROUGE-SU4']:.4f}")




# Save the macro averages to a JSON file
with open(output_file_path, "w", encoding="utf-8") as file:
    json.dump(macro_averages, file, ensure_ascii=False, indent=4)

print(f"Macro averages saved to {output_file_path}")