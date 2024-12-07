import logging
import os
import sys
import json
from typing import List
import ollama

# Config
input_file = './out/taskA_tfidf_title/results_small_ltc.json'
do_golden = True # also use the golden snippets instead of the results from task A
do_non_golden = True # do not use the results from task A, useful to only use the golden ones

ollama_model = 'llama3.1:8b' # Ollama model to use for task B. Needs to be installed. Examples are 'jsk/bio-mistral' 'llama3.1:70b'
output_file = './out/taskB_LLM/results_' + ollama_model.replace(':', '_').replace('/', '_') + '_tfidf_title_small_ltc.json'
top_n = 10 # only use the top n snippets

# SETUP
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s", ' '.join(sys.argv))

def answerQuestionWithContext(question: str, relevant_paper_contents: List[str]) -> str:
    prompt = "You are an expert doctor answering questions for medical professionals. Given the following snippets from paper:\n" + "\n".join(relevant_paper_contents) + "\n\nPlease provide a brief answer to the following question: " + question   
    result = ollama.generate(model=ollama_model, prompt=prompt)
    return result['response']

def main():
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)
        logger.info("Loaded input data from json")
        total_entries = len(data)
        for idx, entry in enumerate(data):
            query = entry.get("query", "")
            
            if do_golden:
                golden_snippets = entry.get("golden_snippets", [])
                if (len(golden_snippets) == 0):
                    logger.warning(f"No golden snippets found for query: {query}")
                else:
                    entry["answer_from_golden_snippet"] = answerQuestionWithContext(query, golden_snippets)
            
            if do_non_golden:
                task1_snippets_raw = entry.get("relevant_snippets", [])
                task1_snippets_sorted = sorted(task1_snippets_raw, key=lambda x: x['score'], reverse=True)
                task1_snippets = [snippet['text'] for snippet in task1_snippets_sorted[:top_n]]
                if (len(task1_snippets) == 0):
                    logger.warning(f"No task1 snippets found for query: {query}")
                else:
                    entry["answer_from_task1_snippet"] = answerQuestionWithContext(query, task1_snippets)
            
            # Calculate and print progress
            progress = (idx + 1) / total_entries * 100
            logger.info(f"Progress: {progress:.2f}% ({idx + 1}/{total_entries})")
        
        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
            logger.info("Saved output data to json")

if __name__ == "__main__":
    logger.info("Starting script")
    main()
    logger.info("done")
