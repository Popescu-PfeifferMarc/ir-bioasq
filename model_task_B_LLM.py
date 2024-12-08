import logging
import os
import sys
import json
from typing import List
import ollama

# Config
ollama_model = "llama3.1:8b"  # Ollama model to use for task B. Needs to be installed. Examples are 'jsk/bio-mistral' 'llama3.1:70b'

input_taskA_results = "./out/taskA_BM25S_title/results.json" # Replace with your results JSON file path
output_results_taskA_file = (
    "./out/taskB_LLM/results_"
    + ollama_model.replace(":", "_").replace("/", "_")
    + "_BM25S_title"
    + ".json"
)
do_taskA_results = True

input_golden_file = "./dataset/12B_golden_combined.json"
output_results_golden_file = (
    "./out/taskB_LLM/results_"
    + ollama_model.replace(":", "_").replace("/", "_")
    + "_golden"
    + ".json"
)
do_golden = True

# Setup
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s")
logging.root.setLevel(level=logging.INFO)
logger.info("running %s", " ".join(sys.argv))
os.makedirs(os.path.dirname(output_results_taskA_file), exist_ok=True)
os.makedirs(os.path.dirname(output_results_golden_file), exist_ok=True)


def answerQuestionWithContext(question: str, relevant_paper_contents: List[str]) -> str:
    prompt = (
        "You are an expert doctor answering questions for medical professionals. Given the following snippets from paper:\n"
        + "\n".join(relevant_paper_contents)
        + "\n\nPlease provide a brief answer to the following question: "
        + question
    )
    result = ollama.generate(model=ollama_model, prompt=prompt)
    return result["response"]


def loadGoldenData():
    with open(input_golden_file, "r", encoding="utf-8") as file:
        data = json.load(file).get("questions")
        logger.info("Loaded golden data from json")
        return [
            {
                "question": row["body"],
                "snippets": [snippet["text"] for snippet in row["snippets"]],
            }
            for row in data
        ]


def loadTaskAData():
    with open(input_taskA_results, "r", encoding="utf-8") as file:
        data = json.load(file).get("questions")
        logger.info("Loaded task A data from json")
        return [
            {
                "question": row["query"],
                "snippets": [snippet["text"] for snippet in row["snippets"]],
            }
            for row in data
        ]


def main():
    # run model on golden snippets
    if do_golden:
        golden_data = loadGoldenData()
        output_golden = []
        for idx, row in enumerate(golden_data):
            question = row["question"]
            snippets = row["snippets"]

            if len(snippets) == 0:
                logger.warning(f"No golden snippets found for question: {question}")

            answer = answerQuestionWithContext(question, snippets)
            output_golden.append({"question": question, "answer": answer})
            logger.info(f"Progress (golden): {idx + 1}/{len(golden_data)}")

        with open(output_results_golden_file, "w", encoding="utf-8") as outfile:
            json.dump(output_golden, outfile, ensure_ascii=False, indent=4)
            logger.info("Saved golden results to json")

    # run model on task A results
    if do_taskA_results:
        results_data = loadTaskAData()
        output_resultsA = []
        for idx, row in enumerate(results_data):
            question = row["question"]
            snippets = row["snippets"]

            if len(snippets) == 0:
                logger.warning(f"No snippets found for question: {question}")

            answer = answerQuestionWithContext(question, snippets)
            output_resultsA.append({"question": question, "answer": answer})
            logger.info(f"Progress (task A): {idx + 1}/{len(results_data)}")

        with open(output_results_taskA_file, "w", encoding="utf-8") as outfile:
            json.dump(output_resultsA, outfile, ensure_ascii=False, indent=4)
            logger.info("Saved taskA results to json")


if __name__ == "__main__":
    main()
    logger.info("All done 🚀")
