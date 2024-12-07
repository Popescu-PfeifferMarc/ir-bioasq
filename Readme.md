# IR-Project

## Requirements
- Python `3.12` incl. library requirements from requirements.txt
- [ollama](https://ollama.com/) incl. models as specified in model_task_B_LLM.py
- `unzip` or equivalent to unpack zip files
- `jq` or equivalent to combine JSON files

## Dataset

### Bioasq training data (~38MB download)
- create BioASQ account: [signup](http://participants-area.bioasq.org/accounts/register/)
- open http://participants-area.bioasq.org/datasets/
- open the section "Datasets for task b"
- download the "Training 12b" zip archive
- extract the file training12b_new.json to the dataset folder
- remove the zip archive 

### Bioasq golden data (~7MB download)
- create BioASQ account: [signup](http://participants-area.bioasq.org/accounts/register/)
- open http://participants-area.bioasq.org/datasets/
- open the section "Datasets for task b"
- download the "12b golden enriched" zip archive
- extract the files `12B1_golden.json`, `12B2_golden.json`, `12B3_golden.json` and `12B4_golden.json`
- combine the files using `jq -s '{questions: map(.questions) | add}' 12B1_golden.json 12B2_golden.json 12B3_golden.json 12B4_golden.json > 12B_golden_combined.json`
- move the resulting `12B_golden_combined.json` file to the dataset folder
- remove the four partial JSON files and the zip archive

### pubmed annual baseline 2024 (~20GB download)
- run `dataset_pubmed_annual_baseline_download.py` to download and unpack XML files
- run `dataset_pubmed_annual_baseline_extract_xml.py` to extract relevant information from the XML files and combine them into a CSV file
- (optional) run `tail -n 100000 pubmed_annual_baseline.csv > pubmed_annual_baseline_tail_100k.csv` to get a subset of the dataset

### Pubmed open access non-commercial subset (~100GB download)
Note that this dataset is not needed to reproduce the final results, as it was used only for experimentation. It is included anyway for completeness and potential further research.
- run `dataset_pumed_oa_noncomm_download.py` to download and unpack article XML files
- run `dataset_pubmed_oa_noncomm_extract_xml.py` to extract relevant information from the XML files and combine them into a CSV file

## Models

### Task A 1.1 - TF-IDF
 - see `model_task_A_TFIDF.py`

### Task A 1.2 - BM25
 - see `model_task_A_BM25.py`


### Task B 1.1 - LLM
 - see `model_task_B_LLM.py`

### Task B 1.2 - Vector DB
 - see `bioasq_project_with_vectorDB.ipynb`

---

## Task B 1.2: Retrier Model Leveraging Embeddings and Vector Database

This project implements a biomedical question-answering (QA) system by leveraging embeddings and a vector database for efficient document retrieval, paired with a generative model for answer generation.  

The main implementation is contained in the Jupyter Notebook: `bioasq_project_with_vectorDB.ipynb`.  

---

### **Pipeline**  
1. **Embedding Creation**: Extracts contexts from the BioASQ dataset and generates embeddings using `flax-sentence-embeddings/all_datasets_v3_mpnet-base`.  
2. **Vector Database**: Stores embeddings in Pinecone, enabling fast similarity-based querying.  
3. **Answer Generation**: Uses the `vblagoje/bart_lfqa` model for generating long-form answers.  
4. **Evaluation**: Evaluates generated answers using the ROUGE-1 metric for lexical similarity.  

---

### **Dependencies**

To run this project, ensure you have the following installed:  

- Required Python libraries:
  ```bash
  pip install -r requirements.txt
  ```

`requirements.txt` includes:
 * `sentence-transformers`
 * `transformers`
 * `pinecone-client`
 * `evaluate`
 * `rouge_score`
 * `numpy`
 * `pandas`

---

### Instructions

1. Clone the Repository
```bash
git clone https://github.com/Popescu-PfeifferMarc/ir-bioasq
cd ir-bioasq
```


2. Set Up Pinecone
   * Create a [Pinecone](https://app.pinecone.io/) account
   * Obtain your API key and region from the Pinecone dashboard.
   * Set up your Pinecone environment variables in the notebook.

3. Run the Jupyter Notebook
   * The main implementation is in `bioasq_project_with_vectorDB.ipynb`.
   1. Open the notebook:
      ```bash
      jupyter notebook bioasq_project_with_vectorDB.ipynb
      ```
   2. Follow the step-by-step instructions in the notebook to:
      * Generate embeddings.
      * Query the Pinecone database.
      * Generate answers using the `vblagoje/bart_lfqa` model.
      * Evaluate the results.

### Reproducibility

The notebook includes all steps to reproduce the results. The dataset, pre-trained models, and evaluation code are integrated into the pipeline.
