# IR-Project

## Requirements
- Python `3.12` (with library from requirements.txt)
- Jupyter
- [ollama](https://ollama.com/) incl. models as specified in `model_task_B_LLM.py`
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
To build a dicitonary, TFIDF model and similarity matrix, then query all documents in the golden file:
 - open `model_task_A_TFIDF.py`
 - configure `smartirs` - `ntc`, `ltc` or `Ltc`; see [gensim docs](https://radimrehurek.com/gensim/models/tfidfmodel.html) for details
 - configure `include_abstract` - `True` or `False`; should the model also include the abstract of the articles or only the title
 - run `model_task_A_TFIDF.py` - checkpoints are available for the dictionary, model and similarity matrix; comment in/out relevant lines in script

To evalute the results from the previous step
 - open `evalute_task_A.py` 
 - configure `file_path` - set it the the results.json produced by `model_task_A_TFIDF.py`
 - configure `output_file_path` - set it to where the results should be saved to
 - run `evalute_task_A.py`

### Task A 1.2 - BM25
To build a BM25 model, then query all documents in the golden file:
 - open `model_task_A_BM25.py`
 - configure `include_abstract` - `True` or `False`; should the model also include the abstract of the articles or only the title

To evalute the results from the previous step
 - open `evalute_task_A.py`
 - configure `file_path` - set it the the results.json produced by `model_task_A_BM25.py`
 - configure `output_file_path` - set it to where the results should be saved to
 - run `evalute_task_A.py`

### Task B 2.1 - LLM
To run a model and query all golden sample questions against either the results from task A or the golden snippets (or both at the same time).
 - install [Ollama](https://ollama.com/)
 - start the ollama service `ollama serve` - or equivalent command depending on your platform
 - pull the model `ollama pull llama3.1:8b`- other model such as `jsk/bio-mistral` or `llama3.1:70b` are also possible
 - open `model_task_B_LLM.py`
 - configure `ollama_model` - set it to the model you pulled in the previous step
 - configure `do_taskA_results` - `True` or `False`; set to `True` if you want to test the model against the results from task A
 - configure `input_taskA_results` - results from task A; only relevant if `do_taskA_results` is set to `True`
 - configure `do_golden` - `True` or `False`; set to `True` if you want to test the model against the golden data snippets

### Task B 2.2 - Retriever Model Leveraging Embeddings and Vector Database
This project implements a biomedical question-answering (QA) system by leveraging embeddings and a vector database for efficient document retrieval, paired with a generative model for answer generation. The main implementation is contained in the Jupyter Notebook: `bioasq_project_with_vectorDB.ipynb`.  

#### Pipeline 
1. **Embedding Creation**: Extracts contexts from the BioASQ dataset and generates embeddings using `flax-sentence-embeddings/all_datasets_v3_mpnet-base`.  
2. **Vector Database**: Stores embeddings in Pinecone, enabling fast similarity-based querying.  
3. **Answer Generation**: Uses the `vblagoje/bart_lfqa` model for generating long-form answers.  
4. **Evaluation**: Evaluates generated answers using the ROUGE-1 metric for lexical similarity.  

#### Instructions

1. Set Up Pinecone
   * Create a [Pinecone](https://app.pinecone.io/) account
   * Obtain your API key and region from the Pinecone dashboard.
   * Set up your Pinecone environment variables in the notebook.

2. Run the Jupyter Notebook
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
