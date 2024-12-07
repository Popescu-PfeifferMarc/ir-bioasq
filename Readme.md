# IR-Project

## Requirements
- Python `3.12` incl. library requirements from requirements.txt
- [ollama](https://ollama.com/) incl. models as specified in model_task_B_LLM.py
- `unzip` or equivalent to unpack zip files
- `jq` or equivalent to combine JSON files

## Dataset

### Bioasq training data (~38MB)
- create BioASQ account: [signup](http://participants-area.bioasq.org/accounts/register/)
- open http://participants-area.bioasq.org/datasets/
- open section "Datasets for task b"
- download the "Training 12b" zip archive
- extract the file training12b_new.json to the dataset folder
- remove the zip archive 

### Bioasq golden data (~7MB)
- create BioASQ account: [signup](http://participants-area.bioasq.org/accounts/register/)
- open http://participants-area.bioasq.org/datasets/
- open section "Datasets for task b"
- download the "12b golden enriched" zip archive
- extract the files `12B1_golden.json`, `12B2_golden.json`, `12B3_golden.json` and `12B4_golden.json`
- combine the files using `jq -s '{questions: map(.questions) | add}' 12B1_golden.json 12B2_golden.json 12B3_golden.json 12B4_golden.json > 12B_golden_combined.json`
- move the resulting `12B_golden_combined.json` file to the dataset folder
- remove the four partial json files and the zip archive

### pubmed annual baseline 2024 (~20GB)
- run `dataset_pubmed_annual_baseline_download.py` to download and unpack XML files
- run `dataset_pubmed_annual_baseline_extract_xml.py` to extract relevant information from the XML files and combine them into a CSV file
- (optional) run `tail -n 100000 pubmed_annual_baseline.csv > pubmed_annual_baseline_tail_100k.csv` to get a subset of the dataset

### pubmed open access non commercial subset (~100GB)
Note that this dataset is not needed to reproduce the final results, as it was used only for experimentation. It is included anyways for completeness and potential further research.
- run `dataset_pumed_oa_noncomm_download.py` to download and unpack article XML files
- run `dataset_pubmed_oa_noncomm_extract_xml.py` to extract relevant information from the XML files and combine them into a CSV file

## Models

### Task A - TF-IDF
 - see `model_task_A_tfidf.py`

### Task A - BM25
 - see `model_task_A_bm25.py`

### Task B - LLM
 - see `model_task_B_LLM.py`

### Task A&B - Vector DB
 - see `bioasq_project_with_vectorDB.ipynb`
