# IR-Project

## Requirements
- Python 3.12
- unzip
- jq

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
- open section "Datasets for task b"
- download the "12b golden enriched" zip archive
- extract the files `12B1_golden.json`, `12B2_golden.json`, `12B3_golden.json` and `12B4_golden.json`
- combine the files `jq -s '{questions: map(.questions) | add}' 12B1_golden.json 12B2_golden.json 12B3_golden.json 12B4_golden.json > 12B_golden_combined.json`
- move the resulting `12B_golden_combined.json` file to the dataset folder
- remove the four partial json files and the zip archive

### pubmed annual baseline 2024 (~20GB)
- run `dataset_pubmed_annual_baseline_download.py` to download and unpack XML files
- run `dataset_pubmed_annual_baseline_extract_xml.py` to extract relevant information from the XML files and combine them into a CSV file
- (optional) run `tail -n 100000 pubmed_annual_baseline.csv > pubmed_annual_baseline_tail_100k.csv`

### pubmed open access non commercial subset (~100GB)
Note that this dataset is not needed to reproduce the final results, as it was used only for experimentation. It is included anyways for completeness and potential further research.
- run `dataset_pumed_oa_noncomm_download.py` to download and unpack article XML files
- run `dataset_pubmed_oa_noncomm_extract_xml.py` to extract relevant information from the XML files and combine them into a CSV file
