import xml.etree.ElementTree as ET
from typing import List, Dict
import json
import gzip
from pathlib import Path

file_template = "pubmed24n{:04d}.xml.gz"
dataset_folder = "./dataset/pubmed_baseline/"
file_count = 1219

def extract_pubmed_info(file_path: str) -> List[Dict]:
    # Un gzip file
    file_path = Path(file_path)
    if file_path.suffix == '.gz':
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            tree = ET.parse(f)
    else:
        tree = ET.parse(file_path)
    
    # Parse XML
    root = tree.getroot()
    articles_data = []
    
    # Find all PubmedArticle elements
    for article in root.findall('.//PubmedArticle'):
        article_data = {}
        
        # Extract Language
        language = article.find('.//Language')
        article_data['language'] = language.text if language is not None else None
        
        # Extract PMID
        pmid = article.find('.//PMID')
        article_data['pmid'] = pmid.text if pmid is not None else None
        
        # Extract ArticleTitle
        title = article.find('.//ArticleTitle')
        article_data['title'] = title.text if title is not None else None
        
        # Extract Abstract
        abstract = article.find('.//Abstract/AbstractText')
        article_data['abstract'] = abstract.text if abstract is not None else None
        
        # Extract ArticleId with type pubmed
        pubmed_id = article.find(".//ArticleId[@IdType='pubmed']")
        article_data['pubmed_id'] = pubmed_id.text if pubmed_id is not None else None
        
        # Extract ArticleId with type doi
        doi = article.find(".//ArticleId[@IdType='doi']")
        article_data['doi'] = doi.text if doi is not None else None
        
        articles_data.append(article_data)
    
    return articles_data

# Example usage
def main():
    for i in range(1, file_count + 1):
        in_file = dataset_folder + file_template.format(i)
        out_file = dataset_folder + file_template.format(i) + '_extracted.json'
        
        articles_data = extract_pubmed_info(in_file)
        with open(out_file, 'w') as json_file:
            json.dump(articles_data, json_file, indent=4)
        
        print (f"[{i}/{file_count}] Extracted {len(articles_data)} articles from {in_file} to {out_file}")
    print ("All files extracted successfully 🚀")

if __name__ == "__main__":
    main()
