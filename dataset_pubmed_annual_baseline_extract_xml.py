import re
import os
import csv
import xml.etree.ElementTree as ET
from typing import Optional, TypedDict


dataset_in_folder = "./dataset/pubmed_annual_baseline"
dataset_out_file = "./dataset/pubmed_annual_baseline.csv"

class ArticleData(TypedDict):
    pmid: str
    title: str
    lang: str
    abstract: Optional[str]

def extract_xml(path: str) -> list[ArticleData]:
    tree = ET.parse(path)
    root = tree.getroot()
    articles = []
    
    for article in root.findall('.//PubmedArticle'):
        pmid = article.findtext('.//PMID')
        title = article.findtext('.//ArticleTitle')
        lang = article.findtext('.//Language')
        abstract = article.findtext('.//Abstract/AbstractText')

        articles.append({
            'pmid': pmid,
            'title': title,
            'lang': lang,
            'abstract': abstract
        })

    return articles

def main():
    with open(dataset_out_file, mode='w', encoding='utf-8') as file:
        writer = csv.writer(file)

        for filename in os.listdir(dataset_in_folder):
            if not filename.endswith(".xml"):
                print(f"Skipping non-XML file: {filename}")
                continue
            
            articles_data = extract_xml(os.path.join(dataset_in_folder, filename))
            
            for article_data in articles_data:
                if article_data['lang'] != 'eng':
                    continue # Skip non-English articles
                
                if article_data['abstract'] is None:
                    continue # Skip articles without abstracts
                
                writer.writerow([article_data["pmid"], article_data["title"], article_data["abstract"]])

if __name__ == "__main__":
    main()
