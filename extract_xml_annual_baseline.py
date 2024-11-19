import xml.etree.ElementTree as ET
from typing import List, Dict
import gzip
from pathlib import Path
import sqlite3
from sqlite3 import Error
from typing import List, Dict, TypedDict, Optional
import time
from datetime import timedelta

file_template = "pubmed24n{:04d}.xml.gz"
dataset_folder = "./dataset/pubmed_baseline/"
db_file = "./dataset/pubmed_baseline_2024_processed.sqlite"
file_count = 1219

class PubMedArticle(TypedDict):
    pmid: int
    language: Optional[str]
    title: Optional[str]
    abstract: Optional[str]
    pubmed_id: Optional[str]
    doi: Optional[str]

def extract_pubmed_info(file_path: str) -> List[PubMedArticle]:
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
        
        # Extract PMID
        pmid = article.find('.//PMID')
        if pmid is None or not pmid.text.isdigit():
            print(f"Error: PMID is missing or not a number in article: {ET.tostring(article, encoding='unicode')}")
            continue
        article_data['pmid'] = int(pmid.text)
        
        # Extract Language
        language = article.find('.//Language')
        article_data['language'] = language.text if language is not None else None
        
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


def create_tables(conn):
    """Create the articles table"""
    try:
        sql = """
        CREATE TABLE IF NOT EXISTS articles (
            pmid INTEGER PRIMARY KEY,
            language TEXT,
            title TEXT,
            abstract TEXT,
            pubmed_id TEXT,
            doi TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
    except Error as e:
        print(f"Error creating table: {e}")

def insert_articles(conn, articles_data):
    cursor = conn.cursor()
    try:
        # Modified to use INSERT OR REPLACE
        sql = """
        INSERT OR REPLACE INTO articles (
            pmid, language, title, abstract, pubmed_id, doi, last_updated
        ) VALUES (
            :pmid, :language, :title, :abstract, :pubmed_id, :doi, CURRENT_TIMESTAMP
        )
        """
        
        cursor.executemany(sql, articles_data)
        conn.commit()
        
    except sqlite3.Error as e:
        print(f"Error inserting articles batch: {e}")
        conn.rollback()
    finally:
        cursor.close()

def main():
    conn = sqlite3.connect(db_file)
    create_tables(conn)
    start_time = time.time()
        
    for i in range(1, file_count + 1):
        current_time = time.time()
        elapsed_time = current_time - start_time
        if i > 1:
            avg_time_per_file = elapsed_time / (i - 1)
            remaining_files = file_count - i + 1
            estimated_remaining_time = avg_time_per_file * remaining_files
            eta = str(timedelta(seconds=int(estimated_remaining_time)))
        else:
            eta = "calculating..."
        
        in_file = dataset_folder + file_template.format(i)        
        articles_data = extract_pubmed_info(in_file)
        insert_articles(conn, articles_data)

        print (f"[{i}/{file_count}] Extracted {len(articles_data)} articles - Elapsed: {str(timedelta(seconds=int(elapsed_time)))} - ETA: {eta}")
    
    conn.close()
    print ("All files extracted successfully 🚀")
    total_elapsed_time = time.time() - start_time
    print(f"Total elapsed time: {str(timedelta(seconds=int(total_elapsed_time)))}")


if __name__ == "__main__":
    main()
