import logging
import os
import sys
import csv
import json
import heapq

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.utils import simple_preprocess
from gensim.similarities import SparseMatrixSimilarity

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

# Config
DICT_SIZE = 100_000
golden_file = "./dataset/12B_golden_combined.json"
articles_file = "./dataset/main_articles_head_10k.csv"
include_abstract = True
output_base_folder = "./out/taskA_tfidf"
smartirs = "ntc" # alternatives are "ltc" and "Ltc"
output_smartirs_name = "large_ltc" if smartirs == "Ltc" else "small_ltc" if smartirs == "ltc" else smartirs # special case for Ltc and ltc to avoid casing
output_folder = output_base_folder + "_title/" if not include_abstract else output_base_folder + "_title_abstract/"
top_n = 10  # Number of top documents/snippets to retrieve
pubmed_baseurl = 'http://www.ncbi.nlm.nih.gov/pubmed/'

# Setup
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s", ' '.join(sys.argv))
os.makedirs(os.path.dirname(output_folder), exist_ok=True)

# Corpus
idx_lookup = {}
class DocumentCorpus:
    def __iter__(self):
        with open(articles_file, mode='r', newline='', errors='ignore') as inp_file:
            reader = csv.reader(inp_file)
            idx = 0
            for row in reader:
                doc_pmid = row[0]
                doc_title = row[1]
                doc_abstract = row[2]
                
                idx_lookup[idx] = doc_pmid
                idx += 1
                
                if include_abstract:
                    yield str(doc_title) + '\n' + str(doc_abstract)
                else:
                    yield str(doc_title)

# Pre-process
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stoplist = set(stopwords.words('english'))
def preprocess(input: str):
    return [lemmatizer.lemmatize(entry) for entry in simple_preprocess(input) if entry not in stoplist]

# Dictionary
corpus = DocumentCorpus()
dictionary = Dictionary(preprocess(doc) for doc in corpus)
dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=DICT_SIZE)
dictionary.compactify()
dictionary.save_as_text(output_folder + '.dictionary')
logger.info("Created dict")

# dictionary = Dictionary.load_from_text(outp + '.dictionary')
# logger.info("Loaded dictionary")

# BOW corpus
class BowCorpus:
    def __iter__(self):
        corpus = DocumentCorpus()
        for doc in corpus:
            yield dictionary.doc2bow(preprocess(doc))

# TF-IDF model
bow_corpus = BowCorpus()
tfidf = TfidfModel(bow_corpus, smartirs=smartirs)
tfidf.save(output_folder + '.model_tfidf_' + output_smartirs_name)
logger.info("Created TfidfModel with smartirs: %s", smartirs)

index = SparseMatrixSimilarity(tfidf[bow_corpus], num_features=len(dictionary))
index.save(output_folder + '.simmat_tfidf_' + output_smartirs_name)
logger.info("Created SparseMatrixSimilarity with smartirs: %s", smartirs)

# Run queries from golden data
output = []
article_idx_to_load = set()
with open(golden_file, "r", encoding="utf-8") as golden_file_data:
    data = json.load(golden_file_data).get("questions")
    total_entries = len(data)
    logger.info(f"Loaded {total_entries} questions from golden data file")
    
    for doc_idx, entry in enumerate(data):
        logger.info(f"Querying: {(doc_idx / total_entries * 100):.2f}% ({doc_idx}/{total_entries})")
        
        query = entry['body']
        query_preprocessed = preprocess(query)
        query_bow = dictionary.doc2bow(query_preprocessed)
        
        sims = index[tfidf[query_bow]]
        top_sims = heapq.nlargest(top_n, enumerate(sims), key=lambda x: x[1])
        for article_idx, _ in top_sims:
            article_idx_to_load.add(article_idx) # Add all article indicies to load
        
        output.append({
            'query': query,
            # 'query_preprocessed': query_preprocessed,
            # 'query_bow': query_bow,
            'documents': [{ 'index': idx, 'score': float(sim) } for idx, sim in top_sims],
            'snippets': [] # processed later
        })

# Load abstracts
article_data = {}
with open(articles_file, mode='r', newline='', errors='ignore') as inp_file:
    reader = csv.reader(inp_file)
    doc_idx = -1
    for row in reader:
        doc_idx += 1
        if doc_idx not in article_idx_to_load:
            continue
        
        doc_pmid = row[0]
        doc_title = row[1]
        doc_abstract = row[2]
        
        article_data[doc_idx] = {
            'pmid': doc_pmid,
            'title': doc_title,
            'abstract': doc_abstract
        }
logger.info(f"Loaded {len(article_data)} article details")

# Add PMIDs to output
for entry in output:
    for doc in entry['documents']:
        doc_idx = doc['index']
        doc['pmid'] = pubmed_baseurl + article_data[doc_idx]['pmid']
        # doc['title'] = article_data[idx]['title']
logger.info("Added PMIDs to output")

# Add snippets to output
for idx, entry in enumerate(output):
    snippets = []
    logger.info(f"Querying snippets: {(idx / len(output) * 100):.2f}% ({doc_idx}/{total_entries})")
    for doc in entry['documents']:
        doc_idx = doc['index']
        pmid = article_data[doc_idx]['pmid']
        abstract = article_data[doc_idx]['abstract']
        sentences = sent_tokenize(abstract)
        
        for sentence in sentences:
            sentence_preprocessed = preprocess(sentence)
            sentence_bow = dictionary.doc2bow(sentence_preprocessed)
            sentence_tfidf = tfidf[sentence_bow]
    
            snippet_score = sum(
                s for term_id, s in sentence_tfidf if term_id in [dictionary.token2id.get(token) for token in sentence_preprocessed]
            )
            snippets.append({ "text": sentence, "pmid": pmid, "score": float(snippet_score) })
    
    top_snippets = heapq.nlargest(top_n, snippets, key=lambda x: x['score'])
    entry['snippets'] = top_snippets
logger.info("Added snippets to output")

# Remove the index from the output
for entry in output:
    for doc in entry['documents']:
        del doc['index']
logger.info("Removed indicies from output")



with open(output_folder + "results.json", "w", encoding="utf-8") as outfile:
    json.dump({ 'questions': output }, outfile, ensure_ascii=False, indent=4)
    logger.info("Saved results to json")

logger.info("All done 🚀")
