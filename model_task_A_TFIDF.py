import logging
import os
import sys
import csv

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.utils import simple_preprocess
from gensim.similarities import SparseMatrixSimilarity

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Config
DICT_SIZE = 100_000
articles_file = "./dataset/pubmed_annual_baseline.csv"
include_abstract = True
output_base_folder = "./out/taskA_tfidf"
smartirs = "ntc" # alternatives are "ltc" and "Ltc"
output_smartirs_name = "large_ltc" if smartirs == "Ltc" else "small_ltc" if smartirs == "ltc" else smartirs # special case for Ltc and ltc to avoid casing
output_folder = output_base_folder + "_title/" if not include_abstract else output_base_folder + "_title_abstract/"

# Setup
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s", ' '.join(sys.argv))
os.makedirs(os.path.dirname(output_folder), exist_ok=True)

# Corpus
class DocumentCorpus:
    def __iter__(self):
        with open(articles_file, mode='r', newline='', errors='ignore') as inp_file:
            reader = csv.reader(inp_file)
            for row in reader:
                # doc_pmid = row[0]
                doc_title = row[1]
                doc_abstract = row[2]
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
"""
dictionary = Dictionary.load_from_text(outp + '.dictionary')
logger.info("Loaded dictionary")
"""

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
logger.info("----- Created TfidfModel with smartirs: %s", smartirs)

index = SparseMatrixSimilarity(tfidf[bow_corpus], num_features=len(dictionary))
index.save(output_folder + '.simmat_tfidf_' + output_smartirs_name)
logger.info("----- Created SparseMatrixSimilarity with smartirs: %s", smartirs)

logger.info("All done 🚀")
