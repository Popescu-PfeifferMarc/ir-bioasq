import logging
import nltk
import sys
import csv
import os
import json

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess

from bm25s import tokenize, BM25

# Check the current working directory
print("Current Working Directory:", os.getcwd())

# Change the current working directory
new_directory = "/Users/onurcanmemis/Desktop/Lectures/Information_Retrieval/Project/ir-bioasq"  # Replace with your desired path
os.chdir(new_directory)

# Verify the change
print("New Working Directory:", os.getcwd())

# Setup

articles_file = "./main_articles_head_1m.csv"
outp = "./out/"
golden_file_path = "./golden_data.json"
include_abstract=True
output_base_folder = "./out/taskA_BM25S"
output_folder = output_base_folder + "_title/" if not include_abstract else output_base_folder + "_title_abstract/"

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s", ' '.join(sys.argv))
os.makedirs(os.path.dirname(output_folder), exist_ok=True)

if not os.path.isdir(os.path.dirname(outp)):
    raise SystemExit("Error: The output directory does not exist. Create the directory and try again.")


# Corpus

lookup = {}
class MyCorpus:
    def __iter__(self):
        global lookup
        count = -1
        with open(articles_file, mode='r', newline='', errors='ignore') as inp_file:
            reader = csv.reader(inp_file)
            is_header = True
            for row in reader:
                if is_header:
                    is_header = False
                    continue
                count += 1
                pmid=row[0]
                doc_title=row[1]
                doc_abstract = row[2]
                lookup[count] = {"pmid":pmid,"title":doc_title,"abstract":doc_abstract}
                
                if include_abstract:
                    yield str(doc_title)+ "\n" + str(doc_abstract)
                else:
                    yield str(doc_title)
                
    def get_first_10_entries(self):
        entries = []
        for i, row in enumerate(self):
            if i >= 10:
                break
            entries.append(row)
        return entries
    
## Preprocessing
"""nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stoplist = set(stopwords.words('english'))
def preprocess(input: str):
    return [lemmatizer.lemmatize(entry) for entry in simple_preprocess(input) if entry not in stoplist];
"""

corpus = MyCorpus()
logger.info("Created corpus")
processed_corpus = tokenize([entry for entry in corpus])

# Creating the model
retriever = BM25()
retriever.index(processed_corpus)

# Retrieving relevant documents and snippets
output=[]
with open(golden_file_path, "r", encoding="utf-8") as golden_file_data:
    data = json.load(golden_file_data).get("questions")
    total_entries = len(data)
    logger.info(f"Loaded {total_entries} questions from golden data file")
    for doc_idx, entry in enumerate(data):
        logger.info(f"Querying: {(doc_idx / total_entries * 100):.2f}% ({doc_idx}/{total_entries})")
        query = entry['body']
        query_preprocessed = tokenize(query)
        doc, scores = retriever.retrieve(query_preprocessed, k=10)
        doc_score_mapping = {doc_id: score for doc_id, score in zip(doc[0], scores[0])}
        top_docs_indices = sorted(doc[0])
        relevant_documents = []
        snippets = []
        # score must be added
        for doc_id in top_docs_indices:
            if doc_id in lookup:
                pmid = lookup[doc_id]["pmid"]
                abstract = lookup[doc_id]["abstract"]
                
                relevant_documents.append({"pmid":f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}",
                                           "score":float(doc_score_mapping[doc_id])})

                # Process abstract for snippets
                sentences = sent_tokenize(abstract)
                for sentence in sentences:
                    snippets.append({
                        "text": sentence,
                        "source": pmid
                    })
        snippets_tokenized=tokenize([snippet["text"] for snippet in snippets])

        # Re-index the BM25 retriever with snippets
        retriever=BM25()
        retriever.index(snippets_tokenized)
        # Retrieve the top k most relevant snippets globally
        snippet_docs, snippet_scores = retriever.retrieve(query_preprocessed, k=10)

        # Extract the top k snippets with scores and source document IDs
        top_snippets = [
            {
                "text": snippets[idx]["text"],
                "pmid": snippets[idx]["source"],
                "score": float(snippet_scores[0, i])

            }
            for i, idx in enumerate(snippet_docs[0])
        ]
        output.append({"query": query, "relevant_documents": relevant_documents, "snippets": top_snippets})

with open(output_folder + "results.json", "w", encoding="utf-8") as outfile:
    json.dump({ 'questions': output }, outfile, ensure_ascii=False, indent=4)
    logger.info("Saved results to json")