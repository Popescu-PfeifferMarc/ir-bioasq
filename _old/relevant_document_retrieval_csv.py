import pandas as pd
import numpy as np
import gensim
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from rank_bm25 import BM25Okapi  # BM25 library
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss  # For approximate nearest neighbor search
import torch
from bm25s import tokenize, BM25


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class QuerySpecificTFIDFModelLogarithmic:
    def __init__(self):
        # Initialize stop words and lemmatizer
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.dictionary = None
        self.tfidf_model = None
        self.index = None

    def preprocess_text(self, text):
        """
        Preprocesses text by tokenizing, removing stopwords, and lemmatizing.
        """

        ######## THIS PART IS CHANGED FROM word_tokenize to .split() it is 2x faster!!!
        ############################

        
        tokens = text.lower().split()  # Tokenize and convert to lowercase
        tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]  # Remove stop words and punctuation
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]  # Apply lemmatization
        return tokens

    def load_and_preprocess_documents(self, file_path):
        """
        Loads and preprocesses documents from a CSV file.
        """
        print(f"Loading documents from {file_path}")
        df = pd.read_csv(file_path)
        df = df.dropna(subset=["pmid", "title", "abstract"])  # Drop rows with missing values

        # Combine title and abstract, and preprocess
        df["raw_text"] = df["title"] + " " + df["abstract"]
        df["tokens"] = df["raw_text"].apply(self.preprocess_text)
        return df

    def build_tfidf_model(self, documents):
        """
        Builds the TF-IDF model and similarity index using Gensim.
        """
        print("Building TF-IDF model...")
        self.dictionary = Dictionary(documents["tokens"])  # Create a Gensim dictionary
        corpus = [self.dictionary.doc2bow(tokens) for tokens in documents["tokens"]]  # Convert to bag-of-words format
        self.tfidf_model = TfidfModel(corpus,smartirs='lfu')  # Build the TF-IDF model
        self.index = SparseMatrixSimilarity(self.tfidf_model[corpus], num_features=len(self.dictionary))  # Build similarity index
        return corpus

    def calculate_relevance(self, query, corpus):
        """
        Calculates relevance scores for the query against the corpus.
        """
        print("Calculating relevance scores...")
        query_tokens = self.preprocess_text(query)
        query_bow = self.dictionary.doc2bow(query_tokens)  # Convert query to bag-of-words
        query_tfidf = self.tfidf_model[query_bow]  # Convert query to TF-IDF
        similarities = self.index[query_tfidf]  # Compute similarities
        return similarities

    def rank_snippets(self, query, top_documents, top_n_snippets=10):
        """
        Extracts and ranks snippets globally based on similarity to the query.
        """
        print("Ranking snippets globally...")
        snippets = []
        query_tokens = self.preprocess_text(query)

        for _, doc in top_documents.iterrows():
            pmid, text = doc["pmid"], doc["raw_text"]
            sentences = sent_tokenize(text)
            for sentence in sentences:
                sentence_tokens = self.preprocess_text(sentence)
                sentence_bow = self.dictionary.doc2bow(sentence_tokens)
                sentence_tfidf = self.tfidf_model[sentence_bow]
                snippet_score = sum(
                    score for term_id, score in sentence_tfidf if term_id in [self.dictionary.token2id.get(token) for token in query_tokens]
                )
                snippets.append({
                    "text": sentence,
                    "source": pmid,
                    "score": snippet_score,
                })

        # Sort snippets globally
        top_snippets = sorted(snippets, key=lambda x: x["score"], reverse=True)[:top_n_snippets]
        return top_snippets

    def get_relevant_documents_and_snippets(self, query, file_path, top_n_docs=10, top_n_snippets=10):
        """
        Retrieves the top N relevant documents and globally ranked snippets.
        """
        # Load and preprocess documents
        documents = self.load_and_preprocess_documents(file_path)
        if len(documents) == 0:
            print("No valid documents found.")
            return [], []

        # Build the TF-IDF model
        corpus = self.build_tfidf_model(documents)

        # Calculate relevance scores
        relevance_scores = self.calculate_relevance(query, corpus)

        # Retrieve the indices of the top N documents
        print("Retrieving top documents...")
        top_indices = np.argsort(relevance_scores)[-top_n_docs:][::-1]  # Get indices of top N scores in descending order

        # Create a DataFrame with the top N documents
        top_documents = documents.iloc[top_indices].copy()
        top_documents["score"] = [relevance_scores[idx] for idx in top_indices]

        # Rank snippets globally
        top_snippets = self.rank_snippets(query, top_documents, top_n_snippets)

        return top_documents, top_snippets

file_path = "./main_articles_head1000.csv"
query = "Effects of interferon on viral infections"

# Instantiate the model
model = QuerySpecificTFIDFModelLogarithmic()

# Get relevant documents and snippets
top_documents, top_snippets = model.get_relevant_documents_and_snippets(query, file_path, top_n_docs=10, top_n_snippets=10)

# Display results
print("Top Documents:")
for doc in top_documents.to_dict('records'):
    print(f"PMID: {doc['pmid']}, Score: {doc['score']:.4f}")

print("\nTop Snippets:")
for snippet in top_snippets:
    print(f"Snippet: {snippet['text']}, Score: {snippet['score']:.4f}")
