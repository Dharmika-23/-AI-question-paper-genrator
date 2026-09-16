# utils.py

import re
import nltk
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

# Download nltk resources if not already present
nltk.download('punkt', quiet=True)

# Load sentence transformer model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text(uploaded_file):
    """Extract text from PDF or TXT file."""
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text()
    elif uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
    else:
        text = ""
    return text.strip()

def split_sentences(text):
    """Split text into sentences."""
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]

def top_k_relevant_sentences(sentences, topic, k=5):
    """Find top-k relevant sentences to the given topic."""
    topic_embedding = embedding_model.encode(topic, convert_to_tensor=True)
    sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(topic_embedding, sentence_embeddings)[0]
    top_results = cosine_scores.argsort(descending=True)[:k]
    return [sentences[i] for i in top_results]
