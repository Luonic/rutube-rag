import os
import nltk
import string
from pathlib import Path
from razdel import tokenize
from transformers import AutoTokenizer
from snowballstemmer import stemmer
from stop_words import get_stop_words
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader

CURRENT_DIR = Path(os.getcwd())
RAG_DIR = CURRENT_DIR / 'rag'

TOKENIZER_PATH = 'intfloat/multilingual-e5-base'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

russian_stopwords = get_stop_words('ru')
stemmer = stemmer('russian')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = tokenize(text)
    tokens = [token.text for token in tokens]
    tokens = [stemmer.stemWord(token) for token in tokens if token not in russian_stopwords]
    processed_text = ' '.join(tokens)
    return processed_text

def read_docs(path, delimiter=';'):
    loader = CSVLoader(
        path, 
        source_column='answer', 
        metadata_columns='question;answer;classifier_1;classifier_2'.split(delimiter), 
        csv_args={'delimiter': delimiter, 'quotechar': '"'}
        )

    docs = loader.load()
    for doc in docs:
        doc.page_content = doc.metadata['answer']
    return docs

def split_docs(docs, count_tokens, chunk_size=256, chunk_overlap=20):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        is_separator_regex=False,
    )

    splits = text_splitter.split_documents(docs)
    return splits

def count_tokens(text):
    return len(tokenizer.tokenize(text, add_special_tokens=False))

def format_docs_with_score(docs):
    print(len(docs))
    concateneted_docs_string = "\n\n".join('\n'.join([f'Id источника: {i}', doc.metadata['question'], doc.page_content]) for i, (doc, score) in enumerate(docs))
    print(concateneted_docs_string)
    print()
    return concateneted_docs_string

def format_docs(docs):
    print(len(docs))
    concateneted_docs_string = "\n\n".join('\n'.join([f'Id источника: {i}', doc.metadata['question'], doc.page_content]) for i, doc in enumerate(docs))
    print(concateneted_docs_string)
    print()
    return concateneted_docs_string
    
def preprocess_text(text):
    text = text.replace('\u2060', '')
    return text

