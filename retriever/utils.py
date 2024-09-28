from pathlib import Path
import uuid
import re
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader


def write_json(data, upload_folder):
    path = Path(upload_folder) / Path(str(uuid.uuid4()) + '.json')
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

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