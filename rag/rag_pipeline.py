import torch
import yaml
import gc
from yaml.loader import SafeLoader
from tqdm import tqdm

from rag_utils import *
from embedding_model import init_embeddings
from handler import init_reranker

config = yaml.load(open('config.yaml', 'r'), Loader=SafeLoader)

RERANKER_MODEL_PATHS = config['RERANKER']['RERANKER_MODEL_PATHS']
RETRIEVAL_MODEL_PATHS = config['RETRIEVER']['RETRIEVAL_MODEL_PATHS']
DATA_PATH = config['DATA_PATH']
RERANKER_TOKENIZER_PATH = config['RETRIEVER']['TOKENIZER_PATH']


if __name__ == "__main__":
    llm = init_llm()
    embeddings = [init_embeddings(path, TOKENIZER_PATH, device=torch.device('cuda:0')) for path in tqdm(RETRIEVAL_MODEL_PATHS)]
    gc.collect()
    retriever = init_retriever(DATA_PATH, embeddings, config['RAG'])
    gc.collect()
    reranker = init_reranker(config['RERANKER'])
    gc.collect()
    rag_chain = init_multiquery_rag_chain(prompt, multiquery_prompt, retriever, reranker, llm)
    question = 'Как удалить канал на рутубе?'
    result_dict = invoke_multiquery_rag_chain(rag_chain, question)
    print(result_dict)