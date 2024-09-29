from __future__ import annotations

from typing import Union

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Union

from api_models import HTTPValidationError, Request, Response, ResponseWithContext

import torch
import yaml
import os
import asyncio
from yaml.loader import SafeLoader
from pathlib import Path
from rag.rag_utils import *
from rag.embedding_model import init_embeddings
from reranker.handler import init_reranker

CURRENT_DIR = Path(os.getcwd())
RAG_DIR = CURRENT_DIR / 'rag'
RERANKER_DIR = CURRENT_DIR / 'reranker'
RETRIEVAL_DIR = CURRENT_DIR / 'retriever'

# Load config
CONFIG_FILENAME = 'config.yaml'
CONFIG = yaml.load(open(RAG_DIR / CONFIG_FILENAME, 'r'), Loader=SafeLoader)

# Load parameters
RETRIEVAL_MODEL_PATHS = CONFIG['RETRIEVER']['RETRIEVAL_MODEL_PATHS']
DATA_PATH = RERANKER_DIR / CONFIG['DATA_PATH']
RETRIEVER_TOKENIZER_PATH = CONFIG['RETRIEVER']['TOKENIZER_PATH']
# RERANKER_TOKENIZER_PATH = RERANKER_DIR / CONFIG['RERANKER']['RERANKER_TOKENIZER_PATH']

# Init all
llm = init_llm(CONFIG['LLM'])
embeddings = [init_embeddings(path, RETRIEVER_TOKENIZER_PATH, device=torch.device('cuda:0')) for path in RETRIEVAL_MODEL_PATHS]
retriever = init_retriever(DATA_PATH, embeddings, CONFIG['RAG'])
reranker = init_reranker(CONFIG['RERANKER'])
rag_chain = init_multiquery_rag_chain(prompt, multiquery_prompt, retriever, reranker, llm)

# Init app
app = FastAPI(
    title='Assistant API',
    version='0.1.0',
)

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 500:
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error. Please try again later."},
        )
    return await request.app.default_exception_handler(request, exc)


@app.post(
    '/get_answer',
    response_model=Response,
    responses={'422': {'model': HTTPValidationError}},
    tags=['default'],
)
async def get_answer(body: Request) -> Union[Response, HTTPValidationError]:
    """
    Assist
    """
    question = body.question
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            result_dict = invoke_multiquery_rag_chain(rag_chain, question, True)
            return result_dict
        except Exception as e:
            if attempt < max_retries:
                continue
            raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post(
    '/get_answer_with_context',
    response_model=ResponseWithContext,
    responses={'422': {'model': HTTPValidationError}},
    tags=['default'],
)
async def get_answer_with_context(body: Request) -> Union[ResponseWithContext, HTTPValidationError]:
    """
    Assist
    """
    question = body.question
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            result_dict = invoke_multiquery_rag_chain(rag_chain, question, False)
            return result_dict
        except Exception as e:
            if attempt < max_retries:
                continue
            raise HTTPException(status_code=500, detail="Internal Server Error")
