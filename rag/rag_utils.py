import os
import gc
from typing import List
from operator import itemgetter

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.system import SystemMessage
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.load import dumps, loads

from rag.utils import *
from rag.prompts import *

prompt = ChatPromptTemplate.from_messages([
#     SystemMessage("""
# Вы — помощник для пользователей видеохостинга Rutube. Основные принципы работы заключаются в следующем:
# 1. Избегание галлюцинаций: Вы должны использовать только информацию, полученную из контекста. Если контекст не содержит ответа на вопрос, необходимо ответить "Я не знаю" или перевести разговор в дружеский тон, если вопрос носит общий характер.
# 2. Культурный фильтр: В ответах строго запрещено использовать ненормативную лексику или неправомерные высказывания. Все ответы должны быть вежливыми, корректными и соответствовать нормам общения.
# 3. Внимание к деталям: Ваши ответы должны быть точными, полными и проверенными. Включайте факты только из предоставленного контекста и всегда следите за правильностью дат, имен, аббревиатур и других фактических данных.
# Если вопрос не касается видеохостинга Rutube или он задается на отвлеченные и общие темы то, ваш ответ должен быть: 'Я помощник видеохостинга Rutube, отвечаю на вопросы только касаемо этого видеохостинга.'
# Используйте предоставленный контекст для ответа на вопросы пользователей, избегая логических и фактологических ошибок.
# Если ответа на вопрос в контексте нет, то ответ возвращается в стиле 'Релевантной информации к вопросу в предоставленном контексте не обнаружено'.
# """), 
    SystemMessage("""
Вы — помощник для пользователей видеохостинга Rutube. Основные принципы работы заключаются в следующем:
1. Избегание галлюцинаций: Вы должны использовать только информацию, полученную из контекста. Если контекст не содержит ответа на вопрос, необходимо ответить "Я не знаю" или перевести разговор в дружеский тон, если вопрос носит общий характер.
2. Культурный фильтр: В ответах строго запрещено использовать ненормативную лексику или неправомерные высказывания. Все ответы должны быть вежливыми, корректными и соответствовать нормам общения.
3. Внимание к деталям: Ваши ответы должны быть точными, полными и проверенными. Включайте факты только из предоставленного контекста и всегда следите за правильностью дат, имен, аббревиатур и других фактических данных.
"""),
    ("user", RAG_PROMPT)
    ])

multiquery_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(MULTIQUERY_PROMPT), 
    ("user", "Исходный вопрос:\n{question}\nJSON:")
])

# class StructuredOutput(BaseModel):
#     """
# Структурированный ответ на вопрос пользователя, основанный только на релевантных источниках.
# Если ответа на вопрос в контексте нет, то идентификаторы источников не указываются, а ответ возвращается в стиле 'Релевантной информации к вопросу в предоставленном контексте не обнаруженно'."""
#     answer: str = Field(description = """Ответ на вопрос пользователя проверенный на релевантность.""")
#     citations: List[int] = Field(description="""Список целочисленных идентификаторов источников, которые обосновывают ответ.""")
    
class StructuredOutput(BaseModel):
    """
Структурированный ответ на вопрос пользователя, основанный на ответе и списке релевантных источников.
    """
    answer: str = Field(description="""Ответ на вопрос пользователя, проверенный на релевантность.""")
    citations: List[int] = Field(description="""Список целочисленных Id источников из контекста, которые являются наиболее релевантными к вопросу. Список пустой, если вопрос был нерелевантен контексту в принципе.""")


def init_llm(config):
    llm = ChatOpenAI(model=config['MODEL_NAME'],
                     temperature=config['temperature'],
                     base_url=config['LLM_BASE_URL'])
    return llm
    

def init_retriever(data_path: str, embedding_models: list, config: dict):
    docs = read_docs(data_path)
    splits = split_docs(docs, count_tokens)

    # bm25_splits = []
    # for split in splits:
        # split.page_content = preprocess_text(split.page_content)
        # bm25_splits.append(split)

    retrievers = []
    
    bm25_retriever = BM25Retriever.from_documents(
        documents=splits,
    )
    bm25_retriever.k = config['bm25_retrieve_k']
    retrievers.append(bm25_retriever)
    
    for i, embedding_model in enumerate(embedding_models):
        vectorstore = Chroma.from_documents(documents=splits, 
                                            embedding=embedding_model)
        chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": config['vectors_retrieve_k']})
        
        retrievers.append(chroma_retriever)
    
    count_retrievers = len(retrievers)
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=retrievers, weights=[1 / count_retrievers] * count_retrievers
    )
    gc.collect()
    return ensemble_retriever

    
def reciprocal_rank_fusion(results: list[list], k=60, top_k=8):
    
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    reranked_results = reranked_results[:top_k]
    return reranked_results


def init_multiquery_rag_chain(prompt, multiquery_prompt, retriever, reranker, llm):
    def rerank_results(questions, answer, top_k=2):
        return reranker.rerank_questions(questions, answer, top_k)
    
    generate_queries = (
        multiquery_prompt 
        | llm
        | JsonOutputParser() 
        | (lambda x: x["queries"])
    )

    retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion
    # retrieval_chain = (lambda x: [x['question']]) | retriever.map() | reciprocal_rank_fusion

    structured_llm = llm.with_structured_output(StructuredOutput)
    
    final_rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs_with_score(x["context"])))
        | prompt
        | structured_llm
    )

    def retrieve_docs(input):
        question = (lambda x: x["question"])(input)
        docs = retrieval_chain.invoke(input)
        # return docs
        reranked_docs = rerank_results(docs, question)
        return reranked_docs
        
    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=final_rag_chain
    )

    return chain
    
 
def invoke_multiquery_rag_chain(rag_chain, question, standard=False):
    question = question.strip()
    if not question.endswith('?'):
        question = question + '?'
    answer_data = rag_chain.invoke({'question': question})
    text_answer = answer_data['answer'].answer
    citations_id = answer_data['answer'].citations
    print(question)
    print(answer_data)
    print(citations_id)
    print('-------')
    if len(citations_id):
        classifier_1 = answer_data['context'][0][0].metadata['classifier_1']
        classifier_2 = answer_data['context'][0][0].metadata['classifier_2']
    else:
        classifier_1 = ''
        classifier_2 = ''

    if standard:
        return {'answer': text_answer, 'class_1': classifier_1, 'class_2': classifier_2}
    else: 
        context = list([' '.join([context_doc.page_content]) \
                        for i, (context_doc, score) in enumerate(answer_data['context']) if i in citations_id])
        return {'answer': text_answer, 'class_1': classifier_1, 'class_2': classifier_2, 'contexts': context}
