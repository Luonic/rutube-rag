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

from utils import *


prompt = ChatPromptTemplate.from_messages([
    SystemMessage("Вы помощник для выполнения ответов на вопросы пользователей видеохостинга Rutube."), 
    ("user", """Используйте следующие части полученного контекста для ответа на поставленный вопрос. 
     Ответ должен быть максимально точным, правильным и релевантным вопросу и контексту. 
     Не совершайте фактологических и логических ошибок относительно контекста и вопроса.
     Если в контексте нет ответа на поставленный вопрос, скажите, что не знаете, либо ведите диалог с пользователем в режиме болталки, если он задает общие вопросы.
     Вопрос: {question}
     Контекст: {context}""")
    ])

multiquery_prompt = ChatPromptTemplate.from_messages([
    SystemMessage("""Вы помощник языковой модели искусственного интеллекта для ответов на вопросы пользователей видеохостинга Rutube. 
                  Ваша задача — сгенерировать пять разных, но близких по смыслу версий заданного вопроса для извлечения соответствующих документов из векторной базы данных. 
                  Можно использовать парафразы или синонимы к понятиям из вопроса, но с высокой степенью релевантности к первоначальному вопросу. 
                  Укажите исходный вопрос и 5 альтернативных вопросов в JSON следующего формата: {"queries": ["Исходный вопрос", "Альтернативный вопрос 1", ..., "Альтернативный вопрос 5"]}."""), 
    ("user", "Исходный вопрос: {question}")
])

class StructuredOutput(BaseModel):
    """Ответьте на вопрос пользователя, основываясь только на указанных источниках, и укажите использованные источники.
    Источники должны быть полностью релевантны к вопросу, если источник нерелевантен вопросу, то его использовать для генерации ответа нельзя!
    Если ответа на вопрос в контексте нет, то идентификаторы источников указывать также не нужно."""
    answer: List[str] = Field(description = """Ответ на вопрос пользователя, основанный только на релевантных источниках.""")
    citations: List[int] = Field(description="""Список целочисленных идентификаторов источников, которые обосновывают ответ.""")
    

def init_llm():
    llm = ChatOpenAI(model="gpt-4o", openai_proxy=os.getenv('PROXY'))
    return llm
    

def init_retriever(data_path: str, embedding_models: list, config: dict):
    docs = read_docs(data_path)
    splits = split_docs(docs, count_tokens)

    bm25_splits = []
    for split in splits:
        split.page_content = preprocess_text(split.page_content)
        bm25_splits.append(split)

    retrievers = []
    
    bm25_retriever = BM25Retriever.from_documents(
        documents=bm25_splits,
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

    
def reciprocal_rank_fusion(results: list[list], k=60):
    
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

    return reranked_results
    

def init_multiquery_rag_chain(prompt, multiquery_prompt, retriever, llm):
    
    generate_queries = (
        multiquery_prompt 
        | llm
        | JsonOutputParser() 
        | (lambda x: x["queries"])
    )

    retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

    structured_llm = llm.with_structured_output(StructuredOutput)
    
    final_rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs_with_score(x["context"])))
        | prompt
        | structured_llm
    )

    retrieve_docs = (lambda x: x["question"]) | retrieval_chain

    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=final_rag_chain
    )

    return chain
    
 
def invoke_multiquery_rag_chain(rag_chain, question, return_context=False):
    answer_data = rag_chain.invoke({'question': question})
    text_answer = answer_data['answer'].answer
    citations_id = answer_data['answer'].citations
    classifier_1_lst = []
    classifier_2_lst = []
    for i, (context_doc, score) in enumerate(answer_data['context']):
        if i in citations_id:
            classifier_1_lst.append(context_doc.metadata['classifier_1'])
            classifier_2_lst.append(context_doc.metadata['classifier_2'])

    if not return_context:
        return {'text': text_answer, 'classifier_1': classifier_1_lst, 'classifier_2': classifier_2_lst}
    else: 
        context = list([' '.join([context_doc.metadata['question'], context_doc.page_content]) for i, (context_doc, score) in enumerate(answer_data['context']) if i in citations_id])
        return {'text': text_answer, 'classifier_1': classifier_1_lst[0], 'classifier_2': classifier_2_lst[0], 'contexts': context}
