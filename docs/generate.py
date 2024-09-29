from tqdm import tqdm
from pathlib import Path
import os, os.path
import uuid
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.system import SystemMessage
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from prompts import DOCX_QUESTIONS_PROMPT, CLASSIFIER_PROMPT
from utils import write_json, read_docs, split_docs


DATA_PATH = 'data/agreement.tsv'
UPLOAD_FOLDER = 'data/generated_qa_for_docs/'
MODEL_NAME = 'intfloat/multilingual-e5-base'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

def parse_and_write_split(split):
    context = split.page_content
    gen_qa = chain.invoke({"context": context})
    gen_qa_clf = chain_2.invoke({"json": gen_qa})
    json_answer = dict()
    json_answer.update(split.metadata)
    json_answer['questions'] = gen_qa_clf
    json_answer['context'] = context
    write_json(json_answer, UPLOAD_FOLDER)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def count_tokens(text):
        return len(tokenizer.tokenize(text, add_special_tokens=False))

    docs = read_docs(DATA_PATH)
    splits = split_docs(docs, count_tokens)

    prompt_1 = ChatPromptTemplate.from_messages(
        [
            SystemMessage(DOCX_QUESTIONS_PROMPT),
                         ("user", "Текст:\n{context}\n\nJSON:")
        ]
    )

    prompt_2 = ChatPromptTemplate.from_messages(
        [
            SystemMessage(CLASSIFIER_PROMPT),
                         ("user", "Вход:\n\n{json}\n\nВыход JSON:")
        ]
    )

    llm = ChatOpenAI(model="gpt-4o", openai_proxy=os.getenv('PROXY'))
    json_parser = JsonOutputParser()
    str_parser = StrOutputParser()

    chain = prompt_1 | llm | str_parser
    chain_2 = prompt_2 | llm | json_parser

    # for split in tqdm(splits):
    with ThreadPoolExecutor(1) as f:
        res = f.map(parse_and_write_split, splits[153:])

    for r in tqdm(res):
        print(r)
        

