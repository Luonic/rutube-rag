from tqdm import tqdm
from pathlib import Path
import os, os.path
import uuid
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.system import SystemMessage
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from prompts import SIMILAR_QUESTIONS_PROMPT
from utils import write_json, read_docs, split_docs


DATA_PATH = '../reranker/data/all_data.csv'
UPLOAD_FOLDER = 'data/generated_similar_questions/'
MODEL_NAME = 'intfloat/multilingual-e5-base'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

def parse_and_write_split(split):
    split_content = split.page_content
    split_question = split.metadata['question']
    ranked_gen_questions = chain.invoke({"question": split_question, "answer": split_content})
    json_answer = dict()
    json_answer.update(split.metadata)
    json_answer['questions'] = ranked_gen_questions
    json_answer['split_answer'] = split_content
    write_json(json_answer, UPLOAD_FOLDER)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def count_tokens(text):
        return len(tokenizer.tokenize(text, add_special_tokens=False))

    docs = read_docs(DATA_PATH)
    splits = split_docs(docs, count_tokens)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(SIMILAR_QUESTIONS_PROMPT),
                         ("user", "Вопрос:\n{question}\n\nОтвет:\n{answer}\nJSON:")
        ]
    )

    llm = ChatOpenAI(model="gpt-4o", openai_proxy=os.getenv('PROXY'))
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    # for split in tqdm(splits):
    with ThreadPoolExecutor(1) as f:
        res = f.map(parse_and_write_split, splits[1:])

    for r in tqdm(res):
        print(r)
        