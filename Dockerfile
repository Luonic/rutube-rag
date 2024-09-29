FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y git

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY app/ /app
COPY rag/ /app/rag/
COPY reranker/ /app/reranker/
COPY retriever/ /app/retriever/

WORKDIR /app

CMD ["fastapi", "run", "main.py", "--port", "31000"]