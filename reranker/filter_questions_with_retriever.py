import json
import os
import yaml
from glob import glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils import write_json, read_docs, split_docs
from model import RelevanceModel

config_path = 'train_config.yaml'
config = yaml.safe_load(open(config_path, 'r'))

DATA_PATH = 'data/rutube_faq_with_themes.tsv'
UPLOAD_FOLDER = 'data/filter_generated_ranked_questions/'
MODEL_NAME = 'tinkoff-ai/response-quality-classifier-large' #config['model_name']
CHECKPOINT_PATH = config['checkpoint_path']
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Загрузка модели Cross-Encoder
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# model = RelevanceModel(MODEL_NAME, 2)
# model.load_state_dict({key.replace('model.', '', 1): value for key, value in torch.load(CHECKPOINT_PATH)['state_dict'].items()})
model.eval()
model.to('cuda')

# Функция для оценки релевантности с помощью Cross-Encoder
def cross_encoder_similarity(question, answer):
    # inputs = tokenizer.encode_plus(question,
    inputs = tokenizer.encode("[CLS]привет[SEP]привет![SEP]как дела?[RESPONSE_TOKEN]норм, у тя как?",
                                   answer,
                                   add_special_tokens=False,
                                   max_length=config['max_len'],
                                   padding='max_length',
                                   return_tensors='pt', 
                                   truncation=True).to('cuda')
    with torch.no_grad():
        logits = model(**inputs).logits
    # Применяем софтмакс для получения вероятности
    probs = torch.nn.functional.softmax(logits, dim=1)
    relevance_score = probs[0][1].item()  # Предполагая, что релевантность соответствует метке '1'
    torch.cuda.empty_cache()
    return relevance_score

# Основной пайплайн обработки
def process_answer(answer, questions_with_relevance):    
    results = []
    
    # Для каждого сгенерированного вопроса
    for qr in questions_with_relevance:
        question = qr["question"]
        relevance = qr["relevance"] / 5
        
        # Оцениваем релевантность сгенерированного вопроса и данного ответа
        relevance_score = cross_encoder_similarity(question, answer)
        
        # Сохраняем результат для каждого вопроса
        results.append({
            "question": question,
            "true_answer": answer,
            "relevance": relevance,
            "cross_encoder_relevance_score": relevance_score,
            "valid_label": 1 if abs(relevance - relevance_score) < 0.2 else 0
        })
    
    # Возвращаем список вопросов с их оценками релевантности и скором Cross-Encoder
    return results

if __name__ == '__main__':

    data_files = glob('data/generated_ranked_questions/*.json')

    # Обработка всех данных
    all_results = []
    for file in data_files[:1]:
        with open(file, 'r') as f:
            generated_questions_data = json.load(f)

        result = process_answer(generated_questions_data['source'], generated_questions_data['questions'])
        write_json(result, UPLOAD_FOLDER)
        all_results.extend(result)
