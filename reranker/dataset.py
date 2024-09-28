import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from glob import glob


class RelevanceDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        """
        data: список словарей с полями 'question', 'answer', 'relevance_gradation'
        tokenizer: токенизатор из HuggingFace (например, для BERT или Roberta)
        max_len: максимальная длина токенов
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        index = item['index']
        question = item['query']
        answer = item['answer']
        relevance = item['relevance']
        
        # Токенизация с объединением вопросов и ответов
        inputs = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'index': torch.tensor(index, dtype=torch.long),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(relevance, dtype=torch.long)
        }

# Функция для преобразования записей в формат query, answer, label
def dataset_to_tidy_data(dataset):
    return [row for idx, (query, pos, neg) in enumerate(zip(dataset['query'], dataset['positive'], dataset['negative'])) \
                for row in ({'index': idx, 'query': query, 'answer': pos, 'relevance': 1}, 
                            {'index': idx, 'query': query, 'answer': neg, 'relevance': 0})]


def create_dataset(config):
    if config['type_of_train'] == 'pretrain':
        full_dataset = load_dataset('unicamp-dl/mmarco', 'russian').shuffle(42)
        full_dataset = full_dataset['train'].select([i for i in range(config['dataset_size'])])
        tidy_full_dataset = dataset_to_tidy_data(full_dataset)
        train_dataset, test_dataset = train_test_split(tidy_full_dataset, test_size=0.1, shuffle=False, random_state=42)
    else:
        data_paths = glob(f'{config["data_path"]}/*.json')
        dataset = []
        for idx, path in enumerate(data_paths):
            with open(path, 'r') as f:
                sample_data = json.load(f)
                answer = sample_data['split_answer']
                for question in sample_data['questions']:
                    if question.get('relevance', 4) < 3:
                        dataset.append({'index': idx, 'query': question['question'], 'answer': answer, 'relevance': question['relevance']})
        train_dataset, test_dataset = train_test_split(dataset, test_size=0.1, shuffle=False, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    return (RelevanceDataset(train_dataset,
                            tokenizer,
                            config['max_len']),
            RelevanceDataset(test_dataset,
                                tokenizer,
                                config['max_len']))