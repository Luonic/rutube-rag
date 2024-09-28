import torch
from torch.utils.data import Dataset
import os
import json
from glob import glob
from collections import defaultdict
from numpy.random import choice

class RutubeRagDataset(Dataset):
    def __init__(self, json_path: str, tokenizer, max_length: int, fold_idx: int, train=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
                
        with open(json_path, "r") as dataset_f:
            data = json.load(dataset_f)
            
        data = data["folds"][fold_idx]
        if train:
            data = data["train"]
        else:
            data = data["val"]
            
        self.examples = data 
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
                
        question = example["question"]
        response = example["response"]

        lines = [
            question, 
            response
        ]
        encoded = self.tokenizer(
            lines, padding="max_length", truncation=True, 
            max_length=self.max_length, return_tensors="pt")
        query_input_ids = encoded["input_ids"][0]
        query_attention_mask = encoded["attention_mask"][0]
        passage_input_ids = encoded["input_ids"][1]
        passage_attention_mask = encoded["attention_mask"][1]
        return {
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "passage_input_ids": passage_input_ids,
            "passage_attention_mask": passage_attention_mask
        }
        
        
if __name__ == "__main__":
    import transformers
    from pprint import pprint
    tokenizer = transformers.AutoTokenizer.from_pretrained('deepvk/USER-bge-m3')
    dataset = RutubeRagDataset("data/train_val_test.json", tokenizer=tokenizer, max_length=512, fold_idx=0, train=True)
    pprint(dataset[0]["_input_ids"])