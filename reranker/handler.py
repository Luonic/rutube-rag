import torch
from transformers import AutoTokenizer
from triton.client import tritonClient


class rerankHandler:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.client = tritonClient(config['checkpoint_names'])
        self.max_len = config['max_len']
        
    async def rerank_questions(self, questions: list, answer: str):
        inputs = self.tokenizer(
            questions,
            [answer] * len(questions),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )

        logits = await self.client.inference(**inputs)
        logits = torch.from_numpy(logits)
        probs = torch.nn.Softmax(dim=1)(logits)[:, -1].detach().cpu().numpy().tolist()
        reranked_questions = list(zip(questions, probs))
        return reranked_questions