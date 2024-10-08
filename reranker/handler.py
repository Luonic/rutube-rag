import torch
from transformers import AutoTokenizer
from reranker.model import RelevanceModel

class rerankHandler:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config['TOKENIZER_PATH'])
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.models = []
        if checkpoint_paths := config.get('RERANKER_MODEL_PATHS', []):
            for path in checkpoint_paths:
                model = RelevanceModel(config['RERANKER_MODEL_PATHS'][0], config['NUM_CLASSES'])
                model.load_state_dict({key: value for key, value in torch.load(path + 'pytorch_model.bin').items()})
                model.to(self.device).eval()
                self.models.append(model)
        self.max_len = config['max_len']
        
    def rerank_questions(self, questions: list, answer: str, top_k: int):
        # print(answer)
        # print([question.page_content for question, _ in questions])
        inputs = self.tokenizer(
            [question.page_content for question, _ in questions],
            [answer] * len(questions),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        for i, model in enumerate(self.models):
            with torch.no_grad():
                fold_logits, _ = model(**inputs)
            if i == 0:
                logits = fold_logits
            else:
                logits += fold_logits

        logits /= len(self.models)
        probs = torch.nn.Softmax(dim=1)(logits)[:, -1].detach().cpu().numpy().tolist()
        torch.cuda.empty_cache()
        reranked_questions_with_probs = sorted(list(zip(questions, probs)), key=lambda x: x[1], reverse=True)
        reranked_questions = []
        for question, prob in reranked_questions_with_probs:
            new_question = question
            new_question = list(new_question)
            retrival_prob = new_question[-1] 
            new_question[-1] =  (prob * retrival_prob) / (prob + retrival_prob)
            new_question = tuple(new_question)
            reranked_questions.append(new_question)
        reranked_questions = sorted(reranked_questions, key=lambda x: x[1], reverse=True)
        # print(reranked_questions)
        return reranked_questions[:top_k]

def init_reranker(config):
    reranker = rerankHandler(config)
    return reranker