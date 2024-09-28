import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

class RelevanceModel(torch.nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.linear_1 = torch.nn.Linear(self.config.hidden_size, int(self.config.hidden_size // 2))
        self.linear_2 = torch.nn.Linear(int(self.config.hidden_size // 2), num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask)
        last_hidden_state = output.last_hidden_state
        output = self.linear_2(self.linear_1(last_hidden_state))
        logits = output[:, 0, :]

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.linear_2.out_features), labels.view(-1))

        return logits, loss
    