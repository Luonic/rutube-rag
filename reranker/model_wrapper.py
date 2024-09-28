import lightning as l
import torch
from torchmetrics import RetrievalNormalizedDCG
from torchmetrics.classification import MulticlassF1Score, MultilabelF1Score
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor
from model import *

class RelevanceLightningModel(l.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.optimizer_type = config['optimizer']
        self.scheduler_usage = config['scheduler']
        self.num_train_steps = config['num_train_steps']
        self.track = config['track']
        self.epochs = config['epochs']
        self.model = RelevanceModel(config['model_name'], config['num_classes'])
        if config.get('checkpoint_path', False):
            state_dict = {key.replace('model.', '', 1): value \
                                for key, value in torch.load(config['checkpoint_path'])['state_dict'].items()}
            state_dict = {key: value for key, value in state_dict.items() if key not in ['linear_2.weight', 'linear_2.bias']}
            self.model.load_state_dict(state_dict, strict=False)
        self.lr = config['lr']
        
        # Метрики
        self.train_f1 = MulticlassF1Score(num_classes=config['num_classes'], average='weighted')
        self.val_f1 = MulticlassF1Score(num_classes=config['num_classes'], average='weighted')
        self.test_f1 = MulticlassF1Score(num_classes=config['num_classes'], average='weighted')
            
        self.train_ndcg = RetrievalNormalizedDCG()
        self.val_ndcg = RetrievalNormalizedDCG()
        self.test_ndcg = RetrievalNormalizedDCG()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        indexes = batch['index']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits, loss = self(input_ids, attention_mask, labels)
        probs = torch.nn.Softmax(dim=1)(logits)
        self.train_f1.update(probs.argmax(1), labels)
        self.train_ndcg.update(probs[:, -1], labels, indexes=indexes)

        self.log('train_loss', loss, prog_bar=True)

        if self.scheduler_usage in ['linear', 'cosine'] and self.optimizer_type not in ['adafactor']:
            self.scheduler.step()
        return loss

    def validation_step(self, batch, batch_idx):
        indexes = batch['index']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits, loss = self(input_ids, attention_mask, labels)
        probs = torch.nn.Softmax(dim=1)(logits)
        self.val_f1.update(probs.argmax(1), labels)
        self.val_ndcg.update(probs[:, -1], labels, indexes=indexes)    
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        indexes = batch['index']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits, loss = self(input_ids, attention_mask, labels)
        probs = torch.nn.Softmax(dim=1)(logits)
        self.test_f1.update(probs.argmax(1), labels)
        self.test_ndcg.update(probs[:, -1], labels, indexes=indexes)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        elif self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['lr'])
        elif self.optimizer_type == 'adafactor':
            optimizer = Adafactor(self.parameters(), warmup_init=False, relative_step=True)
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config['lr'])
        if self.scheduler_usage == 'reduce' and self.optimizer_type in ['adam', 'sgd', 'adamw']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, mode='min' if 'loss' in self.track else 'max')

            return {
                    'optimizer': optimizer, 
                    'lr_scheduler': {
                                     'scheduler': scheduler, 
                                     'monitor': self.track
                                    }
                    }

        elif self.scheduler_usage == 'linear' and self.optimizer_type in ['adam', 'sgd', 'adamw']:
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=int(self.num_train_steps * self.epochs * 0.1), 
                num_training_steps=self.num_train_steps * self.epochs
            )
            return {'optimizer': optimizer, 'lr_scheduler': self.scheduler}

        elif self.scheduler_usage == 'cosine' and self.optimizer_type in ['adam', 'sgd', 'adamw']:
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=int(self.num_train_steps * self.epochs * 0.1), 
                num_training_steps=self.num_train_steps * self.epochs
            )
            return {'optimizer': optimizer, 'lr_scheduler': self.scheduler}
            
        return optimizer

    def on_train_epoch_end(self):
        avg_f1 = self.train_f1.compute()
        avg_ndcg = self.train_ndcg.compute()
        self.log('train_f1', avg_f1)
        self.log('train_ndcg', avg_ndcg)
        self.train_ndcg.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        avg_f1 = self.val_f1.compute()
        avg_ndcg = self.val_ndcg.compute()
        self.log('val_f1', avg_f1)
        self.log('val_ndcg', avg_ndcg)
        self.val_ndcg.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self):
        avg_f1 = self.test_f1.compute()
        avg_ndcg = self.test_ndcg.compute()
        self.log('test_f1', avg_f1)
        self.log('test_ndcg', avg_ndcg)
        self.test_ndcg.reset()
        self.test_f1.reset()