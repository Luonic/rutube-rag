import yaml
import numpy as np
import wandb
import lightning as l
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from dataset import *
from model_wrapper import *


if __name__ == "__main__":
    config = yaml.safe_load(open('train_config.yaml', 'r'))
    # 1. Cоздание датасета
    train_dataset, test_dataset = create_dataset(config)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    # 2. Разбиение на N фолдов с использованием KFold
    kf = KFold(n_splits=config['n_folds'], shuffle=False)

    # 3. Инициализация EarlyStopping
    early_stopping = EarlyStopping(
        monitor=config['track'],  # Останавливаем на основе значения val_loss
        min_delta=0.001,
        mode='min' if 'loss' in config['track'] else 'max',
        patience=2,  # Останавливаем, если за 3 эпохи не было улучшений
        verbose=True
    )

    # 4. Запуск цикла для каждого фолда
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(train_dataset)))):
        print(f"Training fold {fold_idx+1}...")

        # Подготовка train и validation датасетов для текущего фолда
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=config['batch_size'])
        val_loader = DataLoader(val_subset, batch_size=config['batch_size'])
        
        # Инициализация модели
        config['num_train_steps'] = len(train_loader)
        config['fold_idx'] = fold_idx
        model = RelevanceLightningModel(config)

        # Инициализация чекпоинта для текущего фолда
        checkpoint_callback = ModelCheckpoint(
            monitor=config['track'],
            dirpath=f'checkpoints/{config["model_name"]}_fold_{config["fold_idx"]}',
            filename='checkpoint_{epoch}-{val_loss:.2f}-{val_f1:.2f}-{val_ndcg:.2f}',
            mode='min' if 'loss' in config['track'] else 'max',
            save_weights_only=True,
            save_top_k=1,
            verbose=True
            )
        
        logger = WandbLogger(
                    project=config['project_name'], 
                    group=f'{config["model_name"]}_fold_{config["fold_idx"]}',
                    log_model=False
                    )
        
        # Инициализация тренера
        trainer = l.Trainer(
                            max_epochs=config['epochs'], 
                            val_check_interval=config['val_check_interval'],
                            precision=config['precision'],
                            accelerator='gpu',
                            devices=1,
                            logger=logger, 
                            callbacks=[checkpoint_callback, early_stopping]
                            )
        
        # 5. Обучение модели
        trainer.fit(model, train_loader, val_loader)
        
        # 6. Тестирование модели после тренировки
        test_result = trainer.test(model, test_loader, ckpt_path='best')

        wandb.finish()