import os
from pprint import pprint
from statistics import mean
import gc
import copy

import hydra
import ignite.distributed as idist
import torch
import torch.distributed
import torch.multiprocessing as mp
import torch.nn as nn
from hydra.utils import get_original_cwd, instantiate, call, to_absolute_path
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import Average, Precision, Recall, ClassificationReport
from omegaconf import DictConfig
from torch.multiprocessing import Value
from peft import get_peft_model
from metrics.mean_positive_similarity import MeanPositiveSimilarity, MeanPositiveAndNegativeSimilarity
from metrics.ranking_top_n import Ranking

torch.autograd.set_detect_anomaly(True)

def get_dataloaders(cfg: DictConfig):
    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_dataloader = instantiate(cfg.train_dataloader)
    val_dataloader = instantiate(cfg.val_dataloader)
    return train_dataloader, val_dataloader

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def create_trainer(cfg, model, criterion, optimizer):    
    def train_step(engine, batch):
        model.train()
        device = idist.device()

        query_input_ids = batch["query_input_ids"].contiguous().to(device, non_blocking=True)
        query_attention_mask = batch["query_attention_mask"].contiguous().to(device, non_blocking=True)
        passage_input_ids = batch["passage_input_ids"].contiguous().to(device, non_blocking=True)
        passage_attention_mask = batch["passage_attention_mask"].contiguous().to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True):
            qp_input_ids = torch.cat([query_input_ids, passage_input_ids], dim=0)
            qp_attention_mask = torch.cat([query_attention_mask, passage_attention_mask], dim=0)
            qp_logits = model(qp_input_ids, qp_attention_mask, return_dict=False)[0]
            query_logits, passage_logits = torch.split(qp_logits, split_size_or_sections=query_input_ids.size(0), dim=0)
        
            query_logits = query_logits[:, 0]
            passage_logits = passage_logits[:, 0]             
            
        loss = criterion(query_logits, passage_logits)
        (loss / cfg.train.grad_accumulation_steps).backward()

        if engine.state.iteration % cfg.train.grad_accumulation_steps == 0 and engine.state.iteration != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return loss.item()

    trainer = Engine(train_step)

    if idist.get_rank() == 0:
        ProgressBar(persist=True, smoothing=0.0).attach(
            trainer, output_transform=lambda x: {"train batch loss": x})

    return trainer


def create_evaluator(cfg, model, criterion, device=None):
    if device is None:
        device = idist.device()

    def eval_step(engine, batch):
        with torch.inference_mode():
            model.eval()

            query_input_ids = batch["query_input_ids"].contiguous().to(device, non_blocking=True)
            query_attention_mask = batch["query_attention_mask"].contiguous().to(device, non_blocking=True)
            passage_input_ids = batch["passage_input_ids"].contiguous().to(device, non_blocking=True)
            passage_attention_mask = batch["passage_attention_mask"].contiguous().to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False, cache_enabled=True):
                query_logits = model(query_input_ids, query_attention_mask)[0]
                passage_logits = model(passage_input_ids, passage_attention_mask)[0]            
                query_logits = query_logits[:, 0]
                passage_logits = passage_logits[:, 0] 
                
            loss = criterion(query_logits, passage_logits)

            return {'loss': loss, 'query_logits': query_logits, 'passage_logits': passage_logits}

    evaluator = Engine(eval_step)

    loss_metric = Average(output_transform=lambda output: output['loss'])
    loss_metric.attach(evaluator, 'loss')

    inv_loss_metric = Average(output_transform=lambda output: -output['loss'])
    inv_loss_metric.attach(evaluator, 'inv_loss')

    ranking_metric = Ranking(k=3, output_transform=lambda output: (output["query_logits"], output["passage_logits"]), device="cpu")
    ranking_metric.attach(evaluator, 'ranking')

    if idist.get_rank():
        ProgressBar().attach(evaluator, output_transform=lambda x: {
            "val_batch_loss": x['loss']})

    return evaluator


def training(local_rank: int, cfg: DictConfig, best_metric) -> float:
    logs_dir = './'

    train_dataloader, val_dataloader = get_dataloaders(cfg)

    model = call(cfg.model)
    model = get_peft_model(model, instantiate(cfg.lora_config))
    model.print_trainable_parameters()
    model = idist.auto_model(model)

    optimizer = instantiate(cfg.train.optimizer, params=[param for param in model.parameters() if param.requires_grad])
    optimizer = idist.auto_optim(optimizer)

    scheduler = instantiate(
        cfg.train.scheduler, optimizer=optimizer, steps_per_epoch=len(train_dataloader))

    criterion = instantiate(cfg.train.loss)

    # Setup model trainer and evaluator
    trainer = create_trainer(
        cfg=cfg, model=model, optimizer=optimizer, criterion=criterion)
    evaluator = create_evaluator(cfg, model, criterion)

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def unfreeze():
    #     set_backbone_grad(model, True)

    if idist.utils.get_world_size() > 1:
        @trainer.on(Events.EPOCH_STARTED)
        def set_epoch(engine):
            if hasattr(train_dataloader, 'sampler') and isinstance(train_dataloader.sampler, (torch.utils.data.DistributedSampler)):
                train_dataloader.sampler.set_epoch(engine.state.epoch)

    # Run model evaluation every 3 epochs and show results
    # @trainer.on(Events.ITERATION_STARTED(every=1))
    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate_model():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        state = evaluator.run(val_dataloader)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if idist.get_rank() == 0:
            pprint(state.metrics)

            eval_metric = state.metrics['ranking']
            if eval_metric > best_metric.value:
                best_metric.value = float(eval_metric)
                if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                    module = model.module
                else:
                    module = model
                    
                merged_module = copy.deepcopy(module)
                merged_module = merged_module.merge_and_unload()
                merged_module.save_pretrained(os.path.join(logs_dir, str(cfg.train_dataloader.dataset.fold_idx)))
                # merged_module.save_pretrained(os.path.join(logs_dir, str(cfg.train_dataloader.dataset.fold_idx)) ,save_adapters=True, save_embedding_layers=True)
                # merged_module.base_model.save_pretrained(os.path.join(logs_dir, str(cfg.train_dataloader.dataset.fold_idx)))
                del merged_module
                gc.collect()
                print("SAVED NEW BEST MODEL!")
                
                
    @trainer.on(Events.ITERATION_COMPLETED)
    def lr_step():
        scheduler.step()

    # Setup tensorboard experiment tracking
    if idist.get_rank() == 0:
        tb_logger = common.setup_tb_logging(
            logs_dir, trainer, optimizer, evaluators={"validation": evaluator},
        )

    trainer.run(train_dataloader, max_epochs=cfg.train.max_epochs)

    if idist.get_rank() == 0:
        tb_logger.close()

    return best_metric


@hydra.main(config_path="configs/", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> float:
    print("Working directory : {}".format(os.getcwd()), flush=True)

    if cfg.train.all_folds == True:
        folds_to_train = list(range(cfg.train.num_folds))
    else:
        folds_to_train = [cfg.train_dataloader.dataset.fold_idx]

    result_scores = []

    for fold_idx in folds_to_train:
        cfg.train_dataloader.dataset.fold_idx = fold_idx
        cfg.val_dataloader.dataset.fold_idx = fold_idx

        with idist.Parallel(
            backend=cfg.distributed.backend,
            nproc_per_node=cfg.distributed.nproc_per_node,
            master_addr='0.0.0.0' if cfg.distributed.backend is not None else None,
            master_port='16789' if cfg.distributed.backend is not None else None
        ) as parallel:
            best_metric_value = Value('d', float("-inf"))
            parallel.run(training, cfg, best_metric_value)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result_scores.append(float(best_metric_value.value))
            print(f'Finished training fold {fold_idx}.')
    score = mean(result_scores)
    print(f"Score: {score}", flush=True)
    return score


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
