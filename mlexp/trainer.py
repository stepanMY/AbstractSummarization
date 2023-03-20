import torch
import numpy as np
import random
from tqdm.auto import tqdm
from util.tokenizator import tokenize_sentences


class Trainer:
    """
    Class for training of models
    """
    def __init__(self,
                 model_shell,
                 logger,
                 checkpoint_path,
                 metrics_func,
                 metrics,
                 diagnostics,
                 optimizer,
                 scheduler,
                 optimizer_params=None,
                 scheduler_params=None,
                 random_seed=42,
                 epochs=1,
                 train_epoch=0.05,
                 val_epoch=0.1,
                 saveafter_epoch=0.3):
        """
        @param model_shell: model from models directory, model that will be trained;
        Should have calc_loss, generate, save_model, load_model methods
        @param logger: logger object, object that will be used for metrics and parameters logging
        Should have log_metric, log_stepmetric methods
        @param checkpoint_path: string, directory of checkpoints
        @param metrics_func: function, should accept y_true and y_pred and calculate metrics
        Should accept y_true and y_pred as list of lists (tokenized)
        @param metrics: list of strings, metrics to use
        Should be calculated by metrics_func
        @param diagnostics: list of strings, diagnostic metrics to use
        Should be calculated by metrics_func
        @param optimizer: torch optimizer
        @param scheduler: torch scheduler
        @param optimizer_params: dict, parameters of optimizer
        If None, use defaults
        @param scheduler_params: dict, parameters of scheduler
        If None, use defaults
        @param random_seed: int, random seed that will be used for randomization
        @param epochs: int, number of epochs in training procedure
        @param train_epoch: float, after what part of epoch report of train loss happens
        @param val_epoch: float, after what part of epoch report of validation metrics happens
        @param saveafter_epoch: float, after what part of epoch trainer starts to save model checkpoint
        """
        self.model_shell = model_shell
        self.logger = logger
        self.checkpoint_path = checkpoint_path
        self.metrics_func = metrics_func
        self.metrics = metrics
        self.diagnostics = diagnostics
        if optimizer_params is not None:
            self.optimizer = optimizer(self.model_shell.model.parameters(), **optimizer_params)
        else:
            self.optimizer = optimizer(self.model_shell.model.parameters())
        if scheduler_params is not None:
            self.scheduler = scheduler(self.optimizer, **scheduler_params)
        else:
            self.scheduler = scheduler(self.optimizer)
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        self.epochs = epochs
        self.train_epoch = train_epoch
        self.val_epoch = val_epoch
        self.saveafter_epoch = saveafter_epoch

        self.train_loss = []
        self.step_curr = 0
        self.train_curr, self.val_curr = 0, 0
        self.best_valmetrics = {metric: 0 for metric in self.metrics}
        self.was_saved = False

    def log_trainloss(self, step):
        """
        Log intermediate loss value and lr

        @param step: float, current step in epochs
        """
        self.logger.log_stepmetric('training_procedure/trainloss', np.mean(self.train_loss), step)
        self.logger.log_stepmetric('training_procedure/lr', self.scheduler.get_last_lr()[0], step)
        return

    def log_valmetrics(self, val_metrics, step):
        """
        Log intermediate validation metrics

        @param val_metrics: dict, current validation metrics
        @param step: float, current step in epochs
        """
        for metric in self.metrics:
            self.logger.log_stepmetric(f'val/{metric}', val_metrics[metric], step)
        for diagnostic in self.diagnostics:
            self.logger.log_stepmetric(f'val/{diagnostic}', val_metrics[diagnostic], step)
        return

    def validate(self, val_dataloader):
        """
        Validate model, calculate metrics on val_dataloader

        @param val_dataloader
        """
        self.model_shell.model.eval()
        ytrue_val, ypred_val = [], []
        for batch_val in val_dataloader:
            x_val, y_val = list(batch_val[0]), list(batch_val[1])
            ytrue_val.extend(y_val)
            preds_batch = self.model_shell.generate(x_val)
            ypred_val.extend(preds_batch)
        val_metrics = self.metrics_func(tokenize_sentences(ytrue_val),
                                        tokenize_sentences(ypred_val))
        return val_metrics

    def save_condition(self, val_metrics):
        """
        Check whether model should be saved

        @param val_metrics: dict, current validation metrics
        """
        valcounter = 0
        for metric in self.metrics:
            if val_metrics[metric] >= self.best_valmetrics[metric]:
                valcounter += 1
        if valcounter >= len(self.metrics)/2:
            return True
        return False

    def update_best(self, val_metrics):
        """
        Update best validation metrics

        @param val_metrics: dict, current validation metrics
        """
        for metric in self.metrics:
            self.best_valmetrics[metric] = val_metrics[metric]

    def train(self, train_dataloader, val_dataloader):
        """
        Train model while reporting intermediate train loss and validation metrics

        @param train_dataloader: pytorch dataloader, training samples
        @param val_dataloader: pytorch dataloader, validation samples
        """
        self.model_shell.model.train()
        for epoch in range(self.epochs):
            for i, batch in enumerate(tqdm(train_dataloader)):
                epoch_curr = (self.step_curr+1)/len(train_dataloader)
                x, y = list(batch[0]), list(batch[1])
                loss = self.model_shell.calc_loss(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.train_loss.append(loss.item())
                if epoch_curr >= self.train_curr:
                    self.log_trainloss(epoch_curr)
                    self.train_loss = []
                    self.train_curr += self.train_epoch
                if epoch_curr >= self.val_curr:
                    val_metrics = self.validate(val_dataloader)
                    self.log_valmetrics(val_metrics, epoch_curr)
                    if epoch_curr >= self.saveafter_epoch:
                        if self.save_condition(val_metrics):
                            self.update_best(val_metrics)
                            self.model_shell.save_model(self.checkpoint_path)
                            self.was_saved = True
                    self.model_shell.model.train()
                    self.val_curr += self.val_epoch
                self.step_curr += 1
        if self.was_saved:
            self.model_shell.load_model(self.checkpoint_path)
        return
