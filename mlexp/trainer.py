import torch
import numpy as np
import random


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
        self.checkpoints_path = checkpoint_path
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

        def train(self, train_dataloader, val_dataloader):
            """
            Train model

            @param train_dataloader: pytorch dataloader, training samples
            @param val_dataloader: pytorch dataloader, validation samples
            """
            return
