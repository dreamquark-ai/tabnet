import os
from copy import deepcopy

import torch

from pytorch_tabnet.tuner.abstract_worker import AbstractWorker
from pytorch_tabnet.tab_model import TabNetClassifier


DEFAULT_GRID = {
    "cat_emb_dim": [1],
    "n_a": ["n_d"],
    "n_d": {"type": "int", "lower": 3, "upper": 64, "default_value": 8,},
    "n_independent": {"type": "int", "lower": 1, "upper": 6, "default_value": 2,},
    "n_shared": {"type": "int", "lower": 0, "upper": 6, "default_value": 2,},
    "n_steps": {"type": "int", "lower": 2, "upper": 10, "default_value": 3,},
    "clip_value": [1],
    "gamma": {"type": "float", "lower": 1, "upper": 2, "default_value": 1,},
    "momentum": {"type": "float", "lower": 0.01, "upper": 0.5, "log": True},
    "lambda_sparse": {"type": "float", "lower": 1e-6, "upper": 1e-1, "log": True},
    "lr": {"type": "float", "lower": 1e-3, "upper": 1e-1, "log": True},
    "patience": [5],
    "num_workers": [0],
    "batch_size": [1024, 2048, 4096, 8192],
    "virtual_batch_size": [128, 256, 512, 1024],
}


class TabNetWorker(AbstractWorker):
    def get_model_params_keys(self):
        return super().get_model_params_keys()

    def get_classifier_class(self):
        return TabNetClassifier

    def get_budget_desc(self):
        return {"place": "fit", "name": "max_epochs", "type": "int"}

    def format_train_valid(self):
        return {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_valid": self.X_valid,
            "y_valid": self.y_valid,
        }

    @staticmethod
    def get_grid(custom_config):
        grid = deepcopy(DEFAULT_GRID)
        for key, value in custom_config.items():
            if key in DEFAULT_GRID:
                grid[key] = value

        if torch.cuda.is_available():
            grid["num_workers"] = [os.cpu_count()]
        return grid
