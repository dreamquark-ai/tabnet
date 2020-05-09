from pytorch_tabnet.tuner.abstract_worker import AbstractWorker
from xgboost import XGBClassifier
import torch
from copy import deepcopy

DEFAULT_GRID = {
    "verbosity": [0],
    "objective": ["binary:logistic"],
    "booster": ["gbtree"],
    "n_jobs": [-1],
    "seed": [0],
    "random_state": [0],
    "learning_rate": {"type": "float", "lower": 0.1, "upper": 1, "default_value": 0.3,},
    "max_depth": {"type": "int", "lower": 1, "upper": 12, "default_value": 6},
    "min_child_weight": {"type": "int", "lower": 0, "upper": 10, "default_value": 1,},
    "max_delta_step": {"type": "int", "lower": 0, "upper": 10, "default_value": 1,},
    "subsample": {"type": "float", "lower": 0, "upper": 1, "default_value": 1},
    "colsample_bytree": {"type": "float", "lower": 0, "upper": 1, "default_value": 1,},
    "colsample_bylevel": {"type": "float", "lower": 0, "upper": 1, "default_value": 1,},
    "colsample_bynode": {"type": "float", "lower": 0, "upper": 1, "default_value": 1,},
    "reg_alpha": {"type": "int", "lower": 0, "upper": 10, "default_value": 0},
    "reg_lambda": {"type": "int", "lower": 0, "upper": 10, "default_value": 1},
    "gamma": {"type": "int", "lower": 0, "upper": 10, "default_value": 1},
    "early_stopping_rounds": [40],
    "verbose": [0],
    "tree_method": ["hist"],
}


class XGBWorker(AbstractWorker):
    def get_model_params_keys(self):
        return super().get_model_params_keys() + ["tree_method"]

    def get_classifier_class(self):
        return XGBClassifier

    def get_budget_desc(self):
        return {"place": "model", "name": "n_estimators", "type": "int"}

    def format_train_valid(self):
        return {
            "X": self.X_train,
            "y": self.y_train,
            "eval_set": [(self.X_valid, self.y_valid)],
        }

    @staticmethod
    def get_grid(custom_config):
        grid = deepcopy(DEFAULT_GRID)
        for key, value in custom_config.items():
            if key in DEFAULT_GRID:
                grid[key] = value

        if torch.cuda.is_available():
            grid["tree_method"] = ["gpu_hist"]
        return grid


