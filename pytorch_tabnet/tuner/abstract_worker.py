from abc import abstractmethod, abstractstaticmethod

from sklearn.metrics import roc_auc_score
from hpbandster.core.worker import Worker


class AbstractWorker(Worker):
    def __init__(self, X_train, y_train, X_valid, y_valid, X_test, y_test, **kwargs):
        super().__init__(**kwargs)

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test

        self.model_params_keys = self.get_model_params_keys()

    def get_model_params_keys(self):
        instance = self.get_classifier_class()()
        return list(instance.get_params().keys())

    @abstractstaticmethod
    def get_grid(self):
        raise NotImplementedError("This should be overriden")

    @abstractmethod
    def get_classifier_class(self):
        raise NotImplementedError("This should be overriden")

    @abstractmethod
    def get_budget_desc(self):
        raise NotImplementedError("This should be overriden")

    @abstractmethod
    def format_train_valid(self):
        raise NotImplementedError("This should be overriden")

    def compute(self, config, budget, working_directory, *args, **kwargs):

        model_params = {}
        fit_params = {}

        for elt, value in config.items():
            if elt in self.model_params_keys:
                model_params[elt] = value
            else:
                fit_params[elt] = value

        budget_desc = self.get_budget_desc()
        bugdet_value = budget
        if budget_desc["type"] == "int":
            bugdet_value = int(bugdet_value)

        if budget_desc["place"] == "model":
            model_params[budget_desc["name"]] = bugdet_value
        else:
            fit_params[budget_desc["name"]] = bugdet_value

        clf = self.get_classifier_class()(**model_params)

        clf.fit(**self.format_train_valid(), **fit_params)

        valid_score = roc_auc_score(
            y_score=clf.predict_proba(self.X_valid)[:, 1], y_true=self.y_valid
        )
        test_score = roc_auc_score(
            y_score=clf.predict_proba(self.X_test)[:, 1], y_true=self.y_test
        )

        return {
            "loss": -valid_score,  # remember: HpBandSter always minimizes!
            "info": {"test_score": test_score, "valid_score": valid_score,},
        }
