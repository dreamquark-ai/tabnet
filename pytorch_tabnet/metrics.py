from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    log_loss,
    balanced_accuracy_score,
)


@dataclass
class MetricContainer:
    """Container holding a list of metrics.

    Parameters
    ----------
    metric_names : list of str
        List of metric names.
    prefix : str
        Prefix of metric names.

    """

    metric_names: List[str]
    prefix: str = ""

    def __post_init__(self):
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        self.names = [self.prefix + name for name in self.metric_names]

    def __call__(self, y_true, y_pred):
        """Compute all metrics and store into a dict.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_pred: np.ndarray
            Score matrix or vector

        Returns
        -------
        dict
            Dict of metrics ({metric_name: metric_value}).

        """
        logs = {}
        for metric in self.metrics:
            if isinstance(y_pred, list):
                res = np.mean(
                    [metric(y_true[:, i], y_pred[i]) for i in range(len(y_pred))]
                )
            else:
                res = metric(y_true, y_pred)
            logs[self.prefix + metric._name] = res
        return logs


class Metric:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError("Custom Metrics must implement this function")

    @classmethod
    def get_metrics_by_names(cls, names):
        """Get list of metric classes.

        Parameters
        ----------
        cls : Metric
            Metric class.
        names : list
            List of metric names.

        Returns
        -------
        metrics : list
            List of metric classes.

        """
        available_metrics = cls.__subclasses__()
        available_names = [metric()._name for metric in available_metrics]
        metrics = []
        for name in names:
            assert name in available_names, f"{name} is not available, choose in {available_names}"
            idx = available_names.index(name)
            metric = available_metrics[idx]()
            metrics.append(metric)
        return metrics


class AUC(Metric):
    """
    AUC.
    """

    def __init__(self):
        self._name = "auc"
        self._maximize = True

    def __call__(self, y_true, y_score):
        """
        Compute AUC of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            AUC of predictions vs targets.
        """
        return roc_auc_score(y_true, y_score[:, 1])


class Accuracy(Metric):
    """
    Accuracy.
    """

    def __init__(self):
        self._name = "accuracy"
        self._maximize = True

    def __call__(self, y_true, y_score):
        """
        Compute Accuracy of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            Accuracy of predictions vs targets.
        """
        y_pred = np.argmax(y_score, axis=1)
        return accuracy_score(y_true, y_pred)


class BalancedAccuracy(Metric):
    """
    Balanced Accuracy.
    """

    def __init__(self):
        self._name = "balanced_accuracy"
        self._maximize = True

    def __call__(self, y_true, y_score):
        """
        Compute Accuracy of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            Accuracy of predictions vs targets.
        """
        y_pred = np.argmax(y_score, axis=1)
        return balanced_accuracy_score(y_true, y_pred)


class LogLoss(Metric):
    """
    LogLoss.
    """

    def __init__(self):
        self._name = "logloss"
        self._maximize = False

    def __call__(self, y_true, y_score):
        """
        Compute LogLoss of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            LogLoss of predictions vs targets.
        """
        return log_loss(y_true, y_score)


class MAE(Metric):
    """
    Mean Absolute Error.
    """

    def __init__(self):
        self._name = "mae"
        self._maximize = False

    def __call__(self, y_true, y_score):
        """
        Compute MAE (Mean Absolute Error) of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_pred: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            MAE of predictions vs targets.
        """
        return mean_absolute_error(y_true, y_score)


class MSE(Metric):
    """
    Mean Squared Error.
    """

    def __init__(self):
        self._name = "mse"
        self._maximize = False

    def __call__(self, y_true, y_score):
        """
        Compute MSE (Mean Squared Error) of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_pred: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            MSE of predictions vs targets.
        """
        return mean_squared_error(y_true, y_score)


class RMSE(Metric):
    """
    Root Mean Squared Error.
    """

    def __init__(self):
        self._name = "rmse"
        self._maximize = False

    def __call__(self, y_true, y_score):
        """
        Compute RMSE (Root Mean Squared Error) of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_pred: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            RMSE of predictions vs targets.
        """
        return np.sqrt(mean_squared_error(y_true, y_score))


def check_metrics(metrics):
    """Check if custom metrics are provided.

    Parameters
    ----------
    metrics : list of str or classes
        List with built-in metrics (str) or custom metrics (classes).

    Returns
    -------
    val_metrics : list of str
        List of metric names.

    """
    val_metrics = []
    for metric in metrics:
        if isinstance(metric, str):
            val_metrics.append(metric)
        elif issubclass(metric, Metric):
            val_metrics.append(metric()._name)
        else:
            raise TypeError("You need to provide a valid metric format")
    return val_metrics
