import time
import datetime
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any


class Callback:
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_trainer(self, model):
        self.trainer = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


@dataclass
class CallbackContainer:
    """
    Container holding a list of callbacks.
    """

    callbacks: List[Callback] = field(default_factory=list)

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        logs["start_time"] = time.time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)


@dataclass
class EarlyStopping(Callback):
    """EarlyStopping callback to exit the training loop if early_stopping_metric
    does not improve by a certain amount for a certain
    number of epochs.

    Parameters
    ---------
    early_stopping_metric : str
        Early stopping metric name
    is_maximize : bool
        Whether to maximize or not early_stopping_metric
    tol : float
        minimum change in monitored value to qualify as improvement.
        This number should be positive.
    patience : integer
        number of epochs to wait for improvment before terminating.
        the counter be reset after each improvment

    """

    early_stopping_metric: str
    is_maximize: bool
    tol: float = 0.0
    patience: int = 5

    def __post_init__(self):
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.wait = 0
        self.best_weights = None
        self.best_loss = np.inf
        if self.is_maximize:
            self.best_loss = -self.best_loss
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.early_stopping_metric)
        if current_loss is None:
            return
        loss_change = current_loss - self.best_loss
        max_improved = self.is_maximize and loss_change > self.tol
        min_improved = (not self.is_maximize) and (-loss_change > self.tol)
        if max_improved or min_improved:
            self.best_loss = current_loss
            self.wait = 1
            self.best_weights = copy.deepcopy(self.trainer.network.state_dict())
            self.best_epoch = epoch
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.trainer._stop_training = True
            self.wait += 1

    def on_train_end(self, logs=None):
        self.trainer.best_epoch = self.best_epoch
        self.trainer.best_cost = self.best_loss
        final_weights = (
            self.best_weights
            if self.best_weights is not None
            else copy.deepcopy(self.trainer.network.state_dict())
        )
        self.trainer.network.load_state_dict(final_weights)
        if self.stopped_epoch > 0:
            msg = f"\nEarly stopping occured at epoch {self.stopped_epoch}"
            msg += (
                f" with best_epoch = {self.best_epoch} and "
                + f"best_{self.early_stopping_metric} = {round(self.best_loss, 5)}"
            )
            print(msg)
        else:
            msg = f"Stop training because you reached max_epochs = {self.trainer.max_epochs}"
            print(msg)


@dataclass
class History(Callback):
    """Callback that records events into a `History` object.
    This callback is automatically applied to
    every SuperModule.

    Parameters
    ---------
    trainer : DeepRecoModel
        Model class to train
    verbose : int
        Print results every verbose iteration

    """

    trainer: Any
    verbose: int = 1

    def __post_init__(self):
        super().__init__()
        self.samples_seen = 0.0
        self.total_time = 0.0

    def on_train_begin(self, logs=None):
        self.epoch_metrics = {"loss": []}
        self.epoch_metrics.update({"lr": []})
        self.epoch_metrics.update({name: [] for name in self.trainer._metrics_names})
        self.start_time = logs["start_time"]

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_metrics = {"loss": 0.0}
        self.samples_seen = 0.0

    def on_epoch_end(self, epoch, logs=None):
        for k in self.batch_metrics:
            self.epoch_metrics[k].append(self.batch_metrics[k])
        if self.verbose == 0:
            return
        if epoch % self.verbose != 0:
            return
        msg = f"epoch: {epoch:<4}"
        for metric_name, metric_value in self.batch_metrics.items():
            if metric_name != "lr":
                msg += f"|  {metric_name:<5}: {np.round(metric_value, 5):<8}"
        self.total_time = int(time.time() - self.start_time)
        msg += f"|  {str(datetime.timedelta(seconds=self.total_time)) + 's':<6}"
        print(msg)

    def on_batch_end(self, batch, logs=None):
        batch_size = logs["batch_size"]
        for k in self.batch_metrics:
            self.batch_metrics[k] = (
                self.samples_seen * self.batch_metrics[k] + logs[k] * batch_size
            ) / (self.samples_seen + batch_size)
        self.samples_seen += batch_size

    def __getitem__(self, name):
        return self.epoch_metrics[name]

    def __repr__(self):
        return str(self.epoch_metrics)

    def __str__(self):
        return str(self.epoch_metrics)
