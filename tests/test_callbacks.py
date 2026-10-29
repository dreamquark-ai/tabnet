import pytest
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from pytorch_tabnet.callbacks import LRSchedulerCallback


@pytest.fixture
def optimizer():
    return torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)


def test_plateau_scheduler_receives_metric(optimizer):
    callback = LRSchedulerCallback(
        ReduceLROnPlateau, optimizer, {"patience": 0, "factor": 0.5}, "loss"
    )
    callback.on_epoch_end(0, {"loss": 1.0})
    callback.on_epoch_end(1, {"loss": 2.0})
    assert callback.scheduler.best == 1.0
    assert optimizer.param_groups[0]["lr"] == 0.5


@pytest.mark.parametrize("is_batch_level", [False, True])
def test_step_scheduler_without_metric(optimizer, is_batch_level):
    callback = LRSchedulerCallback(
        StepLR, optimizer, {"step_size": 1, "gamma": 0.5}, "loss",
        is_batch_level=is_batch_level,
    )
    optimizer.step()
    callback.on_batch_end(0)
    callback.on_epoch_end(0, {"loss": 10.0})
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_custom_metric_scheduler(optimizer):
    class MetricScheduler:
        def __init__(self, optimizer):
            self.metrics = []

        def is_better(self, current, best):
            return current < best

        def step(self, metric):
            self.metrics.append(metric)

    callback = LRSchedulerCallback(MetricScheduler, optimizer, {}, "loss")
    callback.on_epoch_end(0, {"other": 3.0})
    callback.on_epoch_end(1, {"loss": 2.0})
    assert callback.scheduler.metrics == [2.0]
