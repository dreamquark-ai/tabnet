import numpy as np
import torch
from pytorch_tabnet.metrics import UnsupervisedLoss, UnsupervisedLossNumpy

torch.set_printoptions(precision=10)


def test_equal_losses():
    y_pred = np.random.uniform(low=-2, high=2, size=(20, 100))
    embedded_x = np.random.uniform(low=-2, high=2, size=(20, 100))
    obf_vars = np.random.choice([0, 1], size=(20, 100), replace=True)

    numpy_loss = UnsupervisedLossNumpy(
        y_pred=y_pred,
        embedded_x=embedded_x,
        obf_vars=obf_vars
    )

    torch_loss = UnsupervisedLoss(
        y_pred=torch.tensor(y_pred, dtype=torch.float64),
        embedded_x=torch.tensor(embedded_x, dtype=torch.float64),
        obf_vars=torch.tensor(obf_vars, dtype=torch.float64)
    )

    assert np.isclose(numpy_loss, torch_loss.detach().numpy())
