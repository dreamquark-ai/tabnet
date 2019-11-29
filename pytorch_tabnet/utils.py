from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
from IPython.display import clear_output


class TorchDataset(Dataset):
    """
    Format for numpy array

    Parameters
    ----------
        X: 2D array
            The input matrix
        y: 2D array
            The one-hot encoded target
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return x, y


class PredictDataset(Dataset):
    """
    Format for numpy array

    Parameters
    ----------
        X: 2D array
            The input matrix
    """

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        return x


def create_dataloaders(X_train, y_train, X_valid, y_valid, weights, batch_size):
    """
    Create dataloaders with or wihtout subsampling depending on weights and balanced.

    Parameters
    ----------
        X_train: np.ndarray
            Training data
        y_train: np.array
            Mapped Training targets
        X_valid: np.ndarray
            Validation data
        y_valid: np.array
            Mapped Validation targets
        weights : dictionnary or bool
            Weight for each mapped target class
            0 for no sampling
            1 for balanced sampling
    Returns
    -------
        train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader
            Training and validation dataloaders
    """
    if weights == 0:
        train_dataloader = DataLoader(TorchDataset(X_train, y_train),
                                      batch_size=batch_size, shuffle=True)
    else:
        if weights == 1:
            class_sample_count = np.array(
                [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

            weights = 1. / class_sample_count

            samples_weight = np.array([weights[t] for t in y_train])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
        else:
            # custom weights
            samples_weight = np.array([weights[t] for t in y_train])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_dataloader = DataLoader(TorchDataset(X_train, y_train),
                                      batch_size=batch_size, sampler=sampler)

    valid_dataloader = DataLoader(TorchDataset(X_valid, y_valid),
                                  batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader


def plot_losses(losses_train, losses_valid, metrics_train, metrics_valid):
    """
    Plot train and validation losses.

    Parameters
    ----------
        losses_train : list
            list of train losses per epoch
        losses_valid : list
            list of valid losses per epoch
        metrics_train : list
            list of train metrics per epoch
        metrics_valid : list
            list of valid metrics per epoch
    Returns
    ------
    plot
    """
    clear_output()
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(losses_train)), losses_train, label='Train')
    plt.plot(range(len(losses_valid)), losses_valid, label='Valid')
    plt.grid()
    plt.title('Losses')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(metrics_train)), metrics_train, label='Train')
    plt.plot(range(len(metrics_valid)), metrics_valid, label='Valid')
    plt.grid()
    plt.title('Training Metrics')
    plt.legend()
    plt.show()
