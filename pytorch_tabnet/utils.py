from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
import scipy


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


def create_dataloaders(X_train, y_train, X_valid, y_valid, weights,
                       batch_size, num_workers, drop_last):
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
        weights : either 0, 1, dict or iterable
            if 0 (default) : no weights will be applied
            if 1 : classification only, will balanced class with inverse frequency
            if dict : keys are corresponding class values are sample weights
            if iterable : list or np array must be of length equal to nb elements
                          in the training set
    Returns
    -------
        train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader
            Training and validation dataloaders
    """

    if isinstance(weights, int):
        if weights == 0:
            need_shuffle = True
            sampler = None
        elif weights == 1:
            need_shuffle = False
            class_sample_count = np.array(
                [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

            weights = 1. / class_sample_count

            samples_weight = np.array([weights[t] for t in y_train])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            raise ValueError('Weights should be either 0, 1, dictionnary or list.')
    elif isinstance(weights, dict):
        # custom weights per class
        need_shuffle = False
        samples_weight = np.array([weights[t] for t in y_train])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    else:
        # custom weights
        if len(weights) != len(y_train):
            raise ValueError('Custom weights should match number of train samples.')
        need_shuffle = False
        samples_weight = np.array(weights)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_dataloader = DataLoader(TorchDataset(X_train, y_train),
                                  batch_size=batch_size,
                                  sampler=sampler,
                                  shuffle=need_shuffle,
                                  num_workers=num_workers,
                                  drop_last=drop_last)

    valid_dataloader = DataLoader(TorchDataset(X_valid, y_valid),
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    return train_dataloader, valid_dataloader


def create_explain_matrix(input_dim, cat_emb_dim, cat_idxs, post_embed_dim):
    """
    This is a computational trick.
    In order to rapidly sum importances from same embeddings
    to the initial index.

    Parameters
    ----------
    input_dim: int
        Initial input dim
    cat_emb_dim : int or list of int
        if int : size of embedding for all categorical feature
        if list of int : size of embedding for each categorical feature
    cat_idxs : list of int
        Initial position of categorical features
    post_embed_dim : int
        Post embedding inputs dimension

    Returns
    -------
    reducing_matrix : np.array
        Matrix of dim (post_embed_dim, input_dim)  to performe reduce
    """

    if isinstance(cat_emb_dim, int):
        all_emb_impact = [cat_emb_dim-1]*len(cat_idxs)
    else:
        all_emb_impact = [emb_dim-1 for emb_dim in cat_emb_dim]

    acc_emb = 0
    nb_emb = 0
    indices_trick = []
    for i in range(input_dim):
        if i not in cat_idxs:
            indices_trick.append([i+acc_emb])
        else:
            indices_trick.append(range(i+acc_emb, i+acc_emb+all_emb_impact[nb_emb]+1))
            acc_emb += all_emb_impact[nb_emb]
            nb_emb += 1

    reducing_matrix = np.zeros((post_embed_dim, input_dim))
    for i, cols in enumerate(indices_trick):
        reducing_matrix[cols, i] = 1

    return scipy.sparse.csc_matrix(reducing_matrix)
