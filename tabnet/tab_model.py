import torch
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from torch.autograd import Variable
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


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
        self.timer = []

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return x, y


class Model(object):
    def __init__(self,
                 device_name='auto',):
        """ Class for TabNet model

        Parameters
        ----------
            type: str
                Model type ('classification' or 'regression').
            device_name: str
                'cuda' if running on GPU, 'cpu' if not, 'auto' to autodetect
            save: bool
                If True, save model in path. Model name will be the time stamp
                of its execution.
            load: str
                Name of model that should be loaded.
        """

        # Defining device
        if device_name == 'auto':
            if torch.cuda.is_available():
                device_name = 'cuda'
            else:
                device_name = 'cpu'
        self.device = torch.device(device_name)
        print(f"Device used : {self.device}")

    def def_network(self, network, **kwargs):
        """Defines network architecture and attributes **kwargs to network
        parameters, e.g. `input_dim`, `output_dim` and `layers`.
        If load is passed to model init, it ignores all parameters and load
        the file located at path/load.pt,

        Parameters
        ----------
            network: a :class: `nn.Module`
                The network whose weights will be updated in training.
                See `network.py` for possible networks.
        """
        self.network = network(**kwargs).to(self.device)

    def set_params(self, **kwargs):
        """Sets default hyperparameters and overrides default with
        values set in **kwargs.

        Parameters
        ----------
            loss_fn: :class: `torch.nn.functional`
                The loss function. A few options:
                    - torch.nn.functional.mse_loss
                    - torch.nn.functional.binary_cross_entropy
                    - torch.nn.functional.cross_entropy
                    - torch.nn.functional.l1_loss
            max_epochs: int
                The maximum number of epochs for training
            patience: int
            learning_rate: float
                The initial learning rate
            schedule: str
                The learning rate schedule('lambda', 'cos',
                'exp' or 'step' or None).
            lr_params: dict
                Additional infos on the learning rate scheduler
                See https://pytorch.org/docs/master/optim.html for params
                of each scheduler type.
            optimizer_fn: :class: torch.optim object
                The optimizer function to be used. A few options:
                    - torch.optim.SGD
                    - torch.optim.Adam
                    - torch.optim.Adadelta
            opt_params: dict
                Further parameters to be used by optimizer_fn.
                or None)

        """
        # default params
        self.max_epochs = 100
        self.patience = 15
        self.lr = 2e-2
        self.scheduler = None
        self.lr_params = {}
        self.opt_params = {}
        self.optimizer_fn = None
        self.clip_value = None
        self.model_name = "DQTabNet"
        self.lambda_sparse = 1e-3
        self.scheduler_fn = None
        self.patience_counter = 0
        self.batch_size = 1024
        self.saving_path = "./"
        self.verbose = 1

        # Overrides parametersk
        self.__dict__.update(kwargs)

        self.output_dim = self.network.output_dim

        if self.output_dim == 1:
            self.loss_fn = torch.nn.functional.mse_loss
        else:
            self.loss_fn = torch.nn.functional.cross_entropy

        self.opt_params['lr'] = self.lr

        if self.optimizer_fn is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(),
                                              **self.opt_params)
        else:
            self.optimizer = self.optimizer_fn(self.network.parameters(),
                                               **self.opt_params)

        if self.scheduler_fn:
            self.scheduler = self.scheduler_fn(self.optimizer, **self.scheduler_params)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None,
            balanced=False, weights=None):
        """Train a neural network stored in self.network
        Using train_dataloader for training data and
        valid_dataloader for validation.

        Parameters
        ----------
            X_train: np.ndarray
                Train set
            y_train : np.array
                Train targets
            X_train: np.ndarray
                Train set
            y_train : np.array
                Train targets
            balanced : bool
                If set to True, training will oversample less frequent classes
            weights : dictionnary
                For classification problems only, a dictionnary with keys ranging from
                0 to output_dim - 1, with corresponding weights for each class

        """
        # Initialize counters and histories.
        self.patience_counter = 0
        self.epoch = 0
        self.best_cost = np.inf

        if balanced:
            if weights:
                samples_weight = np.array([weights[t] for t in y_train])
            else:
                class_sample_count = np.array(
                    [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

                weights = 1. / class_sample_count

            samples_weight = np.array([weights[t] for t in y_train])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weigth = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            train_dataloader = DataLoader(TorchDataset(X_train, y_train), batch_size=self.batch_size, sampler=sampler)
            valid_dataloader = DataLoader(TorchDataset(X_valid, y_valid), batch_size=self.batch_size, shuffle=False)

        train_dataloader = DataLoader(TorchDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(TorchDataset(X_valid, y_valid), batch_size=self.batch_size, shuffle=False)

        losses_train = []
        losses_valid = []

        metrics_train = []
        metrics_valid = []

        while (self.epoch < self.max_epochs and
               self.patience_counter < self.patience):
            print(f"EPOCH : {self.epoch}")
            fit_metrics = self.fit_epoch(train_dataloader, valid_dataloader)
            losses_train.append(fit_metrics['train']['loss_avg'])
            losses_valid.append(fit_metrics['valid']['total_loss'])
            metrics_train.append(fit_metrics['train']['stopping_loss'])
            metrics_valid.append(fit_metrics['valid']['stopping_loss'])


            stopping_loss = fit_metrics['valid']['stopping_loss']
            if stopping_loss < self.best_cost:
                self.best_cost = stopping_loss
                self.patience_counter = 0

                print("saving model")
                torch.save(self.network, self.saving_path+f"{self.model_name}.pt")
            else:
                self.patience_counter += 1

            print("Best metric valid: ", self.best_cost)
            self.epoch += 1

            if self.epoch % self.verbose == 0:
                clear_output()
                fig = plt.figure(figsize=(15, 5))
                plt.subplot(1, 2, 1)
                plt.plot(range(len(losses_train)), losses_train, label='Train')
                plt.plot(range(len(losses_valid)), losses_valid, label='Valid')
                plt.grid()
                plt.title('Losses')
                plt.legend()
                #plt.show()

                plt.subplot(1, 2, 2)
                plt.plot(range(len(metrics_train)), metrics_train, label='Train')
                plt.plot(range(len(metrics_valid)), metrics_valid, label='Valid')
                plt.grid()
                plt.title('Training Metrics')
                plt.legend()
                plt.show()

    def fit_epoch(self, train_dataloader, valid_dataloader):
        """
        Evaluates and updates network for one epoch.

        Parameters
        ----------
            train_dataloader: a :class: `torch.utils.data.Dataloader`
                DataLoader with train set
            valid_dataloader: a :class: `torch.utils.data.Dataloader`
                DataLoader with valid set
        """
        train_metrics = self.train_epoch(train_dataloader)
        valid_metrics = self.predict_epoch(valid_dataloader)

        fit_metrics = {'train': train_metrics,
                       'valid': valid_metrics}

        return fit_metrics

    def train_epoch(self, train_loader):
        """
        Trains one epoch of the network in self.network

        Parameters
        ----------
            train_loader: a :class: `torch.utils.data.Dataloader`
                DataLoader with train set
        """

        self.network.train()
        y_preds = []
        ys = []
        total_loss = 0

        with tqdm() as pbar:
            for data, targets in train_loader:
                batch_outs = self.train_batch(data, targets)
                if self.output_dim == 1:
                    y_preds.append(batch_outs["y_preds"].cpu().detach().numpy())
                elif self.output_dim == 2:
                    y_preds.append(batch_outs["y_preds"][:, 1].cpu().detach().numpy())
                else:
                    values, indices = torch.max(batch_outs["y_preds"], dim=1)
                    y_preds.append(indices.cpu().detach().numpy())
                ys.append(batch_outs["y"].cpu().detach().numpy())
                total_loss+=batch_outs["loss"]
                pbar.update(1)

        y_preds = np.hstack(y_preds)
        ys = np.hstack(ys)

        if self.output_dim == 2:
            stopping_loss = -roc_auc_score(y_true=ys, y_score=y_preds)
            # print("AUC train: ", -stopping_loss)
        elif self.output_dim == 1:
            stopping_loss = mean_squared_error(y_true=ys, y_pred=y_preds)
            # print("MSE train: ", stopping_loss)
        else:
            stopping_loss = -accuracy_score(y_true=ys, y_pred=y_preds)
            # print("ACCURACY Train ", -stopping_loss)
        total_loss = total_loss / len(train_loader)
        epoch_metrics = {'loss_avg': total_loss,
                         'stopping_loss': stopping_loss
                         }

        if self.scheduler is not None:
            self.scheduler.step()
            print("Current learning rate: ", self.optimizer.param_groups[-1]["lr"])
        return epoch_metrics

    def train_batch(self, data, targets):
        """
        Trains one batch of data

        Parameters
        ----------
            data: a :tensor: `torch.tensor`
                Input data
            target: a :tensor: `torch.tensor`
                Target data
        """
        self.network.train()
        data = data.to(self.device).float()
        targets = targets.to(self.device).long()
        self.optimizer.zero_grad()

        output, M_loss, M_explain, _ = self.network(data)

        loss = self.loss_fn(output, targets)
        loss -= self.lambda_sparse*M_loss

        loss.backward()
        if self.clip_value:
            clip_grad_norm_(self.network.parameters(), self.clip_value)
        self.optimizer.step()

        loss_value = loss.item()
        batch_outs = {'loss': loss_value,
                       'y_preds': output,
                       'y': targets}
        return batch_outs

    def predict_epoch(self, loader):
        """
        Validates one epoch of the network in self.network

        Parameters
        ----------
            loader: a :class: `torch.utils.data.Dataloader`
                    DataLoader with validation set
        """
        y_preds = []
        ys = []
        self.network.eval()
        total_loss = 0

        for data, targets in loader:
            batch_outs = self.predict_batch(data, targets)
            total_loss += batch_outs["loss"]
            if self.output_dim == 1:
                y_preds.append(batch_outs["y_preds"].cpu().detach().numpy())
            elif self.output_dim == 2:
                y_preds.append(batch_outs["y_preds"][:, 1].cpu().detach().numpy())
            else:
                values, indices = torch.max(batch_outs["y_preds"], dim=1)
                y_preds.append(indices.cpu().detach().numpy())
            ys.append(batch_outs["y"].cpu().detach().numpy())

        y_preds = np.hstack(y_preds)
        ys = np.hstack(ys)

        if self.output_dim == 2:
            stopping_loss = -roc_auc_score(y_true=ys, y_score=y_preds)
            # print("AUC Valid: ", -stopping_loss)
        elif self.output_dim == 1:
            stopping_loss = mean_squared_error(y_true=ys, y_pred=y_preds)
            # print("MSE Valid: ", stopping_loss)
        else:
            stopping_loss = -accuracy_score(y_true=ys, y_pred=y_preds)
            # print("ACCURACY Valid ", -stopping_loss)

        total_loss = total_loss / len(loader)
        epoch_metrics = {'total_loss': total_loss,
                         'stopping_loss': stopping_loss}

        return epoch_metrics

    def predict_batch(self, data, targets):
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
            data: a :tensor: `torch.Tensor`
                Input data
            target: a :tensor: `torch.Tensor`
                Target data

        Returns
        -------
            batch_outs: dict
        """
        self.network.eval()
        data = data.to(self.device).float()
        targets = targets.to(self.device).long()

        output, M_loss, M_explain, _ = self.network(data)

        loss = self.loss_fn(output, targets)
        loss -= self.lambda_sparse*M_loss

        loss_value = loss.item()
        batch_outs = {'loss': loss_value,
                      'y_preds': output,
                      'y': targets}
        return batch_outs

    def load_best_model(self):
        self.network = torch.load(self.saving_path+f"{self.model_name}.pt")

    def predict_proba(self, X):
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
            data: a :tensor: `torch.Tensor`
                Input data
            target: a :tensor: `torch.Tensor`
                Target data

        Returns
        -------
            batch_outs: dict
        """
        self.network.eval()
        data = torch.Tensor(X).to(self.device).float()

        output, M_loss, M_explain, masks = self.network(data)
        predictions = output.cpu().detach().numpy()

        return predictions, M_explain, masks
