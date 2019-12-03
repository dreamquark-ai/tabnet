import torch
import numpy as np
from tqdm import tqdm
from abc import abstractmethod
from pytorch_tabnet import tab_network
from pytorch_tabnet.multiclass_utils import unique_labels
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from torch.nn.utils import clip_grad_norm_
from pytorch_tabnet.utils import PredictDataset, plot_losses, create_dataloaders
from torch.utils.data import DataLoader


class TabModel(object):
    def __init__(self, n_d=8, n_a=8, n_steps=3, gamma=1.3, cat_idxs=[], cat_dims=[], cat_emb_dim=1,
                 n_independent=2, n_shared=2, epsilon=1e-15,  momentum=0.02,
                 lambda_sparse=1e-3, seed=0,
                 clip_value=1, verbose=1,
                 lr=2e-2, optimizer_fn=torch.optim.Adam,
                 scheduler_params=None, scheduler_fn=None,
                 device_name='auto', saving_path="./", model_name="DreamQuarkTabNet"):
        """ Class for TabNet model

        Parameters
        ----------
            device_name: str
                'cuda' if running on GPU, 'cpu' if not, 'auto' to autodetect
        """

        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dim = cat_emb_dim
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.epsilon = epsilon
        self.momentum = momentum
        self.lambda_sparse = lambda_sparse
        self.clip_value = clip_value
        self.verbose = verbose
        self.lr = lr
        self.optimizer_fn = optimizer_fn
        self.device_name = device_name
        self.saving_path = saving_path
        self.model_name = model_name

        self.scheduler_params = scheduler_params
        self.scheduler_fn = scheduler_fn

        self.opt_params = {}
        self.opt_params['lr'] = self.lr

        self.seed = seed
        torch.manual_seed(self.seed)
        # Defining device
        if device_name == 'auto':
            if torch.cuda.is_available():
                device_name = 'cuda'
            else:
                device_name = 'cpu'
        self.device = torch.device(device_name)
        print(f"Device used : {self.device}")

    @abstractmethod
    def construct_loaders(self, X_train, y_train, X_valid, y_valid, weights, batch_size):
        """
        Returns
        -------
        train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader
            Training and validation dataloaders
        -------
        """
        raise NotImplementedError('users must define construct_loaders to use this base class')

    def fit(self, X_train, y_train, X_valid=None, y_valid=None,
            loss_fn=None,
            weights=0, max_epochs=100, patience=10, batch_size=1024, virtual_batch_size=128):
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
            weights : bool or dictionnary
                0 for no balancing
                1 for automated balancing
                dict for custom weights per class
            max_epochs : int
                Maximum number of epochs during training
            patience : int
                Number of consecutive non improving epoch before early stopping
            batch_size : int
                Training batch size
            virtual_batch_size : int
                Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)
        """

        self.update_fit_params(X_train, y_train, X_valid, y_valid, loss_fn,
                               weights, max_epochs, patience, batch_size, virtual_batch_size)

        train_dataloader, valid_dataloader = self.construct_loaders(X_train,
                                                                    y_train,
                                                                    X_valid,
                                                                    y_valid,
                                                                    self.updated_weights,
                                                                    self.batch_size)

        self.network = tab_network.TabNet(self.input_dim, self.output_dim,
                                          n_d=self.n_d, n_a=self.n_d,
                                          n_steps=self.n_steps, gamma=self.gamma,
                                          cat_idxs=self.cat_idxs, cat_dims=self.cat_dims,
                                          cat_emb_dim=self.cat_emb_dim,
                                          n_independent=self.n_independent, n_shared=self.n_shared,
                                          epsilon=self.epsilon,
                                          virtual_batch_size=self.virtual_batch_size,
                                          momentum=self.momentum,
                                          device_name=self.device_name).to(self.device)

        self.optimizer = self.optimizer_fn(self.network.parameters(),
                                           **self.opt_params)

        if self.scheduler_fn:
            self.scheduler = self.scheduler_fn(self.optimizer, **self.scheduler_params)
        else:
            self.scheduler = None

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
                # Saving model
                torch.save(self.network, self.saving_path+f"{self.model_name}.pt")
                # Updating feature_importances_
                self.feature_importances_ = fit_metrics['train']['feature_importances_']
            else:
                self.patience_counter += 1

            print("Best metric valid: ", self.best_cost)
            self.epoch += 1

            if self.epoch % self.verbose == 0:
                plot_losses(losses_train, losses_valid, metrics_train, metrics_valid)

        # load best models post training
        self.load_best_model()

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

    @abstractmethod
    def train_epoch(self, train_loader):
        """
        Trains one epoch of the network in self.network

        Parameters
        ----------
            train_loader: a :class: `torch.utils.data.Dataloader`
                DataLoader with train set
        """
        raise NotImplementedError('users must define train_epoch to use this base class')

    @abstractmethod
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
        raise NotImplementedError('users must define train_batch to use this base class')

    @abstractmethod
    def predict_epoch(self, loader):
        """
        Validates one epoch of the network in self.network

        Parameters
        ----------
            loader: a :class: `torch.utils.data.Dataloader`
                    DataLoader with validation set
        """
        raise NotImplementedError('users must define predict_epoch to use this base class')

    @abstractmethod
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
        raise NotImplementedError('users must define predict_batch to use this base class')

    def load_best_model(self):
        self.network = torch.load(self.saving_path+f"{self.model_name}.pt")

    @abstractmethod
    def predict(self, X):
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
            predictions: np.array
                Predictions of the regression problem or the last class
        """
        raise NotImplementedError('users must define predict to use this base class')

    def explain(self, X):
        """
        Return local explanation

        Parameters
        ----------
            data: a :tensor: `torch.Tensor`
                Input data
            target: a :tensor: `torch.Tensor`
                Target data

        Returns
        -------
            M_explain: matrix
                Importance per sample, per columns.
            masks: matrix
                Sparse matrix showing attention masks used by network.
        """
        self.network.eval()

        dataloader = DataLoader(PredictDataset(X),
                                batch_size=self.batch_size, shuffle=False)

        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()

            output, M_loss, M_explain, masks = self.network(data)
            for key, value in masks.items():
                masks[key] = value.cpu().detach().numpy()

            if batch_nb == 0:
                res_explain = M_explain.cpu().detach().numpy()
                res_masks = masks
            else:
                res_explain = np.vstack([res_explain,
                                         M_explain.cpu().detach().numpy()])
                for key, value in masks.items():
                    res_masks[key] = np.vstack([res_masks[key], value])
        return M_explain, res_masks


class TabNetClassifier(TabModel):

    def __repr__(self):
        repr_ = f"""TabNetClassifier(n_d={self.n_d}, n_a={self.n_a}, n_steps={self.n_steps},
                 lr={self.lr}, seed={self.seed},
                 gamma={self.gamma}, n_independent={self.n_independent}, n_shared={self.n_shared},
                 cat_idxs={self.cat_idxs},
                 cat_dims={self.cat_dims},
                 cat_emb_dim={self.cat_emb_dim},
                 lambda_sparse={self.lambda_sparse}, momentum={self.momentum},
                 clip_value={self.clip_value},
                 verbose={self.verbose}, device_name="{self.device_name}",
                 model_name="{self.model_name}", epsilon={self.epsilon},
                 optimizer_fn={str(self.optimizer_fn)},
                 scheduler_params={self.scheduler_params},
                 scheduler_fn={self.scheduler_fn}, saving_path="{self.saving_path}")"""
        return repr_

    def infer_output_dim(self, y_train, y_valid):
        """
        Infer output_dim from targets

        Parameters
        ----------
            y_train : np.array
                Training targets
            y_valid : np.array
                Validation targets

        Returns
        -------
            output_dim : int
                Number of classes for output
            train_labels : list
                Sorted list of initial classes
        """
        train_labels = unique_labels(y_train)
        output_dim = len(train_labels)

        if y_valid is not None:
            valid_labels = unique_labels(y_train)
            if not set(valid_labels).issubset(set(train_labels)):
                print(f"""Valid set -- {set(valid_labels)} --
                        contains unkown targets from training -- {set(train_labels)}""")
                raise
        return output_dim, train_labels

    def weight_updater(self, weights):
        """
        Updates weights dictionnary according to target_mapper.

        Parameters
        ----------
            weights : bool or dict
                Given weights for balancing training.
        Returns
        -------
            bool or dict
            Same bool if weights are bool, updated dict otherwise.

        """
        if isinstance(weights, int):
            return weights
        elif isinstance(weights, dict):
            return {self.target_mapper[key]: value
                    for key, value in weights.items()}
        else:
            print("Unknown type for weights, please provide 0, 1 or dictionnary")
            raise

    def construct_loaders(self, X_train, y_train, X_valid, y_valid, weights, batch_size):
        """
        Returns
        -------
        train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader
            Training and validation dataloaders
        -------
        """
        y_train_mapped = np.vectorize(self.target_mapper.get)(y_train)
        y_valid_mapped = np.vectorize(self.target_mapper.get)(y_valid)
        train_dataloader, valid_dataloader = create_dataloaders(X_train,
                                                                y_train_mapped,
                                                                X_valid,
                                                                y_valid_mapped,
                                                                weights,
                                                                batch_size)
        return train_dataloader, valid_dataloader

    def update_fit_params(self, X_train, y_train, X_valid, y_valid, loss_fn,
                          weights, max_epochs, patience, batch_size, virtual_batch_size):
        if loss_fn is None:
            self.loss_fn = torch.nn.functional.cross_entropy
        else:
            self.loss_fn = loss_fn
        assert X_train.shape[1] == X_valid.shape[1], "Dimension mismatch X_train X_valid"
        self.input_dim = X_train.shape[1]

        output_dim, train_labels = self.infer_output_dim(y_train, y_valid)
        self.output_dim = output_dim
        self.classes_ = train_labels
        self.target_mapper = {class_label: index
                              for index, class_label in enumerate(self.classes_)}

        self.weights = weights
        self.updated_weights = self.weight_updater(self.weights)

        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        # Initialize counters and histories.
        self.patience_counter = 0
        self.epoch = 0
        self.best_cost = np.inf

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
        feature_importances_ = np.zeros((self.input_dim))
        with tqdm() as pbar:
            for data, targets in train_loader:
                batch_outs = self.train_batch(data, targets)
                if self.output_dim == 2:
                    y_preds.append(batch_outs["y_preds"][:, 1].cpu().detach().numpy())
                else:
                    values, indices = torch.max(batch_outs["y_preds"], dim=1)
                    y_preds.append(indices.cpu().detach().numpy())
                ys.append(batch_outs["y"].cpu().detach().numpy())
                total_loss += batch_outs["loss"]
                feature_importances_ += batch_outs['batch_importance']
                pbar.update(1)

        # Normalize feature_importances_
        feature_importances_ = feature_importances_ / np.sum(feature_importances_)

        y_preds = np.hstack(y_preds)
        ys = np.hstack(ys)

        if self.output_dim == 2:
            stopping_loss = -roc_auc_score(y_true=ys, y_score=y_preds)
        else:
            stopping_loss = -accuracy_score(y_true=ys, y_pred=y_preds)
        total_loss = total_loss / len(train_loader)
        epoch_metrics = {'loss_avg': total_loss,
                         'stopping_loss': stopping_loss,
                         'feature_importances_': feature_importances_
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
                      'y': targets,
                      'batch_importance': M_explain.sum(dim=0).detach().numpy()}
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
            if self.output_dim == 2:
                y_preds.append(batch_outs["y_preds"][:, 1].cpu().detach().numpy())
            else:
                values, indices = torch.max(batch_outs["y_preds"], dim=1)
                y_preds.append(indices.cpu().detach().numpy())
            ys.append(batch_outs["y"].cpu().detach().numpy())

        y_preds = np.hstack(y_preds)
        ys = np.hstack(ys)

        if self.output_dim == 2:
            stopping_loss = -roc_auc_score(y_true=ys, y_score=y_preds)
        else:
            stopping_loss = -accuracy_score(y_true=ys, y_pred=y_preds)

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

    def predict(self, X):
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
            predictions: np.array
                Predictions of the regression problem or the last class
        """
        self.network.eval()
        dataloader = DataLoader(PredictDataset(X),
                                batch_size=self.batch_size, shuffle=False)

        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()
            output, M_loss, M_explain, masks = self.network(data)
            predictions = torch.argmax(torch.nn.Softmax(dim=1)(output),
                                       dim=1)
            predictions = predictions.cpu().detach().numpy().reshape(-1)
            if batch_nb == 0:
                res = predictions
            else:
                res = np.hstack([res, predictions])

        return res

    def predict_proba(self, X):
        """
        Make predictions for classification on a batch (valid)

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

        dataloader = DataLoader(PredictDataset(X),
                                batch_size=self.batch_size, shuffle=False)

        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()

            output, M_loss, M_explain, masks = self.network(data)
            predictions = torch.nn.Softmax(dim=1)(output).cpu().detach().numpy()
            if batch_nb == 0:
                res = predictions
            else:
                res = np.vstack([res, predictions])
        return res


class TabNetRegressor(TabModel):

    def __repr__(self):
        repr_ = f"""TabNetRegressor(n_d={self.n_d}, n_a={self.n_a}, n_steps={self.n_steps},
                lr={self.lr}, seed={self.seed},
                gamma={self.gamma}, n_independent={self.n_independent}, n_shared={self.n_shared},
                cat_idxs={self.cat_idxs},
                cat_dims={self.cat_dims},
                cat_emb_dim={self.cat_emb_dim},
                lambda_sparse={self.lambda_sparse}, momentum={self.momentum},
                clip_value={self.clip_value},
                verbose={self.verbose}, device_name="{self.device_name}",
                model_name="{self.model_name}",
                optimizer_fn={str(self.optimizer_fn)},
                scheduler_params={self.scheduler_params}, scheduler_fn={self.scheduler_fn},
                epsilon={self.epsilon}, saving_path="{self.saving_path}")"""
        return repr_

    def construct_loaders(self, X_train, y_train, X_valid, y_valid, weights, batch_size):
        """
        Returns
        -------
        train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader
            Training and validation dataloaders
        -------
        """
        train_dataloader, valid_dataloader = create_dataloaders(X_train,
                                                                y_train,
                                                                X_valid,
                                                                y_valid,
                                                                0,
                                                                batch_size)
        return train_dataloader, valid_dataloader

    def update_fit_params(self, X_train, y_train, X_valid, y_valid, loss_fn,
                          weights, max_epochs, patience, batch_size, virtual_batch_size):

        if loss_fn is None:
            self.loss_fn = torch.nn.functional.mse_loss
        else:
            self.loss_fn = loss_fn

        assert X_train.shape[1] == X_valid.shape[1], "Dimension mismatch X_train X_valid"
        self.input_dim = X_train.shape[1]

        self.output_dim = 1

        self.weights = 0  # No weights for regression
        self.updated_weights = 0

        if self.scheduler_fn:
            self.scheduler = self.scheduler_fn(self.optimizer, **self.scheduler_params)
        else:
            self.scheduler = None

        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        # Initialize counters and histories.
        self.patience_counter = 0
        self.epoch = 0
        self.best_cost = np.inf

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
        feature_importances_ = np.zeros((self.input_dim))
        with tqdm() as pbar:
            for data, targets in train_loader:
                batch_outs = self.train_batch(data, targets)
                y_preds.append(batch_outs["y_preds"].cpu().detach().numpy().flatten())
                ys.append(batch_outs["y"].cpu().detach().numpy())
                total_loss += batch_outs["loss"]
                feature_importances_ += batch_outs['batch_importance']
                pbar.update(1)

        # Normalize feature_importances_
        feature_importances_ = feature_importances_ / np.sum(feature_importances_)

        y_preds = np.hstack(y_preds)
        ys = np.hstack(ys)

        stopping_loss = mean_squared_error(y_true=ys, y_pred=y_preds)
        total_loss = total_loss / len(train_loader)
        epoch_metrics = {'loss_avg': total_loss,
                         'stopping_loss': stopping_loss,
                         'feature_importances_': feature_importances_
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

        targets = targets.to(self.device).float()
        self.optimizer.zero_grad()

        output, M_loss, M_explain, _ = self.network(data)

        loss = self.loss_fn(output.view(-1), targets)
        loss -= self.lambda_sparse*M_loss

        loss.backward()
        if self.clip_value:
            clip_grad_norm_(self.network.parameters(), self.clip_value)
        self.optimizer.step()

        loss_value = loss.item()
        batch_outs = {'loss': loss_value,
                      'y_preds': output,
                      'y': targets,
                      'batch_importance':  M_explain.sum(dim=0).detach().numpy()}
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
            y_preds.append(batch_outs["y_preds"].cpu().detach().numpy().flatten())
            ys.append(batch_outs["y"].cpu().detach().numpy())

        y_preds = np.hstack(y_preds)
        ys = np.hstack(ys)

        stopping_loss = mean_squared_error(y_true=ys, y_pred=y_preds)

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
        targets = targets.to(self.device).float()

        output, M_loss, M_explain, _ = self.network(data)

        loss = self.loss_fn(output.view(-1), targets)
        loss -= self.lambda_sparse*M_loss

        loss_value = loss.item()
        batch_outs = {'loss': loss_value,
                      'y_preds': output,
                      'y': targets}
        return batch_outs

    def predict(self, X):
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
            predictions: np.array
                Predictions of the regression problem or the last class
        """
        self.network.eval()
        dataloader = DataLoader(PredictDataset(X),
                                batch_size=self.batch_size, shuffle=False)

        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()

            output, M_loss, M_explain, masks = self.network(data)
            predictions = output.cpu().detach().numpy().reshape(-1)
            if batch_nb == 0:
                res = predictions
            else:
                res = np.hstack([res, predictions])

        return res
