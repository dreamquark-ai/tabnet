import torch
import numpy as np
from pytorch_tabnet.multiclass_utils import unique_labels
from torch.nn.utils import clip_grad_norm_
from pytorch_tabnet.utils import (PredictDataset,
                                  create_dataloaders,
                                  filter_weights)
from pytorch_tabnet.abstract_model import TabModel
from torch.utils.data import DataLoader


class TabNetMultiTaskClassifier(TabModel):

    def infer_output_dim(self, y_train, y_valid):
        """
        Infer output_dim from targets
        This is for simple 1D np arrays
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
            valid_labels = unique_labels(y_valid)
            if not set(valid_labels).issubset(set(train_labels)):
                raise ValueError(f"""Valid set -- {set(valid_labels)} --
                                 contains unkown targets from training --
                                 {set(train_labels)}""")
        return output_dim, train_labels

    def infer_multitask_output(self, y_train, y_valid):
        """
        Infer output_dim from targets
        This is for multiple tasks.

        Parameters
        ----------
            y_train : np.ndarray
                Training targets
            y_valid : np.ndarray
                Validation targets

        Returns
        -------
            tasks_dims : list
                Number of classes for output
            tasks_labels : list
                List of sorted list of initial classes
        """

        if len(y_train.shape) < 2:
            raise ValueError(f"""y_train shoud be of shape (n_examples, n_tasks) """ +
                             f"""but got {y_train.shape}""")
        nb_tasks = y_train.shape[1]
        tasks_dims = []
        tasks_labels = []
        for task_idx in range(nb_tasks):
            try:
                output_dim, train_labels = self.infer_output_dim(y_train[:, task_idx],
                                                                 y_valid[:, task_idx])
                tasks_dims.append(output_dim)
                tasks_labels.append(train_labels)
            except ValueError as err:
                raise ValueError(f"""Error for task {task_idx} : {err}""")
        return tasks_dims, tasks_labels

    def construct_loaders(self, X_train, y_train, X_valid, y_valid, weights,
                          batch_size, num_workers, drop_last):
        """
        Returns
        -------
        train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader
            Training and validation dataloaders
        -------
        """
        # all weights are not allowed for this type of model
        filter_weights(weights)
        y_train_mapped = y_train.copy()
        y_valid_mapped = y_valid.copy()
        for task_idx in range(y_train.shape[1]):
            task_mapper = self.target_mapper[task_idx]
            y_train_mapped[:, task_idx] = np.vectorize(task_mapper.get)(y_train[:, task_idx])
            y_valid_mapped[:, task_idx] = np.vectorize(task_mapper.get)(y_valid[:, task_idx])
        train_dataloader, valid_dataloader = create_dataloaders(X_train,
                                                                y_train_mapped,
                                                                X_valid,
                                                                y_valid_mapped,
                                                                weights,
                                                                batch_size,
                                                                num_workers,
                                                                drop_last)
        return train_dataloader, valid_dataloader

    def update_fit_params(self, X_train, y_train, X_valid, y_valid, loss_fn,
                          weights, max_epochs, patience,
                          batch_size, virtual_batch_size, num_workers, drop_last):

        if loss_fn is None:
            self.loss_fn = torch.nn.functional.cross_entropy
        else:
            self.loss_fn = loss_fn
        assert X_train.shape[1] == X_valid.shape[1], "Dimension mismatch X_train X_valid"
        self.input_dim = X_train.shape[1]

        output_dim, train_labels = self.infer_multitask_output(y_train, y_valid)
        self.output_dim = output_dim
        self.classes_ = train_labels
        self.target_mapper = [{class_label: index
                              for index, class_label in enumerate(classes)}
                              for classes in self.classes_]
        self.preds_mapper = [{index: class_label
                             for index, class_label in enumerate(classes)}
                             for classes in self.classes_]
        self.weights = weights
        self.updated_weights = weights

        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        # Initialize counters and histories.
        self.patience_counter = 0
        self.epoch = 0
        self.best_cost = np.inf
        self.num_workers = num_workers
        self.drop_last = drop_last

    def train_epoch(self, train_loader):
        """
        Trains one epoch of the network in self.network

        Parameters
        ----------
            train_loader: a :class: `torch.utils.data.Dataloader`
                DataLoader with train set
        """

        self.network.train()
        total_loss = 0

        for data, targets in train_loader:
            batch_outs = self.train_batch(data, targets)
            total_loss += batch_outs["loss"]
            # TODO : add stopping loss
        total_loss = total_loss / len(train_loader)
        epoch_metrics = {'loss_avg': total_loss,
                         'stopping_loss': total_loss,
                         }

        return epoch_metrics

    def compute_multi_loss(self, output, targets):
        """
            Computes the loss according to network output and targets

            Parameters
            ----------
                output: list of tensors
                    Output of network
                targets: LongTensor
                    Targets label encoded

        """
        loss = 0
        if isinstance(self.loss_fn, list):
            # if you specify a different loss for each task
            for task_loss, task_output, task_id in zip(self.loss_fn,
                                                       output,
                                                       range(len(self.loss_fn))):
                loss += task_loss(task_output, targets[:, task_id])
        else:
            # same loss function is applied to all tasks
            for task_id, task_output in enumerate(output):
                loss += self.loss_fn(task_output, targets[:, task_id])

        loss /= len(output)
        return loss

    def train_batch(self, data, targets):
        """
        Trains one batch of data

        Parameters
        ----------
            data: a :tensor: `torch.tensor`
                Input data
            target: a :tensor: `torch.tensor`
                Target data

        Returns
        -------
            batch_outs = {'loss': loss_value,
                        'y_preds': output,
                        'y': targets}
        """
        self.network.train()
        data = data.to(self.device).float()

        targets = targets.to(self.device).long()
        self.optimizer.zero_grad()

        output, M_loss = self.network(data)

        loss = self.compute_multi_loss(output, targets)
        # Add the overall sparsity loss
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
        self.network.eval()
        total_loss = 0
        for data, targets in loader:
            batch_outs = self.predict_batch(data, targets)
            total_loss += batch_outs["loss"]
            # TODO : add stopping loss
        total_loss = total_loss / len(loader)
        epoch_metrics = {'total_loss': total_loss,
                         'stopping_loss': total_loss,
                         }
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
        output, _ = self.network(data)

        loss = self.compute_multi_loss(output, targets)
        # Here we do not compute the sparsity loss

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
                Predictions of the most probable class
        """
        self.network.eval()
        dataloader = DataLoader(PredictDataset(X),
                                batch_size=self.batch_size, shuffle=False)

        results = {}
        for data in dataloader:
            data = data.to(self.device).float()
            output, _ = self.network(data)
            predictions = [torch.argmax(torch.nn.Softmax(dim=1)(task_output),
                                        dim=1).cpu().detach().numpy().reshape(-1)
                           for task_output in output]

            for task_idx in range(len(self.output_dim)):
                results[task_idx] = results.get(task_idx, []) + [predictions[task_idx]]
        # stack all task individually
        results = [np.hstack(task_res) for task_res in results.values()]
        # map all task individually
        results = [np.vectorize(self.preds_mapper[task_idx].get)(task_res)
                   for task_idx, task_res in enumerate(results)]
        return results

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
                                batch_size=self.batch_size,
                                shuffle=False)

        results = {}
        for data in dataloader:
            data = data.to(self.device).float()
            output, _ = self.network(data)
            predictions = [torch.nn.Softmax(dim=1)(task_output).cpu().detach().numpy()
                           for task_output in output]
            for task_idx in range(len(self.output_dim)):
                results[task_idx] = results.get(task_idx, []) + [predictions[task_idx]]
        res = [np.vstack(task_res) for task_res in results.values()]
        return res
