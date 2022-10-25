import tqdm
import numpy as np

from torch.utils.data import DataLoader
from .activation import BaseFunction
from .optimizer import BaseOptimizer
from .model import Model


class Trainer:
    def __init__(
        self,
        model: Model,
        optimizer: BaseOptimizer,
        loss_func: BaseFunction
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.batch_size = None

    def backward(self, Y: np.ndarray):
        """
        Arguments:
        Y: Ground truth/label vector.

        Return: 
        A list of tuples of matrices (weights' gradient, biases' gradient) both in np.array format.
        The order of this list should be the same as the model's weights. 
        For example: [(dW0, db0), (dW1, db1), ... ].
        """

        n_layers = len(self.model.layers_dims)
        grads = [None] * (n_layers-1)

        # chain rule (derivate last layer)
        dC = self.loss_func.grad(Y, self.model.activations[-1])
        dA = self.model.activation_funcs[-1].grad(self.model.Z_list[-1])
        # dC/dA . dA/dZ
        delta = dC * dA

        # weight and bias gradients
        dW = np.dot(self.model.activations[-2].T, delta) / self.batch_size
        dB = np.sum(delta, axis=0, keepdims=True) / self.batch_size
        grads[-1] = (dW, dB)

        for l in range(2, n_layers):
            # chain rula (derivate hidden layers)
            dC = np.dot(delta, self.model.weights[-l+1].T)
            dA = self.model.activation_funcs[-l].grad(self.model.Z_list[-l])
            delta = dC * dA

            dW = np.dot(
                self.model.activations[-l-1].T, delta) / self.batch_size
            dB = np.sum(delta, axis=0, keepdims=True) / self.batch_size
            grads[-l] = (dW, dB)

        return grads

    def train(
        self,
        n_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        """
        Arguments:
        n_epochs: Number of epochs
        train_loader: Train DataLoader
        val_loader: Validation DataLoader

        Return: 
        A dictionary with the log of train and validation loss along the epochs
        """
        log_dict = {
            'epoch': [],
            'train_loss': [],
            'val_loss': []
        }

        self.batch_size = train_loader.batch_size
        loop = tqdm.trange(n_epochs)
        for epoch in loop:
            train_loss_history = []

            for i, batch in enumerate(train_loader):
                X, Y = batch
                X = X.numpy()
                Y = Y.numpy()

                Y_pred = self.model.forward(X)
                train_loss = self.loss_func(Y, Y_pred)
                train_loss_history.append(train_loss)

                grads = self.backward(Y)
                self.optimizer.step(grads)

            val_loss_history = []
            for i, batch in enumerate(val_loader):
                X, Y = batch
                X = X.numpy()
                Y = Y.numpy()

                Y_pred = self.model.forward(X)
                val_loss = self.loss_func(Y, Y_pred)
                val_loss_history.append(val_loss)

            # appending losses to history
            train_loss = np.array(train_loss_history).mean()
            val_loss = np.array(val_loss_history).mean()

            # tqdm
            loop.set_postfix(train_loss=train_loss, val_loss=val_loss)

            log_dict['epoch'].append(epoch)
            log_dict['train_loss'].append(train_loss)
            log_dict['val_loss'].append(val_loss)

        return log_dict
