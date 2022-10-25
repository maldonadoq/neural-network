import numpy as np
from typing import List
from .activation import BaseFunction


class Model:
    def __init__(
        self,
        layers_dims: List[int],
        activation_funcs: List[BaseFunction],
        initialization_method: str = "random"
    ):
        """
        Arguments:
        layers_dims: A list with the size of each layer
        activation_funcs: A list with the activation functions 
        initialization_method: Indicates how to initialize the parameters
        """

        assert all([isinstance(d, int) for d in layers_dims]), \
            "It is expected a list of int to the param ``layers_dims"
        assert all([isinstance(a, BaseFunction) for a in activation_funcs]),\
            "It is expected a list of BaseFunction to the param ``activation_funcs´´"

        self.layers_dims = layers_dims
        self.activation_funcs = activation_funcs
        self.weights, self.bias = self.initialize_model(initialization_method)

    def __len__(self):
        return len(self.weights)

    def initialize_model(self, method: str = "random", value: int = 0):
        """
        Arguments:
        method: Indicates how to initialize the parameters

        Return: A list of matrices (np.ndarray) of weights and a list of 
        matrices of biases.
        """

        weights = []
        bias = []
        n_layers = len(self.layers_dims)
        for l in range(0, n_layers-1):
            # the weight w_i,j  connects the i-th neuron in the current layer to
            # the j-th neuron in the next layer
            W = np.random.randn(self.layers_dims[l], self.layers_dims[l + 1])
            b = np.random.randn(1, self.layers_dims[l + 1])

            # He et al. Normal initialization
            if method.lower() == 'he':
                n = self.layers_dims[l]
                W = W * np.sqrt(2/n)
                b = b * np.sqrt(2/n)
            # Glorot & Bengio Normal initialization
            elif method.lower() == 'glorot':
                n = self.layers_dims[l] + self.layers_dims[l+1]
                W = W * np.sqrt(2/n)
                b = b * np.sqrt(2/n)
            # Constant initialization
            elif method.lower() == 'constant':
                W = W * value
                b = b * value

            weights.append(W)
            bias.append(b)

        return weights, bias

    def forward(self, X: np.ndarray):
        """
        Arguments:
        X: input data

        Return:
        Predictions for the input data (np.ndarray)
        """
        activation = X
        self.activations = [X]
        self.Z_list = []

        n_layers = len(self.layers_dims)
        for l in range(0, n_layers-1):
            # (W.T x a) + b ?
            Z = np.dot(activation, self.weights[l]) + self.bias[l]
            activation = self.activation_funcs[l](Z)

            self.Z_list.append(Z)
            self.activations.append(activation)

        return activation
