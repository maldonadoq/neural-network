import numpy as np
from typing import List
from abc import ABC, abstractmethod
from .model import Model


class BaseOptimizer(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def step(self, grads):
        pass


class SGDOptimizer(BaseOptimizer):
    def __init__(
        self,
        model: Model,
        lr: float = 1e-3
    ):
        self.model = model
        self.lr = lr

    def step(self, grads: List):
        """
        Arguments:
        grads: A list of tuples of matrices (weights' gradient, biases' gradient)
        both in np.ndarray format.
        """

        for idx, grad in enumerate(grads):
            self.model.weights[idx] -= self.lr * grad[0]
            self.model.bias[idx] -= self.lr * grad[1]


class SGDMomentumOptimizer(BaseOptimizer):
    def __init__(
        self,
        model: Model,
        lr: float = 1e-3,
        momentum: float = 0.2
    ):
        self.previous = [(0, 0)] * len(model)
        self.lr = lr
        self.model = model
        self.momentum = momentum

    def step(self, grads: List):
        """
        Arguments:
        grads: A list of tuples of matrices (weights' gradient, biases' gradient)
        both in np.ndarray format.
        """

        previous = []
        for idx, grad in enumerate(grads):
            deltaW = -self.lr * grad[0] + self.momentum * self.previous[idx][0]
            deltaB = -self.lr * grad[1] + self.momentum * self.previous[idx][1]

            self.model.weights[idx] += deltaW
            self.model.bias[idx] += deltaB

            previous.append((deltaW, deltaB))

        self.previous = previous


class AdaGradOptimizer(BaseOptimizer):
    def __init__(
        self,
        model: Model,
        lr: float = 1e-3,
        ep: float = 1e-6
    ):
        self.ep = ep
        self.lr = lr
        self.model = model
        self.cum_sum = [
            [np.zeros_like(model.weights[i]), np.zeros_like(model.bias[i])] for i in range(len(model))
        ]

    def step(self, grads: List):
        """
        Arguments:
        grads: A list of tuples of matrices (weights' gradient, biases' gradient)
        both in np.ndarray format.
        """

        cum_sum = self.cum_sum
        for idx, grad in enumerate(grads):
            cum_sum[idx][0] += grad[0] ** 2
            cum_sum[idx][1] += grad[1] ** 2

            deltaW = self.lr * grad[0] / (np.sqrt(cum_sum[idx][0] + self.ep))
            deltaB = self.lr * grad[1] / (np.sqrt(cum_sum[idx][1] + self.ep))

            self.model.weights[idx] -= deltaW
            self.model.bias[idx] -= deltaB

        self.cum_sum = cum_sum


class AdamOptimizer(BaseOptimizer):
    def __init__(
        self,
        model: Model,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        ep: float = 1e-8
    ):
        self.ep = ep
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.model = model
        self.t = 1

        # First and second momentum cumulative
        self.momentumM = [
            [np.zeros_like(model.weights[i]), np.zeros_like(model.bias[i])] for i in range(len(model))
        ]
        self.momentumV = [
            [np.zeros_like(model.weights[i]), np.zeros_like(model.bias[i])] for i in range(len(model))
        ]

    def step(self, grads: List):
        """
        Arguments:
        grads: A list of tuples of matrices (weights' gradient, biases' gradient)
        both in np.ndarray format.
        """

        for idx, grad in enumerate(grads):
            # Weight
            self.momentumM[idx][0] = self.beta1 * \
                self.momentumM[idx][0] + (1 - self.beta1) * grad[0]
            self.momentumV[idx][0] = self.beta2 * \
                self.momentumV[idx][0] + (1 - self.beta2) * grad[0] ** 2
            m_corr = self.momentumM[idx][0] / (1 - self.beta1 ** self.t)
            v_corr = self.momentumV[idx][0] / (1 - self.beta2 ** self.t)
            self.model.weights[idx] -= self.lr * \
                m_corr / (np.sqrt(v_corr) + self.ep)

            # Bias
            self.momentumM[idx][1] = self.beta1 * \
                self.momentumM[idx][1] + (1 - self.beta1) * grad[1]
            self.momentumV[idx][1] = self.beta2 * \
                self.momentumV[idx][1] + (1 - self.beta2) * grad[1] ** 2
            m_corr = self.momentumM[idx][1] / (1 - self.beta1 ** self.t)
            v_corr = self.momentumV[idx][1] / (1 - self.beta2 ** self.t)
            self.model.bias[idx] -= self.lr * \
                m_corr / (np.sqrt(v_corr) + self.ep)

        self.t += 1
