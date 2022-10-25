from abc import ABC, abstractmethod
import numpy as np


class BaseFunction(ABC):
    @abstractmethod
    def __call__(self, X):
        pass

    @abstractmethod
    def grad(self, X):
        pass


class ReLU(BaseFunction):
    def __call__(self, X: np.ndarray):
        """
        Arguments:
        X: Input data

        Return:
        ReLU output
        """
        return np.maximum(np.zeros_like(X), X)

    def grad(self, X: np.ndarray):
        return np.where(X >= 0, 1, 0)


class LeakyReLU(BaseFunction):
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def __call__(self, X: np.ndarray):
        """
        Arguments:
        X: Input data

        Return:
        LeakyReLU output
        """
        # np.where(X > 0, X, 0.1*X) also this
        return np.maximum(X*self.alpha, X)

    def grad(self, X: np.ndarray):
        return np.where(X >= 0, 1, self.alpha)


class ExpLU(BaseFunction):
    def __init__(self, alpha: float = 1):
        self.alpha = alpha

    def __call__(self, X: np.ndarray):
        """
        Arguments:
        X: Input data

        Return:
        ExpLU output
        """
        return np.where(X >= 0, X, self.alpha * (np.exp(X) - 1))

    def grad(self, X: np.ndarray):
        return np.where(X >= 0, 1, self(X) + self.alpha)


class TanH(BaseFunction):
    def __call__(self, X: np.ndarray):
        """
        Arguments:
        X: Input data

        Return:
        TanH output
        """
        eX = np.exp(-2*X)
        return (2 / (1 + eX)) - 1

    def grad(self, X: np.ndarray):
        tanh = self(X)
        return 1 - (tanh * tanh)


class ArcTan(BaseFunction):
    def __call__(self, X: np.ndarray):
        """
        Arguments:
        X: Input data

        Return:
        ArcTan output
        """
        return np.arctan(X)

    def grad(self, X: np.ndarray):
        return 1 / (X*X + 1)


class Sigmoid(BaseFunction):
    def __call__(self, X: np.ndarray):
        """
        Arguments:
        X: Input data

        Return:
        Sigmoid output
        """
        return 1 / (1 + np.exp(-X))

    def grad(self, X: np.ndarray):
        s = self(X)
        return s*(1 - s)


class Softplus(BaseFunction):
    def __call__(self, X: np.ndarray):
        """
        Arguments:
        X: Input data

        Return:
        Softplus output
        """
        return np.log(1 + np.exp(X))

    def grad(self, X: np.ndarray):
        return 1 / (1 + np.exp(-X))


class Softmax(BaseFunction):
    def __init__(self, axis=1):
        self.axis = axis

    def __call__(self, X: np.ndarray):
        """
        Arguments:
        X: Input data

        Return:
        Softmax output
        """
        eX = np.exp(X - np.max(X))
        return eX / eX.sum(axis=self.axis, keepdims=True)

    def grad(self, X: np.ndarray):
        return 1


class CrossEntropy(BaseFunction):
    def __call__(
        self,
        Y: np.ndarray,
        Y_pred: np.ndarray
    ):
        """
        Arguments:
        Y: Ground-truth labels
        Y_pred: Predicted labels

        Return:
        Cross-Entropy output
        """
        cE = -np.sum(Y * np.log(Y_pred + 1e-8), axis=1)
        return np.mean(cE)

    def grad(
        self,
        Y: np.ndarray,
        Y_pred: np.ndarray
    ):
        return Y_pred - Y


class L2(BaseFunction):
    def __call__(
        self,
        Y: np.ndarray,
        Y_pred: np.ndarray
    ):
        """
        Arguments:
        Y: Ground-truth labels
        Y_pred: Predicted labels

        Return:
        L2 output
        """
        return np.sum((Y - Y_pred) ** 2)

    def grad(
        self,
        Y: np.ndarray,
        Y_pred: np.ndarray
    ):
        return Y_pred - Y
