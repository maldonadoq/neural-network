from src.model import Model
from src.activation import CrossEntropy, ReLU, Softmax
from src.optimizer import SGDOptimizer
from src.trainer import Trainer
from src.utils import predictTwo

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import balanced_accuracy_score

import torch


class MultiLabel(Dataset):
    def __init__(self, x, y, nlabels):
        self._x = x
        self._y = y.squeeze()
        self._nlabels = nlabels

    def __len__(self):
        return self._x.shape[0]

    def __getitem__(self, idx):
        _y = [0] * self._nlabels
        _y[self._y[idx]] = 1
        return self._x[idx], torch.Tensor(_y)


iris = datasets.load_iris()
x, y = iris.data, iris.target

x_tmp, x_test, y_tmp, y_test = train_test_split(x, y, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_tmp, y_tmp, test_size=0.1)

train_set = MultiLabel(x_train, y_train, 3)
val_set = MultiLabel(x_val, y_val, 3)
test_set = MultiLabel(x_test, y_test, 3)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=True)

model = Model(
    layers_dims=[4, 64, 16, 8, 3],
    activation_funcs=[ReLU(), ReLU(), ReLU(), Softmax()],
    initialization_method="glorot"
)
opt = SGDOptimizer(model, lr=1e-3)
trainer = Trainer(model, opt, CrossEntropy())
history = trainer.train(5000, train_loader, val_loader)

Ytrain_real, Ytrain_pred = predictTwo(model, train_loader)
Yval_real, Yval_pred = predictTwo(model, val_loader)
Ytest_real, Ytest_pred = predictTwo(model, test_loader)

print("train acc balanced:", balanced_accuracy_score(Ytrain_real, Ytrain_pred))
print("val acc balanced:", balanced_accuracy_score(Yval_real, Yval_pred))
print("test acc balanced:", balanced_accuracy_score(Ytest_real, Ytest_pred))
