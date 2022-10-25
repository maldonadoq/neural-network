import matplotlib.pyplot as plt
import numpy as np


def plot_history(history):
    """
    Plot loss history
    """
    _, ax = plt.subplots(figsize=(20, 5))
    ax.plot(history['train_loss'], color='#407cdb', label='Train')
    ax.plot(history['val_loss'], color='#db5740', label='Validation')

    ax.legend(loc='upper left')
    handles, labels = ax.get_legend_handles_labels()
    lgd = dict(zip(labels, handles))
    ax.legend(lgd.values(), lgd.keys())

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss along the training')
    plt.show()


def predictOne(model, loader):
    Y_pred = []
    for X in loader:
        X = X.numpy()
        pred = model.forward(X)

        Y_pred += np.argmax(pred, axis=1).tolist()

    return Y_pred


def predictTwo(model, loader):
    Y_real = []
    Y_pred = []
    for batch in loader:
        X, Y = batch
        X = X.numpy()
        Y = Y.numpy()

        pred = model.forward(X)

        Y_pred += np.argmax(pred, axis=1).tolist()
        Y_real += np.argmax(Y, axis=1).tolist()

    return Y_real, Y_pred
