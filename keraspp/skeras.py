# kerapp/skerass.py

import matplotlib.pyplot as plt

def plot_loss(history, title=None):
    """
    loss / val_loss 그래프 출력
    """
    # Keras History 객체면 dict로 변환
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'], loc=0)
    plt.show()


def plot_acc(history, title=None):
    """
    accuracy / val_accuracy 그래프 출력
    """
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    if title is not None:
        plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'], loc=0)
    plt.show()
