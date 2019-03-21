import numpy as np
from abc import abstractmethod, ABC


# Абстрактный метод слоя
class Layer(ABC):
    def __init__(self):
        pass

    # Обязательный метод для прямого прохода
    @abstractmethod
    def forward(self, signal):
        pass

    # Обязательный метод для обратного прохода
    @abstractmethod
    def back(self, signal):
        pass

    # Обязательный метод для легкой инициализации матриц весов
    @abstractmethod
    def has_neurons(self):
        pass


# Входной слой
class Input(Layer):
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons

    def forward(self, signal):
        return signal

    def back(self, signal):
        return signal

    def has_neurons(self):
        return True


# Полносвязный слой
class Dense(Layer):
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self._bias = []
        self._weights = []

    # TODO: проверить размерности
    def forward(self, signal):
        return np.dot(signal, self.weights)

    def back(self, signal):
        dw = np.dot(signal, self.weights.T)
        db = signal
        return dw, db

    def has_neurons(self):
        return True

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias


class Softmax(Layer):
    def forward(self, signal):
        f = np.exp(signal) / np.sum(np.exp(signal))
        return f

    def back(self, signal):
        # Запишим сигнал в форме столбца
        s = signal.reshape(-1, 1)
        f = np.diagflat(s) - np.dot(s, s.T)
        return f

    def has_neurons(self):
        return False


# Функции активации:
class ReLu(Layer):
    def forward(self, signal):
        f = signal if signal > 0 else 0
        return f

    def back(self, signal):
        f = 1 if signal > 0 else 0
        return f

    def has_neurons(self):
        return False


class Sigmoid(Layer):
    def forward(self, signal):
        f = 1 / (1 + np.exp(-signal))
        return f

    def back(self, signal):
        f = self.forward(signal) * (1 - self.forward(signal))
        return f

    def has_neurons(self):
        return False


# TODO: для каждого типа слоя закрепить абстрактный слой
class LossSoftmax(Layer):
    def __init__(self):
        self._y = []

    def forward(self, signal):
        L = np.sum(-self._y * np.log(signal))
        return L

    def back(self, signal):
        dL = -self._y / signal
        return dL

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y
