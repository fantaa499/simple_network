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
        self._weights = []

    def forward(self, signal):
        return np.dot(self.weights, signal)

    def back(self, signal):
        return np.dot(signal, self.weights.T)

    def has_neurons(self):
        return True

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @weights.deleter
    def weights(self):
        del self.weights


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
