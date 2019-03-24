import numpy as np
from abc import abstractmethod, ABC
from net.learning_rate import Adam, Normal


# Абстрактный метод слоя
class Layer(ABC):
    def __init__(self):
        # Вспомогательная переменная, чтобы предотвратить деление на 0
        self.EPS = 0.001

    # Обязательный метод для прямого прохода
    @abstractmethod
    def forward(self, signal):
        pass

    # Обязательный метод для обратного прохода
    @abstractmethod
    def back(self, signal, ds):
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
        return np.array(signal)

    def back(self, signal, ds):
        return ds * signal

    def has_neurons(self):
        return True


# Полносвязный слой
class Dense(Layer):
    def __init__(self, n_neurons, lr_optimizer="normal"):
        self.n_neurons = n_neurons
        self._bias = []
        self._weights = []
        self._init_optimizer(lr_optimizer)

    def forward(self, signal):
        return np.dot(signal, self.weights) + self.bias

    def back(self, signal, ds):
        dw = np.dot(signal.T, ds)
        db = np.sum(ds, axis=0)
        new_ds = np.dot(ds, self.weights.T)
        # Для обновления весов, берем среднее по батчу
        dw = dw / len(new_ds)
        db = db / len(new_ds)
        return dw, db, new_ds

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

    def _init_optimizer(self, optimizer_type):
        if optimizer_type == "adam":
            self.optimizer = Adam()
        if optimizer_type == "normal":
            self.optimizer = Normal()

    def update_lr(self, gradient, n_epoch):
        dlr = self.optimizer.update(gradient, n_epoch)
        return dlr


class Softmax(Layer):
    def forward(self, signal):
        f = map(lambda x: x/(np.sum(x)), np.exp(signal))
        return list(f)

    def back(self, signal, ds):
        # Запишим сигнал в форме столбца
        gradient = []
        for s, ds_one_sample in zip(signal, ds):
            s = s.reshape(-1, 1)
            f = np.diagflat(s) - np.dot(s, s.T)
            gradient_one_sample = np.dot(f, ds_one_sample)
            gradient.append(gradient_one_sample)
        return np.array(gradient)
        # return signal - ds

    def has_neurons(self):
        return False


# Функции активации:
class ReLu(Layer):
    def forward(self, signal):
        f = signal
        f[signal < 0] = 0
        return f

    def back(self, signal, ds):
        f = signal.copy()
        f[signal > 0] = 1
        f[signal <= 0] = 0
        return f*ds

    def has_neurons(self):
        return False


class Sigmoid(Layer):
    def forward(self, signal):
        f = 1 / (1 + np.exp(-signal))
        return f

    def back(self, signal, ds):
        f = self.forward(signal) * (1 - self.forward(signal))
        return f * ds

    def has_neurons(self):
        return False


# TODO: для каждого типа слоя закрепить абстрактный слой
class LossLayer(Layer):
    def __init__(self):
        self._y = []

    def has_neurons(self):
        return False

    # Обязательный метод для прямого прохода
    @abstractmethod
    def forward(self, signal):
        pass

    # Обязательный метод для обратного прохода
    @abstractmethod
    def back(self, signal):
        pass

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y


class LossSoftmax(LossLayer):
    def forward(self, signal):
        L_sum = np.sum(-1 * (self._y * np.log(signal)))
        L = L_sum / len(signal)
        return L

    def back(self, signal, ds):
        dL = np.array(-1 * np.divide(self._y, signal))
        return ds * dL


class LossMSE(LossLayer):
    def forward(self, signal):
        L = ((signal - self._y)**2).mean()/2
        return L

    def back(self, signal, ds):
        dL = (signal - self._y)
        return dL

