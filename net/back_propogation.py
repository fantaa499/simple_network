import numpy as np


class BackPropagation:
    def __init__(self, network, l_rate=1):
        self.net = network
        self.l_rate = l_rate
        # Вспомогательная переменная в которой будут храниться
        # промежуточные значения во время обучения
        self.current_signals = []
        self.losses = []

    def fit(self, X, y, n_epoch):
        for i in range(n_epoch):
            self._forward_propagation(X)
            self._save_loss(y)
            self._back_propagation(y)
            if i % 100 == 0:
                print(f"Epoch {i}: loss = {self.current_signals[-1]}\n")

    def _forward_propagation(self, X):
        temp_in = X
        # В current_signals хранятся значения сигналов после каждого слоя,
        # начиная со входа, заканчивая выходом.
        self.current_signals[:] = []
        # Прямое распростронение
        for layer in self.net.layers:
            temp_out = layer.forward(temp_in)
            self.current_signals.append(temp_out)
            temp_in = temp_out

    def _back_propagation(self, y):
        # Последний слой считает ошибку, для нее требуются правильные ответы
        self.net.layers[-1].y = y
        l_indexes = range(self.net.n_layers)
        # Инициализаируем обратно распростроняющийся сигнал ds
        ds = 1
        for i in reversed(l_indexes):
            # layer указывает на тоже место в памяти, что и self.net.layers[i]
            layer = self.net.layers[i]
            if self.net.layers[i].has_neurons():
                dw, db = layer.back(ds)
                layer.weights -= dw * l_rate
                layer.bias -= db * l_rate
                ds = dw
            else:
                ds *= layer.back(self.current_signals[i])

    def _save_loss(self):
        loss = self.current_signals[-1]
        self.losses.append(loss)

    def predict(self, X):
        weights = self.net.weights
        f_activations = self.net.f_activations
        X_with_bias = self.__add_bias_to_X(X)
        self.__forward_propagation(X_with_bias, weights, f_activations)
        prediction = self.current_signals[-1]
        return prediction

    def get_loss(self):
        return self.losses

    def get_weights(self):
        return self.net.weights