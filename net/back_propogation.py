from random import randint
import numpy as np


class BackPropagation:
    def __init__(self, network, l_rate=1, l_rate_decay=0.1, l_rate_decay_n_epoch=20):
        self.net = network
        self.l_rate = l_rate
        self._l_rate_decay = l_rate_decay
        self._l_rate_decay_n_epoch = l_rate_decay_n_epoch
        # Вспомогательная переменная в которой будут храниться
        # промежуточные значения во время обучения
        self.current_signals = []
        self.losses = []
        # номер эпохи
        self.i_epoch = 0

    def fit(self, X, y, n_epoch, batch_size = 100):
        for i in range(n_epoch):
            self.i_epoch = i
            batch_x, batch_y = self._gen_batch(X, y)
            self._forward_propagation(batch_x)
            self._save_loss(batch_y)
            self._back_propagation(batch_y)
            if i % 100 == 0:
                print(f"Epoch {i}: loss = {self.current_signals[-1]}\n")
            if i % self._l_rate_decay_n_epoch:
                self._l_rate_reduce()

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
            # Это условие выполняется в слоях, где требуется обновление весов
            if self.net.layers[i].has_neurons():
                dw, db = layer.back(ds)
                # dlr попровка для оптимизации learning rate
                dlr = layer.update_lr(dw, self.i_epoch)
                layer.weights -= dw * self.l_rate * dlr
                layer.bias -= db * self.l_rate
                ds = dw
            else:
                ds *= layer.back(self.current_signals[i])

    def _gen_batch(self, X, y, batch_size=100):
        batch_x = np.array([])
        batch_y = np.array([])
        n_samples = len(X)
        for _ in range(batch_size):
            i = randint(0, n_samples - 1)
            batch_x = np.append(batch_x, X[i])
            batch_y = np.append(batch_y, y[i])
        return batch_x, batch_y

    def _save_loss(self):
        loss = self.current_signals[-1]
        self.losses.append(loss)

    def _l_rate_reduce(self):
        self.l_rate *= self._l_rate_decay

    def predict(self, X):
        self._forward_propagation(X)
        prediction = self.current_signals[-1]
        return prediction

    def get_loss(self):
        return self.losses

    def get_weights(self):
        return self.net.weights
