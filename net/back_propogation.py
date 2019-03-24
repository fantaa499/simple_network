from random import randint
import numpy as np
from net.layer import LossLayer, Input


class BackPropagation:
    def __init__(self, network, ):
        self.net = network
        # Вспомогательная переменная в которой будут храниться
        # промежуточные значения во время обучения
        self.current_signals = []
        self.losses = []
        # номер эпохи
        self.i_epoch = 0

    def fit(self, X, y, n_epoch, batch_size=None, l_rate=1, l_rate_decay=0.1, l_rate_decay_n_epoch=20):
        self.l_rate = l_rate
        self._l_rate_decay = l_rate_decay
        self._l_rate_decay_n_epoch = l_rate_decay_n_epoch
        for i in range(n_epoch):
            self.i_epoch = i
            # Разделим выборку на батчи
            if batch_size is None:
                batch_x = X
                batch_y = y
            else:
                batch_x, batch_y = self._gen_batch(X, y, batch_size)
            # Последний слой считает ошибку, для нее требуются правильные ответы
            self.net.layers[-1].y = batch_y
            self._forward_propagation(batch_x)
            self._save_loss()
            self._back_propagation(batch_y)
            if i % 100 == 0:
                print(f"Epoch {i}: loss = {self.current_signals[-1]}\n")
            if i % self._l_rate_decay_n_epoch == 0:
                self._l_rate_reduce()

    def _forward_propagation(self, X, prediction=False):
        temp_in = X
        # В current_signals хранятся значения сигналов после каждого слоя,
        # начиная со входа, заканчивая выходом.
        self.current_signals[:] = []
        # Прямое распростронение
        for layer in self.net.layers:
            # В случае предсказания, не вычисляем ошибку, так как нет массива ответов
            if prediction and isinstance(layer, LossLayer):
                break
            temp_out = layer.forward(temp_in)
            self.current_signals.append(temp_out)
            temp_in = temp_out

    def _back_propagation(self, y):
        l_indexes = range(self.net.n_layers)
        # Инициализаируем обратно распростроняющийся сигнал ds
        ds = 1
        dws = []
        dbs = []
        for i in reversed(l_indexes):
            # layer указывает на тоже место в памяти, что и self.net.layers[i]
            layer = self.net.layers[i]
            # Это условие выполняется в слоях, где требуется обновление весов
            if self.net.layers[i].has_neurons() and not isinstance(layer, Input):
                dw, db, ds = layer.back(self.current_signals[i-1], ds)
                dw = dw/len(ds)
                db = db/len(ds)
                dws.append(dw)
                dbs.append(db)
            else:
                if isinstance(layer, LossLayer):
                    ds = layer.back(self.current_signals[i-1], ds)
                    continue
                ds = layer.back(self.current_signals[i], ds)
        # dlr попровка для оптимизации learning rate
        for i in l_indexes:
            layer = self.net.layers[i]
            # Это условие выполняется в слоях, где требуется обновление весов
            if layer.has_neurons() and not isinstance(layer, Input):
                dw = dws.pop()
                db = dbs.pop()
                dlr = layer.update_lr(dw, self.i_epoch)
                layer.weights -= dw * self.l_rate * dlr
                layer.bias -= db * self.l_rate

    def _gen_batch(self, X, y, batch_size):
        batch_x = []
        batch_y = []
        n_samples = len(X)
        for _ in range(batch_size):
            i = randint(0, n_samples - 1)
            batch_x.append(X[i])
            batch_y.append(y[i])
        return batch_x, batch_y

    def _save_loss(self):
        loss = self.current_signals[-1]
        self.losses.append(loss)

    def _l_rate_reduce(self):
        self.l_rate *= self._l_rate_decay

    def predict(self, X):
        self._forward_propagation(X, prediction=True)
        prediction = self.current_signals[-1]
        return prediction

    def get_loss(self):
        return self.losses

    def get_weights(self):
        ws = [layer.weights for layer in self.net.layers if layer.has_neurons() and not isinstance(layer, Input)]
        return ws
