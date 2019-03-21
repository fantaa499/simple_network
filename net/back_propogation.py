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
        for _ in range(n_epoch):
            self._forward_propagation(X)
            self._save_loss(y)
            self._back_propagation(y)

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

    def __back_propagation(self, y):
        # результат храниться в последнем элементе массива
        # current_signals
        corrected_ws = []
        reversed_w = weights[::-1]
        reversed_f = f_activation[::-1]
        reversed_cur_sig = self.current_signals[::-1]
        # Флаг для первой итерации, на ней необходимо вычислить ошибку,
        # зная y
        is_first_iteration = True
        answer = y
        # Обратное распростронение начинается с конца, поэтому используем
        # развернутые массивы
        # Также в массиве reversed_cur_sig на один элемент больше чем в других массивах.
        # Вынесем первый элемент который равен предсказанию на последнем слое
        # в отдельную переменную
        current_sig = reversed_cur_sig[0]
        for w_layer, f, follow_sig in zip(reversed_w, reversed_f, reversed_cur_sig[1:]):
            # temp error показывает как сильно нужно изменять веса
            if is_first_iteration:
                temp_error = answer - current_sig
                is_first_iteration = False
            else:
                # На i шаге мы находимся на i слое с конца и хотим обновить
                # i-е веса. А так как мы начинаем распростронение с i-1 слоя
                # с конца, необходимо использовать i-1 веса
                temp_error = np.dot(temp_delta, w_previous_layer.T)
            # temp delta указывает в какую сторону необходимо изменять веса
            temp_delta = temp_error * f(current_sig, deriv=True)
            # Коррекция весов.
            d_w = np.dot(follow_sig.T, temp_delta)
            # Корректированные веса
            w_corr_temp = w_layer + self.l_rate * d_w
            corrected_ws.append(w_corr_temp)
            # Обновление локальных переменных
            w_previous_layer = w_layer
            current_sig = follow_sig
        # обновляем веса сети, необходимо развернуть
        # массив, так как веса добавлялись в обратном порядке
        # TODO: corrected = net.weights ??
        self.net.set_weights(corrected_ws[::-1])


    def __save_loss(self, y):
        # Метрика - среднеквадратичное отклонение
        loss = ((y - self.current_signals[-1])**2).mean()
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