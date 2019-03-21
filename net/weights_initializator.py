import numpy as np


class WeightsInitializator:
    def __init__(self, layers):
        # layers передается как ссылка на массив слоев, в self.layers
        # будет хранится таже ссылка, поэтому при изменении self.layers, будет меняться
        # и переданный массив
        self.layers = layers
        # Начальное значение весов смещения
        self.BIAS = 0.01

    def set_weights_matrix(self):
        n_neurons_first = self.layers[0].n_neurons
        n_neurons_prev = n_neurons_first
        # Проходим циклом по всем слоям, где нужно определить матрицы весов,
        # это все слои у которых есть нейроны исключая первый.
        # Обращаемся через индекс, чтобы изменять значение по ссылке
        n_layers = len(self.layers)
        for i in range(1, n_layers):
            if self.layers[i].has_neurons():
                n_neurons_curr = self.layers[i].n_neurons
                # Инициализируем веса,
                # весов всего n_neurons_prev*n_neurons_curr + 1*n_neurons_curr
                # то есть учитывается смещение
                weights, bias = self._init_weights(n_neurons_prev, n_neurons_curr)
                self.layers[i].weights = weights
                self.layers[i].bias = bias
                n_neurons_prev = n_neurons_curr

    def _init_weights(self, width, height):
        # Хорошая инициализация для случая, когда используется функция активации ReLu
        weights = np.random.randn(width, height)
        weights *= np.sqrt(2/weights.size)
        bias = np.zeros(1, height) + self.BIAS
        return weights, bias
