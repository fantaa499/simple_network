import numpy as np


class Network:
    def __init__(self, layers, init=True):
        self.layers = layers
        self.__add_bias()
        self.f_activations = []
        self.n_layers = len(self.layers)
        self.__check_layers_arch()
        self.n_neurons = self.__count_neurons()
        # Инициализируем веса
        self.n_weights = self.__count_weights()
        if init:
            # Инициализируем веса слуачйными значениями
            init_weights = self.__gen_weights(self.n_weights)
            self.__set_weights(init_weights)
        # Запишем функции активации всех слоев, кроме первого и последнего
        for layer in layers:
            f = layer.f_activation
            if f is not None:
                self.f_activations.append(f)

    def __gen_weights(self, size, low=0.1, high=0.5):
        # Создание массива случайных значений от low до high
        return np.random.uniform(low, high, size=(size))

    def __set_weights(self, data):
        # Метод для ининциализации
        # Сеть полносвязная.
        self.weights = []
        prev_pos = 0
        # Так как мы берем последовательные пары слоев, то их  будет
        # на одно меньше чем всё количество слоев
        for i in range(self.n_layers - 1):
            width = self.layers[i].n_neurons
            height = self.layers[i + 1].n_neurons
            cur_pos = prev_pos + width * height
            # В data хранятся значения в виде массива, необходимо его разделить на
            # несколько частей, каждая часть будет относится к двум соседним слоям.
            # в итоге получится массив weights хранящий массивы весов между слоями i
            # и i+1
            temp = data[prev_pos:cur_pos]
            temp = temp.reshape(width, height)
            self.weights.append(temp)
            prev_pos = cur_pos

    def __check_layers_arch(self):
        try:
            if len(self.layers) < 2:
                raise NetArchitectureException
            first_layer = self.layers[0]
            if first_layer.type != "in":
                raise NetArchitectureException
            last_layer = self.layers[-1]
            if last_layer.type != "out":
                raise NetArchitectureException
            # Проверим все слои между первым и последним имеют тип
            # hidden
            for layer in self.layers[1:-1]:
                if layer.type != "hidden":
                    raise NetArchitectureException
        except NetArchitectureException:
            print('Архитектура сети неправильная, приведите ее к следующей:\n' +
                  'in - hidden - ... - hidden - out')

    def __count_neurons(self):
        quantity = 0
        for layer in self.layers:
            quantity += layer.n_neurons
        return quantity

    def __count_weights(self, ):
        quantity = 0
        # Так как мы берем последовательные пары слоев, то их  будет
        # на одно меньше чем всё количество слоев
        for i in range(self.n_layers - 1):
            width = self.layers[i].n_neurons
            height = self.layers[i + 1].n_neurons
            quantity += width*height
        return quantity

    def __add_bias(self):
        # Добавляем смещение - нейрон который не зависит от входа.
        # Он должен быть во всех слоях кроме выходного.
        for i, layer in enumerate(self.layers):
            if layer.type != "out":
                self.layers[i].n_neurons += 1

    def set_weights(self, weights):
        self.weights = weights


class NetArchitectureException(Exception):
    def __init__(self):
        Exception.__init__(self)
