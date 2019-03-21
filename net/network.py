import numpy as np


class Network:
    def __init__(self, layers, init=True):
        self.layers = layers



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
