from net_exceptions import HiddenLayerException
from net_exceptions import LayerTypeException


class Layer:
    def __init__(self, n_neurons, type, f_activation=None):
        self.n_neurons = n_neurons
        try:
            if type == "in":
                self.type = "in"
            elif type == "out":
                self.type = "out"
            elif type == "hidden":
                self.type = "hidden"
                if f_activation is None:
                    raise(HiddenLayerException)
            else:
                raise(LayerTypeException)
        except LayerTypeException:
            print("Введите корректный тип слоя, in, out или hidden")
        except HiddenLayerException:
            print("Укажите тип функции активации")

        self.f_activation = f_activation
