from net.weights_initializator import WeightsInitializator


class Network:
    def __init__(self, layers):
        self.layers = layers
        self.init_weights()
        self.n_layers = len(layers)

    def init_weights(self):
        w_init = WeightsInitializator(self.layers)
        w_init.set_weights_matrix()
