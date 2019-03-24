import numpy as np
from abc import abstractmethod, ABC


class LROptimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, gradient, n_epoch):
        pass


class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.m = 0
        self.mt = 0
        self.v = 0
        self.vt = 0

    def update(self, ds, t):
        if t == 0:
            t = 1
        self.m = self.b1 * self.m + (1 - self.b1) * ds
        self.mt = self.m / (1 - self.b1 ** t)
        self.v = self.b2 * self.v + (1 - self.b2) * (ds ** 2)
        self.vt = self.v / (1 - self.b2 ** t)
        dlr = self.mt / (np.sqrt(self.vt) + self.eps)
        return dlr


class Normal:
    def update(self, gradient, n_epoch):
        return 1
