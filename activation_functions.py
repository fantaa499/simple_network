import numpy as np


def sigmoid(x, deriv=False):
    if deriv:
        # Производная функции, выглядит так, потому что на проходе вперед
        # уже была вычислена sigmoid и когда мы идем обратно, вместо
        # x придет sigmoid(x)
        f = x * (1 - x)
    else:
        f = 1 / (1 + np.exp(-x))
    return f

def ReLu(x, deriv=False):
    if deriv:
        f = 1 if x>0 else 0
    else:
        f = x if x>0 else 0
    return f
