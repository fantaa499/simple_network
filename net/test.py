import numpy as np
import random

from net.layer import Dense, Input, ReLu, Softmax, LossSoftmax, LossMSE
from net.network import Network
from net.back_propogation import BackPropagation
import matplotlib.pyplot as plt

layers = [Input(2),
          Dense(3),
          Softmax(),
          LossSoftmax()]

net = Network(layers)
solver = BackPropagation(net)

X = np.array([[-0.1, 1.4],
              [-0.5, 0.2],
              [ 1.3, 0.9],
              [-0.6, 0.4],
              [-1.6, 0.2],
              [ 0.2, 0.2],
              [-0.3,-0.4],
              [ 0.7,-0.8],
              [ 1.1,-1.5],
              [-1.0, 0.9],
              [-0.5, 1.5],
              [-1.3,-0.4],
              [-1.4,-1.2],
              [-0.9,-0.7],
              [ 0.4,-1.3],
              [-0.4, 0.6],
              [ 0.3,-0.5],
              [-1.6,-0.7],
              [-0.5,-1.4],
              [-1.0,-1.4]])

y = np.array([0, 0, 1, 0, 2, 1, 1, 1, 1, 0, 0, 2, 2, 2, 1, 0, 1, 2, 2, 2])
Y = np.eye(3)[y]
print(solver.predict([[-0.1, 1.4]]))
solver.fit(X, Y, n_epoch=40, batch_size=20)

# Построим график ошибки, от эпохи, как мы видим, уже после нескольких десятков эпох,
# алгоритм сошелся
losses = solver.get_loss()
plt.plot(losses)
plt.show()

print(solver.predict([[-1.0, -1.4]]))

