import numpy as np
import random

from net.layer import Dense, Input, ReLu, Softmax, LossSoftmax, LossMSE, Sigmoid
from net.network import Network
from net.back_propogation import BackPropagation
import matplotlib.pyplot as plt

layers = [Input(2),
          Dense(100),
          ReLu(),
          Dense(3),
          Softmax(),
          LossSoftmax()]

net = Network(layers)
solver = BackPropagation(net)


# Количество сэмплов
n = 100
# 8 битами можно записать числа от 0 до 255
X_raw = [random.randint(0,255) for _ in range(n)]

# Функция перевода числа в битовый вид
def int2bin(x):
    # Бинаризуем числа, первые два символа означают кодировку,
    # поэтому их учистывать не будем
    x_str = bin(x)[2:]
    # Если количество символов не 8, добавим незначащие нули
    n_char = len(x_str)
    if n_char < 8:
        x_str = '0'*(8 - n_char) + x_str
    # Из строки в массив, поэлементно
    x_str_list = list(x_str)
    # Из строки в целое число
    x_int_list = list(map(int, x_str_list))
    return x_int_list

X = list(map(int2bin, X_raw))

# Сгенерируем ответы
# Последний бит указывает на четность
# y = list(map(lambda x: x[7], X))
y = list(map(lambda x: 1 if x%2 == 0 else 0, X_raw))
# 0 - четное, 1 нечетное
# y = np.eye(2)[y]
y =np.array(y).reshape(n,1)


N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j




solver.fit(X, np.eye(3)[y], n_epoch=10000, l_rate=1, l_rate_decay_n_epoch=10000)

# Построим график ошибки, от эпохи, как мы видим, уже после нескольких десятков эпох,
# алгоритм сошелся
losses = solver.get_loss()
plt.plot(losses)
# plt.show()
# print(solver.get_weights())
#
# print(solver.predict([int2bin(23)]))
# print(solver.predict([int2bin(27)]))




h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = solver.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()