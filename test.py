import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from net.layer import Layer
from net.network import Network
from net.activation_functions import sigmoid
from net.back_propogation import BackPropagation

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.9)

# Определим слои:
# - входной, который будет принимать побитовое представление числа.
# - выходной, который выводит четное или нечетное число.
layers = [Layer(n_neurons=784, type="in"),
          Layer(n_neurons=784, type='hidden', f_activation=sigmoid),
          Layer(n_neurons=1, type='out', f_activation=sigmoid)]
# Определим сеть из слоев
net = Network(layers)
# Выберем в качестве оптимизатора, алгоритм обратного распростронения
solver = BackPropagation(net)

# пройдем по всем данным n_epoch раз
solver.fit(X_train, y_train, n_epoch=100)

# Построим график ошибки, от эпохи, как мы видим, уже после нескольких десятков эпох,
# алгоритм сошелся
losses = solver.get_loss()
plt.plot(losses)
plt.show()

y_pred = solver.predict(X_test)
