#Neural Network: Multi-Layer Perceptrons
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def mlp(c):
    iris_dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                                        iris_dataset['target'], random_state=0)
    x = []
    y = []
    for i in range(c):
        reg = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[i+1, i+1],
                            activation='tanh', alpha=1, max_iter=1000) #l2 regularization
        reg.fit(X_train, y_train)
        x.append(reg.score(X_train, y_train))
        y.append(reg.score(X_test, y_test))
    I = [i+1 for i in range(c)]
    plt.figure()
    plt.plot(I, x, 'r', I, y, 'k')
    plt.show()

if __name__=='__main__':
    mlp(int(input('Enter C:')))