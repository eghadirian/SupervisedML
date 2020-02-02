# K-Neighbor Regressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def regressor(n_neighbor):
    x = np.linspace(0, np.pi, 100)
    z = np.random.random(100)
    y = []
    X = []
    [X.append([x[i]])  for i in range(len(x))]
    [y.append([0.1*z[i]+np.sin(x[i])]) for i in range(len(x))]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    x=[]
    z=[]
    for i in range(n_neighbor):
        reg = KNeighborsRegressor(n_neighbors=i+1)
        reg.fit(X_train, y_train)
        x.append(reg.score(X_train, y_train))
        z.append(reg.score(X_test, y_test))
    I = [i for i in range(n_neighbor)]
    plt.plot(I, x, 'r', I, z, 'k')
    plt.show()

if __name__=='__main__':
    regressor(int(input('Enter n_neighbor:')))