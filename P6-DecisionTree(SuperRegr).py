# Decision Tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def dectree(c):
    x = np.linspace(0, np.pi, 100)
    z = np.random.random(100)
    y = []
    X = []
    [X.append([x[i]**2, np.exp(np.sin(z[i]))])  for i in range(len(x))]
    [y.append([0.1*z[i]+(x[i])]) for i in range(len(x))]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    x=[]
    z=[]
    for i in range(c):
        reg = DecisionTreeRegressor(max_depth=i+1).fit(X_train, y_train)
        x.append(reg.score(X_train, y_train))
        z.append(reg.score(X_test, y_test))
    I = [i+1 for i in range(c)]
    plt.plot(I, x, 'r', I, z, 'k')
    plt.show()

if __name__=='__main__':
    dectree(int(input('Enter max depth of the tree:')))