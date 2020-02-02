# Ridge
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def linridge(alpha):
    x = np.linspace(0, np.pi, 100)
    z = np.random.random(100)
    y = []
    X = []
    [X.append([x[i]**2, np.exp(np.sin(z[i]))])  for i in range(len(x))]
    [y.append([0.1*z[i]+(x[i])]) for i in range(len(x))]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    x=[]
    z=[]
    for i in range(alpha):
        lr = Ridge(alpha=float(i)).fit(X_train, y_train)
            #L2 Regularization (hyperparameter): as alpha increases model underfits
        x.append(lr.score(X_train, y_train))
        z.append(lr.score(X_test, y_test))
    I = [i for i in range(alpha)]
    plt.plot(I, x, 'r', I, z, 'k')
    plt.show()

if __name__=='__main__':
    linridge(int(input('Enter hyperparameter:')))