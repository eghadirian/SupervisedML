# Ordinary Least Square
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

def linols():
    x = np.linspace(0, np.pi, 100)
    z = np.random.random(100)
    y = []
    X = []
    [X.append([x[i]**2, np.sin(z[i])])  for i in range(len(x))]
    [y.append([0.1*z[i]+(x[i])]) for i in range(len(x))]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    lr = LinearRegression().fit(X_train, y_train)
    print('Train set score: {}'.format(lr.score(X_train, y_train)))
    print('Test set score: {}'.format(lr.score(X_test, y_test)))

if __name__=='__main__':
    linols()